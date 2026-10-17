"""
train_synthetic_classifier.py
=============================

Stage B, Level 3: can a molecule classifier trained ONLY on physically simulated JWST spectra
recognise the gases that the literature reports in REAL spectra?

Stage B Level 1 trained on the 53 real spectra (13 planets) and had no skill above a constant. Its
models had too few examples to learn from. Here the training data are unlimited and labelled by
physics (synthetic_grid.py: petitRADTRANS spectra with the real spectra's coverage and noise; a gas
is labelled present only if it is DETECTABLE at that noise). The real spectra are used only as the
test set.

Run (project .venv, after synthetic_grid.py): python stage_b_atmospheres/train_synthetic_classifier.py

Protocol
  * Strict leave-one-planet-out. To score real planet P, every synthetic sample that was anchored
    on P, or that borrowed the coverage and noise of one of P's spectra, is removed from training.
    So the model has never seen P's parameters or its noise pattern. (Without this rule, the
    synthetic twins of P would make the test easier than it is.)
  * Labels: the verified literature labels (outputs/labels/verified_gas_labels.csv, via the Stage B
    dataset). "unknown" labels are not scored.
  * Models:
      - GBM: gradient-boosted trees on depth, uncertainty and mask.
      - CNN: a 1-D convolutional network, the same input as Stage B, with masked multi-label BCE.
    Both are compared with the majority-class baseline and with Stage B Level 1 on the same planets.
  * Planet-level score = mean over that planet's spectra (as in Stage B Level 1).
  * In-domain check: the same models scored on the held-out SYNTHETIC samples of planet P. The gap
    between in-domain and real performance is the simulation-to-reality gap.

Output: outputs/stage_b_synthetic/{level3_predictions.csv, level3_metrics.csv, level3_summary.json}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
SYN = RESEARCH / "outputs" / "stage_b_synthetic"
REAL = RESEARCH / "outputs" / "stage_b" / "stage_b_dataset.npz"
DEPTH_SCALE = 1000.0          # ppm -> numerics, as in Stage B
SEED = 42


def features(X: np.ndarray) -> np.ndarray:
    d, u, m = X[:, 0] / DEPTH_SCALE, X[:, 1] / DEPTH_SCALE, X[:, 2]
    snr = np.where(m > 0, X[:, 0] / np.maximum(X[:, 1], 1e-6), 0.0)
    return np.concatenate([d, u, m, snr], axis=1)


def fit_gbm(Xtr, ytr):
    if len(np.unique(ytr)) < 2:
        return float(ytr.mean())
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, random_state=SEED).fit(features(Xtr), ytr)


def predict_gbm(model, X):
    if isinstance(model, float):
        return np.full(len(X), model)
    return model.predict_proba(features(X))[:, 1]


def train_cnn(Xtr, Ytr, epochs=40):
    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    x = torch.tensor(np.stack([Xtr[:, 0] / DEPTH_SCALE, Xtr[:, 1] / DEPTH_SCALE, Xtr[:, 2]], 1), dtype=torch.float32)
    y = torch.tensor(Ytr, dtype=torch.float32)
    net = nn.Sequential(nn.Conv1d(3, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                        nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                        nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(4), nn.Flatten(),
                        nn.Dropout(0.3), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, Ytr.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(SEED)
    for _ in range(epochs):
        net.train()
        for idx in torch.randperm(len(x), generator=g).split(64):
            xb = x[idx]
            xb = torch.cat([xb[:, :1] + torch.randn_like(xb[:, :1]) * xb[:, 1:2] * 0.3, xb[:, 1:]], 1)  # noise jitter
            opt.zero_grad()
            loss_fn(net(xb), y[idx]).backward()
            opt.step()
    net.eval()

    def predict(X):
        with torch.no_grad():
            t = torch.tensor(np.stack([X[:, 0] / DEPTH_SCALE, X[:, 1] / DEPTH_SCALE, X[:, 2]], 1), dtype=torch.float32)
            return torch.sigmoid(net(t)).numpy()
    return predict


def feature_strength(X: np.ndarray) -> np.ndarray:
    """Gas-BLIND confound baseline: how far each spectrum departs from flat, in units of its own noise
    (reduced chi^2 about the weighted mean). A model that only sensed 'big features' - which giant
    planets have and small planets lack - would score like this for every gas."""
    out = np.zeros(len(X))
    for i, (d, u, m) in enumerate(X):
        c = m > 0
        w = 1 / u[c] ** 2
        mu = np.sum(d[c] * w) / np.sum(w)
        out[i] = np.mean(((d[c] - mu) / u[c]) ** 2)
    return out


def auc_permutation_p(y, p, n_perm=100_000, seed=SEED) -> float:
    """One-sided p-value of the ROC-AUC under random relabelling of the planets (Monte Carlo)."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    if len(np.unique(y)) < 2:
        return np.nan
    obs = roc_auc_score(y, p)
    ranks = pd.Series(p).rank().to_numpy()
    n1, n0 = y.sum(), len(y) - y.sum()
    perm = np.random.default_rng(seed).permuted(np.tile(y, (n_perm, 1)), axis=1)
    a = ((perm * ranks).sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float((a >= obs - 1e-12).mean())


def auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan


def main() -> int:
    syn = np.load(SYN / "synthetic_grid.npz", allow_pickle=True)
    Xs, Ys = syn["X"].astype(np.float64), syn["Y"].astype(int)
    anchors, noise_from = syn["anchors"].astype(str), syn["noise_from"].astype(str)
    mols = list(syn["molecules"].astype(str))
    real = np.load(REAL, allow_pickle=True)
    Xr, Yr, planets = real["X"], real["Y"], real["planets"].astype(str)
    assert list(real["molecules"].astype(str)) == mols

    rows, insyn = [], []
    for p in sorted(set(planets)):
        train = (anchors != p) & (noise_from != p)
        test_syn = ~train                     # every synthetic sample held out for P
        te = planets == p
        cnn = train_cnn(Xs[train], Ys[train])
        pc_real, pc_syn = cnn(Xr[te]), cnn(Xs[test_syn]) if test_syn.any() else None
        for j, g in enumerate(mols):
            gbm = fit_gbm(Xs[train], Ys[train, j])
            pg = predict_gbm(gbm, Xr[te])
            for k, i in enumerate(np.flatnonzero(te)):
                rows.append({"planet": p, "spectrum": int(i), "molecule": g, "label": Yr[i, j],
                             "gbm_prob": float(pg[k]), "cnn_prob": float(pc_real[k, j]),
                             "strength": float(feature_strength(Xr[[i]])[0]),
                             "train_pos_rate": float(Ys[train, j].mean())})
            if test_syn.any():
                insyn.append({"planet": p, "molecule": g, "n": int(test_syn.sum()),
                              "y": Ys[test_syn, j].tolist(), "gbm": predict_gbm(gbm, Xs[test_syn]).tolist(),
                              "cnn": pc_syn[:, j].tolist()})
        print(f"  {p}: {te.sum()} real spectra, {train.sum()} synthetic training samples", flush=True)

    pred = pd.DataFrame(rows)
    pred.to_csv(SYN / "level3_predictions.csv", index=False)
    known = pred[pred.label.notna()]

    metrics = []
    for g in mols:
        k = known[known.molecule == g]
        planet = k.groupby("planet").agg(label=("label", "first"), gbm=("gbm_prob", "mean"), cnn=("cnn_prob", "mean"),
                                         strength=("strength", "mean"), prior=("train_pos_rate", "mean"))
        for level, d in [("planet", planet.reset_index()), ("spectrum", k.rename(columns={"gbm_prob": "gbm", "cnn_prob": "cnn"}))]:
            if d.empty:
                continue
            y = d.label.to_numpy().astype(int)
            maj = float(y.mean() >= 0.5)       # majority of the TEST labels: an optimistic baseline
            for model in ("gbm", "cnn", "strength"):
                p_ = d[model].to_numpy()
                prob = model != "strength"
                metrics.append({"molecule": g, "level": level, "model": model, "n": len(y), "n_pos": int(y.sum()),
                                "roc_auc": auc(y, p_),
                                "perm_p": auc_permutation_p(y, p_) if level == "planet" else np.nan,
                                "accuracy_at_0.5": float(((p_ >= 0.5) == y).mean()) if prob else np.nan,
                                "brier": float(np.mean((p_ - y) ** 2)) if prob else np.nan,
                                "majority_accuracy": float((y == maj).mean()),
                                "majority_brier": float(np.mean((y.mean() - y) ** 2))})
    met = pd.DataFrame(metrics)
    stage_b = pd.read_csv(RESEARCH / "outputs" / "stage_b" / "stage_b_metrics.csv")
    stage_b = stage_b[stage_b.model.isin(["gbm", "cnn"])][["molecule", "level", "model", "roc_auc", "accuracy"]] \
        .rename(columns={"roc_auc": "level1_roc_auc", "accuracy": "level1_accuracy"})
    met = met.merge(stage_b, on=["molecule", "level", "model"], how="left")
    met.to_csv(SYN / "level3_metrics.csv", index=False)

    ins = {}
    for g in mols:
        y = np.concatenate([np.array(r["y"]) for r in insyn if r["molecule"] == g])
        for model in ("gbm", "cnn"):
            p_ = np.concatenate([np.array(r[model]) for r in insyn if r["molecule"] == g])
            ins.setdefault(g, {})[model] = round(auc(y, p_), 3)
    summary = {"n_synthetic": int(len(Xs)), "n_real_spectra": int(len(Xr)), "n_real_planets": int(len(set(planets))),
               "in_domain_synthetic_roc_auc_heldout_planet": ins,
               "real_planet_level": met[met.level == "planet"].round(3).to_dict(orient="records")}
    (SYN / "level3_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    pd.set_option("display.width", 200)
    print(met.round(3).to_string(index=False))
    print("in-domain synthetic ROC-AUC (held-out planet):", json.dumps(ins))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
