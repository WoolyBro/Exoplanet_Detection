"""
methane_specnet.py
==================

SpecNet, the strength-blind CNN + BiLSTM CH4 detector pre-registered in METHANE_SPECNET_PROTOCOL.md.

Input (per spectrum): the noise-whitened residual about its weighted mean, divided by its own RMS (exactly
sqrt of the gas-blind "feature strength"), the coverage mask, and the noise pattern log10(sigma / median sigma).
The network therefore cannot see how strong a spectrum is overall, only where its excess absorption lies.

    python stage_b_atmospheres/methane_specnet.py           # project .venv; about 30-45 min (LOPO trains 45 nets)
Output: outputs/stage_b_methane_ext/{specnet.json, specnet_scores.csv}, eda/methane_specnet.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(HERE))
import methane_extension as me  # noqa: E402
import methane_round2 as m2  # noqa: E402
import methane_round3 as m3  # noqa: E402
import methane_tls_robustness as tr  # noqa: E402
import train_synthetic_classifier as l3  # noqa: E402

OUT = RESEARCH / "outputs" / "stage_b_methane_ext"
SEEDS = [42, 1, 2, 7, 11]
CH4 = me.CH4


def strength_blind(X: np.ndarray) -> np.ndarray:
    """(n, 3, bins): [whitened residual / its RMS, mask, log10(sigma / median sigma)]."""
    d, u, m = X[:, 0], X[:, 1], X[:, 2]
    out = np.zeros_like(X, dtype=np.float32)
    for i in range(len(X)):
        c = m[i] > 0
        if c.sum() < 2:
            continue
        w = 1 / u[i, c] ** 2
        mu = np.sum(d[i, c] * w) / np.sum(w)
        r = (d[i, c] - mu) / u[i, c]
        rms = np.sqrt(np.mean(r ** 2))
        out[i, 0, c] = r / max(rms, 1e-6)
        out[i, 1] = m[i]
        out[i, 2, c] = np.clip(np.log10(u[i, c] / np.median(u[i, c])), -2, 2)
    return out


def build_net(seed: int):
    import torch
    from torch import nn
    torch.manual_seed(seed)

    class SpecNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv1d(3, 32, 5, padding=2), nn.ReLU(), nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(),
                                      nn.MaxPool1d(2), nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2))
            self.lstm = nn.LSTM(64, 32, batch_first=True, bidirectional=True)
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 5))

        def forward(self, x):
            h = self.conv(x).transpose(1, 2)
            out, _ = self.lstm(h)
            return self.head(out.mean(dim=1))
    return SpecNet()


def train_specnet(X: np.ndarray, Y: np.ndarray, seed: int, epochs: int = 40):
    import torch
    from torch import nn
    torch.set_num_threads(2)
    net = build_net(seed)
    rng = np.random.default_rng(seed)
    y = torch.tensor(Y, dtype=torch.float32)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        net.train()
        Xj = X.copy()
        Xj[:, 0] = X[:, 0] + rng.normal(0, 1, X[:, 0].shape) * X[:, 1] * 0.3      # noise jitter, before whitening
        x = torch.tensor(strength_blind(Xj))
        for idx in torch.randperm(len(x), generator=torch.Generator().manual_seed(int(rng.integers(1 << 31)))).split(64):
            opt.zero_grad()
            loss_fn(net(x[idx]), y[idx]).backward()
            opt.step()
    net.eval()

    def predict(Xp):
        with torch.no_grad():
            return torch.sigmoid(net(torch.tensor(strength_blind(Xp)))).numpy()
    return predict


def ensemble(X, Y):
    nets = [train_specnet(X, Y, s) for s in SEEDS]
    return lambda Xp: np.mean([n(Xp) for n in nets], axis=0)[:, CH4]


def main() -> int:
    rng = np.random.default_rng(me.SEED)
    X, Y, present, anchors, noise_from, params = tr.load_grid(tr.GRID["v3"])
    stress = tr.load_grid(tr.STRESS)
    model = ensemble(X, Y)
    res = {"protocol": "stage_b_atmospheres/METHANE_SPECNET_PROTOCOL.md"}
    res["stress_test"] = tr.stress_metrics(model(stress[0]), stress[1], stress[2], stress[5])

    # rounds 1 + 2: the 29 planets
    lab = m2.frozen_labels()
    Xn, meta, _ = me.build_new_spectra(lab.planet.tolist())
    spec = meta.assign(specnet=model(Xn), strength=l3.feature_strength(Xn))
    pl = spec.groupby("planet").agg(specnet=("specnet", "mean"), strength=("strength", "mean")).reset_index()
    pl = pl.merge(lab[["planet", "label", "featureless"]], on="planet")
    old = pd.read_csv(OUT / "tls_real_scores.csv")[["planet", "gbm_v3", "cnn_v3"]]
    pl = pl.merge(old, on="planet", how="left")
    strong = pl.strength > pl.strength.median()
    ev = lambda d: me.evaluate(d.rename(columns={"gbm_v3": "gbm", "cnn_v3": "cnn"}), rng, cols=("specnet", "gbm", "cnn", "strength"))
    res["new_29"] = ev(pl)
    res["confound_test_strong_feature_half"] = {"strength_threshold": round(float(pl.strength.median()), 3), **ev(pl[strong])}

    # the original 9: strict leave-one-planet-out
    real = np.load(RESEARCH / "outputs" / "stage_b" / "stage_b_dataset.npz", allow_pickle=True)
    rX, rp, rY = real["X"], real["planets"].astype(str), real["Y"][:, CH4]
    orig = []
    for p in sorted(set(rp)):
        te = rp == p
        if not np.isfinite(rY[te]).any():
            continue
        trn = (anchors != p) & (noise_from != p)
        orig.append({"planet": p, "label": float(np.nanmax(rY[te])), "specnet": float(np.mean(ensemble(X[trn], Y[trn])(rX[te]))),
                     "strength": float(np.mean(l3.feature_strength(rX[te])))})
        print(f"  LOPO {p}: {orig[-1]['specnet']:.3f} (label {orig[-1]['label']:.0f})", flush=True)
    orig = pd.DataFrame(orig).merge(pd.read_csv(OUT / "tls_original_lopo_v3.csv")[["planet", "gbm", "cnn"]]
                                    .rename(columns={"gbm": "gbm_v3", "cnn": "cnn_v3"}), on="planet", how="left")

    # round 3 (labelled planets only)
    lab3 = m3.frozen()
    lab3 = lab3[lab3.CH4.isin(["0", "1"])]
    X3, meta3 = m3.spectra(lab3.planet.tolist())
    r3 = pd.DataFrame()
    if len(X3):
        s3 = meta3[meta3.scored].reset_index(drop=True).assign(specnet=model(X3), strength=l3.feature_strength(X3))
        r3 = s3.groupby("planet").agg(specnet=("specnet", "mean"), strength=("strength", "mean")).reset_index()
        r3 = r3.merge(lab3.assign(label=lab3.CH4.astype(float))[["planet", "label"]], on="planet")
        prev = json.loads((OUT / "round3.json").read_text())["planets"]
        r3["gbm_v3"] = r3.planet.map(lambda p: prev.get(p, {}).get("gbm_v3"))
    allp = pd.concat([orig.assign(set="original (LOPO)"), pl.assign(set="rounds 1-2"), r3.assign(set="round 3")], ignore_index=True)
    res["all_labelled"] = me.evaluate(allp.assign(gbm=allp.gbm_v3, cnn=allp.cnn_v3.fillna(allp.gbm_v3)), rng, cols=("specnet", "gbm"))
    res["n_all_labelled"] = int(len(allp))
    res["min_ch4_score_specnet"] = float(allp[allp.label == 1].specnet.min())
    res["max_nonch4_score_specnet"] = float(allp[allp.label == 0].specnet.max())
    sp = lambda c: round(float(allp[[c, "strength"]].dropna().corr(method="spearman").iloc[0, 1]), 3)
    res["spearman_with_feature_strength"] = {"specnet": sp("specnet"), "cnn_v3": sp("cnn_v3"), "gbm_v3": sp("gbm_v3")}
    allp.to_csv(OUT / "specnet_scores.csv", index=False)

    s1 = res["new_29"]["specnet"]["roc_auc"] >= 0.95
    ch = res["confound_test_strong_feature_half"]["specnet"]
    s2 = ch is not None and ch["roc_auc"] >= 0.90 and ch["perm_p"] < 0.05
    s3 = res["all_labelled"]["specnet"]["roc_auc"] >= 0.95
    s4 = res["stress_test"]["fpr_at_0.5_no_ch4_present"] <= 0.03
    s5 = abs(res["spearman_with_feature_strength"]["specnet"]) < abs(res["spearman_with_feature_strength"]["cnn_v3"])
    res["expectations"] = {"S1 new-29 ROC-AUC >= 0.95": bool(s1), "S2 confound half >= 0.90, p < 0.05": bool(s2),
                           "S3 all labelled >= 0.95": bool(s3), "S4 stress false-CH4 <= 3 %": bool(s4),
                           "S5 less tied to feature strength than the v3 CNN": bool(s5)}
    res["decision"] = "SpecNet reported as a CH4 detector alongside the GBM" if (s1 and s2 and s3 and s4) else \
        "SpecNet failed a pre-registered criterion; the GBM stays the headline"
    (OUT / "specnet.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in ["expectations", "decision", "spearman_with_feature_strength", "stress_test"]}, indent=2, default=float))
    print("new 29:", res["new_29"]["specnet"], "| confound half:", ch, "| all:", res["all_labelled"]["specnet"])
    make_figure(allp, res)
    return 0


def make_figure(allp: pd.DataFrame, res: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for c, col, lab in [("specnet", "#2a78d6", "SpecNet (CNN+BiLSTM, strength-blind)"), ("gbm_v3", "#8a8984", "GBM v3"), ("cnn_v3", "#eb6834", "CNN v3 (not strength-blind)")]:
        d = allp.dropna(subset=[c])
        f, t, _ = roc_curve(d.label, d[c])
        ax.plot(f, t, color=col, lw=2, label=f"{lab}: {roc_auc_score(d.label, d[c]):.3f} (n={len(d)})")
    ax.plot([0, 1], [0, 1], ":", color="0.7")
    ax.set(xlabel="false-positive rate", ylabel="true-positive rate", title="All labelled real JWST planets (frozen labels)")
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1]
    for c, col, lab in [("specnet", "#2a78d6", "SpecNet"), ("cnn_v3", "#eb6834", "CNN v3")]:
        d = allp.dropna(subset=[c])
        ax.scatter(d.strength, d[c], c=np.where(d.label == 1, col, "none"), edgecolors=col, s=40,
                   label=f"{lab} (filled = CH4); Spearman with strength {res['spearman_with_feature_strength'][c if c != 'specnet' else 'specnet']:+.2f}")
    ax.set(xscale="log", xlabel="feature strength (gas-blind, reduced chi^2 about flat)", ylabel="P(CH4)",
           title="Does the score follow CH4, or just how strong the spectrum is?")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(RESEARCH / "eda" / "methane_specnet.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())
