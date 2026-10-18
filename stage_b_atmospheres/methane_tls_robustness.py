"""
methane_tls_robustness.py
=========================

The pre-registered test of METHANE_TLS_PROTOCOL.md. Does stellar contamination (and haze) make the
CH4 classifier hallucinate methane, and does training with them fix it?

Run (project .venv, after the contaminated grids exist):
    python stage_b_atmospheres/methane_tls_robustness.py

  v1 = the Level 3 models trained on the clean grid (outputs/stage_b_synthetic)
  v3 = the same hyperparameters trained on the contaminated + hazy grid (outputs/stage_b_synthetic_realistic)
  stress set = 800 independent contaminated + hazy spectra (outputs/stage_b_synthetic_stress), never trained on

Output: outputs/stage_b_methane_ext/tls_robustness.json, tls_real_scores.csv, eda/methane_tls_robustness.png
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
import train_synthetic_classifier as l3  # noqa: E402

OUT = RESEARCH / "outputs" / "stage_b_methane_ext"
GRID = {"v1": RESEARCH / "outputs" / "stage_b_synthetic", "v3": RESEARCH / "outputs" / "stage_b_synthetic_realistic"}
STRESS = RESEARCH / "outputs" / "stage_b_synthetic_stress"
CH4 = me.CH4
DIAGNOSTIC = ["TRAPPIST-1 b", "TRAPPIST-1 e", "TOI-260 b"]


def load_grid(folder: Path):
    z = np.load(folder / "synthetic_grid.npz", allow_pickle=True)
    params = pd.read_csv(folder / "params.csv")
    return z["X"].astype(np.float64), z["Y"].astype(int), z["present"].astype(int), z["anchors"].astype(str), \
        z["noise_from"].astype(str), params


def train(X, Y):
    return l3.fit_gbm(X, Y[:, CH4]), l3.train_cnn(X, Y)


def score(models, X):
    gbm, cnn = models
    return {"gbm": l3.predict_gbm(gbm, X), "cnn": cnn(X)[:, CH4]}


def build_any_coverage(planets: list[str]):
    """Stage B preprocessing for diagnostic spectra, keeping any spectrum with >= 1 bin in a CH4 band
    (the coarse TRAPPIST-1 PRISM spectra have fewer than the 3 the labelled tests require)."""
    import stage_b_pipeline as sbp

    X, names = [], []
    for r in sbp.select_spectra(planets).itertuples():
        df, _ = sbp.parse_tbl(sbp.SPEC_DIR / "spectra_tbl" / r.file)
        depth, unc, mask = sbp.resample(df)
        cov = mask > 0
        if not cov.any() or not any(mask[(sbp.CENTERS >= lo) & (sbp.CENTERS <= hi)].sum() for lo, hi in sbp.BANDS["CH4"]):
            continue
        depth = np.where(cov, depth - np.median(depth[cov]), 0.0)
        unc = np.where(cov, np.where(np.isfinite(unc), unc, np.nanmedian(unc[cov])), 0.0)
        X.append(np.stack([depth, unc, mask]))
        names.append(r.pl_name)
    return np.array(X), pd.DataFrame({"planet": names})


def stress_metrics(p: np.ndarray, Y, present, params) -> dict:
    y = Y[:, CH4]
    f = params["tls_f"].fillna(0.0).to_numpy() if "tls_f" in params else np.zeros(len(y))
    no_ch4 = present[:, CH4] == 0
    out = {"roc_auc_all": round(float(roc_auc_score(y, p)), 3),
           "fpr_at_0.5_no_ch4_present": round(float((p[no_ch4] >= 0.5).mean()), 3),
           "fpr_at_0.5_no_ch4_strong_contamination_f>=0.1": round(float((p[no_ch4 & (f >= 0.1)] >= 0.5).mean()), 3),
           "fpr_at_0.5_no_ch4_no_contamination": round(float((p[no_ch4 & (f == 0)] >= 0.5).mean()), 3),
           "n_no_ch4": int(no_ch4.sum()), "n_no_ch4_strong": int((no_ch4 & (f >= 0.1)).sum())}
    cont = f > 0
    if len(np.unique(y[cont])) == 2:
        out["roc_auc_contaminated_only"] = round(float(roc_auc_score(y[cont], p[cont])), 3)
    return out


def lopo_original(grid, real_X, real_planets, real_labels) -> pd.DataFrame:
    """Strict leave-one-planet-out on the original 13 planets (anchored or noise-borrowing samples excluded)."""
    X, Y, _, anchors, noise_from, _ = grid
    rows = []
    for p in sorted(set(real_planets)):
        te = real_planets == p
        if not np.isfinite(real_labels[te]).any():
            continue
        tr = (anchors != p) & (noise_from != p)
        s = score(train(X[tr], Y[tr]), real_X[te])
        rows.append({"planet": p, "label": float(np.nanmax(real_labels[te])), "gbm": float(np.mean(s["gbm"])), "cnn": float(np.mean(s["cnn"]))})
    return pd.DataFrame(rows)


def main() -> int:
    rng = np.random.default_rng(me.SEED)
    grids = {k: load_grid(v) for k, v in GRID.items()}
    stress = load_grid(STRESS)
    models = {k: train(g[0], g[1]) for k, g in grids.items()}
    res = {"protocol": "stage_b_atmospheres/METHANE_TLS_PROTOCOL.md", "stress_test": {}, "real": {}, "diagnostic": {}}

    # ---- H1/H2: the stress test ----
    for k, m in models.items():
        s = score(m, stress[0])
        res["stress_test"][k] = {mdl: stress_metrics(s[mdl], stress[1], stress[2], stress[5]) for mdl in ("gbm", "cnn")}

    # ---- real planets: the 29 new ones (frozen labels, rounds 1 + 2) ----
    lab = m2.frozen_labels()
    Xn, meta, _ = me.build_new_spectra(lab.planet.tolist())
    tab = meta[["planet"]].copy()
    for k, m in models.items():
        s = score(m, Xn)
        tab[f"gbm_{k}"], tab[f"cnn_{k}"] = s["gbm"], s["cnn"]
    new = tab.groupby("planet").mean().reset_index().merge(lab[["planet", "label", "featureless"]], on="planet")
    for k in models:
        res["real"][f"new_{k}"] = me.evaluate(new.rename(columns={f"gbm_{k}": "gbm", f"cnn_{k}": "cnn"}).assign(
            strength=0.0), rng, cols=("gbm", "cnn"))
    new.to_csv(OUT / "tls_real_scores.csv", index=False)

    # ---- the original 9 labelled planets, strict LOPO, for v3 (v1's LOPO scores already exist) ----
    real = np.load(RESEARCH / "outputs" / "stage_b" / "stage_b_dataset.npz", allow_pickle=True)
    orig_v3 = lopo_original(grids["v3"], real["X"], real["planets"].astype(str), real["Y"][:, CH4])
    old = pd.read_csv(RESEARCH / "outputs" / "stage_b_methane_ext" / "all_planet_scores.csv")
    old = old[old.set == "original"][["planet", "label", "gbm", "cnn"]]
    all_v1 = pd.concat([old, new[["planet", "label", "gbm_v1", "cnn_v1"]].rename(columns={"gbm_v1": "gbm", "cnn_v1": "cnn"})])
    all_v3 = pd.concat([orig_v3, new[["planet", "label", "gbm_v3", "cnn_v3"]].rename(columns={"gbm_v3": "gbm", "cnn_v3": "cnn"})])
    res["real"]["all_v1"] = me.evaluate(all_v1.assign(strength=0.0), rng, cols=("gbm", "cnn"))
    res["real"]["all_v3"] = me.evaluate(all_v3.assign(strength=0.0), rng, cols=("gbm", "cnn"))
    orig_v3.to_csv(OUT / "tls_original_lopo_v3.csv", index=False)

    # ---- diagnostics: contamination-dominated spectra with unknown labels ----
    Xd, md = build_any_coverage(DIAGNOSTIC)
    for k, m in models.items():
        s = score(m, Xd)
        d = md.assign(gbm=s["gbm"], cnn=s["cnn"]).groupby("planet")[["gbm", "cnn"]].mean().round(3)
        res["diagnostic"][k] = d.to_dict(orient="index")

    # ---- the pre-registered decision ----
    fpr = {k: res["stress_test"][k]["gbm"]["fpr_at_0.5_no_ch4_present"] for k in models}
    auc = {k: res["real"][f"new_{k}"]["gbm"]["roc_auc"] for k in models}
    adopt = (fpr["v3"] < fpr["v1"]) and (auc["v3"] >= auc["v1"] - 0.03)
    res["decision"] = {"stress_fpr_gbm": fpr, "new_planet_roc_auc_gbm": auc, "v3_becomes_headline": bool(adopt),
                       "rule": "v3 if its stress-test FPR (no-CH4 spectra, P>=0.5) is lower AND its ROC-AUC on the 29 new "
                               "planets is no more than 0.03 below v1's"}
    (OUT / "tls_robustness.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    make_figure(res, new)
    return 0


def make_figure(res, new):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    keys = ["fpr_at_0.5_no_ch4_no_contamination", "fpr_at_0.5_no_ch4_present", "fpr_at_0.5_no_ch4_strong_contamination_f>=0.1"]
    labels = ["no contamination", "all", "strong contamination\n(f >= 0.1)"]
    x = np.arange(len(keys))
    for j, (k, c) in enumerate([("v1", "0.6"), ("v3", "#2e933c")]):
        vals = [res["stress_test"][k]["gbm"][kk] for kk in keys]
        ax.bar(x + (j - 0.5) * 0.35, vals, 0.35, color=c, label=f"{k}: trained on {'clean' if k == 'v1' else 'contaminated + hazy'} spectra")
    ax.set(xticks=x, xticklabels=labels, ylabel="false CH4 rate (P >= 0.5) on spectra WITHOUT CH4",
           title="Stress test: 800 independent contaminated spectra")
    ax.legend(fontsize=7)
    ax = axes[1]
    for k, c in [("v1", "0.6"), ("v3", "#2e933c")]:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(new.label, new[f"gbm_{k}"])
        ax.plot(fpr, tpr, color=c, label=f"{k} GBM: {roc_auc_score(new.label, new[f'gbm_{k}']):.2f}")
    ax.plot([0, 1], [0, 1], ":", color="0.7")
    ax.set(xlabel="false-positive rate", ylabel="true-positive rate", title="The 29 new real JWST planets (frozen labels)")
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[2]
    names = list(res["diagnostic"]["v1"].keys())
    x = np.arange(len(names))
    for j, (k, c) in enumerate([("v1", "0.6"), ("v3", "#2e933c")]):
        vals = [res["diagnostic"][k][n]["gbm"] for n in names]
        ax.bar(x + (j - 0.5) * 0.35, vals, 0.35, color=c, label=k)
        for xi, v in zip(x + (j - 0.5) * 0.35, vals):
            ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=7)
    ax.set(xticks=x, xticklabels=names, ylabel="P(CH4), GBM", ylim=(0, 1),
           title="Diagnostic spectra (labels unknown or flat)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESEARCH / "eda" / "methane_tls_robustness.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    if "--figure-only" in sys.argv:          # redraw from the saved results, no retraining
        make_figure(json.loads((OUT / "tls_robustness.json").read_text()), pd.read_csv(OUT / "tls_real_scores.csv"))
        raise SystemExit(0)
    raise SystemExit(main())
