"""
methane_round2.py
=================

Round 2 of the pre-registered methane test (METHANE_EXTENSION_PROTOCOL_ROUND2.md).
- Adds the planets whose full papers state a CH4 result, mostly feature-rich giants without CH4.
- Asks whether the result survives a model trained on a second, independent synthetic grid.

Run (project .venv, after the seed-2 grid exists):
    python stage_b_atmospheres/methane_round2.py

Output: outputs/stage_b_methane_ext/round2_{scores.csv, metrics.json}, eda/methane_round2.png
"""

from __future__ import annotations

import hashlib
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
import train_synthetic_classifier as l3  # noqa: E402

OUT = RESEARCH / "outputs" / "stage_b_methane_ext"
LAB_DIR = RESEARCH / "outputs" / "labels"
GRIDS = {"grid_A (seed 20261017)": RESEARCH / "outputs" / "stage_b_synthetic" / "synthetic_grid.npz",
         "grid_B (seed 7770001)": RESEARCH / "outputs" / "stage_b_synthetic_seed2" / "synthetic_grid.npz"}


def frozen_labels() -> pd.DataFrame:
    rec = json.loads((OUT / "protocol_hashes.json").read_text())
    frames = []
    for name, rnd in [("ch4_labels_extended.csv", 1), ("ch4_labels_round2.csv", 2)]:
        p = LAB_DIR / name
        if hashlib.sha256(p.read_bytes()).hexdigest() != rec[name]:
            raise SystemExit(f"{name} changed after it was frozen - the pre-registered test is void")
        frames.append(pd.read_csv(p, dtype=str, keep_default_na=False).assign(round=rnd))
    lab = pd.concat(frames)
    lab = lab[lab.CH4.isin(["0", "1"])].drop_duplicates("planet", keep="last")   # round 2 only fills round-1 unknowns
    return lab.rename(columns={"CH4": "label"}).assign(label=lambda d: d.label.astype(int),
                                                       featureless=lambda d: d.featureless.astype(int))


def main() -> int:
    rng = np.random.default_rng(me.SEED)
    lab = frozen_labels()
    Xn, meta, _ = me.build_new_spectra(lab.planet.tolist())
    strength = l3.feature_strength(Xn)
    per_grid, scores = {}, None
    for gname, gpath in GRIDS.items():
        syn = np.load(gpath, allow_pickle=True)
        Xs, Ys = syn["X"].astype(np.float64), syn["Y"].astype(int)
        gbm = l3.fit_gbm(Xs, Ys[:, me.CH4])
        cnn = l3.train_cnn(Xs, Ys)
        spec = meta.assign(gbm=l3.predict_gbm(gbm, Xn), cnn=cnn(Xn)[:, me.CH4], strength=strength)
        pl = spec.groupby("planet").agg(gbm=("gbm", "mean"), cnn=("cnn", "mean"), strength=("strength", "mean")).reset_index()
        pl = pl.merge(lab[["planet", "label", "featureless", "round", "source"]], on="planet")
        strong = pl.strength > pl.strength.median()
        per_grid[gname] = {
            "primary_all_new_labelled": me.evaluate(pl, rng),
            "round2_planets_only": me.evaluate(pl[pl["round"] == 2], rng),
            "confound_test_strong_feature_half": {"strength_threshold": round(float(pl.strength.median()), 3),
                                                  **me.evaluate(pl[strong], rng)},
            "no_featureless": me.evaluate(pl[pl.featureless == 0], rng),
        }
        tag = "A" if gname.startswith("grid_A") else "B"
        pl = pl.rename(columns={"gbm": f"gbm_{tag}", "cnn": f"cnn_{tag}"})
        scores = pl if scores is None else scores.merge(pl[["planet", f"gbm_{tag}", f"cnn_{tag}"]], on="planet")
    a, b = scores.gbm_A.to_numpy(), scores.gbm_B.to_numpy()
    agree = {"spearman_gbm_A_vs_B": round(float(pd.Series(a).corr(pd.Series(b), method="spearman")), 3),
             "max_abs_diff_gbm": round(float(np.max(np.abs(a - b))), 3)}
    metrics = {"protocol": "stage_b_atmospheres/METHANE_EXTENSION_PROTOCOL_ROUND2.md",
               "n_labelled_new": int(len(lab)), "n_scorable": int(len(scores)),
               "by_grid": per_grid, "grid_agreement": agree}
    scores.to_csv(OUT / "round2_scores.csv", index=False)
    (OUT / "round2_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    pd.set_option("display.width", 220)
    print(scores.sort_values("gbm_A", ascending=False)[["planet", "round", "label", "featureless", "gbm_A", "gbm_B", "cnn_A", "cnn_B", "strength"]]
          .round(3).to_string(index=False))
    make_figure(scores)
    return 0


def make_figure(s: pd.DataFrame):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    for lab, c, name in [(1, "#d1495b", "CH4 detected"), (0, "#2a6fdb", "CH4 not detected")]:
        d = s[s.label == lab]
        ax.scatter(d.strength, d.gbm_A, color=c, s=40, label=name, edgecolor="k", linewidth=0.4)
    for _, r in s.iterrows():
        if r.strength > 4 or r.label == 1:
            ax.annotate(r.planet, (r.strength, r.gbm_A), fontsize=6.5, xytext=(3, 2), textcoords="offset points")
    ax.set(xscale="log", xlabel="gas-blind feature strength (reduced chi^2 vs flat)", ylabel="P(CH4), GBM",
           title="Strong features do not imply CH4 to the model")
    ax.axvline(s.strength.median(), color="0.7", ls=":", lw=1)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.scatter(s.gbm_A, s.gbm_B, c=np.where(s.label == 1, "#d1495b", "#2a6fdb"), s=35, edgecolor="k", linewidth=0.4)
    ax.plot([0, 1], [0, 1], ":", color="0.6")
    auc_a, auc_b = roc_auc_score(s.label, s.gbm_A), roc_auc_score(s.label, s.gbm_B)
    ax.set(xlabel="P(CH4), trained on grid A", ylabel="P(CH4), trained on independent grid B",
           title=f"Two independent training grids (ROC-AUC {auc_a:.2f} vs {auc_b:.2f})")
    fig.tight_layout()
    fig.savefig(RESEARCH / "eda" / "methane_round2.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())
