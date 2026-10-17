"""
report_v2.py
============

Figure for the final Stage A evaluation on split v2 (evaluate_v2.py). Reads only the saved test
predictions and final_eval_v2.json; it never re-scores test.

Run: python stage_a_transit_model/report_v2.py   ->   figures/13_stage_a_v2_final.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

R = Path(__file__).resolve().parents[1]
FINAL = R / "stage_a_transit_model" / "runs" / "v2_final"


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ev = json.loads((FINAL / "final_eval_v2.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colours = {"kepler": "#2a6fdb", "tess": "#d1495b"}

    ax = axes[0]
    for m, c in colours.items():
        p = pd.read_csv(FINAL / f"{m}_test_predictions.csv")
        y = (p.true_label != "FALSE POSITIVE").astype(int)
        fpr, tpr, _ = roc_curve(y, p.planet_like_score)
        b = ev[m]["binary"]
        arch = ev[m]["decision"]["family"].replace(f"{m}_", "")
        ax.plot(fpr, tpr, color=c, label=f"{m.capitalize()} ({arch}): {b['roc_auc']:.3f} [{b['roc_auc_ci'][0]:.3f}, {b['roc_auc_ci'][1]:.3f}]")
        cf = p[p.true_label != "CANDIDATE"]
        fpr, tpr, _ = roc_curve((cf.true_label == "CONFIRMED").astype(int), cf.planet_like_score)
        ax.plot(fpr, tpr, color=c, ls=":", label=f"  confirmed vs FP only: {ev[m]['confirmed_vs_fp']['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], color="0.8", lw=0.8)
    ax.set(xlabel="false-positive rate", ylabel="true-positive rate", title="Test set, read once (leakage-free split v2)")
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1]
    bins = ["0-7.1", "7.1-15", "15-50", "50-inf"]
    x = np.arange(len(bins))
    for k, (m, c) in enumerate(colours.items()):
        d = ev[m]["by_depth_snr"]
        auc = [d[b]["roc_auc"] for b in bins]
        lo = [d[b]["roc_auc_ci"][0] if d[b]["roc_auc_ci"][0] == d[b]["roc_auc_ci"][0] else np.nan for b in bins]
        hi = [d[b]["roc_auc_ci"][1] if d[b]["roc_auc_ci"][1] == d[b]["roc_auc_ci"][1] else np.nan for b in bins]
        err = np.array([np.subtract(auc, lo), np.subtract(hi, auc)])
        ax.errorbar(x + (k - 0.5) * 0.15, auc, yerr=np.nan_to_num(err), fmt="o-", color=c, capsize=3,
                    label=f"{m.capitalize()} (n per bin: {', '.join(str(d[b]['n']) for b in bins)})")
    ax.axhline(0.5, color="0.8", lw=0.8)
    ax.set(xticks=x, xticklabels=bins, xlabel="transit depth S/N (pipeline-measured)", ylabel="ROC-AUC (95 % star bootstrap)",
           title="Performance is set by signal strength", ylim=(0.4, 1.0))
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[2]
    labels, ens, gbm, logi = [], [], [], []
    for m in colours:
        labels.append(m.capitalize())
        ens.append(ev[m]["binary"]["roc_auc"])
        gbm.append(ev[m]["baselines"]["shape_gbm"]["roc_auc"])
        logi.append(ev[m]["baselines"]["logistic_views"]["roc_auc"])
    x = np.arange(len(labels))
    ax.bar(x - 0.25, logi, 0.25, color="0.75", label="logistic regression on views")
    ax.bar(x, gbm, 0.25, color="0.5", label="GBM on transit-shape features")
    ax.bar(x + 0.25, ens, 0.25, color="#2e933c", label="CNN ensemble (5 seeds)")
    for i, m in enumerate(colours):
        dd = ev[m]["baselines"]["ensemble_minus_shape_gbm"]
        ax.text(x[i] + 0.25, ens[i] + 0.01, f"+{dd['delta_roc_auc']:.3f}\n[{dd['ci'][0]:.3f}, {dd['ci'][1]:.3f}]",
                ha="center", fontsize=7)
    ax.set(xticks=x, xticklabels=labels, ylabel="test ROC-AUC", ylim=(0.5, 1.08), title="Same-split baselines (paired bootstrap)")
    ax.legend(fontsize=7, loc="upper right", ncol=1, framealpha=0.95)
    fig.tight_layout()
    out = R / "figures" / "13_stage_a_v2_final.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
