"""
evaluate_snr_balance.py
=======================

Evaluation of eda/SNR_BALANCE_PROTOCOL.md: the v2 Kepler ensemble against model N (the v2 recipe with
S/N-balanced loss weights), on Kepler v2 validation and on the pre-declared second read of K2.

    python stage_a_transit_model/evaluate_snr_balance.py --dry-run   # validation + Stage C TESS only, no K2
    python stage_a_transit_model/evaluate_snr_balance.py             # + the K2 second read (once; guarded)
Output: outputs/stage_a_mechanisms/snr_balance.json, figures/19_snr_balance.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from stage_a_transit_model.cnn_lstm import PLANET_LIKE_IDX, consolidate_views, drop_unusable  # noqa: E402
from stage_a_transit_model.critique2_analyses import snr_of  # noqa: E402
from stage_a_transit_model.failure_analysis import load_run_model, score_views  # noqa: E402
from stage_a_transit_model.vetter_v3 import score as score_v3  # noqa: E402

RUNS = R / "stage_a_transit_model" / "runs"
OUT = R / "outputs" / "stage_a_mechanisms"
GUARD = RUNS / "snr_balance" / "K2_READ.json"
V2 = [RUNS / f"v2_kepler_dual_cnn_lstm_err{s}" for s in ["", "_s1", "_s2", "_s7", "_s11"]]
N = [RUNS / f"v3_N{s}" for s in ["", "_s1", "_s2", "_s7", "_s11"]]
DEEP_PPM = 10000.0


def ens_v2(vs) -> np.ndarray:
    return np.mean([score_views(load_run_model(r)[0], vs)[:, PLANET_LIKE_IDX].sum(1) for r in V2], axis=0)


def ens_n(vs) -> np.ndarray:
    return np.mean([score_v3(r, vs)[:, PLANET_LIKE_IDX].sum(1) for r in N], axis=0)


def thr_recall(y, s, recall=0.90) -> float:
    return float(np.quantile(s[y == 1], 1 - recall))


def bootstrap_diff(a_new, a_old, b_new, b_old, n=4000, seed=1) -> list:
    """CI of the change in the deep-minus-shallow median gap (paired within each group)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i, j = rng.integers(0, len(a_new), len(a_new)), rng.integers(0, len(b_new), len(b_new))
        out.append((np.median(b_new[j]) - np.median(a_new[i])) - (np.median(b_old[j]) - np.median(a_old[i])))
    return np.percentile(out, [2.5, 97.5]).round(3).tolist()


def main(argv=None) -> int:
    from scipy.stats import mannwhitneyu, wilcoxon
    from sklearn.metrics import roc_auc_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    missing = [r.name for r in N if not (r / "report.json").is_file()]
    if missing:
        raise SystemExit(f"model N not finished: {missing}")
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"protocol": "eda/SNR_BALANCE_PROTOCOL.md",
           "n_val_roc_auc_per_seed": [json.loads((r / "report.json").read_text())["val_roc_auc"] for r in N]}

    # validation
    vdir = R / "detection_views_v2" / "kepler_val"
    val = drop_unusable(consolidate_views(vdir, verbose=False), verbose=False)
    y = np.isin(val.label_index, PLANET_LIKE_IDX).astype(int)
    snr = snr_of(vdir, val.object_id)
    sv, sn = ens_v2(val), ens_n(val)
    hi_fp = (y == 0) & (snr > 100)
    res["validation"] = {"n": int(len(y)), "roc_auc": {"v2": float(roc_auc_score(y, sv)), "N": float(roc_auc_score(y, sn))},
                         "high_snr_fp_rejected_at_recall90": {
                             "n": int(hi_fp.sum()),
                             "v2": float(np.mean(sv[hi_fp] < thr_recall(y, sv))), "N": float(np.mean(sn[hi_fp] < thr_recall(y, sn)))},
                         "roc_auc_by_snr": {}}
    for lo, hi in [(0, 20), (20, 100), (100, 1e9)]:
        m = (snr >= lo) & (snr < hi)
        if len(set(y[m])) == 2:
            res["validation"]["roc_auc_by_snr"][f"{lo:g}-{hi:g}"] = {"n": int(m.sum()), "v2": float(roc_auc_score(y[m], sv[m])),
                                                                    "N": float(roc_auc_score(y[m], sn[m]))}

    # Stage C TESS light curves (descriptive, not a test set)
    sc = drop_unusable(consolidate_views(R / "detection_views_err" / "stage_c_targets", verbose=False), verbose=False)
    res["tess_stage_c_targets"] = {"n": int(len(sc)), "median_v2": float(np.median(ens_v2(sc))),
                                   "median_N": float(np.median(ens_n(sc)))}

    if not a.dry_run:
        if GUARD.is_file():
            raise SystemExit(f"the K2 second read was already made ({GUARD}); this protocol allows one")
        pred = pd.read_csv(RUNS / "k2_blind" / "k2_predictions.csv").set_index("object_id")
        k2 = drop_unusable(consolidate_views(R / "detection_views_err" / "k2_everest", verbose=False), verbose=False)
        k = pred.loc[k2.object_id].assign(score_N=ens_n(k2))
        prim = k[k.duration_source.isin(["catalogue", "other set"])]
        yb = (prim.true_label != "FALSE POSITIVE").astype(int).to_numpy()
        c = prim[prim.true_label == "CONFIRMED"]
        deep, shal = c[c.depth_ppm > DEEP_PPM], c[c.depth_ppm <= DEEP_PPM]
        gap = lambda col: float(shal[col].median() - deep[col].median())
        res["k2_second_read"] = {
            "n_primary": int(len(prim)),
            "p1_roc_auc": {"v2": float(roc_auc_score(yb, prim.score_kepler)), "N": float(roc_auc_score(yb, prim.score_N))},
            "confirmed_deep": {"n": int(len(deep)), "median_v2": float(deep.score_kepler.median()), "median_N": float(deep.score_N.median()),
                               "wilcoxon_p_N_higher": float(wilcoxon(deep.score_N, deep.score_kepler, alternative="greater").pvalue)},
            "confirmed_shallow": {"n": int(len(shal)), "median_v2": float(shal.score_kepler.median()), "median_N": float(shal.score_N.median())},
            "gap_shallow_minus_deep": {"v2": gap("score_kepler"), "N": gap("score_N"),
                                       "change_ci95": bootstrap_diff(deep.score_N.to_numpy(), deep.score_kepler.to_numpy(),
                                                                     shal.score_N.to_numpy(), shal.score_kepler.to_numpy())},
            "mannwhitney_p_deep_lower": {"v2": float(mannwhitneyu(deep.score_kepler, shal.score_kepler, alternative="less").pvalue),
                                         "N": float(mannwhitneyu(deep.score_N, shal.score_N, alternative="less").pvalue)},
            "fp_median": {"v2": float(prim[prim.true_label == "FALSE POSITIVE"].score_kepler.median()),
                          "N": float(prim[prim.true_label == "FALSE POSITIVE"].score_N.median())}}
        GUARD.parent.mkdir(parents=True, exist_ok=True)
        GUARD.write_text(json.dumps({"read_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "reads": 1, "what": "K2 EVEREST views, model N only"}))
        k[["true_label", "depth_ppm", "depth_snr", "duration_source", "score_kepler", "score_N"]].to_csv(OUT / "snr_balance_k2_scores.csv")
        kr, v = res["k2_second_read"], res["validation"]
        res["expectations"] = {
            "N1 validation ROC-AUC >= v2 - 0.01": v["roc_auc"]["N"] >= v["roc_auc"]["v2"] - 0.01,
            "N2 K2 deep confirmed median >= 0.85": kr["confirmed_deep"]["median_N"] >= 0.85,
            "N3 deep-vs-shallow gap shrinks by >= half": kr["gap_shallow_minus_deep"]["N"] <= 0.5 * kr["gap_shallow_minus_deep"]["v2"],
            "N4 K2 P1 ROC-AUC >= v2 - 0.01": kr["p1_roc_auc"]["N"] >= kr["p1_roc_auc"]["v2"] - 0.01,
            "N5 high-S/N FP rejection >= v2 - 0.05": v["high_snr_fp_rejected_at_recall90"]["N"] >= v["high_snr_fp_rejected_at_recall90"]["v2"] - 0.05}
        figure(k, res)
    name = "snr_balance.json" if not a.dry_run else "snr_balance_dry_run.json"
    (OUT / name).write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))
    return 0


def figure(k: pd.DataFrame, res: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = k[(k.true_label == "CONFIRMED") & k.duration_source.isin(["catalogue", "other set"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    for col, colr, lab in [("score_kepler", "#999999", "v2 (Kepler label prior)"), ("score_N", "#2a78d6", "N (S/N-balanced)")]:
        ax.scatter(c.depth_ppm.clip(lower=10), c[col], s=10, alpha=0.5, color=colr, label=lab)
    ax.axvline(DEEP_PPM, color="k", ls=":", lw=1)
    kr = res["k2_second_read"]
    ax.set(xscale="log", xlabel="measured transit depth (ppm)", ylabel="planet-like score",
           title=f"K2 confirmed planets: deep median {kr['confirmed_deep']['median_v2']:.2f} -> {kr['confirmed_deep']['median_N']:.2f}")
    ax.legend(fontsize=8, loc="lower left")
    ax = axes[1]
    v = res["validation"]["roc_auc_by_snr"]
    x = np.arange(len(v))
    ax.bar(x - 0.18, [v[b]["v2"] for b in v], 0.36, color="#999999", label="v2")
    ax.bar(x + 0.18, [v[b]["N"] for b in v], 0.36, color="#2a78d6", label="N")
    ax.set(xticks=x, xticklabels=[f"S/N {b}\n(n={v[b]['n']})" for b in v], ylim=(0.5, 1.0), ylabel="ROC-AUC (Kepler validation)",
           title="In-survey cost of removing the S/N prior")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(R / "figures" / "19_snr_balance.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())
