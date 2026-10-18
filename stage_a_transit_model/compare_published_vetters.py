"""
compare_published_vetters.py
============================

Stage A's final Kepler ensemble against published vetters, on exactly the same v2 test KOIs.

Run (project .venv): python stage_a_transit_model/compare_published_vetters.py

Scorers compared
  * Ours: the 5-seed CNN+BiLSTM ensemble with the error channel (evaluate_v2.py; test read once).
  * Robovetter: koi_score from the cumulative KOI table (Thompson et al. 2018, ApJS 235, 38).
  * AstroNet, ExoNet, ExoMiner and ExoMiner-TCE: OUT-OF-FOLD scores for every Q1-Q17 DR25 TCE from
    Valizadegan et al. 2022 (ApJ 926, 120), Table 3. That is the machine-readable table, saved as
    external_data/exominer/t3.txt. Their 10-fold cross-validation is grouped by target star.
    Matched to KOIs by (kepid, koi_tce_plnt_num).

Tasks (labels = the cumulative koi_disposition, as everywhere in Stage A)
  A  planet-like (CANDIDATE + CONFIRMED) vs FALSE POSITIVE: our task.
  B  CONFIRMED vs FALSE POSITIVE: the closest to ExoMiner's own PC-vs-non-PC training labels.

Every comparison uses the KOIs that ALL scorers cover, with a star-bootstrap CI (2,000 resamples)
and a paired bootstrap of the AUC difference (ours minus theirs).

Not a level playing field, and quantified as such:
  * Circularity. The archive's FALSE POSITIVE dispositions come largely from the Robovetter itself
    (koi_pdisposition), so koi_score is partly graded against its own decisions. The script reports
    how often the label equals the Robovetter decision.
  * Inputs. ExoMiner also uses centroid motion, odd/even depth, secondary eclipse and stellar
    diagnostics. Our model sees only the phase-folded light-curve views (and the flux error).
  * Training data. The published models trained on DR25 TCE labels from 2018-2021; ours trains on the
    current cumulative table.

Output: outputs/stage_a_comparison/{published_vetters.json, matched_scores.csv}, figures/14_published_vetters.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

R = Path(__file__).resolve().parents[1]
OUT = R / "outputs" / "stage_a_comparison"
EXOMINER = R / "external_data" / "exominer" / "t3.txt"
N_BOOT, SEED = 2000, 20261018
SCORERS = {"ours": "Ours (CNN+BiLSTM ensemble)", "koi_score": "Robovetter (koi_score)", "astronet": "AstroNet",
           "exonet": "ExoNet", "exominer": "ExoMiner", "exominer_tce": "ExoMiner-TCE"}


def read_exominer() -> pd.DataFrame:
    cols = [(0, 9, "target_id"), (10, 12, "tce_plnt_num"), (13, 27, "tce_period"), (28, 38, "mes"),
            (39, 42, "original_label"), (43, 44, "binary_label"), (45, 56, "astronet"), (57, 68, "exonet"),
            (69, 80, "exominer"), (81, 92, "exominer_tce")]
    lines = EXOMINER.read_text().splitlines()
    start = max(i for i, l in enumerate(lines) if l.startswith("-----")) + 1
    rows = []
    for l in lines[start:]:
        if l.strip():
            rows.append({name: l[a:b].strip() for a, b, name in cols})
    d = pd.DataFrame(rows)
    for c in ["target_id", "tce_plnt_num"]:
        d[c] = d[c].astype(int)
    for c in ["tce_period", "mes", "astronet", "exonet", "exominer", "exominer_tce"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def boot_auc(y, s, stars, rng):
    uniq = np.unique(stars)
    idx = {u: np.flatnonzero(stars == u) for u in uniq}
    vals = []
    for _ in range(N_BOOT):
        pick = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq))])
        if len(np.unique(y[pick])) == 2:
            vals.append(roc_auc_score(y[pick], s[pick]))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def paired_diff(y, a, b, stars, rng):
    uniq = np.unique(stars)
    idx = {u: np.flatnonzero(stars == u) for u in uniq}
    diffs = []
    for _ in range(N_BOOT):
        pick = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq))])
        if len(np.unique(y[pick])) == 2:
            diffs.append(roc_auc_score(y[pick], a[pick]) - roc_auc_score(y[pick], b[pick]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"delta_auc": round(float(roc_auc_score(y, a) - roc_auc_score(y, b)), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pred = pd.read_csv(R / "stage_a_transit_model" / "runs" / "v2_final" / "kepler_test_predictions.csv")
    koi = pd.read_csv(R / "splits" / "koi_cumulative_split_v2.csv", low_memory=False,
                      usecols=["kepid", "kepoi_name", "koi_tce_plnt_num", "koi_score", "koi_pdisposition", "koi_disposition", "split"])
    d = pred.merge(koi, left_on="object_id", right_on="kepoi_name", how="left")
    assert (d.split == "test").all(), "predictions must be the v2 test KOIs"
    em = read_exominer()
    d = d.merge(em, left_on=["kepid", "koi_tce_plnt_num"], right_on=["target_id", "tce_plnt_num"], how="left")
    d = d.rename(columns={"planet_like_score": "ours"})
    d.to_csv(OUT / "matched_scores.csv", index=False)

    res = {"n_test_kois": int(len(d)),
           "coverage": {k: int(d[k].notna().sum()) for k in SCORERS},
           "circularity": {
               "FP_label_equals_robovetter_FP": round(float((d[d.true_label == "FALSE POSITIVE"].koi_pdisposition == "FALSE POSITIVE").mean()), 3),
               "planet_like_label_equals_robovetter_candidate": round(float((d[d.true_label != "FALSE POSITIVE"].koi_pdisposition == "CANDIDATE").mean()), 3),
               "robovetter_binary_accuracy_vs_label": round(float(((d.koi_pdisposition == "FALSE POSITIVE") == (d.true_label == "FALSE POSITIVE")).mean()), 3)},
           "tasks": {}}
    common = d.dropna(subset=list(SCORERS))
    res["n_common"] = int(len(common))
    for task, sub, ylab in [("A_planet_like_vs_FP", common, lambda x: (x.true_label != "FALSE POSITIVE").astype(int)),
                            ("B_confirmed_vs_FP", common[common.true_label != "CANDIDATE"], lambda x: (x.true_label == "CONFIRMED").astype(int))]:
        y = ylab(sub).to_numpy()
        stars = sub.kepid.to_numpy()
        t = {"n": int(len(sub)), "n_pos": int(y.sum()), "scorers": {}}
        for k in SCORERS:
            s = sub[k].to_numpy(float)
            t["scorers"][k] = {"roc_auc": round(float(roc_auc_score(y, s)), 4), "ci95": boot_auc(y, s, stars, rng)}
            if k != "ours":
                t["scorers"][k]["ours_minus_this"] = paired_diff(y, sub["ours"].to_numpy(float), s, stars, rng)
        # where the signal is weak: MES < 12 (DR25 TCE MES)
        low = sub.mes.to_numpy(float) < 12
        if low.sum() > 30 and len(np.unique(y[low])) == 2:
            t["low_mes_lt_12"] = {"n": int(low.sum()), **{k: round(float(roc_auc_score(y[low], sub[k].to_numpy(float)[low])), 4) for k in SCORERS}}
        res["tasks"][task] = t
    (OUT / "published_vetters.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    make_figure(common)
    return 0


def make_figure(common: pd.DataFrame):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"ours": "#2e933c", "koi_score": "0.45", "astronet": "#8f5bd6", "exonet": "#f2a541", "exominer": "#d1495b", "exominer_tce": "#2a6fdb"}
    res = json.loads((OUT / "published_vetters.json").read_text())["tasks"]["B_confirmed_vs_FP"]
    sub = common[common.true_label != "CANDIDATE"]
    y = (sub.true_label == "CONFIRMED").astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for k, lab in SCORERS.items():
        fpr, tpr, _ = roc_curve(y, sub[k])
        ax.plot(fpr, tpr, color=colours[k], lw=2.4 if k == "ours" else 1.3, label=f"{lab}: {roc_auc_score(y, sub[k]):.3f}")
    ax.set(xlim=(0, 0.25), ylim=(0.6, 1.0), xlabel="false-positive rate", ylabel="true-positive rate",
           title=f"Confirmed vs false positive, Kepler v2 test KOIs (n = {len(sub)}), zoomed")
    ax.legend(fontsize=7, loc="lower right")
    ax = axes[1]
    keys = list(SCORERS)
    x = np.arange(len(keys))
    auc = [res["scorers"][k]["roc_auc"] for k in keys]
    lo = [res["scorers"][k]["roc_auc"] - res["scorers"][k]["ci95"][0] for k in keys]
    hi = [res["scorers"][k]["ci95"][1] - res["scorers"][k]["roc_auc"] for k in keys]
    ax.errorbar(x - 0.12, auc, yerr=[lo, hi], fmt="o", color="k", capsize=3, label="all common KOIs (95 % star bootstrap)")
    low = res.get("low_mes_lt_12", {})
    if low:
        ax.scatter(x + 0.12, [low[k] for k in keys], marker="s", color="#d1495b", label=f"weak signals, MES < 12 (n = {low['n']})")
    for i, k in enumerate(keys):
        ax.axvspan(i - 0.4, i + 0.4, color=colours[k], alpha=0.12, lw=0)
    ax.set(xticks=x, xticklabels=["Ours", "Robovetter*", "AstroNet", "ExoNet", "ExoMiner", "ExoMiner-TCE"], ylim=(0.9, 1.0),
           ylabel="ROC-AUC", title="Light-curve-only (Ours, AstroNet) vs vetters with extra diagnostics")
    ax.text(0.01, 0.02, "* Robovetter decisions define the false-positive labels (circular)", transform=ax.transAxes, fontsize=7)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(R / "figures" / "14_published_vetters.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    import sys
    if "--figure-only" in sys.argv:
        make_figure(pd.read_csv(OUT / "matched_scores.csv").dropna(subset=list(SCORERS)))
        raise SystemExit(0)
    raise SystemExit(main())
