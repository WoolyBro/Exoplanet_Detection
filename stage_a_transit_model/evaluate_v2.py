"""
evaluate_v2.py
==============

The single, final evaluation of Stage A on split v2 (star- and signal-family-grouped).

Everything that could be tuned is fixed on VALIDATION, before test is read:
  1. Model choice - the pre-registered rule in sweep.py (written before any v2 result existed):
       architecture = simplest of global_cnn < dual_cnn < dual_cnn_lstm whose mean val ROC-AUC is
                      within one seed-std of the best architecture's mean;
       error channel = kept only if it beats the same architecture without it by > 1 seed-std;
       class weighting (TESS, dual_cnn_lstm only) = best mean val ROC-AUC, ties to "inverse".
  2. Final model = the average of the chosen configuration's 5 seed models (an ensemble), so no
     single seed's luck is reported.
  3. Probability calibration (Platt) and the operating thresholds at 90 / 95 % recall are fitted on
     the validation ensemble scores and carried UNCHANGED to test, so the test precision/recall at
     those thresholds is an honest out-of-sample operating point.
Then test is scored exactly once. A guard file (runs/v2_final/TEST_READ.json) makes a second
reading refuse to run unless --reread is passed, and any re-read is logged in the output.

Reported per mission, all with 95 % star-bootstrap intervals (resampling host stars):
  * ROC-AUC, PR-AUC and its lift over the base rate, 3-class accuracy and confusion matrix
  * precision/recall at the validation-fixed thresholds, and P@R on test for reference
  * CONFIRMED vs FALSE POSITIVE only (CANDIDATE is an undecided label, not ground truth)
  * per depth-SNR bin (the pipeline's own measured transit SNR; Kepler also by koi_model_snr)
  * calibration (Brier, ECE) raw vs Platt
  * the per-seed test spread, and the same-split baselines with a paired bootstrap of the difference
  * the cross-mission matrix (each mission's ensemble on the other mission's test views)

Usage
    python stage_a_transit_model/evaluate_v2.py --select-only   # show the selection, read no test
    python stage_a_transit_model/evaluate_v2.py                 # the one test evaluation
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RESEARCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_DIR))

from stage_a_transit_model.calibrate import apply as cal_apply, ece, fit as cal_fit  # noqa: E402
from stage_a_transit_model.cnn_lstm import (  # noqa: E402
    CLASSES, PLANET_LIKE_IDX, consolidate_views, drop_unusable, evaluate, precision_at_recall,
)
from stage_a_transit_model.failure_analysis import load_run_model, score_views  # noqa: E402

RUNS = RESEARCH_DIR / "stage_a_transit_model" / "runs"
VIEWS = RESEARCH_DIR / "detection_views_v2"
OUT = RUNS / "v2_final"
GUARD = OUT / "TEST_READ.json"
ARCH_ORDER = ["global_cnn", "dual_cnn", "dual_cnn_lstm"]
SEED_SUFFIXES = ["", "_s1", "_s2", "_s7", "_s11"]
SNR_BINS = [(0, 7.1), (7.1, 15), (15, 50), (50, np.inf)]
N_BOOT = 2000
RNG = np.random.default_rng(20261017)


# =========================================================================== #
# Selection (validation only)
# =========================================================================== #
def family_stats(table: pd.DataFrame, family: str) -> tuple[float, float, int]:
    rows = table[table.config.isin([f"v2_{family}{s}" for s in SEED_SUFFIXES])]
    v = rows.val_roc_auc.dropna().to_numpy()
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan"), len(v))


def select(mission: str, table: pd.DataFrame) -> dict:
    stats = {a: family_stats(table, f"{mission}_{a}") for a in ARCH_ORDER}
    best = max(ARCH_ORDER, key=lambda a: stats[a][0])
    best_mean, best_std, _ = stats[best]
    arch = next(a for a in ARCH_ORDER if stats[a][0] >= best_mean - best_std)
    family = f"{mission}_{arch}"
    decision = {"architecture_stats": {a: {"mean": m, "std": s, "n": n} for a, (m, s, n) in stats.items()},
                "best_by_mean": best, "chosen_architecture": arch}
    if arch != "global_cnn":
        m0, s0, _ = stats[arch]
        me, se, ne = family_stats(table, f"{mission}_{arch}_err")
        decision["error_channel"] = {"without": {"mean": m0, "std": s0}, "with": {"mean": me, "std": se, "n": ne},
                                     "threshold": m0 + s0}
        if ne and me > m0 + s0:
            family += "_err"
    decision["use_errors"] = family.endswith("_err")
    if mission == "tess" and family == "tess_dual_cnn_lstm":
        cands = {"inverse": stats["dual_cnn_lstm"][0]}
        for cw in ("sqrt", "none"):
            cands[cw] = family_stats(table, f"tess_dual_cnn_lstm_cw{cw}")[0]
        cw = max(cands, key=lambda k: (round(cands[k], 6), k == "inverse"))
        decision["class_weight"] = {"means": cands, "chosen": cw}
        if cw != "inverse":
            family = f"tess_dual_cnn_lstm_cw{cw}"
    decision["family"] = family
    decision["runs"] = [f"v2_{family}{s}" for s in SEED_SUFFIXES]
    return decision


# =========================================================================== #
# Scoring helpers
# =========================================================================== #
def load_split(mission: str, split: str):
    vs = drop_unusable(consolidate_views(VIEWS / f"{mission}_{split}", verbose=False), verbose=False)
    meta = []
    for oid in vs.object_id:
        from preprocessing_pipeline import _npz_name
        with np.load(VIEWS / f"{mission}_{split}" / _npz_name(oid), allow_pickle=False) as z:
            meta.append(json.loads(str(z["metadata"])).get("depth_snr", np.nan))
    return vs, np.array(meta, float)


def ensemble_proba(runs: list[str], vs) -> tuple[np.ndarray, list[np.ndarray]]:
    per = [score_views(load_run_model(RUNS / r)[0], vs) for r in runs]
    return np.mean(per, axis=0), per


def star_boot(stars: np.ndarray, fn, n: int = N_BOOT) -> tuple[float, float]:
    groups = pd.Series(np.arange(len(stars))).groupby(stars).apply(np.array).tolist()
    vals = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in RNG.integers(0, len(groups), len(groups))])
        try:
            vals.append(fn(idx))
        except ValueError:
            continue
    return tuple(np.percentile(vals, [2.5, 97.5]).round(4))


def binary_block(y_bin: np.ndarray, score: np.ndarray, stars: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    if not 0 < y_bin.sum() < len(y_bin):
        return {"n": int(len(y_bin)), "note": "one class only"}
    return {"n": int(len(y_bin)), "positive_rate": float(y_bin.mean()),
            "roc_auc": float(roc_auc_score(y_bin, score)),
            "roc_auc_ci": star_boot(stars, lambda i: roc_auc_score(y_bin[i], score[i])),
            "pr_auc": float(average_precision_score(y_bin, score)),
            "pr_auc_ci": star_boot(stars, lambda i: average_precision_score(y_bin[i], score[i])),
            "pr_auc_lift": float(average_precision_score(y_bin, score) - y_bin.mean())}


def at_threshold(y_bin: np.ndarray, score: np.ndarray, thr: float) -> dict:
    pred = score >= thr
    tp, fp, fn = int((pred & (y_bin == 1)).sum()), int((pred & (y_bin == 0)).sum()), int((~pred & (y_bin == 1)).sum())
    return {"threshold": float(thr), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "false_positive_rate": fp / max(int((y_bin == 0).sum()), 1)}


def paired_auc_diff(y_bin, a, b, stars) -> dict:
    from sklearn.metrics import roc_auc_score

    d = float(roc_auc_score(y_bin, a) - roc_auc_score(y_bin, b))
    lo, hi = star_boot(stars, lambda i: roc_auc_score(y_bin[i], a[i]) - roc_auc_score(y_bin[i], b[i]))
    return {"delta_roc_auc": d, "ci": (lo, hi), "excludes_zero": bool(lo > 0 or hi < 0)}


# =========================================================================== #
# Per-mission evaluation
# =========================================================================== #
def evaluate_mission(mission: str, decision: dict, test_vs, test_snr, val_vs, other: dict) -> dict:
    from stage_a_transit_model.baselines import logistic_baseline, shape_gbm_baseline

    runs = decision["runs"]
    val_p, _ = ensemble_proba(runs, val_vs)
    test_p, per_seed = ensemble_proba(runs, test_vs)
    yv, yt = val_vs.label_index.astype(int), test_vs.label_index.astype(int)
    yv_b, yt_b = np.isin(yv, PLANET_LIKE_IDX).astype(int), np.isin(yt, PLANET_LIKE_IDX).astype(int)
    sv, st = val_p[:, PLANET_LIKE_IDX].sum(1), test_p[:, PLANET_LIKE_IDX].sum(1)
    stars = test_vs.star_id

    # --- fixed on validation ---
    thresholds = {r["target_recall"]: r["threshold"] for r in precision_at_recall(yv_b, sv) if r.get("achievable")}
    cal = cal_fit("platt", sv, yv_b)

    res = {"decision": decision, "n_test": int(len(yt)), "n_test_stars": int(len(set(stars.tolist())))}
    full = evaluate(yt, test_p)
    res["three_class"] = {"accuracy": full["accuracy_3class"], "confusion": full["confusion_3class"],
                          "report": full["report_3class"]}
    res["binary"] = binary_block(yt_b, st, stars)
    res["operating_points_fixed_on_val"] = {f"recall_{int(k * 100)}": at_threshold(yt_b, st, t) for k, t in thresholds.items()}
    res["precision_at_recall_on_test"] = precision_at_recall(yt_b, st)
    cf = np.isin(yt, [0, 2])
    res["confirmed_vs_fp"] = binary_block((yt[cf] == 2).astype(int), st[cf], stars[cf])
    res["by_depth_snr"] = {f"{lo:g}-{hi:g}": binary_block(yt_b[m], st[m], stars[m])
                           for lo, hi in SNR_BINS for m in [(test_snr >= lo) & (test_snr < hi)]}
    calibrated = cal_apply(cal, st)
    res["calibration"] = {"platt": cal, "raw": {"brier": float(np.mean((st - yt_b) ** 2)), "ece": ece(yt_b, st)},
                          "platt_fitted_on_val": {"brier": float(np.mean((calibrated - yt_b) ** 2)),
                                                  "ece": ece(yt_b, calibrated)}}
    from sklearn.metrics import roc_auc_score

    seeds = [roc_auc_score(yt_b, p[:, PLANET_LIKE_IDX].sum(1)) for p in per_seed]
    res["per_seed_test_roc_auc"] = {"values": seeds, "mean": float(np.mean(seeds)), "std": float(np.std(seeds, ddof=1))}

    # --- same-split baselines, trained on v2 train only ---
    tr = drop_unusable(consolidate_views(VIEWS / f"{mission}_train", verbose=False), verbose=False)
    with contextlib.redirect_stdout(io.StringIO()):
        log_p = logistic_baseline(tr, test_vs, 42)
        _, _, gbm_p = shape_gbm_baseline(tr, test_vs, 42)
    res["baselines"] = {
        "logistic_views": {"roc_auc": log_p["roc_auc_binary"], "pr_auc": log_p["pr_auc_binary"]},
        "shape_gbm": {"roc_auc": float(roc_auc_score(yt_b, gbm_p[:, PLANET_LIKE_IDX].sum(1)))},
        "ensemble_minus_shape_gbm": paired_auc_diff(yt_b, st, gbm_p[:, PLANET_LIKE_IDX].sum(1), stars),
    }
    pd.DataFrame({"object_id": test_vs.object_id, "star_id": stars, "true_label": [CLASSES[i] for i in yt],
                  "planet_like_score": st, "calibrated_score": calibrated, "depth_snr": test_snr,
                  **{f"p_{c.replace(' ', '_').lower()}": test_p[:, i] for i, c in enumerate(CLASSES)}}
                 ).to_csv(OUT / f"{mission}_test_predictions.csv", index=False)
    other[mission] = runs
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select-only", action="store_true", help="apply the selection rule; read no test data")
    ap.add_argument("--reread", action="store_true", help="allow a second reading of test (logged)")
    ap.add_argument("--dry-run-on-val", action="store_true",
                    help="exercise every code path with VALIDATION standing in for test; reads no test data, "
                         "writes to runs/v2_final_dryrun/ and leaves the guard untouched")
    args = ap.parse_args(argv)

    table = pd.read_csv(RUNS / "v2_comparison.csv")
    decisions = {m: select(m, table) for m in ("kepler", "tess")}
    for m, d in decisions.items():
        print(f"\n[{m}] selection on validation")
        for a, s in d["architecture_stats"].items():
            print(f"  {a:<14} val ROC-AUC {s['mean']:.4f} +/- {s['std']:.4f} (n={s['n']})")
        print(f"  best by mean: {d['best_by_mean']} -> chosen (simplest within 1 std): {d['chosen_architecture']}")
        if "error_channel" in d:
            e = d["error_channel"]
            print(f"  error channel: with {e['with']['mean']:.4f} vs needed > {e['threshold']:.4f} -> "
                  f"{'USED' if d['use_errors'] else 'not used'}")
        if "class_weight" in d:
            print(f"  class weight: {d['class_weight']['means']} -> {d['class_weight']['chosen']}")
        print(f"  final model: 5-seed ensemble of v2_{d['family']}")
    if args.select_only:
        return 0

    global OUT
    if args.dry_run_on_val:
        OUT = RUNS / "v2_final_dryrun"
    OUT.mkdir(parents=True, exist_ok=True)
    if args.dry_run_on_val:
        pass
    elif GUARD.is_file() and not args.reread:
        raise SystemExit(f"test was already read at {json.loads(GUARD.read_text())['read_at']}; "
                         "pass --reread to read it again (it will be recorded)")
    if args.dry_run_on_val:
        reads = 0
    else:
        reads = json.loads(GUARD.read_text())["reads"] + 1 if GUARD.is_file() else 1
        GUARD.write_text(json.dumps({"read_at": datetime.now().isoformat(timespec="seconds"), "reads": reads}))

    results, data, other = {}, {}, {}
    for m in ("kepler", "tess"):
        val_vs, _ = load_split(m, "val")
        test_vs, test_snr = load_split(m, "val" if args.dry_run_on_val else "test")
        data[m] = test_vs
        results[m] = evaluate_mission(m, decisions[m], test_vs, test_snr, val_vs, other)
        b = results[m]["binary"]
        print(f"\n[{m}] TEST  n={results[m]['n_test']}  ROC-AUC {b['roc_auc']:.4f} {b['roc_auc_ci']}  "
              f"PR-AUC {b['pr_auc']:.4f} (lift {b['pr_auc_lift']:+.3f})  CONF-vs-FP {results[m]['confirmed_vs_fp']['roc_auc']:.4f}")

    # --- cross-mission matrix ---
    from sklearn.metrics import roc_auc_score, average_precision_score

    cross = []
    for trained, runs in other.items():
        for evaluated, vs in data.items():
            p, _ = ensemble_proba(runs, vs)
            y = np.isin(vs.label_index.astype(int), PLANET_LIKE_IDX).astype(int)
            s = p[:, PLANET_LIKE_IDX].sum(1)
            cross.append({"trained_on": trained, "evaluated_on": evaluated, "roc_auc": float(roc_auc_score(y, s)),
                          "pr_auc": float(average_precision_score(y, s)), "base_rate": float(y.mean())})
    results["cross_mission"] = cross
    results["test_reads"] = reads
    (OUT / "final_eval_v2.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nsaved {OUT.relative_to(RESEARCH_DIR)}/ (final_eval_v2.json, *_test_predictions.csv); test reads: {reads}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
