"""
baselines.py
============

Track A Level 2: the baselines that must exist BEFORE the deep model is believed.

The mentor's checklist is explicit: "Baselines first (majority class, logistic
regression) before deep models". This file provides them on exactly the same input
the CNN+LSTM sees - the global/local views - so the comparison is apples to apples.

A separate, already-completed baseline lives in baseline_results/: RandomForest on
the KOI *catalog* table (kepler_all.json, honest ROC-AUC 0.971). That model is not a
competitor to the CNN, it is a different information source: it reads transit
parameters the Kepler pipeline already fitted (period, depth, duration, SNR, stellar
properties). The CNN reads raw folded shape and nothing else. Both numbers are
reported side by side in the results write-up; neither subsumes the other.

Baselines here, in increasing order of what they are allowed to know:
  1. majority class        - the floor. Any model below this has learned nothing.
  2. logistic regression   - linear decision on the standardized raw views.
  3. shape features + GBM  - 11 hand-derived, physically interpretable numbers
                             (depth, V-shape, odd/even, secondary, scatter...).
                             This one doubles as the failure-analysis instrument:
                             its features are exactly the quantities the Kepler
                             false-positive flags encode.

Usage
    python stage_a_transit_model/baselines.py                 # val only (default, safe)
    python stage_a_transit_model/baselines.py --eval-test     # final run only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RESEARCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_DIR))

from stage_a_transit_model.cnn_lstm import (  # noqa: E402  (single source of truth for loading + metrics)
    CLASSES, MODEL_DIR, PLANET_LIKE_IDX, VIEWS_ROOT, ViewSet,
    consolidate_views, drop_unusable, evaluate, planet_like_score,
)


# =========================================================================== #
# Physically interpretable shape features
# =========================================================================== #
def shape_features(vs: ViewSet) -> pd.DataFrame:
    """Eleven numbers per KOI, each one something a human vetter actually looks at.

    Views are already normalised (median 0, deepest bin -1), so these are shape
    descriptors, not absolute depths. Names map onto the Kepler FP flags:
      * v_shape / duration    -> koi_fpflag_nt (not transit-like: grazing, V-shaped)
      * secondary_depth       -> koi_fpflag_ss (stellar eclipse: a real secondary)
      * quadrature_dip        -> ellipsoidal variation / out-of-transit variability
      * global_scatter / snr  -> stellar variability drowning a shallow signal

    Geometry note: build_global_view bins phase from -P/2 to +P/2 with the transit at
    the CENTRE bin, so phase 0.5 - where a circular-orbit secondary eclipse sits - is
    at the two EDGES of the view, and phase +/-0.25 (quadrature) is at the quarter
    points. An earlier version of this function measured "secondary_depth" at the
    quarter points, which is quadrature, not the secondary; both are now measured and
    named for what they actually are.

    What is deliberately NOT here: a true odd/even depth difference. That needs
    alternate transits separated (a fold at 2x the period, or the unfolded series).
    A single phase-folded view cannot express it - splitting this view at its centre
    just returns the two halves of the same averaged transit. `transit_asymmetry`
    measures that left/right difference and is reported as such, not as odd/even.
    """
    g, l = vs.global_view, vs.local_view
    n = len(vs)
    centre = l.shape[1] // 2
    core = slice(max(0, centre - 10), centre + 11)          # in-transit core of the local view
    wings = np.r_[0:max(0, centre - 40), min(l.shape[1], centre + 41):l.shape[1]]

    local_depth = -l[:, core].min(axis=1)
    local_floor = -l[:, core].mean(axis=1)
    wing_level = l[:, wings].mean(axis=1) if len(wings) else np.zeros(n)
    wing_scatter = l[:, wings].std(axis=1) if len(wings) else np.zeros(n)

    # V vs U: a box-like (planetary) transit has floor ~ depth; a V (grazing/EB) has floor << depth.
    with np.errstate(divide="ignore", invalid="ignore"):
        v_shape = np.where(local_depth > 0, local_floor / local_depth, 0.0)
        local_snr = np.where(wing_scatter > 0, local_depth / wing_scatter, 0.0)

    nb = g.shape[1]
    gc = nb // 2          # transit centre (phase 0)
    quarter = nb // 4     # phase +/-0.25 = quadrature
    edge = max(1, int(round(0.04 * nb)))  # outer 4% of each side ~ phase +/-0.5

    # Out-of-transit baseline and robust scatter, excluding the transit itself.
    oot = np.r_[edge:gc - 60, gc + 60:nb - edge]
    base = np.median(g[:, oot], axis=1)
    mad = 1.4826 * np.median(np.abs(g[:, oot] - base[:, None]), axis=1)
    safe_mad = np.where(mad > 0, mad, np.nan)

    def dip_significance(cols: np.ndarray) -> np.ndarray:
        """How many robust sigma below the out-of-transit baseline this region sits.

        A MEDIAN over the region, not a min: a min picks the deepest noise spike, which is
        why an earlier version of these features ranked noisy shallow candidates above real
        eclipsing binaries. Dividing by the out-of-transit MAD makes the number a
        significance rather than an amplitude, so it does not simply track SNR.
        """
        level = np.median(g[:, cols], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sig = (base - level) / safe_mad
        return np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

    # Phase 0.5 = the two EDGES of the view: where a circular-orbit secondary eclipse sits.
    secondary_sig = np.maximum(dip_significance(np.arange(0, edge)),
                               dip_significance(np.arange(nb - edge, nb)))
    # Phase +/-0.25 = quadrature: ellipsoidal variation, not a secondary eclipse.
    quadrature_sig = np.maximum(dip_significance(np.arange(max(0, gc - quarter - 30), max(1, gc - quarter + 30))),
                                dip_significance(np.arange(min(nb - 1, gc + quarter - 30), min(nb, gc + quarter + 30))))

    # Left/right halves of the SAME averaged transit: asymmetry, NOT an odd/even depth test.
    left = -g[:, :gc].min(axis=1)
    right = -g[:, gc:].min(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        asymmetry = np.where((left + right) > 0, np.abs(left - right) / (left + right), 0.0)

    # Width of the dip in the global view: fraction of bins below half depth.
    gdepth = -g.min(axis=1)
    below_half = (g < -(gdepth[:, None] / 2)).sum(axis=1) / g.shape[1]

    # gdepth (= -min of the global view) is 1.0 for every row by construction, because the
    # views are normalised so the deepest bin is exactly -1. It is therefore not a feature
    # and is not returned; it is only used above to scale duration_frac.
    return pd.DataFrame({
        "local_depth": local_depth,
        "local_floor": local_floor,
        "v_shape": v_shape,
        "local_snr": local_snr,
        "wing_level": wing_level,
        "wing_scatter": wing_scatter,
        "global_scatter": g.std(axis=1),
        "oot_mad": np.nan_to_num(mad),
        "duration_frac": below_half,
        "secondary_sig": secondary_sig,
        "quadrature_sig": quadrature_sig,
        "transit_asymmetry": asymmetry,
    })


# =========================================================================== #
# Baselines
# =========================================================================== #
def majority_baseline(y_train: np.ndarray, y_eval: np.ndarray) -> dict:
    """Predict the training-set majority class for everything. The floor."""
    counts = np.array([(y_train == i).sum() for i in range(len(CLASSES))])
    winner = int(counts.argmax())
    proba = np.zeros((len(y_eval), len(CLASSES)), float)
    proba[:, winner] = 1.0
    out = evaluate(y_eval, proba)
    out["_note"] = (f"always predicts {CLASSES[winner]}; AUCs are undefined or 0.5 by construction "
                    f"because the score is constant")
    return out


def logistic_baseline(tr: ViewSet, ev: ViewSet, seed: int) -> dict:
    """Multinomial logistic regression on the standardized, concatenated raw views."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    xtr = np.hstack([tr.global_view, tr.local_view])
    xev = np.hstack([ev.global_view, ev.local_view])
    # sklearn >= 1.7 dropped multi_class=; multinomial is the default for a multiclass target.
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))
    clf.fit(xtr, tr.label_index.astype(int))
    return evaluate(ev.label_index.astype(int), clf.predict_proba(xev))


def shape_gbm_baseline(tr: ViewSet, ev: ViewSet, seed: int) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """Gradient boosting on the 12 interpretable shape features.

    Returns metrics, the feature-importance table, and the eval-set probabilities
    (kept for the failure analysis, which needs per-row predictions).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    ftr, fev = shape_features(tr), shape_features(ev)
    ytr = tr.label_index.astype(int)
    # Balanced sample weights: HistGB has no class_weight argument.
    counts = np.bincount(ytr, minlength=len(CLASSES)).astype(float)
    w = (counts.mean() / np.where(counts > 0, counts, np.nan))[ytr]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=6, random_state=seed)
    clf.fit(ftr.values, ytr, sample_weight=np.nan_to_num(w, nan=0.0))
    proba = clf.predict_proba(fev.values)

    from sklearn.inspection import permutation_importance
    imp = permutation_importance(clf, fev.values, ev.label_index.astype(int),
                                 n_repeats=5, random_state=seed, scoring="accuracy")
    table = (pd.DataFrame({"feature": ftr.columns, "importance": imp.importances_mean,
                           "std": imp.importances_std})
             .sort_values("importance", ascending=False).reset_index(drop=True))
    return evaluate(ev.label_index.astype(int), proba), table, proba


# =========================================================================== #
# Reporting
# =========================================================================== #
def summarise(name: str, m: dict) -> str:
    roc = "n/a" if m["roc_auc_binary"] is None else f"{m['roc_auc_binary']:.4f}"
    pr = "n/a" if m["pr_auc_binary"] is None else f"{m['pr_auc_binary']:.4f}"
    p90 = next((r for r in m["precision_at_recall"] if r["target_recall"] == 0.90), None)
    p90s = f"{p90['precision']:.3f}" if p90 and p90.get("achievable") else "n/a"
    return f"  {name:<28} acc3 {m['accuracy_3class']:.4f}  ROC-AUC {roc:>6}  PR-AUC {pr:>6}  P@R90 {p90s:>5}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-views", type=Path, default=VIEWS_ROOT / "kepler_train")
    p.add_argument("--val-views", type=Path, default=VIEWS_ROOT / "kepler_val")
    p.add_argument("--test-views", type=Path, default=VIEWS_ROOT / "kepler_test")
    p.add_argument("--eval-test", action="store_true",
                   help="also score the held-out test views; off by default so tuning never touches them")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=MODEL_DIR / "runs" / "baselines")
    args = p.parse_args(argv)
    args.out = Path(args.out).resolve()  # so a relative --out still prints and saves correctly

    print("=" * 88)
    print("View-level baselines (majority / logistic / shape+GBM)")
    print("Checklist item: baselines before deep models, on the same input the CNN sees.")
    print("=" * 88)

    print("\n[1] views")
    tr = drop_unusable(consolidate_views(args.train_views))
    va = drop_unusable(consolidate_views(args.val_views))
    print(f"  train {len(tr)} rows / {len(set(tr.star_id.tolist()))} stars | "
          f"val {len(va)} rows / {len(set(va.star_id.tolist()))} stars")
    overlap = set(tr.star_id.tolist()) & set(va.star_id.tolist())
    if overlap:
        raise SystemExit(f"train/val share {len(overlap)} star(s) - splits are leaking: {sorted(overlap)[:5]}")
    print("  star-disjointness train vs val: verified")

    ytr, yva = tr.label_index.astype(int), va.label_index.astype(int)
    results, seed = {}, args.seed

    print("\n[2] baselines on VAL")
    results["majority"] = majority_baseline(ytr, yva)
    print(summarise("majority class", results["majority"]))

    results["logistic"] = logistic_baseline(tr, va, seed)
    print(summarise("logistic regression (views)", results["logistic"]))

    gbm_metrics, importance, val_proba = shape_gbm_baseline(tr, va, seed)
    results["shape_gbm"] = gbm_metrics
    print(summarise("shape features + GBM", gbm_metrics))

    print("\n[3] which shape features carry the signal (permutation importance on val)")
    for _, r in importance.iterrows():
        bar = "#" * max(0, int(round(r.importance * 300)))
        print(f"    {r.feature:<18} {r.importance:+.4f} +/- {r['std']:.4f}  {bar}")

    if args.eval_test:
        print("\n[4] TEST (single evaluation)")
        te = drop_unusable(consolidate_views(args.test_views))
        results["majority_test"] = majority_baseline(ytr, te.label_index.astype(int))
        print(summarise("majority class [test]", results["majority_test"]))
        results["logistic_test"] = logistic_baseline(tr, te, seed)
        print(summarise("logistic regression [test]", results["logistic_test"]))
        results["shape_gbm_test"] = shape_gbm_baseline(tr, te, seed)[0]
        print(summarise("shape features + GBM [test]", results["shape_gbm_test"]))
    else:
        print("\n[4] TEST not evaluated (pass --eval-test for the final run)")

    args.out.mkdir(parents=True, exist_ok=True)
    importance.to_csv(args.out / "shape_feature_importance.csv", index=False)
    np.save(args.out / "val_shape_gbm_proba.npy", val_proba)
    (args.out / "baselines.json").write_text(json.dumps(
        {"generated_at": datetime.now().isoformat(timespec="seconds"), "seed": seed,
         "n_train": len(tr), "n_val": len(va), "results": results}, indent=2, default=float), encoding="utf-8")
    print(f"\nsaved {args.out.relative_to(RESEARCH_DIR)}/ (baselines.json, shape_feature_importance.csv)")
    print("\nThese are the numbers the CNN+LSTM must beat to justify its complexity.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
