"""
calibrate.py
============

Post-hoc probability calibration of the planet-like score, for a finished run.

The ranking from cnn_lstm.py is good (Kepler test ROC-AUC 0.916) but the score is not a
probability: on Kepler test, KOIs scored 0.5-0.7 are planet-like only ~40 % of the time
(figures/05_probability_calibration.png, Brier 0.130). Anyone thresholding the raw score as
"a 70 % chance of a planet" would be wrong. This file fixes the probability scale without
touching the model.

Protocol (the same discipline as the rest of Stage A)
  * Everything is FITTED on the validation views only. The model is not retrained.
  * The calibration method is CHOSEN on validation too, by star-grouped 5-fold cross-validation
    (no star in both the fitting and the scoring fold), comparing:
        none      the raw score
        platt     logistic regression on logit(score)   - 2 parameters, smooth
        isotonic  monotone step function                - non-parametric, needs more data
  * The chosen calibrator is then fitted on ALL of validation and applied ONCE to the test
    predictions already saved by final_eval.py (test_predictions.csv). No model is re-run on
    test. Calibration is monotone, so it cannot change which candidates rank above which: it
    changes what the score MEANS, not what the model can separate. ROC-AUC before and after
    is printed to show exactly that (isotonic can merge near-ties, so it may move slightly).
  * Test was already spent by the final evaluation. Scoring the calibrator on it is a second
    look at test, but at a calibration map fitted and selected on validation; nothing about the
    model or the map is tuned on what test shows.

Metrics: Brier score, log loss, and expected calibration error (ECE, 10 equal-width bins,
bin-count weighted). Lower is better for all three.

Output, in <run>/calibration/:
    calibration.json             method, parameters (or isotonic knots), all metrics
    test_calibrated.csv          the saved test predictions plus `calibrated_score`
    reliability.png              before/after reliability diagram on test

Apply a saved calibrator elsewhere with `load_calibrator(run_dir)(scores)`.

Usage
    python stage_a_transit_model/calibrate.py --run stage_a_transit_model/runs/bn_aug_sched
    python stage_a_transit_model/calibrate.py --run stage_a_transit_model/runs/tess_bn_aug_sched \
           --val-views detection_views/tess_val
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

from stage_a_transit_model.cnn_lstm import (  # noqa: E402
    PLANET_LIKE_IDX, VIEWS_ROOT, consolidate_views, drop_unusable, planet_like_score,
)
from stage_a_transit_model.failure_analysis import load_run_model, score_views  # noqa: E402

METHODS = ("none", "platt", "isotonic")
N_BINS = 10
EPS = 1e-6


# =========================================================================== #
# Metrics
# =========================================================================== #
def ece(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected calibration error: |observed rate - mean prediction| per bin, weighted by count."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.sum() * abs(y[m].mean() - p[m].mean())
    return float(total / len(y))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    q = np.clip(p, EPS, 1 - EPS)
    return {"brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, q, labels=[0, 1])),
            "ece": ece(y, p), "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
            "n": int(len(y)), "positive_rate": float(y.mean())}


# =========================================================================== #
# Calibrators
# =========================================================================== #
def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, EPS, 1 - EPS)
    return np.log(q / (1 - q))


def fit(method: str, score: np.ndarray, y: np.ndarray) -> dict:
    """Fit one calibrator; returns a JSON-serialisable description of it."""
    if method == "none":
        return {"method": "none"}
    if method == "platt":
        from sklearn.linear_model import LogisticRegression

        # Large C = effectively unregularised: two parameters fitted on ~1,000 rows.
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(score)[:, None], y)
        return {"method": "platt", "a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(score, y)
        return {"method": "isotonic", "x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
    raise ValueError(f"unknown method {method!r}")


def apply(cal: dict, score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, float)
    if cal["method"] == "none":
        return score
    if cal["method"] == "platt":
        return 1.0 / (1.0 + np.exp(-(cal["a"] * _logit(score) + cal["b"])))
    if cal["method"] == "isotonic":
        return np.interp(score, cal["x"], cal["y"])  # same as IsotonicRegression.predict with clip
    raise ValueError(f"unknown method {cal['method']!r}")


def load_calibrator(run_dir: str | Path):
    """The saved calibrator of a run, as a function score -> calibrated probability."""
    cal = json.loads((Path(run_dir) / "calibration" / "calibration.json").read_text(encoding="utf-8"))
    return lambda s: apply(cal["calibrator"], s)


def cross_validate(score: np.ndarray, y: np.ndarray, stars: np.ndarray, seed: int) -> pd.DataFrame:
    """Out-of-fold metrics per method, star-grouped so no star is in both fit and score folds."""
    from sklearn.model_selection import StratifiedGroupKFold

    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = {m: np.zeros(len(y)) for m in METHODS}
    for tr, te in folds.split(score, y, stars):
        for m in METHODS:
            oof[m][te] = apply(fit(m, score[tr], y[tr]), score[te])
    return pd.DataFrame({m: metrics(y, oof[m]) for m in METHODS}).T


# =========================================================================== #
# Figure
# =========================================================================== #
def reliability_figure(y: np.ndarray, raw: np.ndarray, cal: np.ndarray, method: str,
                       title: str, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def curve(p):
        edges = np.linspace(0, 1, N_BINS + 1)
        idx = np.clip(np.digitize(p, edges) - 1, 0, N_BINS - 1)
        xs, ys = [], []
        for b in range(N_BINS):
            m = idx == b
            if m.sum() >= 5:          # a bin of a handful of rows says nothing
                xs.append(p[m].mean())
                ys.append(y[m].mean())
        return xs, ys

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.plot([0, 1], [0, 1], color="#b9b8b2", ls="--", lw=1.0, label="perfect calibration")
    for p, lab, col in [(raw, "raw score", "#eb6834"), (cal, f"calibrated ({method})", "#2a78d6")]:
        xs, ys = curve(p)
        ax.plot(xs, ys, "o-", color=col, label=f"{lab}: Brier {metrics(y, p)['brier']:.3f}, "
                                                f"ECE {ece(y, p):.3f}")
    ax.set(xlabel="predicted planet-like probability", ylabel="fraction actually planet-like",
           xlim=(0, 1), ylim=(0, 1.02), title=title)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# =========================================================================== #
# Main
# =========================================================================== #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="stage_a_transit_model/runs/bn_aug_sched")
    ap.add_argument("--val-views", type=Path, default=None,
                    help="validation views (default: kepler_val, or tess_val for a run named tess_*)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    run_dir = Path(args.run).resolve()
    if not (run_dir / "best.pt").is_file():
        run_dir = (RESEARCH_DIR / args.run).resolve()
    if not (run_dir / "best.pt").is_file():
        raise SystemExit(f"no best.pt in {run_dir}")
    mission = "tess" if run_dir.name.startswith("tess") else "kepler"
    val_views = args.val_views or VIEWS_ROOT / f"{mission}_val"
    test_csv = run_dir / "test_predictions.csv"
    if not test_csv.is_file():
        raise SystemExit(f"{test_csv} missing: run final_eval.py for this run first")

    print("=" * 88)
    print(f"Probability calibration | run {run_dir.name} | fitted and selected on {val_views.name}")
    print("=" * 88)

    model, arch = load_run_model(run_dir)
    vs = drop_unusable(consolidate_views(val_views, verbose=False))
    val_score = planet_like_score(score_views(model, vs))
    y_val = np.isin(vs.label_index.astype(int), PLANET_LIKE_IDX).astype(int)
    print(f"\n[1] validation: {len(vs)} rows, {len(set(vs.star_id.tolist()))} stars, "
          f"{y_val.mean():.1%} planet-like")

    print("\n[2] method selection: star-grouped 5-fold CV on validation (out-of-fold metrics)")
    cv = cross_validate(val_score, y_val, vs.star_id, args.seed)
    print(cv[["brier", "log_loss", "ece", "roc_auc"]].round(4).to_string())
    best = str(cv["brier"].astype(float).idxmin())
    print(f"  chosen by out-of-fold Brier: {best}")

    cal = fit(best, val_score, y_val)
    if best == "platt":
        print(f"  fitted on all validation: p = sigmoid({cal['a']:.4f} * logit(score) + {cal['b']:.4f})")
    elif best == "isotonic":
        print(f"  fitted on all validation: {len(cal['x'])} isotonic knots")

    print("\n[3] test: the saved final_eval predictions, calibrated once (no model re-run)")
    test = pd.read_csv(test_csv)
    y_test = test.true_label.isin(["CANDIDATE", "CONFIRMED"]).astype(int).to_numpy()
    raw = test.planet_like_score.to_numpy(float)
    calibrated = apply(cal, raw)
    before, after = metrics(y_test, raw), metrics(y_test, calibrated)
    print(f"  {'':<10} {'Brier':>8} {'log loss':>9} {'ECE':>7} {'ROC-AUC':>8}")
    for name, m in [("raw", before), (best, after)]:
        print(f"  {name:<10} {m['brier']:>8.4f} {m['log_loss']:>9.4f} {m['ece']:>7.4f} {m['roc_auc']:>8.4f}")
    print("  ROC-AUC is a ranking metric: a monotone map cannot improve it, and it should not")
    print("  move except where isotonic merges near-tied scores into one step.")

    out = run_dir / "calibration"
    out.mkdir(exist_ok=True)
    test.assign(calibrated_score=calibrated).to_csv(out / "test_calibrated.csv", index=False)
    reliability_figure(y_test, raw, calibrated, best,
                       f"{run_dir.name}: test reliability, calibrator fitted on validation",
                       out / "reliability.png")
    (out / "calibration.json").write_text(json.dumps({
        "run": run_dir.name, "generated_at": datetime.now().isoformat(timespec="seconds"),
        "architecture": arch, "fitted_on": val_views.name, "selected_by": "star-grouped 5-fold CV Brier on validation",
        "seed": args.seed, "calibrator": cal,
        "validation_cv": {m: {k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v))
                              for k, v in cv.loc[m].items()} for m in METHODS},
        "test_raw": before, "test_calibrated": after,
        "note": ("Monotone post-hoc map of the planet-like score; the model is unchanged. Fitted and "
                 "selected on validation only; applied once to the already-saved test predictions."),
    }, indent=2), encoding="utf-8")
    print(f"\nsaved {out.relative_to(RESEARCH_DIR)}/ (calibration.json, test_calibrated.csv, reliability.png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
