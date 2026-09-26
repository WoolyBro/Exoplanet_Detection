"""
score_priority_targets.py
=========================

Fetch TESS light curves for the 54 Stage C priority planets and score them with the transit
model, so `transit_ML_probability` can be a real prediction instead of the template's 0.99.

Why this is a separate job from the big TESS build: **none of the 54 are in the TOI split.**
They are WASP / HD / HAT-P / GJ planets; the split being downloaded is TOI candidates. The big
build trains a TESS model; this fetches the 54 targets themselves. All 54 TIC ids resolve from
pscomppars locally, so no name lookup against MAST is needed.

Ephemerides come from pscomppars (period, transit midpoint in BJD, duration in hours). Many of
these midpoints are from the early 2000s - propagating one forward ~18 years to the TESS epoch
accumulates the period uncertainty, so a high "unreliable ephemeris" exclusion rate is expected
and is a real limitation, not a bug. The pipeline's own reliability check decides, exactly as
it does for KOIs and TOIs.

**The score this produces is CROSS-MISSION and UNVALIDATED.** The model was trained on Kepler
30-minute photometry; TESS is 2-minute, a different noise regime and a much shorter baseline.
Until a TESS-trained model exists (the big build), these numbers are an indication, not a
measurement, and every output here is labelled that way.

Usage
    python stage_c_priority_fusion/score_priority_targets.py            # fetch + build + score
    python stage_c_priority_fusion/score_priority_targets.py --score-only
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

import preprocessing_pipeline as p  # noqa: E402

TEMPLATE = (RESEARCH_DIR / "exoplanet_research_data" / "05_final_ML_dataset_DO_NOT_USE"
            / "final_priority_table_TEMPLATE.csv")
PSCOMP = (RESEARCH_DIR / "exoplanet_research_data" / "01_candidate_catalogs"
          / "pscomppars_confirmed_planets.csv")
VIEWS_OUT = RESEARCH_DIR / "detection_views" / "stage_c_targets"
OUT_DIR = RESEARCH_DIR / "outputs" / "stage_c"
# The default model run is chosen in main(); see the note there.


def build_input_frame() -> pd.DataFrame:
    """One row per priority planet, shaped the way ephemeris_from_row(mission='TESS') expects."""
    tpl = pd.read_csv(TEMPLATE)
    ps = pd.read_csv(PSCOMP, low_memory=False)
    cols = ["pl_name", "tic_id", "pl_orbper", "pl_tranmid", "pl_trandur",
            "pl_orbpererr1", "pl_tranmiderr1"]
    have = [c for c in cols if c in ps.columns]
    sub = ps[ps.pl_name.isin(tpl.pl_name)][have].copy()
    sub = (sub.assign(_n=sub.notna().sum(axis=1)).sort_values("_n", ascending=False)
              .drop_duplicates("pl_name").drop(columns="_n"))

    before = len(sub)
    sub = sub.dropna(subset=["tic_id", "pl_orbper", "pl_tranmid", "pl_trandur"])
    print(f"  {len(sub)}/{before} planets have TIC id + period + midpoint + duration; "
          f"{before - len(sub)} dropped for an incomplete ephemeris")

    sub["tid"] = sub.tic_id.astype(str).str.replace("TIC", "", regex=False).str.strip().astype(int)
    sub["object_name"] = sub.pl_name
    sub["pl_trandurh"] = sub.pl_trandur          # pscomppars gives hours, same unit as the TOI split
    sub["label_harmonized"] = "CONFIRMED"        # every priority-table planet is a confirmed planet
    sub["toi"] = np.nan
    return sub.reset_index(drop=True)


def score(views_dir: Path, run: Path, column: str) -> pd.DataFrame:
    from stage_a_transit_model.cnn_lstm import PLANET_LIKE_IDX, consolidate_views, drop_unusable
    from stage_a_transit_model.failure_analysis import load_run_model, score_views

    vs = drop_unusable(consolidate_views(views_dir, verbose=False))
    model, _ = load_run_model(run)
    proba = score_views(model, vs)
    return pd.DataFrame({
        "pl_name": vs.object_id, "tic_id": vs.star_id,
        column: proba[:, PLANET_LIKE_IDX].sum(axis=1),
        "p_false_positive": proba[:, 0], "p_candidate": proba[:, 1], "p_confirmed": proba[:, 2],
    }).sort_values(column, ascending=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score-only", action="store_true", help="skip the fetch, score existing views")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--run", type=Path, default=None,
                    help="model run directory. Default: the TESS-trained model, which is the "
                         "right one for these targets - see --help notes")
    args = ap.parse_args(argv)

    # The TESS model is the correct choice here and the Kepler one is demonstrably not: on this
    # exact set the Kepler model rejected KELT-9 b, HD 189733 b, WASP-19 b and the other
    # canonical hot Jupiters, because in Kepler's training split 93% of transits deeper than
    # 10,000 ppm are false positives. In the TOI split that figure is 11.5%, below its own base
    # rate, so the TESS model carries no such prior. These are TESS light curves either way.
    run = Path(args.run).resolve() if args.run else (RESEARCH_DIR / "stage_a_transit_model" / "runs"
                                                     / "tess_bn_aug_sched")
    same_mission = "tess" in run.name.lower()
    column = ("transit_ML_probability_TESS" if same_mission
              else "transit_ML_probability_TESS_crossmission")

    print("=" * 88)
    print("Stage C targets: fetch TESS light curves for the 54 priority planets and score them")
    print("CROSS-MISSION: the model is Kepler-trained. These are indications, not measurements.")
    print("=" * 88)

    print("\n[1] ephemerides from pscomppars")
    frame = build_input_frame()

    if not args.score_only:
        print(f"\n[2] building views for {len(frame)} targets via the AWS mirror")
        index = p.build_dataset(frame, VIEWS_OUT, n_workers=args.workers, discard_raw=True,
                                min_free_ram_gb=0.4, progress_every=10, verbose=True,
                                use_s3=True, isolate_workers=True, stall_timeout_s=600)
        print(f"\n  status: {dict(index.status.value_counts())}")
        if "category" in index.columns:
            cats = index[index.status != "ok"].category.value_counts()
            for c, n in cats.items():
                print(f"    {c}: {n}")
    else:
        print("\n[2] --score-only: using the views already built")

    if not VIEWS_OUT.is_dir() or not any(VIEWS_OUT.glob("*.npz")):
        print("\nno views were produced; nothing to score")
        return 1

    print(f"\n[3] scoring with {run.name} "
          f"({'TESS-trained, same mission as these light curves' if same_mission else 'CROSS-MISSION'})")
    scores = score(VIEWS_OUT, run, column)
    print(f"  scored {len(scores)} of the {len(frame)} attempted targets\n")
    print(f"  {'planet':<18} {'p(planet-like)':>15} {'p(FP)':>7} {'p(CAND)':>8} {'p(CONF)':>8}")
    for _, r in scores.iterrows():
        print(f"  {r.pl_name:<18} {r[column]:>15.4f} "
              f"{r.p_false_positive:>7.3f} {r.p_candidate:>8.3f} {r.p_confirmed:>8.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUT_DIR / "priority_target_transit_scores.csv", index=False)

    # These are all CONFIRMED planets, so a well-behaved model should score them high.
    # That makes the set a sanity check on cross-mission transfer, not a classification test.
    high = float((scores[column] >= 0.5).mean())
    print(f"\n[4] sanity check: every one of these is a CONFIRMED planet, so a transferring model")
    print(f"  should score them high. Fraction scored >= 0.5: {high:.0%}")
    print(f"  median p(planet-like): {scores[column].median():.4f}")
    print("  This measures cross-mission transfer, not classification skill: there are no")
    print("  negatives here, so it cannot produce a precision or a recall.")

    (OUT_DIR / "priority_target_scores_summary.json").write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_run": run.name,
        "trained_on": "TESS 2-minute photometry" if same_mission else "Kepler 30-minute photometry",
        "scored_on": "TESS 2-minute photometry" + ("" if same_mission else " (CROSS-MISSION)"),
        "score_column": column,
        "n_attempted": int(len(frame)), "n_scored": int(len(scores)),
        "fraction_ge_0.5": high,
        "median_planet_like": float(scores[column].median()),
        "caveat": ("All targets are confirmed planets, so this is a transfer sanity check with no "
                   "negative class - it cannot yield precision or recall. Replace with a "
                   "TESS-trained model once the TOI build finishes."),
    }, indent=2), encoding="utf-8")
    print(f"\nsaved {(OUT_DIR / 'priority_target_transit_scores.csv').relative_to(RESEARCH_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
