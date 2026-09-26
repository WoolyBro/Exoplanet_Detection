"""
error_buckets.py
================

Checklist item 7.3: "Error analysis - bucket false negatives (physically-undetectable vs
noise-dominated)."

A false negative is only a model failure if the signal was there to be found. This splits the
planets the model misses into three buckets, using the signal-to-noise the Kepler pipeline
itself measured (koi_model_snr) and the out-of-transit scatter measured from the view:

  1. below detection   - koi_model_snr < 7.1, Kepler's own detection threshold. No algorithm
                         reading this light curve should be expected to recover these; counting
                         them as model errors flatters or damns the model for the wrong reason.
  2. noise-dominated   - SNR above threshold, but the out-of-transit scatter is in the worst
                         quartile of the sample. The signal exists but sits in a bad light curve.
  3. genuine miss      - adequate SNR, ordinary noise. These are the real failures and the only
                         ones worth trying to fix with a better model.

The same split is applied to the planets the model gets RIGHT, because a bucket is only
meaningful next to its base rate: if 40% of everything is low-SNR, a false-negative set that is
40% low-SNR tells you nothing.

Usage
    python stage_a_transit_model/error_buckets.py --run stage_a_transit_model/runs/bn_aug_sched
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

import data_manifest as dm  # noqa: E402
from stage_a_transit_model.baselines import shape_features  # noqa: E402
from stage_a_transit_model.cnn_lstm import (  # noqa: E402
    CLASSES, KOI_SPLIT_FILE, PLANET_LIKE_IDX, VIEWS_ROOT,
    consolidate_views, drop_unusable, planet_like_score, precision_at_recall,
)
from stage_a_transit_model.failure_analysis import load_run_model, score_views  # noqa: E402

KEPLER_SNR_THRESHOLD = 7.1  # the DR25 pipeline's own detection threshold


def bucket(df: pd.DataFrame, noisy_cut: float) -> pd.Series:
    snr = df.get("koi_model_snr")
    if snr is None:
        return pd.Series(["unknown"] * len(df), index=df.index)
    out = np.where(snr.isna(), "unknown",
                   np.where(snr < KEPLER_SNR_THRESHOLD, "below detection",
                            np.where(df.oot_mad > noisy_cut, "noise-dominated", "genuine miss")))
    return pd.Series(out, index=df.index)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="stage_a_transit_model/runs/bn_aug_sched")
    ap.add_argument("--views", type=Path, default=VIEWS_ROOT / "kepler_val")
    ap.add_argument("--recall", type=float, default=0.90)
    args = ap.parse_args(argv)

    run_dir = Path(args.run).resolve()
    if not (run_dir / "best.pt").is_file():
        run_dir = (RESEARCH_DIR / args.run).resolve()

    print("=" * 88)
    print(f"False-negative buckets | run {run_dir.name} | {args.views.name}")
    print("=" * 88)

    model, _ = load_run_model(run_dir)
    vs = drop_unusable(consolidate_views(args.views, verbose=False))
    proba = score_views(model, vs)
    score = planet_like_score(proba)
    y = vs.label_index.astype(int)
    y_bin = np.isin(y, PLANET_LIKE_IDX).astype(int)

    par = precision_at_recall(y_bin, score, targets=(args.recall,))[0]
    thr = par["threshold"]
    print(f"\n[1] operating point at recall >= {args.recall:.0%}: threshold {thr:.6f} "
          f"(precision {par['precision']:.3f})")

    with contextlib.redirect_stdout(io.StringIO()):
        koi = dm.load_stage1_catalogs()[KOI_SPLIT_FILE]
    cols = [c for c in ["kepoi_name", "koi_model_snr", "koi_depth", "koi_period", "koi_prad"]
            if c in koi.columns]
    df = pd.DataFrame({"kepoi_name": vs.object_id, "true": y, "score": score,
                       "planet": y_bin.astype(bool)})
    df = pd.concat([df, shape_features(vs)[["oot_mad", "local_snr"]]], axis=1)
    df = df.merge(koi[cols], on="kepoi_name", how="left")

    noisy_cut = df.oot_mad.quantile(0.75)
    df["bucket"] = bucket(df, noisy_cut)
    df["missed"] = df.planet & (df.score < thr)

    planets = df[df.planet]
    fn = planets[planets.missed]
    print(f"\n[2] of {len(planets)} true planet-like rows, {len(fn)} are missed at this operating point")
    print(f"  noise cut = worst-quartile out-of-transit scatter (oot_mad > {noisy_cut:.3f})")

    print(f"\n  {'bucket':<18} {'missed':>8} {'all planets':>12} {'miss rate':>10}")
    print("  " + "-" * 52)
    for b in ["below detection", "noise-dominated", "genuine miss", "unknown"]:
        n_all = int((planets.bucket == b).sum())
        n_fn = int((fn.bucket == b).sum())
        if n_all == 0:
            continue
        print(f"  {b:<18} {n_fn:>8} {n_all:>12} {100 * n_fn / n_all:>9.1f}%")

    share = (fn.bucket.value_counts(normalize=True) * 100).round(1).to_dict()
    base = (planets.bucket.value_counts(normalize=True) * 100).round(1).to_dict()
    print(f"\n  composition of the misses : {share}")
    print(f"  composition of all planets: {base}")
    print("  A bucket only matters where its share of the misses exceeds its share of the whole.")

    genuine = fn[fn.bucket == "genuine miss"]
    print(f"\n[3] the {len(genuine)} genuine misses - adequate SNR, ordinary noise")
    if len(genuine):
        show = genuine.sort_values("score").head(10)
        print(f"  {'KOI':<12} {'score':>7} {'SNR':>9} {'depth ppm':>10} {'period d':>9} {'Rp Re':>7}")
        for _, r in show.iterrows():
            print(f"  {r.kepoi_name:<12} {r.score:>7.3f} {r.get('koi_model_snr', np.nan):>9.1f} "
                  f"{r.get('koi_depth', np.nan):>10.0f} {r.get('koi_period', np.nan):>9.2f} "
                  f"{r.get('koi_prad', np.nan):>7.2f}")
        for c in ["koi_period", "koi_prad", "koi_depth"]:
            if c in genuine.columns and genuine[c].notna().any():
                print(f"  median {c:<12} genuine misses {genuine[c].median():>10.2f}  "
                      f"vs all planets {planets[c].median():>10.2f}")

    out = run_dir / "error_buckets"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "val_buckets.csv", index=False)
    (out / "summary.json").write_text(json.dumps({
        "run": run_dir.name, "views": args.views.name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "recall_target": args.recall, "threshold": thr,
        "snr_threshold": KEPLER_SNR_THRESHOLD, "noise_cut_oot_mad": float(noisy_cut),
        "n_planets": int(len(planets)), "n_missed": int(len(fn)),
        "missed_composition_pct": share, "all_planets_composition_pct": base,
        "n_genuine_miss": int(len(genuine)),
    }, indent=2, default=float), encoding="utf-8")
    print(f"\nsaved {out.relative_to(RESEARCH_DIR)}/ (val_buckets.csv, summary.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
