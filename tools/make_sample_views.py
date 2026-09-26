#!/usr/bin/env python3
"""
make_sample_views.py
====================

Builds `detection_views_sample/` - a small, label-stratified slice of the built views that is
committed to the repository, so anyone who clones it can run the model, the baselines and the
figures immediately on real data.

The problem this solves: rebuilding the full view set means re-downloading ~120 GB of
photometry over roughly 30 hours. No reviewer will do that, and they should not have to in
order to check that the code works and does what the reports say.

What the sample is and is not:
  * IS   - real views, straight from the full build, with their real labels; enough of each
           class that every script runs end to end and produces sensible output.
  * IS NOT - a basis for any result. ~40 rows per split cannot reproduce a 0.9158 ROC-AUC,
           and every script that reads it should be understood as being smoke-tested, not
           evaluated. The reported numbers come from the full set.

Rows whose label_index is -1 (no harmonised label) are excluded: drop_unusable() would discard
them anyway, so including them would only make the sample smaller than it looks.

Usage
    python tools/make_sample_views.py                 # default 14 per class per split
    python tools/make_sample_views.py --per-class 25
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

RESEARCH = Path(__file__).resolve().parents[1]
VIEWS = RESEARCH / "detection_views"
SAMPLE = RESEARCH / "detection_views_sample"
SPLITS = ["kepler_train", "kepler_val", "kepler_test", "tess_train", "tess_val", "tess_test"]
CLASSES = ["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=14,
                    help="views per class per split (default 14 -> ~42 per split)")
    ap.add_argument("--scan", type=int, default=800,
                    help="how many files to scan per split when looking for each class")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not VIEWS.is_dir():
        raise SystemExit(f"no built views at {VIEWS}; nothing to sample")

    if SAMPLE.exists():
        shutil.rmtree(SAMPLE)
    SAMPLE.mkdir(parents=True)

    rng = np.random.default_rng(args.seed)
    manifest: dict[str, dict] = {}
    grand_files = grand_bytes = 0

    print("building a committable sample of the built views\n")
    for split in SPLITS:
        src = VIEWS / split
        if not src.is_dir():
            continue
        files = sorted(src.glob("*.npz"))
        if not files:
            continue
        # Scan a spread across the split rather than the first N, so the sample is not all
        # from one region of the catalogue.
        idx = rng.permutation(len(files))[:args.scan]
        by_class: dict[int, list[Path]] = {}
        for i in idx:
            f = files[i]
            try:
                with np.load(f, allow_pickle=False) as z:
                    lab = int(z["label_index"])
            except Exception:
                continue
            if lab < 0:                      # unlabelled: drop_unusable() would remove it
                continue
            bucket = by_class.setdefault(lab, [])
            if len(bucket) < args.per_class:
                bucket.append(f)
            if all(len(by_class.get(c, [])) >= args.per_class for c in range(len(CLASSES))):
                break

        dst = SAMPLE / split
        dst.mkdir(parents=True, exist_ok=True)
        picked, size = 0, 0
        counts = {}
        for lab, fs in sorted(by_class.items()):
            counts[CLASSES[lab] if lab < len(CLASSES) else str(lab)] = len(fs)
            for f in fs:
                shutil.copy2(f, dst / f.name)
                picked += 1
                size += f.stat().st_size
        grand_files += picked
        grand_bytes += size
        manifest[split] = {"n": picked, "bytes": size, "by_class": counts,
                           "of_total": len(files)}
        print(f"  {split:<14} {picked:>3} of {len(files):>5} views  {size/1e6:>5.2f} MB   {counts}")

    (SAMPLE / "SAMPLE_MANIFEST.json").write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "per_class": args.per_class, "seed": args.seed,
        "total_files": grand_files, "total_bytes": grand_bytes,
        "purpose": ("A committable slice of the full build so the code can be run immediately "
                    "after cloning. NOT a basis for any result - the reported numbers come "
                    "from the full view set."),
        "splits": manifest,
    }, indent=2), encoding="utf-8")

    readme = SAMPLE / "README.md"
    readme.write_text(
        "# Sample views\n\n"
        f"{grand_files} real views ({grand_bytes/1e6:.1f} MB) taken from the full build, "
        "label-stratified, so the code in this repository can be run straight after cloning.\n\n"
        "**These are for smoke-testing, not for results.** About 40 rows per split cannot "
        "reproduce the reported metrics; they exist so a reviewer can confirm the pipeline "
        "runs and produces sensible output without rebuilding the full set, which means "
        "re-downloading roughly 120 GB of photometry over about 30 hours.\n\n"
        "```bash\n"
        "python stage_a_transit_model/baselines.py \\\n"
        "    --train-views detection_views_sample/kepler_train \\\n"
        "    --val-views   detection_views_sample/kepler_val\n\n"
        "python stage_a_transit_model/cnn_lstm.py --epochs 3 \\\n"
        "    --train-views detection_views_sample/kepler_train \\\n"
        "    --val-views   detection_views_sample/kepler_val\n"
        "```\n\n"
        "The full view set is linked from the top-level README.\n",
        encoding="utf-8")

    print(f"\n  total: {grand_files} views, {grand_bytes/1e6:.2f} MB -> "
          f"{SAMPLE.relative_to(RESEARCH)}/")
    print("  wrote SAMPLE_MANIFEST.json and README.md alongside")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
