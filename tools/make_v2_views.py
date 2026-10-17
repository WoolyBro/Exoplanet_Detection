#!/usr/bin/env python3
"""
make_v2_views.py
================

Lays out detection_views_v2/<mission>_<split>/ for split v2 (split_signal_families.py) WITHOUT
downloading or rebuilding anything.

A view depends only on the KOI/TOI, never on which split it is in, so v2 reuses the views already
built. The source is detection_views_err/ - the error-channel rebuild of all six v1 splits - because
those files carry both the depth views (identical to the originals to 3e-8) and the per-bin error
channel, so one v2 layout serves both the with- and without-errors experiments. Each v2 folder holds
symbolic links, one per KOI/TOI, named exactly as preprocessing_pipeline names its views.

KOIs with no built view (excluded for an unreliable ephemeris, no light curve, ...) simply have no
link, exactly as in v1. The script prints how many rows of each v2 split have a view, and refuses to
link a view into a split its object does not belong to.

Usage
    python tools/make_v2_views.py            # (re)create detection_views_v2/
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH))

from preprocessing_pipeline import _npz_name  # noqa: E402  (the one naming rule views use)

SOURCE = RESEARCH / "detection_views_err"
TARGET = RESEARCH / "detection_views_v2"
MISSIONS = {
    "kepler": ("splits/koi_cumulative_split_v2.csv", lambda d: d["kepoi_name"].astype(str)),
    "tess": ("splits/toi_tess_candidates_split_v2.csv", lambda d: "TOI " + d["toi"].astype(str)),
}


def main() -> int:
    for mission, (split_file, object_id) in MISSIONS.items():
        cat = pd.read_csv(RESEARCH / split_file, low_memory=False)
        cat["object_id"] = object_id(cat)
        # Every view built for this mission, from any v1 split folder.
        available = {}
        for folder in sorted(SOURCE.glob(f"{mission}_*")):
            for f in folder.glob("*.npz"):
                if f.name in available:
                    raise SystemExit(f"{f.name} exists in two source folders")
                available[f.name] = f
        for split in ("train", "val", "test"):
            out = TARGET / f"{mission}_{split}"
            if out.exists():
                shutil.rmtree(out)
            out.mkdir(parents=True)
            rows = cat[cat.split == split]
            linked = 0
            for oid in rows.object_id:
                src = available.get(_npz_name(oid))
                if src is not None:
                    (out / src.name).symlink_to(src.resolve())
                    linked += 1
            print(f"  {mission}_{split:<5}  {len(rows):>5} catalogue rows -> {linked:>5} views linked "
                  f"({100 * linked / max(len(rows), 1):.1f}%)")
    print(f"\nwrote {TARGET.relative_to(RESEARCH)}/ (symbolic links into {SOURCE.relative_to(RESEARCH)}/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
