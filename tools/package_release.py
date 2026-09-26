#!/usr/bin/env python3
"""
package_release.py
==================

Bundles the built views into a single archive for a GitHub Release, so a reviewer can
reproduce the reported numbers with one ~143 MB download instead of a 30-hour rebuild.

Why a Release asset rather than a commit: git keeps every version of every file forever, so
committing 15,185 binaries and rebuilding them once would leave the repository permanently
carrying both copies. A Release asset sits beside the repository, can be replaced, and never
enters git history.

The archive is stored (not deflated). .npz files are already zlib-compressed internally, so
re-compressing them gains about 2% for a large cost in time.

Usage
    python tools/package_release.py
    python tools/package_release.py --out D:/somewhere/views.zip
"""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from datetime import datetime
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]
VIEWS = RESEARCH / "detection_views"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=RESEARCH / "detection_views_full.zip")
    ap.add_argument("--compress", action="store_true",
                    help="deflate instead of store (~2%% smaller, much slower on 15k files)")
    args = ap.parse_args()

    if not VIEWS.is_dir():
        raise SystemExit(f"no views at {VIEWS}")

    # Include the per-star checkpoints: they are what make a build resumable, and they carry
    # the exclusion reasons that the reports refer to.
    files = [f for f in VIEWS.rglob("*") if f.is_file()]
    npz = [f for f in files if f.suffix == ".npz"]
    total = sum(f.stat().st_size for f in files)
    print(f"packaging {len(files)} files ({len(npz)} views), {total/1e6:.1f} MB")
    print(f"  -> {args.out}")

    mode = zipfile.ZIP_DEFLATED if args.compress else zipfile.ZIP_STORED
    started = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", mode) as zf:
        # A manifest first, so whoever downloads it knows what they have.
        by_split = {}
        for d in sorted(p for p in VIEWS.iterdir() if p.is_dir()):
            n = len(list(d.glob("*.npz")))
            if n:
                by_split[d.name] = n
        zf.writestr("VIEWS_MANIFEST.json", json.dumps({
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "views_by_split": by_split,
            "total_view_files": len(npz),
            "unpack_to": "detection_views/ at the repository root",
            "note": ("Distilled from roughly 120 GB of Kepler and TESS photometry, which was "
                     "streamed and discarded per star and never stored. Each view pair is a "
                     "2001-bin global and 201-bin local phase-folded light curve."),
        }, indent=2))
        for i, f in enumerate(files, 1):
            zf.write(f, str(f.relative_to(VIEWS.parent)))
            if i % 3000 == 0:
                print(f"    {i}/{len(files)} ...", flush=True)
    tmp.replace(args.out)

    size = args.out.stat().st_size
    print(f"\n  wrote {size/1e6:.1f} MB in {time.time()-started:.0f}s "
          f"({100*size/total:.0f}% of the loose files)")
    print(f"  splits: {by_split}")
    print("\n  upload as a GitHub Release asset (2 GB per-file limit), then put its URL")
    print("  into the README where the full-view-set link is described.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
