#!/usr/bin/env python3
"""
prune_s3_cache.py
=================

Frees space in the light-curve cache by removing raw S3 downloads that a build no longer
needs. This exists because of a real bug: raw_cache_paths() named only the Kepler S3 folder
(`s3/kplr<id>`), so --discard-raw silently never removed any TESS download (`s3/tic<id>`).
The cache reached 22.7 GB over 1,485 stars before it was noticed on 2026-09-25. The pipeline
is fixed; this clears what accumulated meanwhile.

A star's folder is listed as safe ONLY if all of these hold, re-checked when the script runs:
  1. the star has at least one final record in that build's progress.jsonl;
  2. EVERY record for the star is final - "ok" with its .npz present on disk, or "excluded".
     A star with any failed or missing record is kept, because it will be retried and the
     download would otherwise have to be fetched again;
  3. nothing in the folder was modified in the last 10 minutes (it could be in flight).

Only folders under <cache>/s3/ are touched. Nothing in detection_views/, csv/, mast/ or any
build log is read for writing or modified.

Usage:
    python tools/prune_s3_cache.py                    # dry run: report only (default)
    python tools/prune_s3_cache.py --delete           # remove the folders listed as safe
    python tools/prune_s3_cache.py --build tess_train # pick the build (default: all builds)
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]
VIEWS = RESEARCH / "detection_views"
RECENT_S = 600  # a folder touched this recently may belong to a star still being processed


def cache_dir_for(build: Path) -> Path | None:
    """The cache this build recorded in its latest session_start event."""
    progress = build / "progress.jsonl"
    if not progress.is_file():
        return None
    cache = None
    with progress.open(encoding="utf-8") as fh:
        for line in fh:
            if '"session_start"' in line:
                try:
                    c = json.loads(line).get("cache_dir")
                    if c:
                        cache = Path(c)
                except json.JSONDecodeError:
                    pass
    return cache


def finished_stars(build: Path) -> tuple[set[int], set[int]]:
    """(stars whose records are all final, stars with any unfinished record)."""
    final: dict[int, bool] = {}
    for line in (build / "progress.jsonl").open(encoding="utf-8"):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        star, status = rec.get("star_id"), rec.get("status")
        if star is None or status is None:
            continue
        if status == "ok":
            ok = bool(rec.get("file")) and (build / rec["file"]).is_file()
        elif status == "excluded":
            ok = True
        else:
            ok = False
        # A star is clean only if every one of its records is final.
        final[star] = final.get(star, True) and ok
    done = {s for s, ok in final.items() if ok}
    return done, set(final) - done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delete", action="store_true", help="actually remove (default: dry run)")
    ap.add_argument("--build", default=None, help="a folder under detection_views/ (default: all)")
    args = ap.parse_args()

    builds = ([VIEWS / args.build] if args.build
              else [d for d in sorted(VIEWS.glob("*")) if (d / "progress.jsonl").is_file()])
    print("=" * 84)
    print(f"S3 cache prune  |  {'DELETE' if args.delete else 'DRY RUN (nothing is removed)'}")
    print("=" * 84)

    total_free, total_keep, now = 0, 0, time.time()
    for build in builds:
        cache = cache_dir_for(build)
        s3root = (cache / "s3") if cache else None
        if not s3root or not s3root.is_dir():
            continue
        done, pending = finished_stars(build)
        mission_prefix = "tic" if build.name.startswith("tess") else "kplr"
        print(f"\n{build.name}: cache {cache}")
        print(f"  stars with all records final: {len(done)}   still unfinished: {len(pending)}")

        freed = kept = skipped_recent = 0
        for folder in s3root.glob(f"{mission_prefix}*"):
            if not folder.is_dir():
                continue
            try:
                star = int(folder.name[len(mission_prefix):])
            except ValueError:
                continue
            size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
            if star not in done:
                kept += size
                continue
            if folder.stat().st_mtime > now - RECENT_S:
                skipped_recent += 1
                kept += size
                continue
            freed += size
            if args.delete:
                try:
                    shutil.rmtree(folder)
                except OSError as exc:
                    print(f"    could not remove {folder.name}: {exc}")
                    freed -= size
        print(f"  {'freed' if args.delete else 'would free'}: {freed / 1e9:.2f} GB")
        print(f"  kept (unfinished star, or touched in the last {RECENT_S // 60} min): "
              f"{kept / 1e9:.2f} GB" + (f", {skipped_recent} recent" if skipped_recent else ""))
        total_free += freed
        total_keep += kept

    print(f"\ntotal {'freed' if args.delete else 'reclaimable'}: {total_free / 1e9:.2f} GB")
    print(f"total kept: {total_keep / 1e9:.2f} GB")
    if not args.delete and total_free > 0:
        print("\nre-run with --delete to remove them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
