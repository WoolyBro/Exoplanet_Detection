#!/usr/bin/env python3
"""
clean_workspace.py
==================

Removes clutter that no longer serves anything: compiled-python caches, finished builds'
stdout/stderr, and the September backup folder. Dry run by default.

What it will NOT touch, by design:
  * detection_views/        the built views - 153 MB, regenerable, but only by re-downloading
                            ~30 GB of photometry over many hours. Gitignored instead.
  * stage_a_transit_model/cache/  116 MB, but rebuilt automatically in ~80 s. Gitignored; delete by
                            hand if the disk is tight.
  * exoplanet_research_data/  the mentor's pack. Never touched by anything in this project.
  * any .npz, .pt, report, figure, split catalog or run_log.txt.

Each category is listed with its size before anything happens, and `--delete` is required to
act. Categories can be chosen individually.

Usage
    python tools/clean_workspace.py                       # dry run, show everything
    python tools/clean_workspace.py --delete              # remove all categories below
    python tools/clean_workspace.py --delete --only pycache logs
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]


def gather() -> dict[str, tuple[str, list[Path]]]:
    """Each category: a human description and the exact paths that would be removed."""
    cats: dict[str, tuple[str, list[Path]]] = {}

    cats["pycache"] = (
        "__pycache__ folders and .pyc files (regenerated on next import)",
        [p for p in RESEARCH.rglob("__pycache__")
         if p.is_dir() and "exoplanet_research_data" not in str(p)],
    )
    cats["logs"] = (
        "stdout_/stderr_ files from finished builds (the builds' own "
        "detection_views/*/build_progress.txt keeps the real history)",
        sorted(RESEARCH.glob("stdout_*.txt")) + sorted(RESEARCH.glob("stderr_*.txt")),
    )
    cats["backup"] = (
        "_cleanup_backup_2026-09-19/ - superseded by the current tree",
        [p for p in RESEARCH.glob("_cleanup_backup_*") if p.is_dir()],
    )
    # splits/<mission>/{train,val,test}.csv are exact duplicates of the master catalogue's rows
    # (verified: same columns, same counts, same star->KOI mapping). The only code that read
    # them - prune_fits_cache.py - now filters the master on its own `split` column instead.
    dup_splits = []
    for mission in ("kepler", "tess", "k2"):
        d = RESEARCH / "splits" / mission
        if d.is_dir() and any(d.glob("*.csv")):
            dup_splits.append(d)
    cats["dupsplits"] = (
        "splits/{kepler,tess,k2}/ - exact duplicates of the master split catalogues, which "
        "carry the same rows under a `split` column. Nothing reads them any more.",
        dup_splits,
    )
    cats["trainlogs"] = (
        "stage_a_transit_model/logs/ - per-run training console output; every run's numbers are "
        "already in its report.json",
        [RESEARCH / "stage_a_transit_model" / "logs"] if (RESEARCH / "stage_a_transit_model" / "logs").is_dir() else [],
    )
    return cats


def size_of(paths: list[Path]) -> tuple[int, int]:
    total = count = 0
    for p in paths:
        if p.is_dir():
            for f in p.rglob("*"):
                if f.is_file():
                    try:
                        total += f.stat().st_size
                        count += 1
                    except OSError:
                        pass
        elif p.is_file():
            try:
                total += p.stat().st_size
                count += 1
            except OSError:
                pass
    return total, count


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delete", action="store_true", help="actually remove (default: dry run)")
    ap.add_argument("--only", nargs="+", metavar="CATEGORY",
                    help="limit to these categories (see the listing)")
    args = ap.parse_args()

    cats = gather()
    if args.only:
        unknown = set(args.only) - set(cats)
        if unknown:
            raise SystemExit(f"unknown category {sorted(unknown)}; choose from {sorted(cats)}")
        cats = {k: v for k, v in cats.items() if k in args.only}

    print("=" * 84)
    print(f"Workspace clean  |  {'DELETING' if args.delete else 'DRY RUN - nothing is removed'}")
    print("=" * 84)

    grand = 0
    for name, (desc, paths) in cats.items():
        size, count = size_of(paths)
        grand += size
        print(f"\n  [{name}]  {size/1e6:.2f} MB in {count} file(s)")
        print(f"    {desc}")
        for p in paths[:5]:
            print(f"      {p.relative_to(RESEARCH)}")
        if len(paths) > 5:
            print(f"      ... and {len(paths) - 5} more")
        if args.delete:
            removed = 0
            for p in paths:
                try:
                    if p.is_dir():
                        shutil.rmtree(p)
                    elif p.is_file():
                        p.unlink()
                    removed += 1
                except OSError as exc:
                    print(f"      could not remove {p.name}: {exc}")
            print(f"    removed {removed}/{len(paths)} entries")

    print(f"\n  total {'freed' if args.delete else 'reclaimable'}: {grand/1e6:.2f} MB")
    if not args.delete:
        print("\n  re-run with --delete to remove them")
    print("\n  Note: the big items (detection_views 153 MB, stage_a_transit_model/cache 116 MB,")
    print("  exoplanet_research_data 371 MB) are deliberately NOT touched here. They are")
    print("  excluded from git by .gitignore instead, because they are either regenerable")
    print("  at real cost or not ours to redistribute.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
