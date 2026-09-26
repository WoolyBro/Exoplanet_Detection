#!/usr/bin/env python3
"""
prune_fits_cache.py
===================

Frees disk space in the light-curve cache by removing raw Kepler FITS files that
the running Kepler TRAIN build no longer needs.

A star's FITS files are listed as safe ONLY if all of these hold, re-checked at
the moment the script runs:
  1. the star belongs to the Kepler train split (the build being run);
  2. every one of the star's train KOIs has a final record in the build's own
     checkpoint (detection_views/kepler_train/progress.jsonl): status "ok" with its
     .npz view present, or status "excluded";
  3. the star's stitched CSV (csv/kepler_KIC<id>_api.csv) exists, is non-empty and
     has no temporary sibling (the pipeline only reads this CSV after download);
  4. nothing in the star's FITS folder was modified in the last 15 minutes.
Anything else is skipped and reported: stars with failed or missing records (in
flight, not reached, or to be retried), non-train stars downloaded for
diagnostics, and any folder that does not parse as kplr<KIC>_lc_*.

Only *.fits files inside mast/mastDownload/Kepler/<star folder>/ are touched.
Nothing in detection_views/, the csv/ folder, or the build's logs is modified.

Usage:
    python tools/prune_fits_cache.py            # dry run: report only (default)
    python tools/prune_fits_cache.py --delete   # delete the files listed as safe
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path

import pandas as pd

RESEARCH = Path(__file__).resolve().parents[1]
BUILD_DIR = RESEARCH / "detection_views" / "kepler_train"
def _build_cache() -> Path:
    """The cache recorded in the build's latest session_start event (D:/lightkurve_cache if none)."""
    cache = Path("D:/lightkurve_cache")
    progress = BUILD_DIR / "progress.jsonl"
    if progress.is_file():
        with progress.open(encoding="utf-8") as fh:
            for line in fh:
                if '"session_start"' in line:
                    try:
                        cache = Path(json.loads(line).get("cache_dir", cache))
                    except json.JSONDecodeError:
                        pass
    return cache


CACHE = _build_cache()
FITS_ROOT = CACHE / "mast" / "mastDownload" / "Kepler"
CSV_DIR = CACHE / "csv"
RECENT_SECONDS = 15 * 60
FOLDER_RE = re.compile(r"^kplr(\d{9})_lc_Q\d+$")


def latest_records() -> dict[str, dict]:
    latest = {}
    with (BUILD_DIR / "progress.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "object_id" in rec:
                latest[rec["object_id"]] = rec
    return latest


def classify() -> tuple[list[dict], list[dict]]:
    # Read the master catalogue and filter on its own `split` column, rather than the
    # splits/kepler/train.csv copy. That per-mission file is an exact duplicate of these rows
    # (same columns, same count), and this was the only code still reading it.
    koi = pd.read_csv(RESEARCH / "splits" / "koi_cumulative_split.csv",
                      usecols=["kepid", "kepoi_name", "split"], dtype={"kepoi_name": str},
                      low_memory=False)
    train = koi[koi.split == "train"]
    kois_by_star = train.groupby("kepid").kepoi_name.apply(list).to_dict()
    latest = latest_records()
    now = time.time()
    safe, skipped = [], []
    for folder in sorted(p for p in FITS_ROOT.iterdir() if p.is_dir()):
        fits = sorted(folder.glob("*.fits"))
        size = sum(f.stat().st_size for f in fits)
        m = FOLDER_RE.match(folder.name)
        entry = {"folder": folder.name, "n_fits": len(fits), "bytes": size}
        if not m:
            skipped.append({**entry, "reason": "folder name does not parse as kplr<KIC>_lc_Q*"})
            continue
        kepid = int(m.group(1))
        entry["kepid"] = kepid
        if any(f.name[4:13] != m.group(1) for f in fits):
            skipped.append({**entry, "reason": "a FITS filename does not match the folder's KIC"})
            continue
        if kepid not in kois_by_star:
            skipped.append({**entry, "reason": "not a Kepler train star (val/test or diagnostic download)"})
            continue
        states = []
        for koi in kois_by_star[kepid]:
            rec = latest.get(koi)
            if rec is None:
                states.append(f"{koi}: no record (in flight or not reached)")
            elif rec["status"] == "ok" and (BUILD_DIR / f"{koi}.npz").is_file():
                continue
            elif rec["status"] == "excluded":
                continue
            else:
                states.append(f"{koi}: status {rec['status']} {rec.get('category', '')}".strip())
        if states:
            skipped.append({**entry, "reason": "; ".join(states)})
            continue
        csv = CSV_DIR / f"kepler_KIC{kepid}_api.csv"
        if not csv.is_file() or csv.stat().st_size == 0 or list(CSV_DIR.glob(f"kepler_KIC{kepid}_api.tmp*")):
            skipped.append({**entry, "reason": "stitched CSV missing, empty or still being written"})
            continue
        newest = max([folder.stat().st_mtime] + [f.stat().st_mtime for f in fits])
        if now - newest < RECENT_SECONDS:
            skipped.append({**entry, "reason": "folder modified in the last 15 minutes"})
            continue
        safe.append({**entry, "kois": kois_by_star[kepid]})
    return safe, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delete", action="store_true", help="delete the FITS files listed as safe")
    args = ap.parse_args()

    free_before = shutil.disk_usage(CACHE).free
    safe, skipped = classify()
    safe_bytes = sum(s["bytes"] for s in safe)
    print(f"stars with FITS in cache: {len(safe) + len(skipped)}")
    print(f"SAFE to delete: {len(safe)} stars, {sum(s['n_fits'] for s in safe)} FITS files, {safe_bytes / 1e9:.2f} GB")
    reasons: dict[str, int] = {}
    for s in skipped:
        key = s["reason"] if ":" not in s["reason"] else re.sub(r"K\d{5}\.\d{2}: ", "", s["reason"]).split(";")[0]
        reasons[key] = reasons.get(key, 0) + 1
    print(f"SKIPPED: {len(skipped)} stars, {sum(s['bytes'] for s in skipped) / 1e9:.2f} GB")
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {v:>4}  {k}")

    report = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "delete" if args.delete else "dry-run",
              "free_gb_before": round(free_before / 1e9, 2), "safe": safe, "skipped": skipped}
    if args.delete:
        deleted, failed, freed = 0, [], 0
        for s in safe:
            for f in sorted((FITS_ROOT / s["folder"]).glob("*.fits")):
                try:
                    size = f.stat().st_size
                    f.unlink()
                    deleted += 1
                    freed += size
                except OSError as exc:  # e.g. file still open: leave it
                    failed.append(f"{f.name}: {exc}")
        free_after = shutil.disk_usage(CACHE).free
        print(f"deleted {deleted} FITS files ({freed / 1e9:.2f} GB); failures {len(failed)}")
        for line in failed[:10]:
            print(f"  {line}")
        print(f"free space on D: {free_before / 1e9:.2f} GB -> {free_after / 1e9:.2f} GB")
        report.update({"deleted_files": deleted, "freed_gb": round(freed / 1e9, 3), "failures": failed,
                       "free_gb_after": round(free_after / 1e9, 2)})
    out = RESEARCH / "scripts" / f"prune_fits_cache_{report['mode']}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report: {out.relative_to(RESEARCH)}")
    if not args.delete:
        print("dry run only - nothing deleted. Run with --delete to remove the SAFE files.")


if __name__ == "__main__":
    main()
