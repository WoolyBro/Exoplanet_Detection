"""
build_pyatmos_final.py
======================

Builds the corrected PyATMOS Stage 2 training table.

D:/Files/Dataset/run_summary_final.csv (the previous Stage 2 source) missed
dir_0 entirely and has blank temperature/pressure for 62,040 runs. The
original PyATMOS summary (Downloads/pyatmos_summary.csv, one row per run with
surface-level gas composition, fluxes, temperature and pressure) is filtered
here to runs whose folder actually exists on disk under D:/Files/Dataset.

Output: datasets/pyatmos_final.csv (PyATMOS is external archive data, so it is
kept out of the mentor-only splits/ and derived/ folders).
Everything printed is appended to run_log.txt.

Inputs are only read, never modified.
"""

from __future__ import annotations

import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

RESEARCH_DIR = Path(__file__).resolve().parents[1]  # stage_d_habitability/ -> project root
sys.path.insert(0, str(RESEARCH_DIR))  # shared modules
from research_log import banner, run_logged  # noqa: E402
SUMMARY_CSV = Path.home() / "Downloads" / "pyatmos_summary.csv"
DATASET_DIR = Path("D:/Files/Dataset")
OUTPUT_CSV = RESEARCH_DIR / "datasets" / "pyatmos_final.csv"

EXPECTED_ROWS = 108_839   # 101,011 runs in run_summary_final.csv + 7,828 in dir_0
TOLERANCE = 300           # allowed difference before the build is stopped
PARENT_DIR_RE = re.compile(r"^(dir_\d+|Dir_alpha)$")
HASH_RE = re.compile(r"^[0-9a-f]{32}$")


def subdirs(path: Path) -> list[str]:
    with os.scandir(path) as it:
        return [e.name for e in it if e.is_dir()]


def scan_run_folders() -> dict[str, list[str]]:
    """Return {parent dir name: [run folder names]} for dir_N / Dir_alpha.

    dir_0 holds run folders directly; the others were extracted with a wrapper
    (dir_1/dir_1/<hash>/). A lone child folder named like its parent is treated
    as that wrapper and scanned instead.
    """
    found = {}
    parents = sorted(n for n in subdirs(DATASET_DIR) if PARENT_DIR_RE.match(n))
    for parent in parents:
        base = DATASET_DIR / parent
        children = subdirs(base)
        if children == [parent]:
            base = base / parent
            children = subdirs(base)
            print(f"  {parent}: run folders are nested one extra level ({parent}/{parent}/<hash>)")
        found[parent] = children
    return found


def main() -> None:
    print(f"\n\n{'#' * 88}\n# PYATMOS FIX: replace run_summary_final.csv with pyatmos_final.csv "
          f"(build_pyatmos_final.py, run {datetime.now():%Y-%m-%d %H:%M:%S})\n{'#' * 88}")

    # ---- Step 1: load summary, identify the hash column ------------------ #
    banner("STEP 1: Load pyatmos_summary.csv")
    summary = pd.read_csv(SUMMARY_CSV, low_memory=False)
    print(f"  file : {SUMMARY_CSV}")
    print(f"  shape: {summary.shape}")
    print(f"  columns: {list(summary.columns)}")
    looks_like_hash = [c for c in summary.columns
                       if summary[c].astype(str).str.fullmatch(r"[0-9a-fA-F]{32}").mean() > 0.99]
    print(f"  columns whose values are 32-char hex hashes: {looks_like_hash}")
    if looks_like_hash != ["hash"]:
        raise RuntimeError(f"expected the run identifier in column 'hash', found {looks_like_hash}")
    hash_col = "hash"
    print(f"  run identifier column: '{hash_col}' "
          f"(unique: {summary[hash_col].nunique()}, duplicated: {int(summary[hash_col].duplicated().sum())}, "
          f"blank: {int(summary[hash_col].isna().sum())})")

    # ---- Step 2: run folders on disk ------------------------------------- #
    banner("STEP 2: Scan run folders under D:/Files/Dataset")
    found = scan_run_folders()
    on_disk: set[str] = set()
    for parent, names in found.items():
        non_hash = [n for n in names if not HASH_RE.match(n)]
        extra = f"  (non-hash folder names: {non_hash[:5]})" if non_hash else ""
        print(f"  {parent:<10} {len(names):>7} folders{extra}")
        on_disk.update(names)
    total = sum(len(v) for v in found.values())
    print(f"  {'TOTAL':<10} {total:>7} folders  ({len(on_disk)} unique names)")
    dupes = [h for h, n in Counter(n for v in found.values() for n in v).items() if n > 1]
    if dupes:
        print(f"  NOTE: {len(dupes)} hash(es) appear in more than one parent dir, e.g. {dupes[:3]}")

    # ---- Step 3: keep runs that exist on disk ---------------------------- #
    banner("STEP 3: Keep summary rows whose hash has a run folder on disk")
    in_disk = summary[hash_col].isin(on_disk)
    final = summary[in_disk].copy()
    print(f"  rows kept: {len(final)}  (expected ~{EXPECTED_ROWS}, difference {len(final) - EXPECTED_ROWS:+d})")
    print(f"  summary rows with no folder on disk: {int((~in_disk).sum())}")
    print(f"  on-disk folders with no summary row: {len(on_disk - set(summary[hash_col]))}")

    if abs(len(final) - EXPECTED_ROWS) > TOLERANCE:
        normalized = summary[hash_col].astype(str).str.strip().str.lower()
        print("\n  STOPPING: row count is outside tolerance. Diagnostics:")
        print(f"    matches after strip()+lower(): {int(normalized.isin({h.lower() for h in on_disk}).sum())}")
        print(f"    sample unmatched summary hashes: {summary.loc[~in_disk, hash_col].head(5).tolist()}")
        print(f"    sample on-disk folder names    : {sorted(on_disk)[:5]}")
        raise RuntimeError(f"pyatmos_final would have {len(final)} rows, expected ~{EXPECTED_ROWS}")

    source = {h: p for p, names in found.items() for h in names}
    by_parent = final[hash_col].map(source).value_counts().sort_index()
    print("\n  kept rows by parent dir: " + ", ".join(f"{p}: {n}" for p, n in by_parent.items()))
    for col in ["temperature_kelvin", "pressure_bar"]:
        print(f"  blank {col}: {int(final[col].isna().sum())}")

    # ---- Step 4: save ---------------------------------------------------- #
    banner("STEP 4: Save")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUTPUT_CSV, index=False)
    print(f"  wrote {OUTPUT_CSV.relative_to(RESEARCH_DIR)}  ({final.shape[0]} rows x {final.shape[1]} cols)")


if __name__ == "__main__":
    run_logged(main)
