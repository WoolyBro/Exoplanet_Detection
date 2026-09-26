"""
build_stellarhosts_deduplicated.py
==================================

Rebuilds outputs/derived/stellarhosts_deduplicated.csv: one row per host star from the
mentor pack's stellarhosts.csv, which lists one row per PUBLISHED SOLUTION
(47,903 rows for 5,243 stars).

Rebuilt 2026-09-20. The handoff describes this dedup as already done, but no
output file and no run_log.txt entry existed - see the run_log entry appended by
this script. Nothing about the tie-break is invented here: it is the rule the
handoff documents, implemented and reported step by step so the result is
checkable.

Tie-break, applied in order, first row wins:
  1. rows where st_teff, st_rad and st_mass are ALL present
  2. most recent reference year (parsed out of st_refname)
  3. most filled parameters (count of non-blank fields in the row)
  4. earliest row in the file (stable, so the result is deterministic)

SCOPE WARNING - this file is NOT a feature source.
  stellarhosts.csv lists only CONFIRMED-planet hosts, so whether a star appears
  in it tracks the label almost perfectly. Joining it onto a detection catalog
  leaks the disposition. data_manifest.py classifies it STAGE_1_SUPPLEMENTARY,
  and the project's hard rules forbid using it (or pscomppars /
  ps_transiting_default) to build model features.
  Legitimate uses are reference and cross-checking only - for example the
  cb_flag lookup that identified the circumbinary systems. Use the stellar
  columns already inside each detection catalog for features instead.

Run:
    python catalog_preparation/build_stellarhosts_deduplicated.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root: shared modules
import data_manifest as dm  # noqa: E402
from research_log import banner, run_logged  # noqa: E402

RESEARCH_DIR = dm.RESEARCH_DIR
SOURCE = RESEARCH_DIR / "exoplanet_research_data" / "03_stellar_parameters" / "stellarhosts.csv"
OUTPUT_DIR = RESEARCH_DIR / "outputs" / "derived"
OUTPUT = OUTPUT_DIR / "stellarhosts_deduplicated.csv"

KEY = "hostname"
CORE_PARAMS = ["st_teff", "st_rad", "st_mass"]
YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")


def reference_year(refname: object) -> float:
    """Latest 4-digit year mentioned in an archive reference string, NaN if none.

    st_refname holds an HTML anchor such as
    '<a refstr=SMITH_ET_AL__2019 href=... > Smith et al. 2019 </a>'; the year is
    taken as the largest plausible year in the string, so a stray id number in
    the href cannot beat the actual publication year.
    """
    if not isinstance(refname, str):
        return float("nan")
    years = [int(y) for y in YEAR_RE.findall(refname)]
    return float(max(years)) if years else float("nan")


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per `KEY`. Returns (deduplicated, per-row ranking keys for reporting)."""
    missing = [c for c in CORE_PARAMS + [KEY, "st_refname"] if c not in df.columns]
    if missing:
        raise KeyError(f"{SOURCE.name} is missing expected columns: {missing}")

    rank = pd.DataFrame({
        KEY: df[KEY],
        "has_core": df[CORE_PARAMS].notna().all(axis=1),
        "ref_year": df["st_refname"].map(reference_year),
        "n_filled": df.notna().sum(axis=1),
        "orig_row": range(len(df)),
    })
    # NaN years must lose to any real year, never win by accident.
    rank["ref_year_sort"] = rank["ref_year"].fillna(-1)

    order = rank.sort_values(
        ["has_core", "ref_year_sort", "n_filled", "orig_row"],
        ascending=[False, False, False, True],
        kind="mergesort",  # stable: equal rows keep file order, so reruns match
    )
    winners = order.drop_duplicates(subset=KEY, keep="first")
    out = df.loc[winners["orig_row"].to_numpy()].reset_index(drop=True)
    return out, rank


STAGES = [("1. core params present", "has_core"),
          ("2. reference year", "ref_year_sort"),
          ("3. most filled fields", "n_filled"),
          ("4. earliest row", "orig_row")]


def tie_break_report(rank: pd.DataFrame) -> pd.DataFrame:
    """Which tie-break step actually decided each multi-row star.

    Survivors are narrowed stage by stage; a star is credited to the first stage
    that leaves exactly one row. Stage 4 (earliest row) always resolves, so the
    counts sum to the number of multi-row stars.
    """
    multi = rank[rank.duplicated(KEY, keep=False)]
    decided = dict.fromkeys(name for name, _ in STAGES)
    decided = {name: 0 for name in decided}
    for _, group in multi.groupby(KEY, sort=False):
        survivors = group
        for name, col in STAGES:
            # every stage prefers the maximum, except the last which prefers the earliest row
            best = survivors[col].min() if col == "orig_row" else survivors[col].max()
            survivors = survivors[survivors[col] == best]
            if len(survivors) == 1:
                decided[name] += 1
                break
    return pd.DataFrame([{"stage": name, "stars_uniquely_decided": decided[name]} for name, _ in STAGES])


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    banner(f"STELLARHOSTS DEDUPLICATION (rebuild, {SOURCE.name})")
    dm.assert_role(SOURCE, "STAGE_1_SUPPLEMENTARY")
    print(f"  manifest role: {dm.role_of(SOURCE)} (reference/cross-check only - NOT a feature source)")

    df = pd.read_csv(SOURCE, comment="#", low_memory=False)
    print(f"  loaded {SOURCE.relative_to(RESEARCH_DIR)}: {df.shape[0]:,} rows x {df.shape[1]} columns")

    counts = df[KEY].value_counts()
    print(f"  unique {KEY}: {counts.size:,} | stars with >1 row: {int((counts > 1).sum()):,} "
          f"| max rows for one star: {int(counts.iloc[0])} ({counts.index[0]})")

    out, rank = deduplicate(df)
    print(f"\n  deduplicated: {len(df):,} -> {len(out):,} rows")
    if len(out) != counts.size:
        raise RuntimeError(f"expected {counts.size} rows (one per star), produced {len(out)}")
    if out[KEY].duplicated().any():
        raise RuntimeError("duplicate hostname survived deduplication")

    print("\n  tie-break resolution (multi-row stars only):")
    print(tie_break_report(rank).to_string(index=False))

    print("\n  parameter completeness, before vs after:")
    for col in CORE_PARAMS + ["st_logg", "st_spectype", "cb_flag"]:
        if col in df.columns:
            before = 100 * df[col].notna().mean()
            after = 100 * out[col].notna().mean()
            print(f"    {col:<12} rows {before:5.1f}% -> stars {after:5.1f}%")
    print(f"    all of {'+'.join(CORE_PARAMS):<24} "
          f"rows {100 * df[CORE_PARAMS].notna().all(axis=1).mean():5.1f}% -> "
          f"stars {100 * out[CORE_PARAMS].notna().all(axis=1).mean():5.1f}%")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)
    print(f"\n  wrote {OUTPUT.relative_to(RESEARCH_DIR)} ({len(out):,} rows x {out.shape[1]} columns, "
          f"{OUTPUT.stat().st_size / 1e6:.1f} MB)")
    print("  REMINDER: reference and cross-checks only; never join this onto a detection catalog as features.")


if __name__ == "__main__":
    run_logged(main)
