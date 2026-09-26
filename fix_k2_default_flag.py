"""
fix_k2_default_flag.py
======================

One-off K2 correction to the splits made by split_catalogs.py.

k2_planets_candidates.csv has one row per PUBLISHED SOLUTION, not one row per
planet (WASP-47: 50 rows for 4 planets), so heavily studied planets were
over-represented. This script keeps only `default_flag == 1` rows (one per
planet), then re-splits K2 with the unchanged `grouped_stratified_split`, the
same star_group_id grouping, label_harmonized strata, 70/15/15 and
random_state=42.

Only K2 outputs are rewritten. Kepler and TESS outputs are not read or written.
Everything printed is also appended to run_log.txt.

split_catalogs.py now applies the same filter (CATALOGS["k2"]["row_filter"]),
so a full re-run reproduces these K2 files exactly.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

import split_catalogs as sc
from research_log import LOG_PATH, banner, run_logged

NAME = "k2"
CFG = sc.CATALOGS[NAME]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Same label and star-ID preparation split_catalogs.py applies."""
    df = sc.harmonize_labels(df, CFG["disp_col"], CFG["label_map"])
    return sc.add_star_id(df, CFG["group_col"], CFG["group_fallback_col"])


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train, val, test = sc.grouped_stratified_split(
        df, group_col=sc.STAR_ID, label_col="label_harmonized",
        test_size=sc.TEST_SIZE, val_size=sc.VAL_SIZE, random_state=sc.RANDOM_STATE,
    )
    return {"train": train, "val": val, "test": test}


def split_stats(parts: dict[str, pd.DataFrame], n_total: int) -> pd.DataFrame:
    """Rows, unique stars and class balance per split (the Step 6/8 format)."""
    rows = []
    for s, part in parts.items():
        labels = part["label_harmonized"].fillna(sc.UNLABELED)
        row = {
            "split": s,
            "rows": len(part),
            "% rows": round(100 * len(part) / n_total, 1),
            "stars": part[sc.STAR_ID].nunique(),
        }
        for c in sc.CLASSES:
            n = int((labels == c).sum())
            row[c] = f"{n} ({100 * n / len(part):.1f}%)"
        rows.append(row)
    return pd.DataFrame(rows).set_index("split")


def main() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(f"\n\n{'#' * 88}\n# K2 FIX: keep default_flag == 1 only "
          f"(fix_k2_default_flag.py, run {datetime.now():%Y-%m-%d %H:%M:%S})\n{'#' * 88}")

    # ---- Step 1: filter K2 before anything else -------------------------- #
    banner("FIX STEP 1: Filter K2 to default_flag == 1")
    raw = sc.load_catalog(sc.DATA_DIR / CFG["file"])
    old_df = prepare(raw.copy())          # unfiltered, for the before/after comparison
    new_df = prepare(sc.apply_row_filter(raw, CFG))

    print(f"\n  rows before filter: {len(old_df)}")
    print(f"  rows after filter : {len(new_df)}")
    print(f"  unique planets (pl_name)      : {new_df['pl_name'].nunique()}  "
          f"(duplicated pl_name rows: {int(new_df['pl_name'].duplicated().sum())})")
    print(f"  unique stars ({sc.STAR_ID}): {new_df[sc.STAR_ID].nunique()}")
    if new_df["pl_name"].duplicated().any():
        raise RuntimeError("default_flag == 1 still leaves more than one row for some planet")

    wasp47 = new_df[new_df["hostname"] == "WASP-47"]
    print(f"\n  WASP-47 rows after filter: {len(wasp47)} (was "
          f"{int((old_df['hostname'] == 'WASP-47').sum())})")
    print(wasp47[CFG["show_cols"] + ["hostname", "label_harmonized"]].to_string())
    if len(wasp47) != 4 or wasp47["pl_name"].nunique() != 4:
        raise RuntimeError("WASP-47 should have exactly 4 rows (one per planet) after filtering")

    # ---- Step 2: re-split filtered K2 with the unchanged function -------- #
    banner("FIX STEP 2: Re-split filtered K2 (grouped_stratified_split, unchanged)")
    old_parts = split(old_df)
    new_parts = split(new_df)
    for s, part in new_parts.items():
        print(f"  {s:<5} {len(part):>5} rows ({100 * len(part) / len(new_df):.1f}%)")

    # Sanity: the in-memory "old" split must equal the file currently on disk,
    # otherwise the comparison would be against the wrong baseline.
    on_disk_path = sc.OUTPUT_DIR / CFG["output_file"]
    if on_disk_path.exists():
        on_disk = pd.read_csv(on_disk_path, usecols=["default_flag", "split"], low_memory=False)
        if (on_disk["default_flag"] == 1).all():
            print("  note: on-disk K2 split is already filtered (fix was applied before)")
        else:
            old_split = pd.Series(index=old_df.index, dtype="object")
            for s, part in old_parts.items():
                old_split.loc[part.index] = s
            # Compare values, not dtypes (pandas 3 reads text back as `str`, not object).
            same = (old_split.to_numpy(dtype=str) == on_disk["split"].to_numpy(dtype=str)).all()
            print(f"  old (unfiltered) split recomputed in memory matches file on disk: {same}")
            if not same:
                raise RuntimeError("baseline mismatch: recomputed old split != on-disk K2 split")

    # ---- Step 3: leakage check ------------------------------------------- #
    banner("FIX STEP 3: K2 leakage check on the filtered split")
    sc.check_leakage(NAME, new_parts, len(new_df), CFG["group_col"])
    print("\n  K2: all 3 star-ID intersections are empty (train/val, train/test, val/test).")

    # ---- Step 4: before/after class balance ------------------------------ #
    banner("FIX STEP 4: K2 class balance, OLD (all solutions) vs NEW (default_flag == 1)")
    old_tab = split_stats({**old_parts, "all": old_df}, len(old_df))
    new_tab = split_stats({**new_parts, "all": new_df}, len(new_df))
    print("\n  OLD - unfiltered, one row per published solution")
    print(old_tab.to_string())
    print("\n  NEW - default_flag == 1, one row per planet")
    print(new_tab.to_string())
    print("\n  Side by side")
    print(pd.concat({"OLD": old_tab, "NEW": new_tab}, axis=1).to_string())

    # ---- Step 5: overwrite K2 outputs only ------------------------------- #
    banner("FIX STEP 5: Overwrite K2 outputs (Kepler/TESS untouched)")
    sc.save_split_outputs(NAME, new_df, new_parts)
    print(f"  appended this output to {LOG_PATH.name}")


if __name__ == "__main__":
    run_logged(main)
