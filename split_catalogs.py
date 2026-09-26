"""
split_catalogs.py
=================

Host-star-grouped, label-stratified train/val/test splits for the Kepler (KOI),
K2 and TESS (TOI) candidate catalogs, as a leakage-safe foundation for the
CNN+LSTM transit-detection model.

Why group by star?  Many stars host several planet candidates, so a catalog has
several rows per star. If rows from one star land in both train and test, the
model can memorise that star's light-curve "fingerprint" and test scores become
optimistic. Every split below therefore keeps all rows of a star together.

Inputs  (read-only, mentor-approved pack):
    exoplanet_research_data/01_candidate_catalogs/koi_cumulative.csv
    exoplanet_research_data/01_candidate_catalogs/k2_planets_candidates.csv
    exoplanet_research_data/01_candidate_catalogs/toi_tess_candidates.csv

Outputs (written next to this script, never into the data pack):
    splits/koi_cumulative_split.csv            full catalog + label_harmonized + split
    splits/k2_planets_candidates_split.csv
    splits/toi_tess_candidates_split.csv
    splits/{kepler,k2,tess}/{train,val,test}.csv

Re-run after the source catalogs are updated:
    python split_catalogs.py
"""

from __future__ import annotations

import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from research_log import banner

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RESEARCH_DIR = Path(__file__).resolve().parent
DATA_DIR = RESEARCH_DIR / "exoplanet_research_data" / "01_candidate_catalogs"
OUTPUT_DIR = RESEARCH_DIR / "splits"

RANDOM_STATE = 42
TEST_SIZE = 0.15  # fraction of the WHOLE catalog
VAL_SIZE = 0.15   # fraction of the WHOLE catalog (train gets the remaining 0.70)

# Largest number of StratifiedGroupKFold folds used to approximate a fraction.
# 0.15 -> 3 of 20 folds; 0.15/0.85 -> 3 of 17 folds (both exact).
MAX_FOLDS = 20

CLASSES = ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"]
UNLABELED = "UNLABELED"  # stratum for rows whose raw disposition is blank
SPLITS = ["train", "val", "test"]

# Column added to every catalog and used as the grouping key. It equals the
# catalog's star-ID column; where that is blank, `group_fallback_col` is used so
# no row is left without a star (a NaN group cannot be kept together).
STAR_ID = "star_group_id"

CATALOGS = {
    "kepler": {
        "file": "koi_cumulative.csv",
        "output_file": "koi_cumulative_split.csv",
        "group_col": "kepid",
        "group_fallback_col": None,
        # koi_quarters is a 32-character quarter bitmask ('0111...'); read as text so the
        # leading zeros (and so the quarter positions) survive.
        "string_columns": ["koi_quarters"],
        "disp_col": "koi_disposition",
        "label_map": {
            "CONFIRMED": "CONFIRMED",
            "CANDIDATE": "CANDIDATE",
            "FALSE POSITIVE": "FALSE POSITIVE",
        },
        # Handful of readable columns for printing a star's rows.
        "show_cols": ["kepid", "kepoi_name", "kepler_name", "koi_disposition", "koi_period"],
    },
    "k2": {
        "file": "k2_planets_candidates.csv",
        "output_file": "k2_planets_candidates_split.csv",
        "group_col": "epic_hostname",
        "group_fallback_col": "hostname",
        # The K2 table has one row per PUBLISHED SOLUTION (WASP-47: 50 rows for
        # 4 planets). default_flag == 1 keeps exactly one row per planet.
        "row_filter": {"default_flag": 1},
        "disp_col": "disposition",
        "label_map": {
            "CONFIRMED": "CONFIRMED",
            "CANDIDATE": "CANDIDATE",
            "FALSE POSITIVE": "FALSE POSITIVE",
            "REFUTED": "FALSE POSITIVE",  # refuted planets are treated as false positives
        },
        "show_cols": ["epic_hostname", "pl_name", "default_flag", "disposition", "pl_orbper"],
    },
    "tess": {
        "file": "toi_tess_candidates.csv",
        "output_file": "toi_tess_candidates_split.csv",
        "group_col": "tid",
        "group_fallback_col": None,
        "disp_col": "tfopwg_disp",
        "label_map": {
            "CP": "CONFIRMED",       # confirmed planet
            "KP": "CONFIRMED",       # known planet
            "PC": "CANDIDATE",       # planet candidate
            "APC": "CANDIDATE",      # ambiguous planet candidate
            "FP": "FALSE POSITIVE",  # false positive
            "FA": "FALSE POSITIVE",  # false alarm
        },
        "show_cols": ["tid", "toi", "tfopwg_disp", "pl_orbper", "pl_rade"],
    },
}


class LeakageError(RuntimeError):
    """Raised when a star appears in more than one split."""


# --------------------------------------------------------------------------- #
# Step 1: load and inspect
# --------------------------------------------------------------------------- #
def load_catalog(path: Path, string_columns: list[str] | None = None) -> pd.DataFrame:
    """Read a catalog CSV, skipping NASA-archive style '#' header lines if present.

    `string_columns` are read as text so digit strings keep their leading zeros
    (e.g. koi_quarters '0111...' would otherwise be parsed as a number).
    """
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        first_data_line = next((ln for ln in fh if ln.strip()), "")
    has_comments = first_data_line.lstrip().startswith("#")
    dtype = {c: str for c in string_columns or []}
    # .copy() de-fragments wide frames so adding columns later doesn't warn.
    df = pd.read_csv(path, comment="#" if has_comments else None, low_memory=False, dtype=dtype).copy()
    print(f"  file: {path.name}  (leading '#' comment lines: {'yes' if has_comments else 'no'})")
    return df


def apply_row_filter(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Keep only rows matching the catalog's `row_filter` ({column: value}), if any."""
    for col, value in cfg.get("row_filter", {}).items():
        before = len(df)
        df = df[df[col] == value].reset_index(drop=True).copy()
        print(f"  row filter {col} == {value!r}: {before} -> {len(df)} rows")
    return df


# --------------------------------------------------------------------------- #
# Step 2: harmonise labels
# --------------------------------------------------------------------------- #
def harmonize_labels(df: pd.DataFrame, disp_col: str, label_map: dict) -> pd.DataFrame:
    """Add `label_harmonized`; the original disposition column is left untouched.

    Blank dispositions stay NaN (they are reported, not guessed). Any non-blank
    value missing from `label_map` raises, so a catalog update that introduces a
    new code cannot slip through silently.
    """
    raw = df[disp_col]
    unknown = sorted(set(raw.dropna().unique()) - set(label_map))
    if unknown:
        raise ValueError(f"{disp_col}: unmapped disposition values {unknown}")
    df["label_harmonized"] = raw.map(label_map)
    return df


def mapping_table(df: pd.DataFrame, disp_col: str) -> pd.DataFrame:
    table = (
        df.groupby([disp_col, "label_harmonized"], dropna=False)
        .size()
        .reset_index(name="rows")
        .rename(columns={disp_col: "old_value", "label_harmonized": "new_value"})
        .sort_values(["new_value", "old_value"], na_position="last")
    )
    return table.fillna("<blank>")


def add_star_id(df: pd.DataFrame, group_col: str, fallback_col: str | None) -> pd.DataFrame:
    """Create the grouping key `star_group_id` (string) from the star-ID column."""
    star = df[group_col].astype("string")
    n_missing = int(star.isna().sum())
    if n_missing and fallback_col:
        star = star.fillna(fallback_col + ":" + df[fallback_col].astype("string"))
        print(f"  {n_missing} row(s) with blank {group_col} grouped by {fallback_col} instead")
    if star.isna().any():
        raise ValueError(f"{int(star.isna().sum())} row(s) have no star ID in {group_col}")
    df[STAR_ID] = star
    return df


# --------------------------------------------------------------------------- #
# Step 3: star-sharing diagnostics
# --------------------------------------------------------------------------- #
def star_sharing_summary(df: pd.DataFrame) -> dict:
    rows_per_star = df[STAR_ID].value_counts()
    multi = rows_per_star[rows_per_star > 1]
    return {
        "total_rows": len(df),
        "unique_stars": rows_per_star.size,
        "stars_with_>1_row": multi.size,
        "%_rows_on_multi-row_stars": round(100 * multi.sum() / len(df), 2),
        "max_rows_one_star": int(rows_per_star.iloc[0]),
        "busiest_star": rows_per_star.index[0],
    }


# --------------------------------------------------------------------------- #
# Step 4: grouped + stratified 3-way split
# --------------------------------------------------------------------------- #
def _hold_out_groups(strata: np.ndarray, groups: np.ndarray, fraction: float,
                     random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """Hold out ~`fraction` of rows, whole groups only, stratified by `strata`.

    StratifiedGroupKFold produces n equal-ish, group-disjoint, stratified folds.
    `fraction` is written as k/n (n <= MAX_FOLDS) and k of the n folds are held
    out, e.g. 0.15 = 3/20. Because the folds partition the groups, the kept and
    held-out positions can never share a group.

    Returns (kept_positions, held_out_positions) as positional indices.
    """
    frac = Fraction(fraction).limit_denominator(MAX_FOLDS)
    k, n = frac.numerator, frac.denominator
    if not 0 < k < n:
        raise ValueError(f"fraction {fraction} cannot be approximated as k/n with 0<k<n<={MAX_FOLDS}")

    sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=random_state)
    with warnings.catch_warnings():
        # Tiny strata (e.g. 14 blank TESS dispositions) trigger a "least populated
        # class" warning; it only means that stratum can't be spread perfectly.
        warnings.filterwarnings("ignore", message="The least populated class")
        folds = [held for _, held in sgkf.split(np.zeros(len(strata)), strata, groups)]

    held_out = np.sort(np.concatenate(folds[:k]))
    kept = np.setdiff1d(np.arange(len(strata)), held_out)
    return kept, held_out


def grouped_stratified_split(df: pd.DataFrame, group_col: str, label_col: str,
                             test_size: float, val_size: float, random_state: int
                             ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split `df` into train/val/test with no group shared between splits.

    test_size and val_size are fractions of the whole dataframe.
      (a) hold out test_size of rows as test (whole groups, stratified);
      (b) from the remaining 1 - test_size, hold out val_size / (1 - test_size)
          as val (0.15 / 0.85 = 17.6% -> 15% of the whole); the rest is train.
    Rows with a blank label are stratified as their own UNLABELED stratum so
    they still get a split (and their star stays in one place).
    """
    if df[group_col].isna().any():
        raise ValueError(f"{group_col} contains NaN; every row needs a group")

    df = df.reset_index(drop=True)
    groups = df[group_col].to_numpy()
    strata = df[label_col].fillna(UNLABELED).to_numpy()

    # (a) test vs. train+val
    trainval_pos, test_pos = _hold_out_groups(strata, groups, test_size, random_state)

    # (b) val vs. train, within train+val only
    inner_val_frac = val_size / (1 - test_size)
    inner_train, inner_val = _hold_out_groups(
        strata[trainval_pos], groups[trainval_pos], inner_val_frac, random_state
    )
    train_pos, val_pos = trainval_pos[inner_train], trainval_pos[inner_val]

    return df.iloc[train_pos].copy(), df.iloc[val_pos].copy(), df.iloc[test_pos].copy()


# --------------------------------------------------------------------------- #
# Step 5: leakage check
# --------------------------------------------------------------------------- #
def check_leakage(name: str, parts: dict[str, pd.DataFrame], n_rows: int,
                  group_col: str) -> None:
    """Raise LeakageError unless the three splits are star-disjoint and complete."""
    stars = {s: set(parts[s][STAR_ID]) for s in SPLITS}
    raw_ids = {s: set(parts[s][group_col].dropna()) for s in SPLITS}
    failures = []
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = stars[a] & stars[b]
        raw_overlap = raw_ids[a] & raw_ids[b]  # same check on the untouched column
        status = "PASS" if not overlap and not raw_overlap else "FAIL"
        print(f"  {name:<7} {a:>5} & {b:<5} shared stars: {len(overlap)} "
              f"(on raw {group_col}: {len(raw_overlap)})  {status}")
        if status == "FAIL":
            failures.append(f"{a}/{b}: {sorted(overlap | raw_overlap)[:10]}")

    # Every row must land in exactly one split.
    n_assigned = sum(len(parts[s]) for s in SPLITS)
    if n_assigned != n_rows:
        failures.append(f"{n_assigned} rows assigned but catalog has {n_rows}")

    if failures:
        raise LeakageError(f"{name}: star leakage / assignment error -> {failures}")


# --------------------------------------------------------------------------- #
# Step 6 helpers: class balance
# --------------------------------------------------------------------------- #
def label_columns(df: pd.DataFrame) -> list[str]:
    extra = [UNLABELED] if df["label_harmonized"].isna().any() else []
    return CLASSES + extra


def class_balance(parts: dict[str, pd.DataFrame], cols: list[str]) -> pd.DataFrame:
    rows = []
    for s, part in parts.items():
        labels = part["label_harmonized"].fillna(UNLABELED)
        counts = labels.value_counts().reindex(cols, fill_value=0)
        row = {"split": s, "rows": len(labels)}
        for c in cols:
            row[c] = f"{counts[c]} ({100 * counts[c] / len(labels):.1f}%)"
        rows.append(row)
    return pd.DataFrame(rows).set_index("split")


# --------------------------------------------------------------------------- #
# Step 7 helper: save
# --------------------------------------------------------------------------- #
def save_split_outputs(name: str, df: pd.DataFrame, parts: dict[str, pd.DataFrame]) -> None:
    """Add a `split` column to `df` (in place) and write the full catalog plus one CSV per split."""
    split_col = pd.Series(pd.NA, index=df.index, dtype="object")
    for s, part in parts.items():
        split_col.loc[part.index] = s
    if split_col.isna().any():
        raise RuntimeError(f"{name}: {int(split_col.isna().sum())} rows without a split")
    df["split"] = split_col

    full_path = OUTPUT_DIR / CATALOGS[name]["output_file"]
    df.to_csv(full_path, index=False)
    print(f"  wrote {full_path.relative_to(RESEARCH_DIR)}  ({len(df)} rows)")

    split_dir = OUTPUT_DIR / name
    split_dir.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        part = df[df["split"] == s]
        part.to_csv(split_dir / f"{s}.csv", index=False)
        print(f"  wrote {(split_dir / f'{s}.csv').relative_to(RESEARCH_DIR)}  ({len(part)} rows)")


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_colwidth", 40)

    # ---- Step 1 ---------------------------------------------------------- #
    banner("STEP 1: Load and inspect")
    data = {}
    for name, cfg in CATALOGS.items():
        print(f"\n[{name}]")
        df = load_catalog(DATA_DIR / cfg["file"], cfg.get("string_columns"))
        print(f"  shape: {df.shape}")
        print(f"  columns ({df.shape[1]}): {', '.join(df.columns)}")
        print(f"  raw {cfg['disp_col']} value counts:")
        for value, count in df[cfg["disp_col"]].value_counts(dropna=False).items():
            print(f"    {'<blank>' if pd.isna(value) else value:<16} {count}")
        data[name] = apply_row_filter(df, cfg)

    # ---- Step 2 ---------------------------------------------------------- #
    banner("STEP 2: Harmonize dispositions -> CONFIRMED / CANDIDATE / FALSE POSITIVE")
    for name, cfg in CATALOGS.items():
        df = harmonize_labels(data[name], cfg["disp_col"], cfg["label_map"])
        print(f"\n[{name}]  {cfg['disp_col']} -> label_harmonized")
        print(mapping_table(df, cfg["disp_col"]).to_string(index=False))
        n_blank = int(df["label_harmonized"].isna().sum())
        if n_blank:
            print(f"  NOTE: {n_blank} row(s) have a blank {cfg['disp_col']}; label_harmonized "
                  f"is left blank and they are stratified as '{UNLABELED}'.")

    # ---- Step 3 ---------------------------------------------------------- #
    banner("STEP 3: Star-sharing diagnostics (before splitting)")
    summaries = {}
    for name, cfg in CATALOGS.items():
        df = add_star_id(data[name], cfg["group_col"], cfg["group_fallback_col"])
        summaries[name] = star_sharing_summary(df)
    print("\n" + pd.DataFrame(summaries).to_string())
    for name, cfg in CATALOGS.items():
        busiest = summaries[name]["busiest_star"]
        star_rows = data[name].loc[data[name][STAR_ID] == busiest, cfg["show_cols"] + ["label_harmonized"]]
        print(f"\n[{name}] star with the most rows: {cfg['group_col']} = {busiest} ({len(star_rows)} rows)")
        print(star_rows.to_string())

    # ---- Step 4 ---------------------------------------------------------- #
    banner(f"STEP 4: Grouped + stratified split "
           f"({1 - TEST_SIZE - VAL_SIZE:.0%}/{VAL_SIZE:.0%}/{TEST_SIZE:.0%}, random_state={RANDOM_STATE})")
    results = {}
    for name in CATALOGS:
        df = data[name].reset_index(drop=True)
        data[name] = df
        train, val, test = grouped_stratified_split(
            df, group_col=STAR_ID, label_col="label_harmonized",
            test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE,
        )
        results[name] = {"train": train, "val": val, "test": test}
        sizes = "  ".join(f"{s}={len(p)} ({100 * len(p) / len(df):.1f}%)" for s, p in results[name].items())
        print(f"  {name:<7} {sizes}")

    # ---- Step 5 ---------------------------------------------------------- #
    banner("STEP 5: Leakage check (star-ID intersections must all be empty)")
    for name, cfg in CATALOGS.items():
        check_leakage(name, results[name], len(data[name]), cfg["group_col"])
    print("\n  All 9 checks passed (3 catalogs x 3 split pairs): no star is shared between splits.")

    # ---- Step 6 ---------------------------------------------------------- #
    banner("STEP 6: Class balance per split")
    for name in CATALOGS:
        cols = label_columns(data[name])
        table = class_balance({**results[name], "all (pre-split)": data[name]}, cols)
        print(f"\n[{name}]")
        print(table.to_string())

    # ---- Step 7 ---------------------------------------------------------- #
    banner("STEP 7: Save outputs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in CATALOGS:
        save_split_outputs(name, data[name], results[name])

    # ---- Step 8 ---------------------------------------------------------- #
    banner("STEP 8: Final summary")
    rows = []
    for name in CATALOGS:
        df = data[name]
        for s in SPLITS:
            part = df[df["split"] == s]
            labels = part["label_harmonized"].fillna(UNLABELED)
            row = {
                "catalog": name,
                "split": s,
                "rows": len(part),
                "% rows": round(100 * len(part) / len(df), 1),
                "stars": part[STAR_ID].nunique(),
            }
            for c in label_columns(df):
                row[f"% {c}"] = round(100 * (labels == c).mean(), 1)
            rows.append(row)
    summary = pd.DataFrame(rows).fillna("-")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
