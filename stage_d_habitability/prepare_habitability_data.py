"""
prepare_habitability_data.py
============================

End-to-end preparation of the INARA habitability data:
load -> explore -> label (placeholder rule) -> leakage checks -> star-grouped split -> save.

Written to run unchanged on any size of INARA file. Nothing depends on the
current row count; every count, threshold and warning is computed at runtime.

Data access: INARA only via data_manifest.load_stage2_catalogs().
Outputs:     outputs/habitability_prepared/{train,val,test}.csv and label_rule.json
             (derived data - not a manifest entry). label_rule.json is MERGED,
             not replaced: manual exclusions, exclude_reasons and leakage_audit
             survive re-runs. Printed output is appended to run_log.txt.

The labeling rule used here is an exploratory PLACEHOLDER, derived from INARA's
own O2/CO2 distribution. It is not a final habitability definition.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root: shared modules
import data_manifest as dm  # noqa: E402
import split_catalogs as sc  # noqa: E402  reuse the Stage 1 grouped split + leakage check (read-only import)
from research_log import banner, run_logged  # noqa: E402

RESEARCH_DIR = dm.RESEARCH_DIR
OUTPUT_DIR = RESEARCH_DIR / "outputs" / "habitability_prepared"
INARA_FILE = "psg_models.csv"

# INARA gas mixing-ratio columns and molar masses (g/mol), used to check closure
# and to detect columns computed from the composition.
MOLAR_MASS = {
    "H2O": 18.015, "CO2": 44.009, "O2": 31.998, "N2": 28.014, "CH4": 16.043, "N2O": 44.013,
    "CO": 28.010, "O3": 47.998, "SO2": 64.066, "NH3": 17.031, "C2H6": 30.069, "NO2": 46.006,
}
# Columns that describe the host star; together they identify it when no star ID exists.
STAR_ID_CANDIDATES = ["star_id", "host_star", "hostname", "star_name", "StarIndex"]
STAR_PARAM_COLS = ["star_class", "star_temperature_in_Kelvin", "star_radius_in_Solar_radii",
                   "distance_from_Earth_to_the_system_in_parsecs"]
TEMP_COL, PRES_COL = "planet_surface_temperature_Kelvin", "planet_surface_pressure_bars"

# Name patterns that would indicate spectrum (wavelength / signal) columns.
SPECTRUM_PATTERN = re.compile(r"(wavelength|wave|^wl|spectr|signal|radiance|transit_depth|"
                              r"^flux|contrast|^\d+(\.\d+)?_?(um|micron|nm)$)", re.IGNORECASE)

MIN_MEANINGFUL_SPLIT_ROWS = 10  # below this a split is not statistically meaningful
LABEL_COL = "habitability_label"
LIQUID_WATER_K = (273.15, 373.15)  # physical constant range at 1 bar, used by one candidate rule


def rel(path) -> str:
    """Path relative to Research when possible (for readable output)."""
    try:
        return str(path.relative_to(RESEARCH_DIR))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- #
# Labeling
# --------------------------------------------------------------------------- #
RULE_COLUMNS = {
    "o2_min": ("O2", ">="), "o2_max": ("O2", "<="),
    "co2_min": ("CO2", ">="), "co2_max": ("CO2", "<="),
    "t_min": (TEMP_COL, ">="), "t_max": (TEMP_COL, "<="),
    "p_min": (PRES_COL, ">="), "p_max": (PRES_COL, "<="),
}


def derive_habitability_label(df: pd.DataFrame, rule_params: dict) -> pd.DataFrame:
    """Return a copy of `df` with a 0/1 `habitability_label` column.

    `rule_params` holds the thresholds; any of o2_min, o2_max, co2_min, co2_max,
    t_min, t_max, p_min, p_max (inclusive bounds; omit or None = unbounded) plus
    an optional "name". A row is labelled 1 only if it satisfies every bound
    given. Rows with a blank value in a column the rule uses get a blank label.
    """
    unknown = set(rule_params) - set(RULE_COLUMNS) - {"name"}
    if unknown:
        raise ValueError(f"unknown rule parameters: {sorted(unknown)}")
    bounds = {k: v for k, v in rule_params.items() if k in RULE_COLUMNS and v is not None}
    if not bounds:
        raise ValueError("rule_params must set at least one threshold")

    out = df.copy()
    passes = pd.Series(True, index=out.index)
    blank = pd.Series(False, index=out.index)
    for key, value in bounds.items():
        col, op = RULE_COLUMNS[key]
        if col not in out.columns:
            raise KeyError(f"rule parameter {key} needs column {col}, which is not in the data")
        passes &= (out[col] >= value) if op == ">=" else (out[col] <= value)
        blank |= out[col].isna()
    out[LABEL_COL] = passes.astype("Int8").mask(blank)
    return out


def describe_rule(params: dict) -> str:
    parts = []
    for key, value in params.items():
        if key in RULE_COLUMNS and value is not None:
            col, op = RULE_COLUMNS[key]
            parts.append(f"{col} {op} {value:g}")
    return " AND ".join(parts)


# --------------------------------------------------------------------------- #
# Leakage checks
# --------------------------------------------------------------------------- #
def leakage_checks(df: pd.DataFrame, gas_cols: list[str]) -> list[str]:
    """Check that no non-composition column is a copy of, or computed from, the gases.

    Returns the columns that must be excluded from model features whenever the
    label is derived from the composition (the gases themselves plus anything
    found to be derived from them).
    """
    exclude = list(gas_cols)
    other = [c for c in df.select_dtypes("number").columns if c not in gas_cols and c != LABEL_COL]

    # (a) Spectrum columns: the composition generates the spectrum (PSG radiative
    # transfer); a spectrum column must not simply repeat a gas fraction.
    spectrum_cols = [c for c in df.columns if SPECTRUM_PATTERN.search(str(c))]
    if spectrum_cols:
        print(f"  spectrum-like columns found: {len(spectrum_cols)} (e.g. {spectrum_cols[:5]})")
    else:
        print("  spectrum-like columns found: NONE. This file holds composition + planet/star "
              "parameters only,")
        print("  so the composition -> spectrum separation CANNOT be verified on it. The same checks "
              "below will")
        print("  run on spectrum columns automatically once a file that contains them is loaded.")

    # (b) Exact copies: a column whose values equal a gas column in (nearly) every row.
    copies = []
    for c in other:
        for g in gas_cols:
            if np.isclose(df[c], df[g], rtol=1e-9, atol=0).mean() > 0.9:
                copies.append((c, g))
    print(f"  columns that repeat a gas fraction verbatim (>90% of rows equal): "
          f"{copies if copies else 'none'}")

    # (c) Near-perfect monotonic relation with a single gas.
    rank_links = []
    for c in other:
        for g in gas_cols:
            rho = df[c].corr(df[g], method="spearman")
            if pd.notna(rho) and abs(rho) >= 0.95:
                rank_links.append((c, g, round(rho, 3)))
    print(f"  columns with |Spearman| >= 0.95 against a single gas: {rank_links if rank_links else 'none'}")

    # (d) Columns computed from the whole composition. Mean molecular weight is
    # sum(x_i * M_i); check any column that matches that formula.
    mmw = sum(df[g] * MOLAR_MASS[g] for g in gas_cols)
    derived = [c for c in other if np.allclose(df[c], mmw, rtol=1e-3, atol=1e-2)]
    print(f"  columns equal to the composition's mean molecular weight sum(x_i*M_i): "
          f"{derived if derived else 'none'}")
    if derived:
        max_diff = max(float((df[c] - mmw).abs().max()) for c in derived)
        print(f"    (max |difference| {max_diff:.2g} g/mol) -> derived from the gases; "
              f"must not be a feature next to a gas-based label")

    for c, *_ in copies + rank_links:
        if c not in exclude:
            exclude.append(c)
    for c in derived:
        if c not in exclude:
            exclude.append(c)
    for c in spectrum_cols:
        if any(c == pair[0] for pair in copies):
            print(f"  LEAK: spectrum column {c} copies a gas fraction")
    return exclude


# --------------------------------------------------------------------------- #
# label_rule.json merge
# --------------------------------------------------------------------------- #
def merge_rule_record(existing: dict, script_record: dict, auto_exclude: list[str],
                      gas_cols: list[str]) -> tuple[dict, list[str]]:
    """Merge this script's output into an existing label_rule.json instead of replacing it.

    The script owns only the keys in `script_record` (rule parameters, split info, ...)
    plus the exclusions it derives itself, tracked in `auto_exclusions`. Everything
    else survives untouched: `leakage_audit`, existing `exclude_reasons` entries, and
    any manually added exclusions.
    """
    merged = dict(existing)
    report = []
    merged.update(script_record)

    old_list = list(existing.get("exclude_from_features", []))
    if "auto_exclusions" in existing:
        previous_auto = set(existing["auto_exclusions"])
    else:
        # Files written before auto_exclusions existed: treat what this script derives
        # now as its own, and everything else in the list as a manual addition.
        previous_auto = set(old_list) & set(auto_exclude)
    manual = [c for c in old_list if c not in previous_auto]
    dropped = sorted(previous_auto - set(auto_exclude) - set(manual))

    new_list = [c for c in old_list if c in manual or c in auto_exclude]
    new_list += [c for c in auto_exclude if c not in new_list]
    merged["exclude_from_features"] = new_list
    # A column excluded manually stays manual even if the script also derives it, so it
    # is never removed just because a later run stops deriving it.
    merged["auto_exclusions"] = [c for c in auto_exclude if c not in manual]

    reasons = dict(existing.get("exclude_reasons", {}))
    added_reasons = []
    for c in auto_exclude:
        if c not in reasons:  # never overwrite an existing (possibly manual) reason
            reasons[c] = ("raw gas fraction; defines the composition-based label" if c in gas_cols else
                          "derived from the gas composition (prepare_habitability_data.py leakage check)")
            added_reasons.append(c)
    merged["exclude_reasons"] = reasons

    report.append(f"manual exclusions kept: {manual or 'none'}")
    report.append(f"script-derived exclusions: {list(auto_exclude)}")
    if dropped:
        report.append(f"script-derived exclusions no longer derived (removed): {dropped}")
    report.append(f"exclude_reasons added for: {added_reasons or 'none'} (existing reasons left unchanged)")
    kept = sorted(k for k in existing if k not in script_record
                  and k not in ("exclude_from_features", "auto_exclusions", "exclude_reasons"))
    report.append(f"other existing keys preserved untouched: {kept or 'none'}")
    return merged, report


# --------------------------------------------------------------------------- #
# Star grouping + split
# --------------------------------------------------------------------------- #
def add_host_star_group(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Add `star_group_id` from a host-star ID column, or from the star parameters if none exists."""
    out = df.copy()
    id_col = next((c for c in STAR_ID_CANDIDATES if c in out.columns), None)
    if id_col:
        out[sc.STAR_ID] = out[id_col].astype("string")
        return out, f"existing column '{id_col}'"

    missing = [c for c in STAR_PARAM_COLS if c not in out.columns]
    if missing:
        raise KeyError(f"no host-star ID column and star parameter columns missing: {missing}")
    # Rows with identical star parameters (rounded to absorb float noise) share a star.
    key = out[STAR_PARAM_COLS].round(6).astype(str).agg("|".join, axis=1)
    out[sc.STAR_ID] = key.map(lambda s: "star_" + hashlib.sha1(s.encode()).hexdigest()[:12])
    return out, f"derived from star parameters {STAR_PARAM_COLS}"


def split_by_host_star(df: pd.DataFrame, test_size: float, val_size: float,
                       random_state: int) -> dict[str, pd.DataFrame]:
    """Star-grouped, label-stratified train/val/test split (same function as Stage 1)."""
    n_groups = df[sc.STAR_ID].nunique()
    if n_groups < sc.MAX_FOLDS:
        raise ValueError(f"only {n_groups} host stars; the grouped split needs at least "
                         f"{sc.MAX_FOLDS} (sc.MAX_FOLDS) to form its folds")
    train, val, test = sc.grouped_stratified_split(
        df, group_col=sc.STAR_ID, label_col=LABEL_COL,
        test_size=test_size, val_size=val_size, random_state=random_state,
    )
    return {"train": train, "val": val, "test": test}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print(f"\n\n{'#' * 88}\n# HABITABILITY DATA PREP (prepare_habitability_data.py, "
          f"run {datetime.now():%Y-%m-%d %H:%M:%S})\n{'#' * 88}")

    # ---- Step 1: load ---------------------------------------------------- #
    banner("STEP 1: Load INARA via data_manifest.load_stage2_catalogs()")
    inara = dm.load_stage2_catalogs()[INARA_FILE]
    n_rows = len(inara)
    print(f"  rows: {n_rows}")
    print(f"  columns ({inara.shape[1]}): {list(inara.columns)}")
    gas_cols = [g for g in MOLAR_MASS if g in inara.columns]
    print(f"  gas columns present ({len(gas_cols)}): {gas_cols}")
    if not {"O2", "CO2"} <= set(gas_cols):
        raise KeyError("INARA must contain O2 and CO2 columns")

    # ---- Step 2: explore INARA's own composition ------------------------- #
    banner("STEP 2: INARA gas composition (INARA only)")
    stats = inara[gas_cols].describe(percentiles=[.25, .5, .75]).T[
        ["min", "25%", "50%", "75%", "max", "mean", "std"]]
    print(stats.to_string(float_format=lambda v: f"{v:.4g}"))
    closure = inara[gas_cols].sum(axis=1)
    off = int(((closure - 1).abs() > 0.05).sum())
    print(f"\n  gas sum per row: min {closure.min():.6f}, max {closure.max():.6f}; "
          f"rows off 1.0 by >5%: {off} of {n_rows}")
    if off:
        print("  WARNING: composition is not closed in every row; the complete-composition "
              "assumption does not hold for this file")
    else:
        print("  composition is complete (sums to 1) in every row")
    rho = inara["O2"].corr(inara["CO2"], method="spearman")
    print(f"  Spearman(O2, CO2) = {rho:+.2f}  (closed compositions push fractions to be negatively related)")
    print(f"  {TEMP_COL}: {inara[TEMP_COL].min():.1f}-{inara[TEMP_COL].max():.1f} K; "
          f"{PRES_COL}: {inara[PRES_COL].min():.3g}-{inara[PRES_COL].max():.3g} bar")

    # ---- Step 3: candidate rules from INARA's own quantiles -------------- #
    banner("STEP 3: Candidate rules from INARA's own O2/CO2 distribution (exploratory, NOT final)")
    q = inara[["O2", "CO2"]].quantile([.25, .5, .75])
    candidates = {
        "C1": {"name": "O2 at/above INARA median AND CO2 at/below INARA median",
               "o2_min": q.loc[.5, "O2"], "co2_max": q.loc[.5, "CO2"]},
        "C2": {"name": "O2 in INARA top quartile AND CO2 in INARA bottom quartile",
               "o2_min": q.loc[.75, "O2"], "co2_max": q.loc[.25, "CO2"]},
        "C3": {"name": "C1 AND liquid-water surface temperature (273.15-373.15 K)",
               "o2_min": q.loc[.5, "O2"], "co2_max": q.loc[.5, "CO2"],
               "t_min": LIQUID_WATER_K[0], "t_max": LIQUID_WATER_K[1]},
    }
    rows = []
    for key, params in candidates.items():
        labels = derive_habitability_label(inara, params)[LABEL_COL]
        rows.append({"rule": key, "definition": describe_rule(params),
                     "flagged": int(labels.sum()), "% of rows": round(100 * labels.mean(), 1)})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n  Thresholds are quantiles of the CURRENT {n_rows}-row file, so they will move when "
          f"INARA grows.")
    print("  A final rule should freeze numeric thresholds once chosen.")

    # ---- Step 4: apply the placeholder rule ------------------------------ #
    placeholder_key = "C1"
    placeholder = {k: (float(v) if not isinstance(v, str) else v)
                   for k, v in candidates[placeholder_key].items()}
    banner(f"STEP 4: Label with PLACEHOLDER rule {placeholder_key} (not a final habitability definition)")
    labelled = derive_habitability_label(inara, placeholder)
    print(f"  rule: {describe_rule(placeholder)}")
    counts = labelled[LABEL_COL].value_counts(dropna=False).sort_index()
    print("  label counts: " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
    if labelled[LABEL_COL].nunique(dropna=True) < 2:
        print("  WARNING: the placeholder rule produces a single class; stratification is meaningless")

    # ---- Step 5: leakage checks ------------------------------------------ #
    banner("STEP 5: Leakage checks (composition vs other columns)")
    exclude = leakage_checks(labelled, gas_cols)
    print(f"\n  columns to EXCLUDE from features while the label is composition-based: {exclude}")

    # ---- Step 6: star-grouped split -------------------------------------- #
    banner("STEP 6: Host-star grouped, label-stratified split (70/15/15, random_state=42)")
    grouped, how = add_host_star_group(labelled)
    per_star = grouped[sc.STAR_ID].value_counts()
    print(f"  host-star identifier: {how}")
    print(f"  host stars: {per_star.size} for {n_rows} rows; stars with >1 planet: {int((per_star > 1).sum())}")
    parts = split_by_host_star(grouped, sc.TEST_SIZE, sc.VAL_SIZE, sc.RANDOM_STATE)
    sc.check_leakage("inara", parts, n_rows, sc.STAR_ID)

    small = []
    for s, part in parts.items():
        c = part[LABEL_COL].value_counts().reindex([0, 1], fill_value=0)
        print(f"  {s:<5} rows {len(part):>4} ({100 * len(part) / n_rows:5.1f}%)  stars "
              f"{part[sc.STAR_ID].nunique():>4}  label=1: {c[1]}  label=0: {c[0]}")
        if len(part) < MIN_MEANINGFUL_SPLIT_ROWS:
            small.append(s)
    for s in small:
        print(f"  WARNING: '{s}' has {len(parts[s])} rows (< {MIN_MEANINGFUL_SPLIT_ROWS}); "
              f"a split this small is not statistically meaningful.")

    # ---- Step 7: save ---------------------------------------------------- #
    banner("STEP 7: Save")
    OUTPUT_DIR.mkdir(exist_ok=True)
    for s, part in parts.items():
        part = part.assign(split=s)
        part.to_csv(OUTPUT_DIR / f"{s}.csv", index=False)
        print(f"  wrote {rel(OUTPUT_DIR / f'{s}.csv')} ({len(part)} rows)")
    script_record = {
        "generated_by": "prepare_habitability_data.py",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "load_stage2_catalogs()['psg_models.csv']",
        "n_rows": n_rows,
        "placeholder_rule": placeholder_key,
        "rule_params": placeholder,
        "rule_is_final": False,
        "group_column": sc.STAR_ID,
        "group_column_source": how,
        "split": {"test_size": sc.TEST_SIZE, "val_size": sc.VAL_SIZE, "random_state": sc.RANDOM_STATE,
                  "rows": {s: len(p) for s, p in parts.items()}},
        "splits_below_meaningful_size": small,
    }
    rule_path = OUTPUT_DIR / "label_rule.json"
    existing = json.loads(rule_path.read_text(encoding="utf-8")) if rule_path.exists() else {}
    merged, report = merge_rule_record(existing, script_record, exclude, gas_cols)
    rule_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    exclude = merged["exclude_from_features"]
    print(f"  wrote {rel(rule_path)} ({'merged into existing file' if existing else 'new file'})")
    for line in report:
        print(f"    {line}")

    # ---- Step 8: explicit status ----------------------------------------- #
    banner("SUMMARY")
    print(f"  rows {n_rows} | placeholder rule {placeholder_key}: {describe_rule(placeholder)}")
    print(f"  label=1: {int(labelled[LABEL_COL].sum())} ({100 * labelled[LABEL_COL].mean():.1f}%) | "
          f"split rows " + ", ".join(f"{s} {len(p)}" for s, p in parts.items()))
    print(f"  star-grouped leakage check: passed | feature exclusions: {exclude}")
    print(f"\nPipeline verified end-to-end on N={n_rows} rows. Results are NOT meaningful for "
          f"training or evaluation until INARA is expanded — re-run this same script once a "
          f"larger file is in place.")


if __name__ == "__main__":
    run_logged(main)
