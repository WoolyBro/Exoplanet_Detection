"""
audit_inara_leakage.py
======================

Systematic derived-column audit of INARA (replaces checking columns by hand).

Check 1 (gas -> column): can a non-gas column be predicted from the 12 gas
         fractions? Single-gas correlation, all 12 gases linearly, and the best
         1-3 gas subset in linear and log-log (power-law) form.
Check 2 (column -> column): can a non-gas column be predicted from OTHER
         non-gas columns?
           2a  best 1-4 column subset, linear and log-log form
           2b  low-cardinality columns (e.g. star_class) that are exact bins of
               another column (non-overlapping value ranges per class)
           2c  explicit physics formulas (equilibrium temperature)

Small-sample safeguards: fits are scored by leave-one-out (LOO) R^2, never
in-sample R^2, and each search is repeated on permuted targets. A column is
flagged only if LOO R^2 >= 0.98 (or |r| >= 0.98) AND it beats the 99.9th
percentile of that permutation null, so chance fits on a tiny file are not
reported as derivations. Near misses (LOO R^2 >= 0.90) are listed for review.

Exclusions: check 1, 2b and 2c hits are added to exclude_from_features. Check 2a
hits flag every member of a relation, so they are recorded as relation groups
needing a human decision rather than excluded automatically.

Reads INARA via data_manifest.load_stage2_catalogs(). The only file written is
outputs/habitability_prepared/label_rule.json (skipped with --dry-run). Output is
appended to run_log.txt.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root: shared modules
import data_manifest as dm  # noqa: E402
from prepare_habitability_data import INARA_FILE, LABEL_COL, MOLAR_MASS  # noqa: E402
from research_log import banner, run_logged  # noqa: E402

RESEARCH_DIR = dm.RESEARCH_DIR
RULE_JSON = RESEARCH_DIR / "outputs" / "habitability_prepared" / "label_rule.json"

FLAG_R2 = 0.98
REVIEW_R2 = 0.90
MAX_SUBSET_GAS = 3
MAX_SUBSET_COLUMNS = 4   # 4 so e.g. T_eq(T_star, R_star, a, albedo) is inside the search
MAX_BIN_LEVELS = 10      # columns with <= this many distinct values are tested as bins
N_PERMUTATIONS = 500
RNG_SEED = 42
SOLAR_RADIUS_AU = 0.00465047


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def loo_r2(X: np.ndarray, Y: np.ndarray, intercept: bool = True) -> np.ndarray:
    """Leave-one-out R^2 of OLS fits of each column of Y on X (closed form via the hat matrix).

    Y may be (n,) or (n, k); returns a scalar or an array of k scores.
    """
    if intercept:
        X = np.column_stack([np.ones(len(X)), X])
    H = X @ np.linalg.pinv(X)
    h = np.clip(np.diag(H), None, 1 - 1e-9)
    Y2 = Y.reshape(len(Y), -1)
    resid = Y2 - H @ Y2
    press = ((resid / (1 - h)[:, None]) ** 2).sum(axis=0)
    ss = ((Y2 - Y2.mean(axis=0)) ** 2).sum(axis=0)
    r2 = np.where(ss > 0, 1 - press / ss, np.nan)
    return r2 if Y.ndim > 1 else r2[0]


def best_subset(y: np.ndarray, pool: pd.DataFrame, max_k: int, log_form: bool, perms: np.ndarray):
    """Best LOO R^2 over every 1..max_k column subset of `pool`, plus its permutation null.

    log_form fits log(y) on log(predictors) (power laws become linear); only
    strictly positive columns take part. Returns (score, columns, null) where
    null[i] is the best score the same search reaches on the i-th permuted y.
    """
    cols = list(pool.columns)
    if log_form:
        if (y <= 0).any():
            return np.nan, (), None
        y = np.log(y)
        cols = [c for c in cols if (pool[c] > 0).all()]
    X_all = np.log(pool[cols].to_numpy(float)) if log_form else pool[cols].to_numpy(float)
    Yp = y[perms].T
    best, best_cols = -np.inf, ()
    minimal = None  # smallest subset reaching FLAG_R2, so relations aren't padded with extra columns
    null = np.full(len(perms), -np.inf)
    for k in range(1, min(max_k, len(cols)) + 1):
        for idx in combinations(range(len(cols)), k):
            X = X_all[:, idx]
            score = loo_r2(X, y)
            if score > best:
                best, best_cols = score, tuple(cols[i] for i in idx)
            if score >= FLAG_R2 and (minimal is None or (len(idx) == len(minimal[1]) and score > minimal[0])):
                minimal = (score, tuple(cols[i] for i in idx))
            null = np.maximum(null, loo_r2(X, Yp))
    if minimal is not None:
        return minimal[0], minimal[1], null
    return best, best_cols, null


def summarise(column, pear, options, extra):
    """Pick the best fit among options and decide the flag against its own null."""
    options = [o for o in options if o[3] is not None and np.isfinite(o[0])]
    score, cols, form, null = max(options, key=lambda o: o[0])
    p999 = float(np.nanpercentile(null, 99.9))
    row = {"column": column, **extra,
           "best LOO R2": round(float(score), 4),
           "best fit": form if not cols else f"{form}: {', '.join(cols)}",
           "null p99.9": round(p999, 3)}
    row["flag"] = bool((score >= FLAG_R2 or pear >= FLAG_R2) and score > p999)
    return row


def _separable(levels: np.ndarray, values: np.ndarray) -> bool:
    """True if the value ranges of the levels do not overlap (levels are threshold bins of values)."""
    order = pd.DataFrame({"l": levels, "v": values}).groupby("l")["v"].agg(["min", "max"]).sort_values("min")
    return bool((order["min"].to_numpy()[1:] > order["max"].to_numpy()[:-1]).all())


def bin_derivation(df: pd.DataFrame, target: str, candidates: list[str], perms: np.ndarray) -> list[dict]:
    """Columns that `target` could be an exact binning of, with a permutation p-value.

    Levels with only one or two rows can fall into a gap by chance, so each hit
    is compared with how often shuffled levels are also separable.
    """
    found = []
    levels = df[target].to_numpy()
    for c in candidates:
        if c == target:
            continue
        values = df[c].to_numpy(float)
        if not _separable(levels, values):
            continue
        p = (1 + sum(_separable(levels[idx], values) for idx in perms)) / (1 + len(perms))
        ranges = df.groupby(target)[c].agg(["min", "max", "count"])
        found.append({"column": target, "binned_from": c, "p_value": round(float(p), 4),
                      "flag": p < 0.001,
                      "ranges": {str(k): [round(float(v["min"]), 4), round(float(v["max"]), 4), int(v["count"])]
                                 for k, v in ranges.iterrows()}})
    return found


def main(write: bool) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 100)
    print(f"\n\n{'#' * 88}\n# INARA DERIVED-COLUMN AUDIT (audit_inara_leakage.py, run "
          f"{datetime.now():%Y-%m-%d %H:%M:%S})\n{'#' * 88}")

    df = dm.load_stage2_catalogs()[INARA_FILE]
    n = len(df)
    gases = [g for g in MOLAR_MASS if g in df.columns]
    numeric = df.select_dtypes("number").columns
    candidates = [c for c in numeric if c not in gases and c != LABEL_COL]
    print(f"  rows: {n}; gas columns ({len(gases)}): {gases}")
    print(f"  columns audited ({len(candidates)}): {candidates}")
    print(f"  non-numeric columns: {[c for c in df.columns if c not in numeric] or 'none'}; "
          f"constant columns: {[c for c in candidates if df[c].nunique() <= 1] or 'none'}")
    print(f"  identifier-like columns (unique integers): "
          f"{[c for c in candidates if pd.api.types.is_integer_dtype(df[c]) and df[c].is_unique] or 'none'}")

    rng = np.random.default_rng(RNG_SEED)
    perms = np.array([rng.permutation(n) for _ in range(N_PERMUTATIONS)])
    G = df[gases].to_numpy(float)

    # ---- Check 1: gas fractions -> column ---------------------------------- #
    banner("CHECK 1: predictable from the gas fractions?")
    rows1 = []
    for c in candidates:
        y = df[c].to_numpy(float)
        pear = df[gases].corrwith(df[c]).abs()
        # The fractions sum to 1, which already acts as an intercept.
        full = (loo_r2(G, y, intercept=False), (), "linear, all gases", loo_r2(G, y[perms].T, intercept=False))
        lin = best_subset(y, df[gases], MAX_SUBSET_GAS, False, perms)
        log = best_subset(y, df[gases], MAX_SUBSET_GAS, True, perms)
        rows1.append(summarise(c, pear.max(), [full, (*lin[:2], "linear", lin[2]), (*log[:2], "log-log", log[2])],
                               {"max |r| (gas)": round(float(pear.max()), 3), "gas": pear.idxmax(),
                                "LOO R2 all gases": round(float(full[0]), 4)}))
    t1 = pd.DataFrame(rows1).sort_values("best LOO R2", ascending=False)
    print(t1.to_string(index=False))

    # ---- Check 2a: other non-gas columns -> column ------------------------ #
    banner(f"CHECK 2a: predictable from 1-{MAX_SUBSET_COLUMNS} OTHER non-gas columns?")
    rows2 = []
    for c in candidates:
        y = df[c].to_numpy(float)
        pool = df[[o for o in candidates if o != c]]
        pear = pool.corrwith(df[c]).abs()
        lin = best_subset(y, pool, MAX_SUBSET_COLUMNS, False, perms)
        log = best_subset(y, pool, MAX_SUBSET_COLUMNS, True, perms)
        rows2.append(summarise(c, pear.max(), [(*lin[:2], "linear", lin[2]), (*log[:2], "log-log", log[2])],
                               {"max |r| (column)": round(float(pear.max()), 3), "most correlated": pear.idxmax()}))
    t2 = pd.DataFrame(rows2).sort_values("best LOO R2", ascending=False)
    print(t2.to_string(index=False))

    # ---- Check 2b: binned columns ----------------------------------------- #
    banner(f"CHECK 2b: low-cardinality columns (<= {MAX_BIN_LEVELS} levels) that are exact bins of another column")
    low_card = [c for c in candidates if 1 < df[c].nunique() <= MAX_BIN_LEVELS]
    print("  low-cardinality columns: " + (", ".join(f"{c} ({df[c].nunique()} levels: "
                                                        f"{df[c].value_counts().sort_index().to_dict()})"
                                                        for c in low_card) or "none"))
    bin_hits = [b for c in low_card for b in bin_derivation(df, c, candidates, perms)]
    for hit in bin_hits:
        print(f"  {hit['column']} levels have non-overlapping ranges of {hit['binned_from']} "
              f"[min, max, rows] {hit['ranges']}; permutation p = {hit['p_value']} -> "
              f"{'FLAG' if hit['flag'] else 'not significant (small levels can separate by chance)'}")
    if low_card and not bin_hits:
        print("  none: every column's value ranges overlap across the levels")
    bins = [hit for hit in bin_hits if hit["flag"]]

    # ---- Check 2c: explicit physics --------------------------------------- #
    banner("CHECK 2c: explicit physics formula (equilibrium temperature)")
    physics = []
    needed = ["star_temperature_in_Kelvin", "star_radius_in_Solar_radii", "semimajor_axis_of_the_planet_in_AU",
              "planets_mean_surface_albedo", "planet_surface_temperature_Kelvin"]
    if all(c in df for c in needed):
        t_eq = (df.star_temperature_in_Kelvin * np.sqrt(df.star_radius_in_Solar_radii * SOLAR_RADIUS_AU
                                                          / (2 * df.semimajor_axis_of_the_planet_in_AU))
                * (1 - df.planets_mean_surface_albedo) ** 0.25)
        y = df.planet_surface_temperature_Kelvin.to_numpy(float)
        r = float(np.corrcoef(t_eq, y)[0, 1])
        score = float(loo_r2(t_eq.to_numpy()[:, None], y))
        null = loo_r2(t_eq.to_numpy()[:, None], y[perms].T)
        ratio = y / t_eq
        print(f"  T_eq = T_star * sqrt(R_star / 2a) * (1 - albedo)^0.25: range {t_eq.min():.1f}-{t_eq.max():.1f} K")
        print(f"  planet_surface_temperature_Kelvin vs T_eq: r = {r:.3f}, LOO R2 = {score:.4f}, "
              f"null p99.9 = {np.nanpercentile(null, 99.9):.3f}")
        print(f"  surface T / T_eq: {ratio.min():.3f}-{ratio.max():.3f} (median {np.median(ratio):.3f})")
        physics.append({"column": "planet_surface_temperature_Kelvin", "formula": "T_eq(T_star, R_star, a, albedo)",
                        "r": round(r, 3), "loo_r2": round(score, 4),
                        "flag": bool((score >= FLAG_R2 or abs(r) >= FLAG_R2) and score > np.nanpercentile(null, 99.9))})

    # ---- Step 3: consolidated report --------------------------------------- #
    banner("REPORT: every column that trips a check")
    flagged1 = t1[t1.flag].to_dict("records")
    flagged2 = t2[t2.flag].to_dict("records")
    flagged_phys = [p for p in physics if p["flag"]]
    print(f"\n  Check 1 (gas -> column): {len(flagged1)} flagged")
    for r in flagged1:
        print(f"    - {r['column']}: LOO R2 {r['best LOO R2']} ({r['best fit']}), max |r| {r['max |r| (gas)']}, "
              f"null p99.9 {r['null p99.9']}")
    print(f"\n  Check 2a (column -> column): {len(flagged2)} flagged")
    for r in flagged2:
        print(f"    - {r['column']}: LOO R2 {r['best LOO R2']} ({r['best fit']}), null p99.9 {r['null p99.9']}")
    print(f"  Check 2b (binned columns): {len(bins)} flagged")
    for b in bins:
        print(f"    - {b['column']} is a binning of {b['binned_from']}")
    print(f"  Check 2c (physics formula): {len(flagged_phys)} flagged")

    review = [(r["column"], "1", r["best LOO R2"], r["best fit"]) for r in t1.to_dict("records")
              if not r["flag"] and r["best LOO R2"] >= REVIEW_R2]
    review += [(r["column"], "2a", r["best LOO R2"], r["best fit"]) for r in t2.to_dict("records")
               if not r["flag"] and r["best LOO R2"] >= REVIEW_R2]
    # Binnings that are separable but not significant (tiny levels): worth a human look.
    review += [(h["column"], "2b", None, f"non-overlapping bins of {h['binned_from']}, permutation p = "
                f"{h['p_value']} (levels too small to confirm): {h['ranges']}")
               for h in bin_hits if not h["flag"] and h["p_value"] < 0.05]
    print(f"\n  near misses for review (LOO R2 >= {REVIEW_R2}, or binning p < 0.05; below the flag): {'none' if not review else ''}")
    for col, check, score, fit in review:
        print(f"    - {col} (check {check}): " + (f"LOO R2 {score} ({fit})" if score is not None else fit))

    print(f"\n  Method limits: with {n} rows, relations of up to {MAX_SUBSET_GAS} gases / "
          f"{MAX_SUBSET_COLUMNS} columns in linear or power-law form, exact binnings, and the")
    print("  T_eq formula were tested. Other nonlinear derivations could still go undetected; re-run this")
    print("  audit whenever INARA is replaced with a larger file.")

    # ---- Step 4: update label_rule.json ------------------------------------ #
    banner("UPDATE label_rule.json")
    rule = json.loads(RULE_JSON.read_text(encoding="utf-8"))
    exclude = list(rule.get("exclude_from_features", []))
    reasons = dict(rule.get("exclude_reasons", {}))
    for g in gases:
        reasons.setdefault(g, "raw gas fraction; defines the composition-based label")
    added = []

    def add(col, reason):
        reasons[col] = reason
        if col not in exclude:
            exclude.append(col)
            added.append(col)

    for r in flagged1:
        add(r["column"], f"derived from the gas fractions (audit check 1): LOO R2 {r['best LOO R2']} via "
                         f"{r['best fit']}; max single-gas |r| {r['max |r| (gas)']}; permutation null 99.9th pct "
                         f"{r['null p99.9']}")
    # A column-to-column relation flags every member (target and predictors alike), and the
    # data cannot say which member was computed from the others. Group the members into
    # relation sets for a human decision instead of excluding them all automatically.
    groups: list[dict] = []
    for r in flagged2:
        members = {r["column"], *r["best fit"].split(": ", 1)[1].split(", ")}
        for g in groups:
            if g["members"] & members:
                g["members"] |= members
                g["fits"].append(f"{r['column']} ~ {r['best fit']} (LOO R2 {r['best LOO R2']})")
                break
        else:
            groups.append({"members": members,
                           "fits": [f"{r['column']} ~ {r['best fit']} (LOO R2 {r['best LOO R2']})"]})
    for b in bins:
        add(b["column"], f"exact binning of {b['binned_from']} (audit check 2b): non-overlapping ranges {b['ranges']}")
    for p in flagged_phys:
        add(p["column"], f"computable by {p['formula']} (audit check 2c): r {p['r']}, LOO R2 {p['loo_r2']}")

    rule["exclude_from_features"] = exclude
    rule["exclude_reasons"] = reasons
    rule["leakage_audit"] = {
        "run_by": "audit_inara_leakage.py",
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "n_rows": n,
        "method": (f"LOO R2 of OLS fits: all gases, best 1-{MAX_SUBSET_GAS} gas subsets, best "
                   f"1-{MAX_SUBSET_COLUMNS} non-gas column subsets (linear and log-log); exact binning of "
                   f"low-cardinality columns; T_eq formula. Flag if LOO R2 >= {FLAG_R2} or |r| >= {FLAG_R2} and "
                   f"above the 99.9th pct of a {N_PERMUTATIONS}-permutation null."),
        "flagged": {"check1_gas_to_column": [r["column"] for r in flagged1],
                    "check2a_column_to_column": [r["column"] for r in flagged2],
                    "check2a_relation_groups_needing_decision": [
                        {"members": sorted(g["members"]), "fits": g["fits"],
                         "action": "keep at most one member per group as a feature; not auto-excluded"}
                        for g in groups],
                    "check2b_binned": [b["column"] for b in bins],
                    "check2c_physics": [p["column"] for p in flagged_phys]},
        "near_misses_for_review": [{"column": c, "check": k, "loo_r2": s, "fit": f} for c, k, s, f in review],
        "note": "JSON has no comments, so each exclusion's reasoning is in exclude_reasons.",
    }
    print(f"  newly excluded: {added or 'none'}")
    print(f"  column-to-column relation groups needing a decision (not auto-excluded): "
          f"{[sorted(g['members']) for g in groups] or 'none'}")
    print(f"  near misses recorded for review (not excluded): {[c for c, *_ in review] or 'none'}")
    if write:
        RULE_JSON.write_text(json.dumps(rule, indent=2), encoding="utf-8")
        print(f"  wrote {RULE_JSON.relative_to(RESEARCH_DIR)}")
    else:
        print("  --dry-run: label_rule.json not written")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    run_logged(lambda: main(write=not dry), append_to_log=not dry)
