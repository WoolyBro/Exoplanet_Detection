"""
split_signal_families.py
========================

Split version 2: train/val/test grouped by host star AND by signal family.

Why v1 is not enough
    split_catalogs.py (v1) keeps every KOI of one star in one split. That cannot stop the SAME
    astrophysical signal appearing on two different stars: a bright eclipsing binary whose light
    contaminates its neighbours produces a KOI on each of them, with the same period and epoch.
    Measured on v1 (2026-10-17): 301 of the 1,406 scored Kepler test KOIs have such a twin in
    another split, almost all false positives flagged koi_fpflag_ec, and removing them moves test
    ROC-AUC from 0.9158 to 0.8974. Star grouping is blind to this by construction.

The rule (signal twin)
    Two candidates on DIFFERENT stars are twins if
        |P1 - P2| / P < 1e-4                      (same period)
        |epoch offset folded on P| < 0.5 x the longer transit duration   (transits coincide)
    The tolerances were chosen against a null: re-running the match with each candidate's epoch
    scrambled to a random phase estimates the chance-coincidence rate. With a 1e-3 period
    tolerance 93-95 % of TESS "twins" were chance; at 1e-4 about 77 % of Kepler matches and 50 %
    of TESS matches are in excess of chance. The rule is deliberately inclusive: a false merge only
    keeps two unrelated stars in the same split (harmless), a missed twin is leakage.

Groups = connected components of {same star} U {signal twin}. The existing, verified
StratifiedGroupKFold machinery of split_catalogs.py then splits on those groups, with the same
70/15/15 fractions, label strata and random_state=42.

Outputs (v1 files are never touched)
    splits/koi_cumulative_split_v2.csv        full catalogue; `split` is the v2 assignment,
    splits/toi_tess_candidates_split_v2.csv   plus `split_v1` and `signal_family_id`
Each is checked before writing: no star and no signal twin in two splits.

Usage
    python split_signal_families.py            # build + check + write
    python split_signal_families.py --check    # only re-check the written v2 files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import split_catalogs as sc
from research_log import banner, run_logged

RESEARCH_DIR = Path(__file__).resolve().parent
SPLIT_DIR = RESEARCH_DIR / "splits"

PERIOD_REL_TOL = 1e-4
EPOCH_TOL_DURATIONS = 0.5

MISSIONS = {
    "kepler": {"v1": "koi_cumulative_split.csv", "v2": "koi_cumulative_split_v2.csv", "star": "kepid",
               "id": "kepoi_name", "period": "koi_period", "epoch": "koi_time0bk",
               "duration": "koi_duration", "duration_unit_h": True},
    "tess":   {"v1": "toi_tess_candidates_split.csv", "v2": "toi_tess_candidates_split_v2.csv", "star": "tid",
               "id": "toi", "period": "pl_orbper", "epoch": "pl_tranmid",
               "duration": "pl_trandurh", "duration_unit_h": True},
}


def signal_twins(df: pd.DataFrame, cfg: dict) -> list[tuple[int, int]]:
    """Positional index pairs (i, j) of candidates on different stars that share one signal."""
    P = df[cfg["period"]].to_numpy(float)
    T = df[cfg["epoch"]].to_numpy(float)
    D = df[cfg["duration"]].to_numpy(float) / (24.0 if cfg["duration_unit_h"] else 1.0)
    S = df[cfg["star"]].to_numpy()
    ok = np.isfinite(P) & np.isfinite(T) & np.isfinite(D) & (P > 0)
    order = np.flatnonzero(ok)[np.argsort(P[ok])]
    pairs = []
    for a, i in enumerate(order):
        for j in order[a + 1:]:
            if P[j] > P[i] * (1 + PERIOD_REL_TOL):
                break
            if S[i] == S[j]:
                continue
            offset = abs(((T[j] - T[i]) + 0.5 * P[i]) % P[i] - 0.5 * P[i])
            if offset < EPOCH_TOL_DURATIONS * max(D[i], D[j]):
                pairs.append((int(i), int(j)))
    return pairs


def signal_families(df: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, int]:
    """Connected components of {same star} U {signal twin}; returns (family id per row, n twin pairs)."""
    parent = np.arange(len(df))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for idx in df.groupby(cfg["star"]).indices.values():
        for k in idx[1:]:
            union(idx[0], k)
    pairs = signal_twins(df, cfg)
    for i, j in pairs:
        union(i, j)
    roots = np.array([find(i) for i in range(len(df))])
    # Stable, readable ids: family number in order of first appearance.
    _, fam = np.unique(roots, return_inverse=True)
    return fam, len(pairs)


def cross_split_twins(df: pd.DataFrame, cfg: dict, split_col: str = "split") -> int:
    s = df[split_col].to_numpy()
    return sum(1 for i, j in signal_twins(df, cfg) if s[i] != s[j])


def check(df: pd.DataFrame, cfg: dict, name: str) -> None:
    """Raise unless no star and no signal twin spans two splits."""
    star_split = df.groupby(cfg["star"])["split"].nunique()
    shared_stars = int((star_split > 1).sum())
    twins = cross_split_twins(df, cfg)
    print(f"  {name:<7} stars in >1 split: {shared_stars}   signal twins across splits: {twins}")
    if shared_stars or twins:
        raise sc.LeakageError(f"{name}: {shared_stars} shared stars, {twins} cross-split twins")


def build(name: str, cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(SPLIT_DIR / cfg["v1"], low_memory=False,
                     dtype={"koi_quarters": str} if name == "kepler" else None)
    fam, n_pairs = signal_families(df, cfg)
    df["signal_family_id"] = [f"{name}_fam{f}" for f in fam]
    sizes = pd.Series(fam).value_counts()
    print(f"\n[{name}] {len(df)} rows, {df[cfg['star']].nunique()} stars, {n_pairs} signal-twin pairs")
    print(f"  groups: {sizes.size} (v1 had {df[cfg['star']].nunique()} star groups); "
          f"largest {sizes.head(5).tolist()}; rows in groups > 20: {int(sizes[sizes > 20].sum())}")
    print(f"  v1 cross-split twins: {cross_split_twins(df, cfg, 'split')}")

    df = df.rename(columns={"split": "split_v1"}).reset_index(drop=True)
    train, val, test = sc.grouped_stratified_split(
        df, group_col="signal_family_id", label_col="label_harmonized",
        test_size=sc.TEST_SIZE, val_size=sc.VAL_SIZE, random_state=sc.RANDOM_STATE)
    split = pd.Series(index=df.index, dtype=object)
    for s, part in (("train", train), ("val", val), ("test", test)):
        split.loc[part.index] = s
    df.insert(df.columns.get_loc("split_v1"), "split", split)
    if df["split"].isna().any():
        raise RuntimeError(f"{name}: rows without a v2 split")

    labels = df.label_harmonized.fillna(sc.UNLABELED)
    tab = pd.crosstab(df.split, labels, normalize="index").mul(100).round(1)
    tab["rows"] = df.split.value_counts()
    print(tab.reindex(["train", "val", "test"]).to_string())
    moved = (df.split != df.split_v1).mean()
    print(f"  rows whose split changed from v1: {100 * moved:.1f}%")
    check(df, cfg, name)
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only re-check the v2 files on disk")
    args = ap.parse_args(argv)
    banner(f"SPLIT V2: grouped by star AND signal family (|dP|/P < {PERIOD_REL_TOL:g}, "
           f"epoch within {EPOCH_TOL_DURATIONS} x duration)")
    for name, cfg in MISSIONS.items():
        if args.check:
            check(pd.read_csv(SPLIT_DIR / cfg["v2"], low_memory=False), cfg, name)
            continue
        df = build(name, cfg)
        out = SPLIT_DIR / cfg["v2"]
        df.to_csv(out, index=False)
        print(f"  wrote {out.relative_to(RESEARCH_DIR)}")


if __name__ == "__main__":
    run_logged(lambda: main(sys.argv[1:]), append_to_log="--check" not in sys.argv)
