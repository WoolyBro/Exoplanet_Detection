#!/usr/bin/env python3
"""
train_transit_baseline.py  (Level-1 tabular baseline)
=====================================================

RandomForest on catalog features: 3-class disposition report plus a binary
planet-like probability. Rebuilt from the mentor's starter script with three
differences:

1. Data: Stage 1 split files via data_manifest.load_stage1_catalogs(), not the
   quarantined 05_final_ML_dataset_DO_NOT_USE/ tables.
2. Split: the existing star-grouped train/val/test column is used as-is (no
   train_test_split), so KOIs of one star never sit in both train and test.
3. Leakage: columns that encode the human/pipeline verdict are dropped.

Labels: label_harmonized (FALSE POSITIVE / CANDIDATE / CONFIRMED). Rows with no
label (14 TESS TOIs) are left out.
Binary mapping: planet-like = CANDIDATE or CONFIRMED, not planet-like = FALSE
POSITIVE (the mentor's label >= 1). P(planet-like) = P(CANDIDATE) + P(CONFIRMED)
from the same 3-class model, so the two reports come from one model.

Usage (from the Research folder or anywhere):
    python tools/train_transit_baseline.py                  # Kepler, all honest numeric features
    python tools/train_transit_baseline.py --toi            # TESS TOI
    python tools/train_transit_baseline.py --features mentor   # mentor's physical feature list
    python tools/train_transit_baseline.py --include-leaky     # CONTROL: keep verdict columns
Options: --n-jobs N (default 2), --no-save
Writes: baseline_results/<run>.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, roc_auc_score)

RESEARCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_DIR))
import data_manifest as dm  # noqa: E402

CLASSES = ["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"]
PLANET_LIKE = ["CANDIDATE", "CONFIRMED"]

# ---- Columns removed from features ------------------------------------------ #
# Verdict leakage named by the mentor (Kepler only; the TOI table has no equivalents).
LEAKY = {
    "kepler": [r"^koi_fpflag_", r"^koi_score$", r"^koi_pdisposition$"],
    "toi": [],
}
# Flagged by the Step 3 scan: verdict/vetting text and provenance metadata. All are text
# columns, so a numeric feature set excludes them anyway; listed so the exclusion is explicit.
REVIEW_FLAGGED = {
    "kepler": ["kepler_name",        # only filled for CONFIRMED planets -> perfect label leak
               "koi_comment",        # Robovetter comment flags (e.g. DEEP_V_SHAPED)
               "koi_vet_stat", "koi_vet_date", "koi_disp_prov", "koi_parm_prov", "koi_sparprov",
               "koi_delivname", "koi_tce_delivname", "koi_datalink_dvr", "koi_datalink_dvs", "koi_fittype"],
    "toi": ["toi_created", "rowupdate", "release_date"],
}
# Labels, split bookkeeping, identifiers and sky position (position identifies the star).
NON_FEATURES = {
    "kepler": ["koi_disposition", "label_harmonized", "split", "star_group_id", "kepid", "kepoi_name",
               "ra", "dec", "koi_quarters"],
    "toi": ["tfopwg_disp", "label_harmonized", "split", "star_group_id", "tid", "toi", "toidisplay",
            "toipfx", "ctoi_alias", "ra", "dec", "rastr", "decstr"],
}
# The mentor's hand-picked physical features, minus the fpflags.
MENTOR_FEATURES = {
    "kepler": ["koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_teq", "koi_insol", "koi_impact",
               "koi_ror", "koi_dor", "koi_model_snr", "koi_num_transits", "koi_steff", "koi_slogg", "koi_srad"],
    "toi": ["pl_orbper", "pl_trandurh", "pl_trandep", "pl_rade", "pl_insol", "pl_eqt", "st_tmag", "st_teff",
            "st_logg", "st_rad"],
}
RF_PARAMS = dict(n_estimators=300, random_state=42, class_weight="balanced_subsample")  # as in the mentor's script


def load(tag: str) -> pd.DataFrame:
    name = "toi_tess_candidates_split.csv" if tag == "toi" else "koi_cumulative_split.csv"
    with contextlib.redirect_stdout(io.StringIO()):
        df = dm.load_stage1_catalogs()[name]
    return df[df["label_harmonized"].isin(CLASSES)].copy()


def select_features(df: pd.DataFrame, tag: str, feature_set: str, include_leaky: bool) -> tuple[list[str], dict]:
    leaky = [c for c in df.columns if any(re.search(p, c) for p in LEAKY[tag])]
    dropped = {"leaky (mentor-named)": [] if include_leaky else leaky,
               "review-flagged (Step 3)": [c for c in REVIEW_FLAGGED[tag] if c in df.columns],
               "label/split/id/position": [c for c in NON_FEATURES[tag] if c in df.columns]}
    if feature_set == "mentor":
        feats = [c for c in MENTOR_FEATURES[tag] if c in df.columns] + (leaky if include_leaky else [])
    else:
        removed = set().union(*dropped.values())
        numeric = [c for c in df.columns if c not in removed and pd.api.types.is_numeric_dtype(df[c])]
        train = df[df.split == "train"]
        feats = [c for c in numeric if train[c].notna().any() and train[c].nunique(dropna=True) > 1]
        if include_leaky:
            feats += [c for c in leaky if c not in feats and pd.api.types.is_numeric_dtype(df[c])]
        dropped["non-numeric (text)"] = [c for c in df.columns if c not in removed and c not in numeric]
        dropped["empty/constant in train"] = [c for c in numeric if c not in feats]
    # Guard: no verdict column may slip into an honest run.
    if not include_leaky:
        bad = [c for c in feats if any(re.search(p, c) for p in LEAKY[tag]) or c in REVIEW_FLAGGED[tag]]
        assert not bad, f"leaky columns in features: {bad}"
    return feats, dropped


def evaluate(clf, X: pd.DataFrame, y: pd.Series) -> dict:
    proba = pd.DataFrame(clf.predict_proba(X), columns=clf.classes_, index=X.index)
    pred = proba.idxmax(axis=1)
    y_bin = y.isin(PLANET_LIKE).astype(int)
    p_planet = proba[PLANET_LIKE].sum(axis=1)
    pred_bin = (p_planet >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "accuracy_3class": accuracy_score(y, pred),
        "report_3class": classification_report(y, pred, labels=CLASSES, digits=3, zero_division=0),
        "confusion_3class": confusion_matrix(y, pred, labels=CLASSES).tolist(),
        "accuracy_binary": accuracy_score(y_bin, pred_bin),
        "roc_auc_binary": roc_auc_score(y_bin, p_planet),
        "pr_auc_binary": average_precision_score(y_bin, p_planet),
        "report_binary": classification_report(y_bin, pred_bin, labels=[0, 1], digits=3, zero_division=0,
                                               target_names=["FALSE POSITIVE", "PLANET-LIKE"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toi", action="store_true", help="use the TESS TOI split instead of Kepler")
    ap.add_argument("--features", choices=["all", "mentor"], default="all")
    ap.add_argument("--include-leaky", action="store_true", help="CONTROL ONLY: keep verdict columns")
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    tag = "toi" if args.toi else "kepler"
    run = f"{tag}_{args.features}" + ("_LEAKY_CONTROL" if args.include_leaky else "")

    df = load(tag)
    feats, dropped = select_features(df, tag, args.features, args.include_leaky)
    parts = {s: df[df.split == s] for s in ["train", "val", "test"]}
    for s in ["val", "test"]:  # the star-grouped split must hold
        assert not set(parts["train"].star_group_id) & set(parts[s].star_group_id), f"star leak train/{s}"

    print(f"=== {run} ===")
    print(f"source: load_stage1_catalogs() -> {'toi_tess_candidates_split.csv' if args.toi else 'koi_cumulative_split.csv'}"
          f" | existing star-grouped split, no re-split")
    print("rows: " + ", ".join(f"{s} {len(p)}" for s, p in parts.items()))
    print("train class balance: " + ", ".join(f"{c} {int((parts['train'].label_harmonized == c).sum())}" for c in CLASSES))
    if args.include_leaky:
        print("*** CONTROL RUN: verdict columns INCLUDED - results are not an honest baseline ***")
    print(f"features ({len(feats)}): {feats}")
    for k, v in dropped.items():
        print(f"dropped {k} ({len(v)}): {v}")

    started = time.time()
    clf = RandomForestClassifier(**RF_PARAMS, n_jobs=args.n_jobs)  # handles NaN natively (sklearn >= 1.4)
    clf.fit(parts["train"][feats], parts["train"]["label_harmonized"])
    results = {s: evaluate(clf, parts[s][feats], parts[s]["label_harmonized"]) for s in ["val", "test"]}

    for s in ["val", "test"]:
        r = results[s]
        print(f"\n--- {s.upper()} (n={r['n']}) ---")
        print(f"3-class accuracy {r['accuracy_3class']:.3f}")
        print(r["report_3class"])
        print("confusion (rows true, cols pred; FP, CAND, CONF):", r["confusion_3class"])
        print(f"binary planet-like (CANDIDATE+CONFIRMED vs FALSE POSITIVE): accuracy {r['accuracy_binary']:.3f}, "
              f"ROC-AUC {r['roc_auc_binary']:.3f}, PR-AUC {r['pr_auc_binary']:.3f}")
        print(r["report_binary"])
    imp = pd.Series(clf.feature_importances_, index=feats).sort_values(ascending=False)
    print("top features:", ", ".join(f"{k} {v:.3f}" for k, v in imp.head(10).items()))
    print(f"fit+eval {time.time() - started:.1f} s")

    if not args.no_save:
        out = RESEARCH_DIR / "baseline_results"
        out.mkdir(exist_ok=True)
        payload = {"run": run, "generated_at": datetime.now().isoformat(timespec="seconds"),
                   "features": feats, "dropped": dropped, "rf_params": RF_PARAMS,
                   "binary_mapping": "planet-like = CANDIDATE or CONFIRMED; P = P(CANDIDATE) + P(CONFIRMED)",
                   "top_features": imp.head(15).round(4).to_dict(),
                   "results": {s: {k: v for k, v in r.items()} for s, r in results.items()}}
        (out / f"{run}.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
        print(f"saved baseline_results/{run}.json")


if __name__ == "__main__":
    main()
