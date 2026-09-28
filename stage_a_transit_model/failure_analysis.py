"""
failure_analysis.py
===================

Checklist item: "Failure analysis: which false positives fool you? (eclipsing
binaries with koi_fpflag_ec, centroid offsets koi_fpflag_co, stellar variability)"
plus "Compare against published dispositions (KOI score, TFOPWG) as reference -
not truth".

What it does, for one trained run, on the VALIDATION rows only:

  1. Re-scores the val views with that run's best.pt.
  2. Picks the operating point by RECALL (default 90 %), never 0.5, matching how the
     rest of the project reports performance.
  3. Splits the true FALSE POSITIVEs into the ones the model wrongly calls
     planet-like ("fooled") and the ones it correctly rejects, then compares the
     Kepler false-positive flags between the two groups. A flag that is commoner
     among the fooled group is a failure mode the model has not learned to see.
        koi_fpflag_nt  not transit-like     (wrong shape: grazing, V, artefact)
        koi_fpflag_ss  stellar eclipse      (real eclipsing binary, secondary present)
        koi_fpflag_co  centroid offset      (signal comes from a neighbouring star)
        koi_fpflag_ec  ephemeris match      (contamination from another known signal)
     koi_fpflag_co is the honest blind spot to expect: centroid information is not
     in a folded light curve at all, so no model reading only these views can see it.
  4. Adds a stellar-variability proxy (out-of-transit scatter from the views) and
     reports it per group.
  5. Compares the model score against koi_score - as a reference point, not ground
     truth. Both are estimates; where they disagree is the interesting part.

Usage
    python stage_a_transit_model/failure_analysis.py --run stage_a_transit_model/runs/base_01
    python stage_a_transit_model/failure_analysis.py --run <dir> --recall 0.95
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RESEARCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_DIR))

import data_manifest as dm  # noqa: E402
from stage_a_transit_model.baselines import shape_features  # noqa: E402
from stage_a_transit_model.cnn_lstm import (  # noqa: E402
    CLASSES, KOI_SPLIT_FILE, PLANET_LIKE_IDX, VIEWS_ROOT,
    build_model, consolidate_views, drop_unusable, planet_like_score, precision_at_recall,
)

FLAGS = {
    "koi_fpflag_nt": "not transit-like (shape)",
    "koi_fpflag_ss": "stellar eclipse (EB)",
    "koi_fpflag_co": "centroid offset (wrong star)",
    "koi_fpflag_ec": "ephemeris match (contamination)",
}


def load_run_model(run_dir: Path):
    """Rebuild the architecture recorded in the run and load its best weights.

    `arch` must be read back too. Without it every checkpoint was rebuilt as the default
    dual_cnn_lstm, so the architecture-ablation runs (global_cnn, dual_cnn and their seed
    repeats) failed to load with a state_dict mismatch: those models have no local branch
    and/or no LSTM, so the tensor names simply are not there. Runs saved before `arch`
    existed have no such key and correctly fall back to the default.
    """
    import torch

    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    saved = ckpt.get("args", {})
    dropout = float(saved.get("dropout", 0.3))
    width = float(saved.get("width", 1.0))
    batchnorm = str(saved.get("batchnorm", "False")).lower() in ("true", "1")
    arch = str(saved.get("arch", "dual_cnn_lstm"))
    model = build_model(dropout, batchnorm=batchnorm, width=width, arch=arch)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, {"dropout": dropout, "width": width, "batchnorm": batchnorm, "arch": arch}


def score_views(model, vs, batch: int = 256) -> np.ndarray:
    import torch

    g = torch.from_numpy(vs.global_view)
    l = torch.from_numpy(vs.local_view)
    out = []
    with torch.no_grad():
        for i in range(0, len(g), batch):
            out.append(torch.softmax(model(g[i:i + batch], l[i:i + batch]), dim=1))
    return torch.cat(out).numpy()


def koi_table() -> pd.DataFrame:
    """KOI catalog columns needed here, read through the manifest loader."""
    with contextlib.redirect_stdout(io.StringIO()):
        koi = dm.load_stage1_catalogs()[KOI_SPLIT_FILE]
    cols = ["kepoi_name", "split", "label_harmonized", "koi_score", *FLAGS]
    have = [c for c in cols if c in koi.columns]
    missing = [c for c in cols if c not in koi.columns]
    if missing:
        print(f"  NOTE: catalog is missing {missing}; those parts are skipped")
    return koi[have].copy()


def group_compare(df: pd.DataFrame, mask_fooled: pd.Series) -> pd.DataFrame:
    """Flag prevalence and variability, fooled vs correctly-rejected false positives."""
    rows = []
    n_f, n_c = int(mask_fooled.sum()), int((~mask_fooled).sum())
    for col, label in FLAGS.items():
        if col not in df.columns:
            continue
        f = df.loc[mask_fooled, col].fillna(0).astype(float).mean() * 100
        c = df.loc[~mask_fooled, col].fillna(0).astype(float).mean() * 100
        rows.append({"signal": f"{col}  {label}", "fooled_pct": f, "rejected_pct": c, "diff_pp": f - c})
    for col, label in [("global_scatter", "out-of-transit scatter (variability)"),
                       ("local_snr", "local depth / wing scatter"),
                       ("secondary_depth", "secondary eclipse depth (phase 0.5)"),
                       ("quadrature_dip", "dip at quadrature (phase 0.25)"),
                       ("transit_asymmetry", "left/right transit asymmetry"),
                       ("v_shape", "floor/depth (1=box, low=V)")]:
        if col in df.columns:
            rows.append({"signal": f"{col}  {label}",
                         "fooled_pct": df.loc[mask_fooled, col].median(),
                         "rejected_pct": df.loc[~mask_fooled, col].median(),
                         "diff_pp": df.loc[mask_fooled, col].median() - df.loc[~mask_fooled, col].median()})
    out = pd.DataFrame(rows)
    out.attrs["n_fooled"], out.attrs["n_rejected"] = n_f, n_c
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="a run directory containing best.pt")
    p.add_argument("--val-views", type=Path, default=VIEWS_ROOT / "kepler_val")
    p.add_argument("--recall", type=float, default=0.90, help="operating point, chosen by recall not 0.5")
    args = p.parse_args(argv)
    # A relative --run is tried against the caller's cwd first, then the project root, so
    # "stage_a_transit_model/runs/x" works whether or not the shell happens to sit in Research/.
    run_dir = Path(args.run).resolve()
    if not (run_dir / "best.pt").is_file() and not Path(args.run).is_absolute():
        alt = (RESEARCH_DIR / args.run).resolve()
        if (alt / "best.pt").is_file():
            run_dir = alt
    if not (run_dir / "best.pt").is_file():
        raise SystemExit(f"no best.pt in {run_dir}")

    print("=" * 88)
    print(f"Failure analysis | run {run_dir.name} | validation rows only")
    print("=" * 88)

    print("\n[1] model and views")
    model, arch = load_run_model(run_dir)
    print(f"  architecture from checkpoint: {arch}")
    vs = drop_unusable(consolidate_views(args.val_views, verbose=True))
    proba = score_views(model, vs)
    score = planet_like_score(proba)
    y = vs.label_index.astype(int)
    y_bin = np.isin(y, PLANET_LIKE_IDX).astype(int)
    print(f"  scored {len(vs)} val rows")

    print(f"\n[2] operating point at recall >= {args.recall:.0%} (not 0.5)")
    par = precision_at_recall(y_bin, score, targets=(args.recall,))[0]
    if not par.get("achievable"):
        raise SystemExit(f"recall {args.recall:.0%} not achievable on these rows")
    thr = par["threshold"]
    print(f"  threshold {thr:.6f} -> precision {par['precision']:.3f}, recall {par['recall']:.3f}")

    print("\n[3] joining the KOI catalog flags")
    koi = koi_table()
    df = pd.DataFrame({"kepoi_name": vs.object_id, "star_id": vs.star_id,
                       "true_index": y, "true_label": [CLASSES[i] for i in y],
                       "model_score": score, "pred_index": proba.argmax(axis=1)})
    df = df.merge(koi, on="kepoi_name", how="left", validate="many_to_one")
    unmatched = int(df["label_harmonized"].isna().sum())
    if unmatched:
        print(f"  WARNING: {unmatched} val row(s) did not match the catalog")
    bad_split = df.loc[df["split"].notna() & (df["split"] != "val"), "kepoi_name"]
    if len(bad_split):
        raise SystemExit(f"{len(bad_split)} row(s) in the val views are not official val KOIs: "
                         f"{list(bad_split[:5])}")
    print(f"  matched {len(df) - unmatched}/{len(df)} rows; all are official val KOIs")
    df = pd.concat([df, shape_features(vs)], axis=1)

    print(f"\n[4] which FALSE POSITIVEs fool the model, at recall {args.recall:.0%}")
    fp = df[df.true_label == "FALSE POSITIVE"].copy()
    fp["fooled"] = fp.model_score >= thr
    cmp = group_compare(fp, fp.fooled)
    n_f, n_c = cmp.attrs["n_fooled"], cmp.attrs["n_rejected"]
    print(f"  {len(fp)} true false positives: {n_f} scored planet-like (fooled), {n_c} correctly rejected")
    print(f"\n  {'signal':<46} {'fooled':>9} {'rejected':>9} {'diff':>9}")
    print("  " + "-" * 76)
    for _, r in cmp.iterrows():
        print(f"  {r.signal:<46} {r.fooled_pct:>9.3f} {r.rejected_pct:>9.3f} {r.diff_pp:>+9.3f}")
    print("\n  (first four rows are % of the group carrying that flag; the rest are medians)")

    flagged = [c for c in FLAGS if c in fp.columns]
    if flagged:
        no_flag = fp[fp[flagged].fillna(0).sum(axis=1) == 0]
        print(f"\n  false positives with NO Kepler flag set: {len(no_flag)} "
              f"({100 * len(no_flag) / max(len(fp), 1):.1f}%), of which "
              f"{int(no_flag.model_score.ge(thr).sum())} fool the model")

    print("\n[5] model score vs koi_score - reference, not truth")
    if "koi_score" in df.columns and df.koi_score.notna().any():
        sub = df[df.koi_score.notna()]
        from scipy.stats import spearmanr
        rho = spearmanr(sub.model_score, sub.koi_score).statistic
        print(f"  Spearman rho = {rho:.3f} on {len(sub)} rows with a published koi_score")
        print("  Both are estimates. Disagreement is not automatically the model being wrong:")
        hi_lo = sub[(sub.model_score >= thr) & (sub.koi_score < 0.5)]
        lo_hi = sub[(sub.model_score < thr) & (sub.koi_score >= 0.5)]
        print(f"    model planet-like, koi_score < 0.5 : {len(hi_lo):>4}  "
              f"(true labels: {dict(hi_lo.true_label.value_counts())})")
        print(f"    model rejects,     koi_score >= 0.5: {len(lo_hi):>4}  "
              f"(true labels: {dict(lo_hi.true_label.value_counts())})")
    else:
        print("  koi_score not available in the catalog; skipped")

    print("\n[6] per-flag recall of the model's rejection")
    for col, label in FLAGS.items():
        if col not in fp.columns:
            continue
        grp = fp[fp[col].fillna(0) == 1]
        if len(grp) == 0:
            print(f"  {col:<16} no val false positives carry this flag")
            continue
        caught = int((grp.model_score < thr).sum())
        print(f"  {col:<16} {caught:>4}/{len(grp):<4} correctly rejected "
              f"({100 * caught / len(grp):5.1f}%)  - {label}")

    out_dir = run_dir / "failure_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = ["kepoi_name", "star_id", "true_label", "model_score", "pred_index", "koi_score",
            *[c for c in FLAGS if c in df.columns],
            "global_scatter", "local_snr", "secondary_depth", "quadrature_dip",
            "transit_asymmetry", "v_shape", "local_floor"]
    df[[c for c in keep if c in df.columns]].to_csv(out_dir / "val_predictions.csv", index=False)
    cmp.to_csv(out_dir / "fooled_vs_rejected.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps({
        "run": run_dir.name, "generated_at": datetime.now().isoformat(timespec="seconds"),
        "architecture": arch, "recall_target": args.recall, "threshold": thr,
        "precision_at_target": par["precision"], "n_val": int(len(df)),
        "n_false_positives": int(len(fp)), "n_fooled": int(n_f), "n_rejected": int(n_c),
    }, indent=2, default=float), encoding="utf-8")
    print(f"\nsaved {out_dir.relative_to(RESEARCH_DIR)}/ "
          f"(val_predictions.csv, fooled_vs_rejected.csv, summary.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
