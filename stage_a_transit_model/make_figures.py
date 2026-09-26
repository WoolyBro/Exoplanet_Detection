"""
make_figures.py
===============

Figures for the Track A Level 2 write-up. Everything is drawn from files already on
disk (run reports, sweep table, failure-analysis CSVs) - nothing is retrained here, so
this is safe to re-run and cannot change any result.

Panels
  1. Model ladder: majority -> logistic -> shape+GBM -> CNN variants, on val ROC-AUC,
     with the seed spread drawn as an error bar so the reader can see which gaps are real.
  2. Precision-recall on the binary planet-like score for the chosen run, with the
     fixed-recall operating points marked (90/95/99 %), because that is how the
     project reports performance - never at 0.5.
  3. Training curves for the chosen run (loss and val AUC), with the selected epoch marked.
  4. Failure analysis: Kepler false-positive flag prevalence among the false positives
     the model is fooled by vs those it correctly rejects.

Usage
    python stage_a_transit_model/make_figures.py --run stage_a_transit_model/runs/bn_aug_sched
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESEARCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_DIR = RESEARCH_DIR / "stage_a_transit_model"
RUNS = MODEL_DIR / "runs"
EDA = RESEARCH_DIR / "eda"

FLAG_LABELS = {
    "koi_fpflag_nt": "not transit-like",
    "koi_fpflag_ss": "stellar eclipse (EB)",
    "koi_fpflag_co": "centroid offset",
    "koi_fpflag_ec": "ephemeris match",
}


def resolve_run(arg: str) -> Path:
    p = Path(arg).resolve()
    if not (p / "report.json").is_file() and not Path(arg).is_absolute():
        alt = (RESEARCH_DIR / arg).resolve()
        if (alt / "report.json").is_file():
            return alt
    return p


def seed_groups(df: pd.DataFrame) -> dict[str, list[float]]:
    """Group repeat runs (name_s<seed>) with their seed-42 parent."""
    import re

    out: dict[str, list[float]] = {}
    for _, r in df.iterrows():
        # Trailing seed suffix only: "bn_aug_sched_s1" -> "bn_aug_sched", but
        # "bn_aug_sched" itself must not be cut at its internal "_sched".
        base = re.sub(r"_s\d+$", "", str(r["config"]))
        out.setdefault(base, []).append(r["val_roc_auc"])
    return out


def panel_ladder(ax, sweep: pd.DataFrame, baselines: dict) -> None:
    rows = [("majority", baselines["majority"]["roc_auc_binary"], 0.0),
            ("logistic\n(raw views)", baselines["logistic"]["roc_auc_binary"], 0.0),
            ("11 shape feats\n+ GBM", baselines["shape_gbm"]["roc_auc_binary"], 0.0)]
    groups = seed_groups(sweep)
    for name in ["base_01", "bn", "bn_aug_sched", "drop45"]:
        if name in groups:
            vals = [v for v in groups[name] if v is not None]
            if vals:
                label = {"base_01": "CNN+LSTM\nbaseline", "bn": "+ batchnorm",
                         "bn_aug_sched": "+ aug + sched", "drop45": "+ dropout .45"}.get(name, name)
                rows.append((label, float(np.mean(vals)), float(np.std(vals)) if len(vals) > 1 else 0.0))

    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    errs = [r[2] for r in rows]
    colors = ["#999999"] * 3 + ["#2d6fa8"] * (len(rows) - 3)
    ax.bar(range(len(rows)), vals, yerr=errs, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, fontsize=7, rotation=30, ha="right")
    ax.set_ylim(0.45, 0.96)
    ax.axhline(0.5, color="crimson", lw=0.8, ls=":", label="chance")
    ax.set_ylabel("validation ROC-AUC (planet-like)")
    ax.set_title("Model ladder: does the deep model earn its complexity?", fontsize=9)
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(i, v + e + 0.008, f"{v:.3f}", ha="center", fontsize=7)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="y", alpha=0.3)


def panel_pr(ax, run_dir: Path) -> None:
    sweep = pd.read_csv(run_dir / "threshold_sweep.csv")
    s = sweep.dropna(subset=["precision", "recall"]).sort_values("recall")
    ax.plot(s.recall, s.precision, color="#2d6fa8", lw=1.6)
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    for row in (report["val_metrics"] or {}).get("precision_at_recall", []):
        if row.get("achievable"):
            ax.plot(row["recall"], row["precision"], "o", ms=6, color="crimson")
            ax.annotate(f"R{row['target_recall']:.0%}\nP={row['precision']:.3f}",
                        (row["recall"], row["precision"]), textcoords="offset points",
                        xytext=(-38, -6), fontsize=7, color="crimson")
    ax.set_xlabel("recall (completeness)")
    ax.set_ylabel("precision")
    ax.set_title(f"Precision-recall, {run_dir.name}\noperating points chosen by recall, never 0.5", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1.02)


def panel_history(ax, run_dir: Path) -> None:
    h = pd.read_csv(run_dir / "history.csv")
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    best = (report.get("training") or {}).get("best_epoch")
    ax.plot(h.epoch, h.train_loss, color="#c06000", lw=1.4, label="train loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss", color="#c06000")
    ax.tick_params(axis="y", labelcolor="#c06000")
    ax2 = ax.twinx()
    ax2.plot(h.epoch, h.val_roc_auc_binary, color="#2d6fa8", lw=1.4, label="val ROC-AUC")
    ax2.set_ylabel("val ROC-AUC", color="#2d6fa8")
    ax2.tick_params(axis="y", labelcolor="#2d6fa8")
    if best:
        ax.axvline(best, color="crimson", ls="--", lw=1.0)
        ax.text(best, ax.get_ylim()[1], f" selected epoch {best}", color="crimson",
                fontsize=7, va="top")
    ax.set_title("Training curve (early stopping on val AUC)", fontsize=9)
    ax.grid(alpha=0.3)


def panel_failure(ax, run_dir: Path) -> None:
    path = run_dir / "failure_analysis" / "fooled_vs_rejected.csv"
    if not path.is_file():
        ax.text(0.5, 0.5, "no failure analysis yet\n(run failure_analysis.py)",
                ha="center", va="center", fontsize=8)
        ax.axis("off")
        return
    cmp = pd.read_csv(path)
    flags = cmp[cmp.signal.str.startswith("koi_fpflag")].copy()
    flags["short"] = [FLAG_LABELS.get(s.split()[0], s.split()[0]) for s in flags.signal]
    y = np.arange(len(flags))
    ax.barh(y - 0.2, flags.fooled_pct, height=0.38, color="#c23b22", label="fooled the model")
    ax.barh(y + 0.2, flags.rejected_pct, height=0.38, color="#3b7a57", label="correctly rejected")
    ax.set_yticks(y)
    ax.set_yticklabels(flags.short, fontsize=8)
    ax.set_xlabel("% of group carrying the flag")
    ax.set_title("Which false positives fool it?", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(axis="x", alpha=0.3)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="stage_a_transit_model/runs/bn_aug_sched")
    p.add_argument("--out", type=Path, default=EDA / "kepler_cnn_lstm_results.png")
    args = p.parse_args(argv)
    run_dir = resolve_run(args.run)

    sweep = pd.read_csv(RUNS / "sweep_comparison.csv")
    baselines = json.loads((RUNS / "baselines" / "baselines.json").read_text(encoding="utf-8"))["results"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panel_ladder(axes[0, 0], sweep, baselines)
    panel_pr(axes[0, 1], run_dir)
    panel_history(axes[1, 0], run_dir)
    panel_failure(axes[1, 1], run_dir)
    fig.suptitle(f"Kepler transit vetting, Track A Level 2 - validation set (1414 KOIs, star-disjoint) "
                 f"| chosen run: {run_dir.name}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out.relative_to(RESEARCH_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
