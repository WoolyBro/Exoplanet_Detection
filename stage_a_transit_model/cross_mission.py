"""
cross_mission.py
================

Checklist item 7.2: "Cross-mission test - train on Kepler, evaluate on TESS/K2."

Scores each trained model on BOTH missions' held-out views, so transfer is measured in all
four directions rather than asserted. The views are mission-agnostic by construction (2001-bin
global + 201-bin local, depth-normalised), so a Kepler model can be applied to TESS views
without any adaptation - which is exactly what makes the comparison meaningful.

Read alongside the base rates, which differ sharply between the missions:
Kepler test is ~50% planet-like, TESS test is ~87%. A PR-AUC of 0.95 on TESS is barely above
the no-skill floor of 0.87, while 0.95 on Kepler would be a strong result. Every number here
is therefore reported against its own mission's floor, never in isolation.

Usage
    python stage_a_transit_model/cross_mission.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RESEARCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_DIR))

from stage_a_transit_model.cnn_lstm import (  # noqa: E402
    PLANET_LIKE_IDX, VIEWS_ROOT, consolidate_views, drop_unusable, evaluate, planet_like_score,
)
from stage_a_transit_model.failure_analysis import load_run_model, score_views  # noqa: E402

RUNS = RESEARCH_DIR / "stage_a_transit_model" / "runs"
MODELS = {"Kepler": RUNS / "bn_aug_sched", "TESS": RUNS / "tess_bn_aug_sched"}
TEST_VIEWS = {"Kepler": VIEWS_ROOT / "kepler_test", "TESS": VIEWS_ROOT / "tess_test"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=RUNS / "cross_mission")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("Cross-mission transfer: every model on every mission's held-out test views")
    print("=" * 88)

    # Load each test set once.
    data = {}
    for mission, path in TEST_VIEWS.items():
        if not path.is_dir() or not any(path.glob("*.npz")):
            print(f"  {mission}: no test views at {path}; skipped")
            continue
        vs = drop_unusable(consolidate_views(path, verbose=False))
        y = vs.label_index.astype(int)
        y_bin = np.isin(y, PLANET_LIKE_IDX).astype(int)
        data[mission] = (vs, y, y_bin)
        print(f"\n  {mission} test: {len(vs)} rows, {y_bin.mean():.1%} planet-like "
              f"(this is the no-skill PR-AUC floor)")

    rows = []
    for trained_on, run_dir in MODELS.items():
        if not (run_dir / "best.pt").is_file():
            print(f"\n  no model at {run_dir}; skipped")
            continue
        model, arch = load_run_model(run_dir)
        for eval_on, (vs, y, y_bin) in data.items():
            proba = score_views(model, vs)
            m = evaluate(y, proba)
            par = {r["target_recall"]: r for r in m["precision_at_recall"] if r.get("achievable")}
            rows.append({
                "trained_on": trained_on, "evaluated_on": eval_on,
                "same_mission": trained_on == eval_on,
                "n": int(len(vs)), "base_rate": float(y_bin.mean()),
                "roc_auc": m["roc_auc_binary"], "pr_auc": m["pr_auc_binary"],
                "acc3": m["accuracy_3class"],
                "p_at_r90": par.get(0.90, {}).get("precision"),
                "pr_auc_lift": (m["pr_auc_binary"] - float(y_bin.mean())
                                if m["pr_auc_binary"] is not None else None),
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "cross_mission.csv", index=False)

    print("\n" + "=" * 88)
    print("RESULTS  (PR-AUC lift = PR-AUC minus that mission's planet-like base rate)")
    print("=" * 88)
    print(f"\n  {'trained on':<11} {'evaluated on':<13} {'ROC-AUC':>8} {'PR-AUC':>8} "
          f"{'base':>6} {'lift':>7} {'acc3':>7} {'P@R90':>7}")
    print("  " + "-" * 76)
    for _, r in df.sort_values(["evaluated_on", "trained_on"]).iterrows():
        mark = "  <- same mission" if r.same_mission else ""
        print(f"  {r.trained_on:<11} {r.evaluated_on:<13} {r.roc_auc:>8.4f} {r.pr_auc:>8.4f} "
              f"{r.base_rate:>6.3f} {r.pr_auc_lift:>+7.4f} {r.acc3:>7.4f} "
              f"{r.p_at_r90 if r.p_at_r90 is not None else float('nan'):>7.3f}{mark}")

    print("\n  transfer penalty (same-mission minus cross-mission, on the same test set):")
    for eval_on in df.evaluated_on.unique():
        sub = df[df.evaluated_on == eval_on]
        same = sub[sub.same_mission]
        cross = sub[~sub.same_mission]
        if len(same) and len(cross):
            d_roc = float(same.roc_auc.iloc[0] - cross.roc_auc.iloc[0])
            print(f"    on {eval_on:<7} ROC-AUC drops {d_roc:+.4f} when the model comes from "
                  f"{cross.trained_on.iloc[0]}")

    (args.out / "cross_mission.json").write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "note": ("PR-AUC must be read against each mission's base rate: Kepler test is ~50% "
                 "planet-like, TESS test ~87%, so the same PR-AUC means very different things."),
        "results": rows,
    }, indent=2, default=float), encoding="utf-8")
    print(f"\nsaved {args.out.relative_to(RESEARCH_DIR)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
