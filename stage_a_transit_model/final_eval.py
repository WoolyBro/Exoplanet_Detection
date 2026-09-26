"""
final_eval.py
=============

The single, final evaluation of one chosen checkpoint on the held-out TEST views.

Why this exists rather than `cnn_lstm.py --eval-test`: that flag retrains the model
before scoring. This loads the exact best.pt that model selection picked, so the number
reported is the number produced by the selected artefact, with no retraining in between.

Protocol
  * Selection happened on VALIDATION only, across the sweep (see runs/sweep_comparison.csv).
  * Test is scored ONCE, here. Re-running this file after changing anything about the
    model would make the test number a tuning signal, which is exactly what it must not be.
  * The val number is also recomputed, and both are printed together, because the honest
    comparison is val-vs-test generalisation, not test alone.
  * Selecting the best of N runs on val is itself optimistic; the seed spread from the
    sweep is reported alongside so the reader can judge the size of that bias.

Usage
    python stage_a_transit_model/final_eval.py --run stage_a_transit_model/runs/bn_aug_sched
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
    CLASSES, PLANET_LIKE_IDX, VIEWS_ROOT, consolidate_views, drop_unusable,
    evaluate, planet_like_score, threshold_sweep,
)
from stage_a_transit_model.failure_analysis import load_run_model, score_views  # noqa: E402

RUNS = RESEARCH_DIR / "stage_a_transit_model" / "runs"


def seed_spread(config: str) -> tuple[float, float, int]:
    """Mean and std of val ROC-AUC across the seed repeats of this configuration."""
    path = RUNS / "sweep_comparison.csv"
    if not path.is_file():
        return float("nan"), float("nan"), 0
    df = pd.read_csv(path)
    # Strip only a trailing seed suffix ("_s1"), not any "_s" inside the name - a plain
    # rsplit("_s") cuts "bn_aug_sched" at "_sched" and silently drops the seed-42 parent.
    base_name = df.config.str.replace(r"_s\d+$", "", regex=True)
    vals = df.loc[base_name == config, "val_roc_auc"].dropna().values
    return (float(vals.mean()), float(vals.std()), len(vals)) if len(vals) else (float("nan"), float("nan"), 0)


def report_block(name: str, m: dict) -> None:
    print(f"\n  --- {name} ---")
    print(f"  n = {m['n']}   class counts {m['class_counts']}")
    roc = "n/a" if m["roc_auc_binary"] is None else f"{m['roc_auc_binary']:.4f}"
    pr = "n/a" if m["pr_auc_binary"] is None else f"{m['pr_auc_binary']:.4f}"
    print(f"  3-class accuracy {m['accuracy_3class']:.4f} | binary ROC-AUC {roc} | PR-AUC {pr}")
    print(m["report_3class"])
    print(f"  confusion (rows true, cols pred; {', '.join(CLASSES)}):")
    for row, cls in zip(m["confusion_3class"], CLASSES):
        print(f"    {cls:<15} {row}")
    print("  precision at fixed recall:")
    for r in m["precision_at_recall"]:
        if r.get("achievable"):
            print(f"    recall >= {r['target_recall']:.0%}: precision {r['precision']:.4f} "
                  f"at threshold {r['threshold']:.6f} (actual recall {r['recall']:.4f})")
        else:
            print(f"    recall >= {r['target_recall']:.0%}: not achievable")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="stage_a_transit_model/runs/bn_aug_sched")
    p.add_argument("--val-views", type=Path, default=VIEWS_ROOT / "kepler_val")
    p.add_argument("--test-views", type=Path, default=VIEWS_ROOT / "kepler_test")
    args = p.parse_args(argv)

    run_dir = Path(args.run).resolve()
    if not (run_dir / "best.pt").is_file() and not Path(args.run).is_absolute():
        run_dir = (RESEARCH_DIR / args.run).resolve()
    if not (run_dir / "best.pt").is_file():
        raise SystemExit(f"no best.pt in {run_dir}")

    out_path = run_dir / "final_eval.json"
    print("=" * 88)
    print(f"FINAL EVALUATION | checkpoint {run_dir.name}")
    print("Selection was made on validation only. Test is scored once, here.")
    if out_path.is_file():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        print(f"NOTE: this checkpoint was already finally evaluated at {prev.get('generated_at')}. "
              f"Re-running does not re-tune anything, but the first number is the reportable one.")
    print("=" * 88)

    model, arch = load_run_model(run_dir)
    print(f"\narchitecture: {arch}")
    mean, std, n = seed_spread(run_dir.name)
    if n:
        print(f"seed spread for this configuration: val ROC-AUC {mean:.4f} +/- {std:.4f} over {n} run(s)")
        print("The single val number below is the best of those; the gap to it is selection bias.")

    results = {}
    for split, views in [("validation", args.val_views), ("test", args.test_views)]:
        vs = drop_unusable(consolidate_views(views, verbose=False))
        proba = score_views(model, vs)
        y = vs.label_index.astype(int)
        m = evaluate(y, proba)
        results[split] = m
        report_block(f"{split.upper()}  ({views.name}, {len(vs)} rows, "
                     f"{len(set(vs.star_id.tolist()))} stars)", m)
        if split == "test":
            threshold_sweep(y, proba).to_csv(run_dir / "test_threshold_sweep.csv", index=False)
            pd.DataFrame({"kepoi_name": vs.object_id, "star_id": vs.star_id,
                          "true_label": [CLASSES[i] for i in y],
                          "planet_like_score": planet_like_score(proba),
                          "pred_index": proba.argmax(axis=1)}).to_csv(
                run_dir / "test_predictions.csv", index=False)

    v, t = results["validation"], results["test"]
    print("\n" + "=" * 88)
    print("GENERALISATION (val -> test)")
    print("=" * 88)
    for key, label in [("roc_auc_binary", "binary ROC-AUC"), ("pr_auc_binary", "binary PR-AUC"),
                       ("accuracy_3class", "3-class accuracy")]:
        a, b = v.get(key), t.get(key)
        if a is not None and b is not None:
            print(f"  {label:<18} val {a:.4f}  ->  test {b:.4f}   ({b - a:+.4f})")
    if n and not np.isnan(std):
        drop = v["roc_auc_binary"] - t["roc_auc_binary"]
        print(f"\n  The val->test drop of {drop:+.4f} is {abs(drop) / std:.1f}x the seed spread "
              f"({std:.4f}).")
        print("  A drop of order the seed spread is consistent with selection bias alone;")
        print("  a much larger drop would indicate the model does not transfer.")

    payload = {"run": run_dir.name, "generated_at": datetime.now().isoformat(timespec="seconds"),
               "architecture": arch, "seed_spread": {"mean": mean, "std": std, "n": n},
               "validation": v, "test": t}
    out_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"\nsaved {out_path.relative_to(RESEARCH_DIR)}, test_predictions.csv, test_threshold_sweep.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
