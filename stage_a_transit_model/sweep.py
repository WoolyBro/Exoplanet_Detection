"""
sweep.py
========

Unattended configuration sweep for the CNN+LSTM transit classifier.

Every configuration is scored on the OFFICIAL VALIDATION views only. The test views
are never touched here - cnn_lstm.py needs --eval-test for that, and this driver
never passes it. Model selection happens once, at the end, by reading the val metric
off each run's report.json; only the winner is then run against test, by hand.

Runs are sequential on purpose: each torch process holds ~500 MB, and this machine
has under 1.5 GB free. Running them concurrently is what caused the build OOM.

Resumable: a configuration whose run directory already contains report.json is
skipped, so the sweep can be interrupted and restarted without losing work.

Usage
    python stage_a_transit_model/sweep.py              # run every pending configuration
    python stage_a_transit_model/sweep.py --table      # just print the comparison table
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

RESEARCH_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = RESEARCH_DIR / "stage_a_transit_model"
RUNS = MODEL_DIR / "runs"
LOGS = MODEL_DIR / "logs"

# Each entry: (name, extra CLI flags). The baseline is listed so the table is complete;
# it is skipped automatically if its run directory already holds a report.
# Wave 1 isolates the three ingredients one at a time (batchnorm, LR schedule,
# augmentation) so the table shows what each is worth on its own before anything is
# stacked. Wave 2 is chosen from wave 1's results rather than guessed in advance.
WAVE1: list[tuple[str, list[str]]] = [
    ("base_01",      []),
    ("bn",           ["--batchnorm"]),
    ("bn_sched",     ["--batchnorm", "--scheduler"]),
    ("aug",          ["--augment"]),
    ("bn_aug_sched", ["--batchnorm", "--augment", "--scheduler"]),
]

# Wave 2, chosen from wave 1: batchnorm was worth +0.015 ROC-AUC on its own, augmentation
# was worth -0.002 on its own but +0.008 on top of batchnorm, and the schedule did nothing
# alone. So bn+aug+sched is the base to tune, and what is left to vary is capacity and
# regularisation strength.
BN_AUG = ["--batchnorm", "--augment", "--scheduler"]
WAVE2: list[tuple[str, list[str]]] = [
    ("w15",      BN_AUG + ["--width", "1.5"]),
    ("w20",      BN_AUG + ["--width", "2.0"]),
    ("drop20",   BN_AUG + ["--dropout", "0.2"]),
    ("drop45",   BN_AUG + ["--dropout", "0.45"]),
    ("lr3e3",    BN_AUG + ["--lr", "0.003"]),
    ("w15_pr",   BN_AUG + ["--width", "1.5", "--monitor", "pr"]),
]

# Wave 3: seed repeats, not new hyperparameters. Wave 2's top seven configurations span only
# 0.006 ROC-AUC, which is meaningless unless run-to-run scatter is smaller than that. These
# repeat the two best settings across four seeds so the spread can be measured instead of
# assumed, and the "winner" is only called a winner if it clears that spread.
WAVE3: list[tuple[str, list[str]]] = [
    (f"{name}_s{seed}", flags + ["--seed", str(seed)])
    for name, flags in [("bn_aug_sched", BN_AUG), ("drop45", BN_AUG + ["--dropout", "0.45"])]
    for seed in (1, 2, 7)
]

# Wave 4: the architecture ablation the guide sequences (5.1 plain CNN on the global view ->
# 6.1 add the local branch -> 6.2 add the LSTM). Everything else is held at the winning
# configuration so ONLY the architecture varies, and each is run on two seeds because the
# run-to-run spread is +/-0.0013 and these differences may be of that order.
WAVE4: list[tuple[str, list[str]]] = [
    (f"arch_{a}{'' if seed == 42 else f'_s{seed}'}",
     BN_AUG + ["--arch", a] + ([] if seed == 42 else ["--seed", str(seed)]))
    for a in ("global_cnn", "dual_cnn", "dual_cnn_lstm")
    for seed in (42, 1)
]

CONFIGS: list[tuple[str, list[str]]] = WAVE1 + WAVE2 + WAVE3 + WAVE4

COMMON = ["--threads", "8", "--epochs", "60", "--patience", "10"]


def run_one(name: str, flags: list[str], python: str) -> bool:
    """Run one configuration; True if it produced a report."""
    run_dir = RUNS / name
    if (run_dir / "report.json").is_file():
        print(f"  [skip] {name}: report.json already present")
        return True
    LOGS.mkdir(parents=True, exist_ok=True)
    cmd = [python, "stage_a_transit_model/cnn_lstm.py", "--out", str(run_dir.relative_to(RESEARCH_DIR))] + COMMON + flags
    print(f"  [run ] {name}: {' '.join(flags) or '(defaults)'}", flush=True)
    started = time.time()
    with (LOGS / f"{name}.log").open("w", encoding="utf-8") as out, \
         (LOGS / f"{name}.err").open("w", encoding="utf-8") as err:
        rc = subprocess.call(cmd, cwd=RESEARCH_DIR, stdout=out, stderr=err)
    took = (time.time() - started) / 60
    ok = rc == 0 and (run_dir / "report.json").is_file()
    print(f"  [{'done' if ok else 'FAIL'}] {name} in {took:.1f} min (exit {rc})", flush=True)
    if not ok:
        tail = (LOGS / f"{name}.err").read_text(encoding="utf-8", errors="replace").strip().splitlines()[-6:]
        for line in tail:
            print(f"        {line}")
    return ok


def collect() -> pd.DataFrame:
    """Read every finished run's val metrics into one comparison table."""
    rows = []
    for name, flags in CONFIGS:
        path = RUNS / name / "report.json"
        if not path.is_file():
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        v = r.get("val_metrics") or {}
        par = {x["target_recall"]: x for x in v.get("precision_at_recall", []) if x.get("achievable")}
        rows.append({
            "config": name,
            "flags": " ".join(flags) or "(defaults)",
            "val_roc_auc": v.get("roc_auc_binary"),
            "val_pr_auc": v.get("pr_auc_binary"),
            "val_acc3": v.get("accuracy_3class"),
            "P@R90": par.get(0.90, {}).get("precision"),
            "P@R95": par.get(0.95, {}).get("precision"),
            "P@R99": par.get(0.99, {}).get("precision"),
            "best_epoch": (r.get("training") or {}).get("best_epoch"),
            "n_params": (r.get("training") or {}).get("n_params"),
            "test_evaluated": r.get("test_metrics") is not None,
        })
    df = pd.DataFrame(rows)
    return df.sort_values("val_roc_auc", ascending=False).reset_index(drop=True) if len(df) else df


def print_table(df: pd.DataFrame) -> None:
    if df.empty:
        print("no finished runs yet")
        return
    print(f"\n{'config':<16} {'ROC-AUC':>8} {'PR-AUC':>8} {'acc3':>7} {'P@R90':>7} {'P@R95':>7} "
          f"{'P@R99':>7} {'epoch':>6} {'params':>8}")
    print("-" * 88)
    for _, r in df.iterrows():
        def f(x, nd=4):
            return f"{x:.{nd}f}" if isinstance(x, (int, float)) and pd.notna(x) else "   n/a"
        print(f"{r.config:<16} {f(r.val_roc_auc):>8} {f(r.val_pr_auc):>8} {f(r.val_acc3,3):>7} "
              f"{f(r['P@R90'],3):>7} {f(r['P@R95'],3):>7} {f(r['P@R99'],3):>7} "
              f"{str(r.best_epoch):>6} {r.n_params if pd.notna(r.n_params) else 0:>8,}")
    leaked = df[df.test_evaluated]
    if len(leaked):
        print(f"\nNOTE: {len(leaked)} run(s) already evaluated test: {list(leaked.config)}. "
              f"Selection below is still made on val only.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", action="store_true", help="print the comparison table and exit")
    p.add_argument("--python", default=sys.executable)
    args = p.parse_args(argv)

    if not args.table:
        print("=" * 88)
        print(f"CNN+LSTM sweep | {len(CONFIGS)} configurations | started {datetime.now():%Y-%m-%d %H:%M:%S}")
        print("Scored on VALIDATION only - the test views are not touched by this driver.")
        print("=" * 88)
        for name, flags in CONFIGS:
            run_one(name, flags, args.python)

    df = collect()
    print_table(df)
    if not df.empty:
        out = RUNS / "sweep_comparison.csv"
        df.to_csv(out, index=False)
        best = df.iloc[0]
        # best["flags"], not best.flags: pandas reserves .flags on a Series.
        print(f"\nbest on val: {best['config']} ({best['flags']}) -> ROC-AUC {best['val_roc_auc']:.4f}")
        print("Run that one configuration once with --eval-test to get the reportable test number.")
        print(f"saved {out.relative_to(RESEARCH_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
