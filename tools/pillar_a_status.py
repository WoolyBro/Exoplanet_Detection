"""
pillar_a_status.py
==================

One-screen progress of Pillar A (vetter v3, eda/VETTER_V3_PROTOCOL.md) and of the S/N-balance test
(eda/SNR_BALANCE_PROTOCOL.md): view builds, training runs, and the one-time evaluation.

    .venv/bin/python tools/pillar_a_status.py
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

R = Path(__file__).resolve().parents[1]
RUNS = R / "stage_a_transit_model" / "runs"
BUILDS = {"NTP training false alarms": "v3_ntp", "fresh INJ1 (planets)": "v3_eval_inj1", "fresh INV (false alarms)": "v3_eval_inv",
          "fresh INJ2 (off-target)": "v3_eval_inj2", "fresh INJ3 (EB mimics)": "v3_eval_inj3"}
SEEDS = ["", "_s1", "_s2", "_s7", "_s11"]


def running(pattern: str) -> bool:
    return subprocess.run(["pgrep", "-f", "--", pattern], capture_output=True).returncode == 0


def build_line(name: str, folder: str) -> str:
    d = R / "detection_views_err" / folder
    prog = d / "progress.jsonl"
    if not prog.is_file():
        return f"  {name:28s} not started"
    recs = [json.loads(x) for x in prog.read_text().splitlines() if x.strip()]
    ok_stars = {r["kic"] for r in recs if all(x["status"] == "ok" for x in r["records"])}
    views = len(list(d.glob("*.npz")))
    last = (d / "build_progress.txt").read_text().strip().splitlines()[-1] if (d / "build_progress.txt").is_file() else ""
    m = re.search(r"stars (\d+)/(\d+).*ETA ([\d.]+) h", last)
    state = "running" if running(f"out {folder}") else "stopped"
    eta = f", {m.group(1)}/{m.group(2)} of this pass, ETA {float(m.group(3)) * 60:.0f} min" if m and state == "running" else ""
    return f"  {name:28s} {state}: {views} views from {len(ok_stars)} complete stars{eta}"


def train_line(cfg: str, label: str) -> str:
    done = [s for s in SEEDS if (RUNS / f"v3_{cfg}{s}" / "report.json").is_file()]
    aucs = [json.loads((RUNS / f"v3_{cfg}{s}" / "report.json").read_text())["val_roc_auc"] for s in done]
    live = sum(1 for s in ["42", "1", "2", "7", "11"] if running(f"config {cfg} --seed {s}( |$)"))
    a = f", val ROC-AUC {min(aucs):.4f}-{max(aucs):.4f}" if aucs else ""
    return f"  config {cfg} ({label}): {len(done)}/5 seeds done, {live} training now{a}"


def main() -> int:
    print("Pillar A: vetter v3 (NTP false alarms + Robovetter-measurement branch)\n")
    print("1. Light curves -> views")
    for k, v in BUILDS.items():
        print(build_line(k, v))
    print("\n2. Training (5 seeds each; validation only)")
    for c, lab in [("B", "diagnostics only"), ("A", "NTP only"), ("C", "NTP + diagnostics, primary"), ("N", "S/N-balanced v2 recipe")]:
        print(train_line(c, lab))
    print("\n3. One-time evaluations")
    for name, guard, out in [("v3 (T1 fresh sims, T2 NTP test, T3 Kepler test)", RUNS / "v3_final" / "TEST_READ.json", R / "stage_a_transit_model" / "runs" / "v3_final" / "v3_eval.json"),
                             ("S/N balance (K2 second read)", RUNS / "snr_balance" / "K2_READ.json", R / "outputs" / "stage_a_mechanisms" / "snr_balance.json")]:
        if out.is_file():
            exp = json.loads(out.read_text()).get("expectations", {})
            held = sum(bool(v) for v in exp.values())
            print(f"  {name}: DONE, {held}/{len(exp)} expectations held")
        else:
            print(f"  {name}: {'READ ALREADY (guard present)' if guard.is_file() else 'not yet run'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
