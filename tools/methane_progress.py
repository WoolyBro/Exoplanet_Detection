"""
Progress of the methane-classifier extension (more JWST planets + calibration).

    python tools/methane_progress.py          # one snapshot
    python tools/methane_progress.py --watch  # refresh every 30 s (Ctrl+C to stop)

Reads outputs/stage_b_methane_ext/progress.json (phases, written as the work advances) and
outputs/labels/ch4_labels_extended.csv (one row per planet labelled so far). Time left comes from
the measured labelling rate plus fixed estimates for the remaining phases.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1]
PROG = R / "outputs" / "stage_b_methane_ext" / "progress.json"
LABELS = R / "outputs" / "labels" / "ch4_labels_extended.csv"


def ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def snapshot() -> str:
    if not PROG.is_file():
        return "not started"
    p = json.loads(PROG.read_text())
    now = datetime.now()
    lines = [f"Methane extension — {now:%H:%M:%S}", ""]
    left_min = 0.0
    for ph in p["phases"]:
        st = ph.get("status", "pending")
        mark = {"done": "[x]", "running": "[>]", "pending": "[ ]"}.get(st, "[?]")
        extra = ""
        if "grid" in ph["name"] and st == "running":
            parts = R / "outputs" / "stage_b_synthetic_seed2" / "parts"
            n = len(list(parts.glob("*.pkl"))) if parts.is_dir() else 0
            spent = (now - ts(ph["started"])).total_seconds() / 60
            if n > 30:
                remaining = spent / n * (3000 - n)
                extra = f"  {n}/3000 spectra (~{remaining:.0f} min left)"
            else:
                remaining = max(ph["estimate_min"] - spent, 1)
                extra = f"  {n}/3000 spectra (starting up)"
            left_min = max(left_min, remaining)          # runs in parallel with the other phases
            lines.append(f"  {mark} {ph['name']}{extra}")
            continue
        if ph["name"] == "label planets":
            labels = R / p.get("labels_file", str(LABELS.relative_to(R)))
            n_done = max(len(pd.read_csv(labels)), 0) if labels.is_file() else 0
            total = p["planets_total"]
            extra = f"  {n_done}/{total} planets"
            if st == "running" and n_done > 0:
                rate = (now - ts(ph["started"])).total_seconds() / 60 / n_done
                remaining = rate * (total - n_done)
                extra += f"  (~{rate:.1f} min/planet, ~{remaining:.0f} min left)"
                left_min += remaining
            elif st != "done":
                left_min += ph["estimate_min"]
        elif st == "running":
            spent = (now - ts(ph["started"])).total_seconds() / 60
            left_min += max(ph["estimate_min"] - spent, 1)
            extra = f"  running {spent:.0f} min (estimate {ph['estimate_min']} min)"
        elif st == "pending":
            left_min += ph["estimate_min"]
            extra = f"  estimate {ph['estimate_min']} min"
        if st == "done" and ph.get("finished"):
            extra = f"  finished {ts(ph['finished']):%H:%M}" + extra
        lines.append(f"  {mark} {ph['name']}{extra}")
    lines += ["", f"Time left (estimate): ~{left_min:.0f} min"
              + (f"  -> around {datetime.fromtimestamp(time.time() + left_min * 60):%H:%M}" if left_min else "")]
    if p.get("note"):
        lines += [f"Note: {p['note']}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    a = ap.parse_args()
    while True:
        out = snapshot()
        if a.watch:
            print("\033[2J\033[H" + out, flush=True)
            time.sleep(30)
        else:
            print(out)
            break


if __name__ == "__main__":
    main()
