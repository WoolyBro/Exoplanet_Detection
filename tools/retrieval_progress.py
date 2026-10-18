"""
Progress of the TRAPPIST-1 e retrievals (stage_b_atmospheres/retrieval_trappist1e.py).

    python tools/retrieval_progress.py          # one snapshot
    python tools/retrieval_progress.py --watch  # refresh every 30 s (Ctrl+C to stop)

Time left = remaining retrievals x the average time of the finished ones (or an estimate before any
finishes). Within one retrieval, dynesty's remaining "delta ln Z" falls towards the stopping value 0.1;
its log is shown as a rough progress bar.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path

R = Path(__file__).resolve().parents[1]
SETS = {"clean (no stellar contamination)": R / "outputs" / "retrieval",
        "with stellar contamination (linear, marginalised)": R / "outputs" / "retrieval_tls",
        "REAL TRAPPIST-1 e data (4 JWST visits)": R / "outputs" / "retrieval_real",
        "REAL e, only >= 3 um": R / "outputs" / "retrieval_real_wl3",
        "REAL e, quiet visits 1+2 only": R / "outputs" / "retrieval_real_v12",
        "null test b (airless)": R / "outputs" / "retrieval_real_trappist1b",
        "null test b, only >= 3 um": R / "outputs" / "retrieval_real_trappist1b_wl3",
        "null test c": R / "outputs" / "retrieval_real_trappist1c",
        "null test d": R / "outputs" / "retrieval_real_trappist1d"}
FIRST_GUESS_MIN = 25.0


def snapshot() -> str:
    return "\n\n".join(f"[{name}]\n" + one(d) for name, d in SETS.items())


def one(d: Path) -> str:
    PROG, RES = d / "progress.json", d / "results.json"
    if not PROG.is_file():
        return "  not started (starts automatically after the clean set)" if "tls" in d.name else "  not started"
    p = json.loads(PROG.read_text())
    runs = json.loads(RES.read_text())["runs"] if RES.is_file() else []
    total, done = p.get("total", 0), len(runs)
    lines = [f"  {datetime.now():%H:%M:%S}  finished {done}/{total}"]
    for r in runs[-4:]:
        tag = (f"N={r['n_transits']} {r['model']}" + (f" noise#{r['realisation']}" if r.get("realisation") is not None else "")
               if "n_transits" in r else r["model"])
        lines.append(f"    [x] {tag:28s} lnZ {r['logz']:8.2f}  ({r['seconds'] / 60:.0f} min)")
    avg = (sum(r["seconds"] for r in runs) / done / 60) if done else FIRST_GUESS_MIN
    cur_left = 0.0
    if p.get("current") and p["current"] != "done":
        dz = p.get("current_dlogz")
        frac = 0.0 if dz is None else min(max(1 - math.log10(max(dz, 0.1) / 0.1) / 4, 0), 1)
        spent = p.get("current_seconds", 0) / 60
        bar = "#" * int(frac * 20) + "." * (20 - int(frac * 20))
        lines.append(f"    [>] {p['current']:28s} [{bar}] dlogz {dz}  {spent:.0f} min, {p.get('current_ncall', 0)} calls")
        cur_left = max(avg - spent, 1)
    remaining = max(total - done - (1 if cur_left else 0), 0)
    left = cur_left + remaining * avg
    if p.get("current") == "done":
        lines.append("  complete")
    else:
        lines.append(f"  time left ~{left:.0f} min -> around {datetime.now() + timedelta(minutes=left):%H:%M} "
                     f"(avg {avg:.0f} min per retrieval)")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    a = ap.parse_args()
    while True:
        s = snapshot()
        print(("\033[2J\033[H" + s) if a.watch else s, flush=True)
        if not a.watch:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
