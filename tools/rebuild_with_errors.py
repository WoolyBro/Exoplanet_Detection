#!/usr/bin/env python3
"""
rebuild_with_errors.py
======================

Rebuilds every Kepler and TESS split with the per-bin flux-error channel, unattended, into
detection_views_err/<mission>_<split>. The original detection_views/ - the views behind every
reported result - is never written to.

The depth channel of a rebuilt view is the same as before (checked on Kepler and TESS val stars:
max difference 3e-8, float32 rounding between platforms); what is new is `global_err` /
`local_err`. That is what makes the later comparison clean: a model trained with --use-errors
differs from the existing one only in the error channel.

Same design as run_tess_remaining.py: builds stop early BY DESIGN (worker pool died, network down
longer than the guard waits, RAM), every stop is resumable from progress.jsonl, so each split is
re-run until its log ends with a clean "build end", up to --max-attempts. Order: val and train
for each mission first - that is all the validation-only comparison needs - and the test splits
last, so the experiment can start while they finish.

Safe to start while another build runs: it waits rather than launching a second one.

Keep the machine awake. On macOS each build runs under `caffeinate -i -s`, which stops idle and
system sleep while it runs - but nothing in software stops CLAMSHELL sleep: closing the lid on
battery suspends everything (seen 2026-10-13: a test build froze for 31 min at 'Clamshell Sleep'
and resumed on wake). Run it plugged in, lid open. A sleep only pauses the build; it never loses
work, because every finished star is already in progress.jsonl.

Usage
    python tools/rebuild_with_errors.py                       # everything, Kepler then TESS
    python tools/rebuild_with_errors.py --missions TESS --splits val
    python tools/rebuild_with_errors.py --dry-run
Progress: python tools/status.py, or tail rebuild_errors_chain.log
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]
VIEWS_ROOT = "detection_views_err"
LOG = RESEARCH / "rebuild_errors_chain.log"

# --workers 4: on the 16 GB / 10-core machine this was written on, 4 workers ran at ~16 s per
# star each (~900 stars/h) with the build's own RAM guard as the backstop.
BUILD_ARGS = ["--workers", "4", "--discard-raw", "--min-free-ram", "1.0", "--stall-timeout", "15",
              "--s3", "--views-root", VIEWS_ROOT]


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def build_running() -> int | None:
    """PID of a running preprocessing_pipeline build, if any."""
    try:
        import psutil
    except ImportError:
        return None
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(proc.info.get("cmdline") or [])
        except Exception:
            continue
        if "preprocessing_pipeline.py" in cmd and "build-split" in cmd:
            return proc.info["pid"]
    return None


def transient_failures(mission: str, split: str) -> int:
    """KOIs whose LATEST record is a network fetch failure. These are worth retrying (a re-run
    re-queues them); "no light curve found" and bad ephemerides are permanent and are not."""
    import json

    progress = RESEARCH / VIEWS_ROOT / f"{mission.lower()}_{split}" / "progress.jsonl"
    latest = {}
    if progress.is_file():
        for line in progress.open(encoding="utf-8"):
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("object_id"):
                latest[rec["object_id"]] = rec
    return sum(1 for r in latest.values() if r.get("status") == "failed" and r.get("category") == "api fetch failure")


def split_complete(mission: str, split: str) -> bool:
    """True if the split's own log ends with a clean 'build end' AND no KOI is left on a transient
    network failure. A brief outage during a long build can fail a hundred stars (101 Kepler train
    KOIs on 2026-10-13, all MAST ConnectionError) without stopping it; without this check those
    views would simply be missing from the rebuilt set."""
    log_file = RESEARCH / VIEWS_ROOT / f"{mission.lower()}_{split}" / "build_progress.txt"
    if not log_file.is_file():
        return False
    lines = [l for l in log_file.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    for line in reversed(lines[-40:]):
        if "build end" in line:
            return "stopped early" not in line and transient_failures(mission, split) == 0
        if "build start" in line:
            return False        # a newer session started and has not ended
    return False


def run_split(mission: str, split: str, attempts: int, dry_run: bool) -> bool:
    name = f"{mission.lower()}_{split}"
    for attempt in range(1, attempts + 1):
        if split_complete(mission, split):
            log(f"{name}: already complete")
            return True
        cmd = [sys.executable, "preprocessing_pipeline.py", "build-split", split, "--mission", mission] + BUILD_ARGS
        log(f"{name}: attempt {attempt}/{attempts} -> {' '.join(cmd[1:])}")
        if sys.platform == "darwin":
            cmd = ["caffeinate", "-i", "-s"] + cmd   # no idle/system sleep while this build runs
        if dry_run:
            return True
        out_path = RESEARCH / f"stdout_err_{name}_{attempt}.txt"
        with out_path.open("w", encoding="utf-8") as out, \
             (RESEARCH / f"stderr_err_{name}_{attempt}.txt").open("w", encoding="utf-8") as err:
            rc = subprocess.call(cmd, cwd=RESEARCH, stdout=out, stderr=err)
        if split_complete(mission, split):
            log(f"{name}: COMPLETE (exit {rc})")
            return True
        tail = [l for l in out_path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
        log(f"{name}: stopped early (exit {rc}); last line: {tail[-1][:160] if tail else '<none>'}")
        time.sleep(60)          # let the machine settle before resuming
    log(f"{name}: gave up after {attempts} attempts - needs a look")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--missions", nargs="+", default=["Kepler", "TESS"], choices=["Kepler", "TESS"])
    ap.add_argument("--splits", nargs="+", default=["val", "train", "test"], choices=["train", "val", "test"])
    ap.add_argument("--max-attempts", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log("=" * 76)
    log(f"error-channel rebuild: {args.missions} x {args.splits} -> {VIEWS_ROOT}/")
    pid = build_running()
    if pid is not None:
        log(f"a build is already running (PID {pid}); waiting for it to finish")
        while build_running() is not None:
            time.sleep(60)

    order = {"val": 0, "train": 1, "test": 2}
    jobs = sorted(((m, s) for m in args.missions for s in args.splits),
                  key=lambda ms: (ms[1] == "test", args.missions.index(ms[0]), order[ms[1]]))
    ok = True
    for mission, split in jobs:
        if not run_split(mission, split, args.max_attempts, args.dry_run):
            ok = False
    log(f"rebuild finished; all requested splits complete: {ok}")
    if ok:
        log("next: python stage_a_transit_model/sweep.py --errors   (error-channel models, val only)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
