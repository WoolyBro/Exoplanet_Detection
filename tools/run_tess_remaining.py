#!/usr/bin/env python3
"""
run_tess_remaining.py
=====================

Runs the remaining TESS splits unattended: waits for whatever build is already going, then
works through val and test in order, restarting any split that stops early.

Why the restart loop: these builds stop early by design when something goes wrong - the worker
pool dies, the network drops for longer than the guard waits, RAM runs out. Every one of those
is resumable, and re-running the same command picks up from progress.jsonl. Doing that by hand
has cost several hours across this project, so it is automated here.

A split is considered finished when its build prints a "build end" line with no "stopped early",
or when re-running it reports nothing left to process. Each split gets at most --max-attempts
tries, so a genuinely broken build cannot spin for ever.

Safe to run while a build is already going: it waits rather than starting a second one.

Usage
    python tools/run_tess_remaining.py                  # val then test
    python tools/run_tess_remaining.py --splits test    # just test
    python tools/run_tess_remaining.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]
VIEWS = RESEARCH / "detection_views"

BUILD_ARGS = ["--mission", "TESS", "--workers", "2", "--discard-raw",
              "--min-free-ram", "0.5", "--stall-timeout", "12", "--s3"]


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with (RESEARCH / "tess_chain.log").open("a", encoding="utf-8") as fh:
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


def wait_for_current(poll_s: int = 60) -> None:
    pid = build_running()
    if pid is None:
        return
    log(f"a build is already running (PID {pid}); waiting for it to finish")
    while build_running() is not None:
        time.sleep(poll_s)
    log("that build has exited")


def split_complete(split: str) -> bool:
    """True if the split's own log ends with a clean 'build end'."""
    log_file = VIEWS / f"tess_{split}" / "build_progress.txt"
    if not log_file.is_file():
        return False
    lines = [l for l in log_file.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    for line in reversed(lines[-40:]):
        if "build end" in line:
            return "stopped early" not in line
        if "build start" in line:
            return False        # a newer session started and has not ended
    return False


def run_split(split: str, attempts: int, dry_run: bool) -> bool:
    for attempt in range(1, attempts + 1):
        if split_complete(split):
            log(f"{split}: already complete")
            return True
        cmd = [sys.executable, "preprocessing_pipeline.py", "build-split", split] + BUILD_ARGS
        log(f"{split}: attempt {attempt}/{attempts} -> {' '.join(cmd[1:])}")
        if dry_run:
            return True
        out_path = RESEARCH / f"stdout_tess_{split}_chain{attempt}.txt"
        with out_path.open("w", encoding="utf-8") as out, \
             (RESEARCH / f"stderr_tess_{split}_chain{attempt}.txt").open("w", encoding="utf-8") as err:
            rc = subprocess.call(cmd, cwd=RESEARCH, stdout=out, stderr=err)
        if split_complete(split):
            log(f"{split}: COMPLETE (exit {rc})")
            return True
        tail = [l for l in out_path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
        log(f"{split}: stopped early (exit {rc}); last line: {tail[-1][:160] if tail else '<none>'}")
        time.sleep(60)          # let the machine settle before resuming
    log(f"{split}: gave up after {attempts} attempts - needs a look")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--max-attempts", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log("=" * 76)
    log(f"TESS chain starting for splits: {', '.join(args.splits)}")
    wait_for_current()

    ok = True
    for split in args.splits:
        if not run_split(split, args.max_attempts, args.dry_run):
            ok = False
            break
    log(f"chain finished; all splits complete: {ok}")
    log("next: train the TESS model (cnn_lstm.py needs a --mission option first)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
