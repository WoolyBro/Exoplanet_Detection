"""Read-only status of a Kepler detection-views build: is it running, how far, how much longer.

    python tools/build_status.py            # the most recently active build (train, val or test)
    python tools/build_status.py val        # a specific split

Reads only the checkpoint log, the split file and the process list; it never touches the build or its outputs.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

RESEARCH = Path(__file__).resolve().parent.parent
VIEWS = RESEARCH / "detection_views"
CACHE = Path("D:/lightkurve_cache")  # fallback only; the build's own session_start event says which cache it uses

def pick_split(argv: list[str]) -> str:
    """The split named on the command line, else the one whose checkpoint was written most recently."""
    if len(argv) > 1:
        return argv[1]
    logs = sorted(VIEWS.glob("kepler_*/progress.jsonl"), key=lambda p: p.stat().st_mtime)
    return logs[-1].parent.name.removeprefix("kepler_") if logs else "train"


def split_star_count(split: str) -> int:
    """Distinct Kepler stars in the split (from splits/koi_cumulative_split.csv)."""
    with (RESEARCH / "splits" / "koi_cumulative_split.csv").open(encoding="utf-8") as fh:
        return len({r["kepid"] for r in csv.DictReader(fh) if r["split"] == split})


SPLIT = pick_split(sys.argv)
OUT = VIEWS / f"kepler_{SPLIT}"
TOTAL_STARS = split_star_count(SPLIT)


def running_pids() -> list[int]:
    """PIDs of running build processes (the preprocessing_pipeline build parent plus its pool workers).

    Matched on the command line, not the process name: on Windows `python` can start through a launcher stub,
    so "any python process" would count this status script's own launcher as a running build."""
    try:
        cmd = ["powershell", "-NoProfile", "-Command",
               "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Where-Object { "
               "$_.CommandLine -match 'preprocessing_pipeline.py build' -or "
               "($_.CommandLine -match 'multiprocessing' -and $_.CommandLine -match 'spawn') } | "
               "ForEach-Object { $_.ProcessId }"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
        return [int(x) for x in out.split()]
    except Exception:
        return []


def folder_gb(path: Path) -> float | None:
    if not path.exists():
        return None
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024 ** 3


def main() -> int:
    progress = OUT / "progress.jsonl"
    if not progress.is_file():
        print(f"no checkpoint log at {progress} - the build has never run")
        return 1

    records, sessions = {}, []
    with progress.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # a half-written last line while the build is mid-write
            if r.get("event") == "session_start":
                sessions.append(r)
            else:
                records[r["object_id"]] = r  # keep the latest record per KOI

    counts = Counter(r["status"] if r["status"] == "ok" else f"{r['status']}: {r.get('category')}"
                     for r in records.values())
    stars_seen = len({r.get("star_id") for r in records.values()})
    started = sessions[-1]["time"] if sessions else None
    this_run = [r for r in records.values() if started and r.get("time", "") >= started]
    stars_this_run = len({r.get("star_id") for r in this_run})
    to_process = sessions[-1].get("stars_to_process", 0) if sessions else 0

    # Finished? The pipeline logs "build end" when it stops for any reason.
    log_text = ""
    for name in ("build_progress.txt",):
        p = OUT / name
        if p.is_file():
            log_text = p.read_text(encoding="utf-8", errors="replace")
    ended = [ln for ln in log_text.splitlines() if "build end" in ln]
    last_end = ended[-1] if ended else ""
    pids = running_pids()

    resume = f"python preprocessing_pipeline.py build-split {SPLIT} --workers 2 --discard-raw --min-free-ram 0.3"
    print("=" * 78)
    print(f"Kepler {SPLIT.upper()} build ({OUT.relative_to(RESEARCH)})")
    if pids:
        print(f"STATUS: RUNNING  ({len(pids)} python process(es): {', '.join(map(str, pids))})")
    elif last_end and started and last_end[1:20] > started.replace("T", " "):  # log uses a space, JSON a 'T'
        print("STATUS: NOT RUNNING - the last session ended:")
        print("   " + last_end.strip())
        if "stopped early" in last_end:
            print(f"   -> resume with:  {resume}")
    else:
        print("STATUS: NOT RUNNING - no build process found, and no 'build end' line for this session.")
        print(f"   It stopped without finishing. Resume with:  {resume}")
    print("=" * 78)

    print(f"\nstars with at least one saved/decided KOI: {stars_seen} / {TOTAL_STARS} "
          f"({100 * stars_seen / TOTAL_STARS:.1f}%)")
    print(f"KOI outcomes so far: {dict(counts)}")
    print(f"view files (.npz) on disk: {len(list(OUT.glob('*.npz')))}")

    progress_lines = [ln for ln in log_text.splitlines() if "| KOIs" in ln]
    if progress_lines:
        print(f"\nlast checkpoint line:\n   {progress_lines[-1].strip()}")

    if started and stars_this_run:
        t0 = datetime.fromisoformat(started)
        t1 = max(datetime.fromisoformat(r["time"]) for r in this_run)
        hours = max((t1 - t0).total_seconds() / 3600, 1e-3)
        rate = stars_this_run / hours
        left = max(to_process - stars_this_run, 0)
        eta_h = left / rate if rate else float("inf")
        print(f"\nthis session started {t0:%Y-%m-%d %H:%M}, {stars_this_run} stars in {hours:.2f} h "
              f"= {rate:.0f} stars/h")
        if pids:
            print(f"remaining: {left} stars -> about {eta_h:.1f} h "
                  f"(finishes around {datetime.now() + timedelta(hours=eta_h):%a %d %b %H:%M})")

    # The cache the build is actually using is recorded in its latest session_start event.
    cache = Path(sessions[-1].get("cache_dir", CACHE)) if sessions else CACHE
    discard = bool(sessions[-1].get("discard_raw")) if sessions else False
    gb = folder_gb(cache)
    if gb is not None:
        import shutil

        free = shutil.disk_usage(cache).free / 1024 ** 3
        print(f"\nlight-curve cache: {cache} = {gb:.2f} GB, {free:.1f} GB free on that drive")
        if discard:
            print("   raw downloads are deleted after each star (--discard-raw), so this stays small")
        else:
            per_star = gb / max(stars_seen, 1)
            need = per_star * max(TOTAL_STARS - stars_seen, 0)
            print(f"   at {per_star * 1000:.0f} MB/star, the remaining stars need roughly {need:.0f} GB")
            if need > free * 0.9:
                print("   WARNING: that is close to the free space. Resume with --discard-raw, or free some "
                      "with tools/prune_fits_cache.py --delete (you must run the delete).")

    print("\nWhen it finishes, this says NOT RUNNING with a 'build end' line listing final statuses.")
    print(f"Its report is written automatically; to redo it:  python preprocessing_pipeline.py summarize-split {SPLIT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
