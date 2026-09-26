"""
status.py
=========

One command that answers "what is running and how far along is it?" for every build,
Kepler or TESS, without needing to know which log file is current.

    python tools/status.py

Reads only progress files and process lists - it never touches a build, so it is safe to
run at any time, as often as you like.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

RESEARCH_DIR = Path(__file__).resolve().parents[1]
VIEWS = RESEARCH_DIR / "detection_views"

SPLITS = {"Kepler": "koi_cumulative_split.csv", "TESS": "toi_tess_candidates_split.csv"}
STAR_COL = {"Kepler": "kepid", "TESS": "tid"}


def running_builds() -> list[tuple[int, str]]:
    """PIDs of live build processes, matched on the command line (not just 'python')."""
    out = []
    try:
        import psutil
    except ImportError:
        return out
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(proc.info.get("cmdline") or [])
        except Exception:
            continue
        if "preprocessing_pipeline.py" in cmd and "build" in cmd:
            mission = "TESS" if "--mission TESS" in cmd else "Kepler"
            split = next((s for s in ("train", "val", "test") if f" {s}" in cmd), "?")
            out.append((proc.info["pid"], f"{mission} {split}"))
    return out


def target_stars(mission: str, split: str) -> int | None:
    """How many distinct stars this split should end up with."""
    path = RESEARCH_DIR / "splits" / SPLITS[mission]
    if not path.is_file():
        return None
    try:
        import pandas as pd

        df = pd.read_csv(path, usecols=["split", STAR_COL[mission]])
        return int(df.loc[df.split == split, STAR_COL[mission]].nunique())
    except Exception:
        return None


def rate_and_eta(progress: Path, remaining: int) -> tuple[float, float] | None:
    """Stars per hour over the current session, and hours left, from the checkpoint timestamps.

    Measured from progress.jsonl rather than parsed out of the log line, so it stays right
    after a resume (the log's own ETA counts only that session's queue).
    """
    stamps = []
    for line in progress.open(encoding="utf-8"):
        try:
            rec = json.loads(line)
        except Exception:
            continue
        t = rec.get("time") or rec.get("timestamp")
        if t and rec.get("status"):
            stamps.append(str(t))
    if len(stamps) < 25:
        return None
    recent = sorted(stamps)[-300:]
    times = []
    for s in recent:
        try:
            times.append(datetime.fromisoformat(s.replace(" ", "T")))
        except ValueError:
            continue
    if len(times) < 25:
        return None

    # Median gap between consecutive checkpoints, NOT total span / count. The machine is
    # switched off, paused by the memory guard and stopped between sessions, and a span-based
    # rate counts all of that as working time: after a 3 h shutdown it reported 75 stars/h
    # against the build's own 280. Gaps longer than IDLE_GAP_S are dropped as "not running",
    # and the median ignores whatever stragglers remain.
    IDLE_GAP_S = 600
    gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
    working = sorted(g for g in gaps if 0 < g <= IDLE_GAP_S)
    if not working:
        return None
    median_gap = working[len(working) // 2]
    per_hour = 3600 / median_gap
    return per_hour, (remaining / per_hour if per_hour else float("nan"))


def summarise(views_dir: Path) -> None:
    name = views_dir.name
    mission = "TESS" if name.startswith("tess") else "Kepler"
    # "kepler_train" -> "train"; a name that is not <mission>_<split> keeps its own label.
    split = name.split("_", 1)[1] if name.split("_", 1)[0] in ("kepler", "tess") else name
    progress = views_dir / "progress.jsonl"
    npz = len(list(views_dir.glob("*.npz")))

    # progress.jsonl is append-only, so a KOI retried after a failure appears twice. Only the
    # LAST record for each object is its current state: counting every line would report
    # failures that have since been fixed (Kepler test really has 0, not 205).
    latest: dict[str, dict] = {}
    if progress.is_file():
        for line in progress.open(encoding="utf-8"):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("status") is not None and rec.get("object_id"):
                latest[rec["object_id"]] = rec

    counts: dict[str, int] = {}
    stars: set = set()
    for rec in latest.values():
        status = rec["status"]
        key = status if status == "ok" else f"{status}: {rec.get('category', '')}".strip(": ")
        counts[key] = counts.get(key, 0) + 1
        if rec.get("star_id") is not None:
            stars.add(rec["star_id"])

    last_line = None

    log = views_dir / "build_progress.txt"
    if log.is_file():
        lines = [l for l in log.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
        last_line = lines[-1] if lines else None

    total = target_stars(mission, split)
    done = len(stars)
    pct = f"{100 * done / total:.1f}%" if total else "?"
    header = f"{mission} {split}" if split in ("train", "val", "test") else split
    print(f"\n  {header}  ({views_dir.relative_to(RESEARCH_DIR)})")

    bar = ""
    if total:
        filled = int(round(30 * done / total))
        bar = "  [" + "#" * filled + "-" * (30 - filled) + "]"
    print(f"    progress      : {done}" + (f" / {total} stars  ({pct}){bar}" if total else " stars"))
    print(f"    view files    : {npz} .npz")
    if counts:
        parts = [f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]
        print(f"    outcomes      : {', '.join(parts)}")

    if total and done < total and progress.is_file():
        est = rate_and_eta(progress, total - done)
        if est:
            per_hour, hours_left = est
            finish = datetime.now() + timedelta(hours=hours_left)
            print(f"    rate          : {per_hour:.0f} stars/hour (recent)")
            print(f"    REMAINING     : {total - done} stars  ->  about "
                  f"{hours_left:.1f} h, finishing around {finish:%a %d %b %H:%M}")
        else:
            print("    remaining     : not enough recent checkpoints to estimate a rate yet")
    elif total and done >= total:
        print("    REMAINING     : none - this split is complete")

    if last_line:
        print(f"    last log line : {last_line.strip()[:150]}")


def main() -> int:
    print("=" * 84)
    print(f"Build status  |  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 84)

    live = running_builds()
    if live:
        for pid, what in live:
            print(f"  RUNNING: {what} build, PID {pid}")
    else:
        print("  nothing running (no preprocessing_pipeline build process found)")

    dirs = sorted(d for d in VIEWS.glob("*") if d.is_dir() and not d.name.startswith("_"))
    if not dirs:
        print("\n  no view folders yet")
        return 0
    for d in dirs:
        summarise(d)

    print("\n  resume any stopped build by re-running its own command; all builds checkpoint")
    print("  per star in progress.jsonl, so nothing already finished is repeated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
