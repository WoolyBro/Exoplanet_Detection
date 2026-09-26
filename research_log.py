"""
research_log.py
===============

Shared console + run_log.txt capture for the Research scripts that document each run in run_log.txt.

    from research_log import banner, run_logged

    if __name__ == "__main__":
        run_logged(main)

run_logged() prints to the console as usual and, when main() ends (normally or with an exception, whose
traceback is captured too), appends everything that was printed to run_log.txt.
"""

from __future__ import annotations

import io
import sys
import traceback
from pathlib import Path
from typing import Callable

RESEARCH_DIR = Path(__file__).resolve().parent
LOG_PATH = RESEARCH_DIR / "run_log.txt"


class Tee(io.TextIOBase):
    """Write to the console and keep a copy for run_log.txt."""

    def __init__(self, console):
        self.console, self.buffer_ = console, io.StringIO()

    def write(self, text):
        self.console.write(text)
        self.buffer_.write(text)
        return len(text)

    def flush(self):
        self.console.flush()


def banner(title: str) -> None:
    print("\n" + "=" * 88 + "\n" + title + "\n" + "=" * 88)


def run_logged(main: Callable[[], None], append_to_log: bool = True, log_path: Path = LOG_PATH) -> None:
    """Run main() with stdout teed; append the captured output (and any traceback) to log_path."""
    tee = Tee(sys.stdout)
    sys.stdout = tee
    try:
        main()
    except Exception:
        traceback.print_exc(file=tee.buffer_)
        raise
    finally:
        sys.stdout = tee.console
        if append_to_log:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(tee.buffer_.getvalue())
