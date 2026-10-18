"""
build_k2_views.py
=================

Views for the K2 blind test (eda/K2_BLIND_TEST_PROTOCOL.md): every K2 planet/candidate with a complete
ephemeris (tools/k2_ephemerides.py), through exactly the Kepler/TESS pipeline, with the flux-error channel.

    python tools/build_k2_views.py                      # EVEREST light curves (primary)
    python tools/build_k2_views.py --author K2SFF       # robustness

Output: detection_views_err/k2_<author>/ (one .npz per signal, index.csv, progress.jsonl). Each star's raw
download is deleted once its views are written. The build is resumable: rerun the same command.
Progress and time left:  tail -3 detection_views_err/k2_everest/build_progress.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", choices=["EVEREST", "K2SFF"], default="EVEREST")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="first N stars only (a smoke test)")
    a = ap.parse_args()
    os.environ["K2_LC_AUTHOR"] = a.author          # read at import, also by every worker process
    sys.path.insert(0, str(R))
    import pandas as pd
    import preprocessing_pipeline as p

    eph = pd.read_csv(p.K2_EPHEMERIDES)
    frame = eph[eph.ephemeris_complete].copy()
    if a.limit:
        frame = frame[frame.epic.isin(frame.epic.drop_duplicates().head(a.limit))]
    out = R / "detection_views_err" / f"k2_{a.author.lower()}"
    print(f"K2 blind-test views: {len(frame)} signals on {frame.epic.nunique()} stars, light curves {a.author} -> {out}")
    index = p.build_dataset(frame, out, n_workers=a.workers, discard_raw=True, min_free_ram_gb=0.5,
                            progress_every=25, verbose=True, use_s3=False, isolate_workers=True, stall_timeout_s=900)
    print(index.status.value_counts().to_dict())
    if "category" in index.columns:
        print(index[index.status != "ok"].category.value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
