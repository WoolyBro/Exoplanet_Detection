"""
design_baseline.py
==================

eda/POST_REVIEW_PROTOCOL.md, analysis F (post hoc): does the information ranking recover the open Rocky Worlds targets because of
the retention model, or simply because both favor easy targets? Ranks the same planets by observability alone.

    python stage_c_priority_fusion/design_baseline.py
Output: outputs/post_review/design_baseline.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
OUT = R / "outputs" / "post_review"


def main() -> int:
    d = pd.read_csv(R / "outputs" / "stage_c" / "shoreline_design.csv")
    meta = json.loads((R / "outputs" / "stage_c" / "shoreline_design.json").read_text())
    t = meta["ddt_targets"]
    names = [x["pl_name"] if isinstance(x, dict) else x for x in (t if isinstance(t, list) else t.keys())]
    d["snr_per_visit"] = (d.depth_bare_ppm - d.depth_atm_ppm).abs() / d.sigma_visit_ppm
    d["reachable"] = d.verdict_error < 0.5 if "verdict_error" in d else True
    ranks = {"information_per_hour": d["rank"].rank(method="min"),
             "shortest_program": d.jwst_hours.rank(method="min"),
             "snr_per_visit": (-d.snr_per_visit).rank(method="min"),
             "brightest_host": d.sy_kmag.rank(method="min")}
    sel = d.pl_name.isin(names)
    out = {"protocol": "eda/POST_REVIEW_PROTOCOL.md (F), post hoc", "pool": int(len(d)), "targets": sorted(d.pl_name[sel]),
           "median_rank": {k: float(np.median(v[sel])) for k, v in ranks.items()},
           "ranks": {k: dict(zip(d.pl_name[sel], v[sel].astype(int).tolist())) for k, v in ranks.items()},
           "top45_count": {k: int((v[sel] <= 45).sum()) for k, v in ranks.items()}}
    base = min(out["median_rank"][k] for k in ("shortest_program", "snr_per_visit"))
    out["independently_recovers_stays"] = bool(out["median_rank"]["information_per_hour"] < base - 10)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "design_baseline.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "ranks"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
