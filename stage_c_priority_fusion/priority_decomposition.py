"""
priority_decomposition.py
=========================

eda/POST_REVIEW_PROTOCOL.md, analysis L (post hoc): the 44 ranked planets with and without the observability term, so that follow-up
priority and the habitability-related terms can be read apart.

    python stage_c_priority_fusion/priority_decomposition.py
Output: outputs/post_review/priority_decomposition.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

R = Path(__file__).resolve().parents[1]
OUT = R / "outputs" / "post_review"
WEIGHTS = {"hz": 0.40, "atm": 0.30, "obs": 0.30}                # the c_* columns are unweighted means of the three terms


def main() -> int:
    t = pd.read_csv(R / "outputs" / "stage_c" / "final_priority_table.csv")
    t = t[t.eligible.astype(bool)].copy()
    t["without_obs"] = (WEIGHTS["hz"] * t.c_hz + WEIGHTS["atm"] * t.c_terrestrial_atm) / (WEIGHTS["hz"] + WEIGHTS["atm"])
    t["without_hz"] = (WEIGHTS["atm"] * t.c_terrestrial_atm + WEIGHTS["obs"] * t.c_observable) / (WEIGHTS["atm"] + WEIGHTS["obs"])
    t["without_atm"] = (WEIGHTS["hz"] * t.c_hz + WEIGHTS["obs"] * t.c_observable) / (WEIGHTS["hz"] + WEIGHTS["obs"])
    t["rank_no_obs"] = t.without_obs.rank(ascending=False, method="min").astype(int)
    t["rank_all"] = t.rank_median.rank(method="min").astype(int)
    top = t.sort_values("rank_median").head(10)
    res = {"protocol": "eda/POST_REVIEW_PROTOCOL.md (L), post hoc", "n": int(len(t)), "spearman_with_full": round(float(spearmanr(t.rank_median, t.rank_no_obs)[0]), 3),
           "top10_full": dict(zip(top.pl_name, top.rank_no_obs.tolist())),
           "top10_without_observability": t.sort_values("rank_no_obs").head(10)[["pl_name", "without_obs", "rank_all"]].round(3).to_dict("records"),
           "shared_top10": int(len(set(top.pl_name) & set(t.sort_values("rank_no_obs").head(10).pl_name))),
           "dropped_term": {k: {"spearman_with_full": round(float(spearmanr(t.rank_median, -t[k]).statistic), 3),
                                "shared_top10": int(len(set(top.pl_name) & set(t.nlargest(10, k).pl_name)))}
                            for k in ("without_hz", "without_atm", "without_obs")}}
    (OUT / "priority_decomposition.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
