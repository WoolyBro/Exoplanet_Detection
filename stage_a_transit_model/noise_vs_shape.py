"""
noise_vs_shape.py
=================

eda/POST_REVIEW_PROTOCOL.md, analysis C (post hoc): does added noise raise deep K2 planets through S/N or by hiding shape?
Frozen Kepler ensemble; no performance is measured.

    python stage_a_transit_model/noise_vs_shape.py
Output: outputs/post_review/noise_vs_shape.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from stage_a_transit_model.cnn_lstm import ViewSet, consolidate_views, drop_unusable  # noqa: E402
from stage_a_transit_model.critique2_analyses import RUNS, SEEDS, per_seed  # noqa: E402
from stage_a_transit_model.critique2_analyses import load_run_model  # noqa: E402
from preprocessing_pipeline import _npz_name  # noqa: E402

OUT = R / "outputs" / "post_review"
VIEWS = R / "detection_views_err" / "k2_everest"
SIGMA, REPEATS = 0.05, 5


def meta(oid) -> dict:
    with np.load(VIEWS / _npz_name(oid), allow_pickle=False) as z:
        return json.loads(str(z["metadata"]))


def regions(vs: ViewSet, info: list[dict]) -> dict:
    """Boolean masks per signal, on the global and local views: in transit, around phase 0.5, and the rest."""
    n, ng, nl = len(vs), vs.global_view.shape[1], vs.local_view.shape[1]
    gin, gsec, lin = np.zeros((n, ng), bool), np.zeros((n, ng), bool), np.zeros((n, nl), bool)
    cg, cl = ng // 2, nl // 2
    lin[:, cl - 15:cl + 16] = True                              # the local view spans +/- 4 durations: transit is +/- 12.5 bins
    for i, d in enumerate(info):
        frac = float(d["duration_days"]) / float(d["period_days"])
        hw = max(3, int(np.ceil(0.6 * frac * ng)))
        gin[i, cg - hw:cg + hw + 1] = True
        gsec[i, :2 * hw], gsec[i, -2 * hw:] = True, True
    return {"gin": gin, "gsec": gsec, "lin": lin}


def noisy(vs: ViewSet, gm: np.ndarray, lm: np.ndarray, rng) -> ViewSet:
    g = vs.global_view + (rng.normal(0, SIGMA, gm.shape) * gm).astype(np.float32)
    l = vs.local_view + (rng.normal(0, SIGMA, lm.shape) * lm).astype(np.float32)
    ge = np.sqrt(vs.global_err.astype(np.float64) ** 2 + (SIGMA * gm) ** 2).astype(np.float32)
    le = np.sqrt(vs.local_err.astype(np.float64) ** 2 + (SIGMA * lm) ** 2).astype(np.float32)
    return ViewSet(g, l, vs.label_index, vs.object_id, vs.star_id, ge, le)


def main() -> int:
    models = [load_run_model(RUNS / r)[0] for r in SEEDS]
    k2 = drop_unusable(consolidate_views(VIEWS, verbose=False), verbose=False)
    info = [meta(o) for o in k2.object_id]
    dep = np.array([d.get("depth_ppm", np.nan) for d in info], float)
    print("metadata keys:", sorted(info[0])[:30], flush=True)
    res = {"protocol": "eda/POST_REVIEW_PROTOCOL.md (C), post hoc", "sigma_depth_units": SIGMA, "groups": {}}
    for name, idx in [("deep_confirmed_gt_10000ppm", np.flatnonzero((k2.label_index == 2) & (dep > 10000))),
                      ("shallow_confirmed_lt_3000ppm", np.flatnonzero((k2.label_index == 2) & (dep < 3000))[:200])]:
        sub = k2.take(idx)
        r = regions(sub, [info[i] for i in idx])
        all_g, all_l = np.ones_like(r["gin"]), np.ones_like(r["lin"])
        out_g, out_l = ~(r["gin"] | r["gsec"]), ~r["lin"]
        conds = {"none": (0 * all_g, 0 * all_l), "everywhere": (all_g, all_l), "in_transit_only": (r["gin"], r["lin"]),
                 "out_of_transit_only": (out_g, out_l), "phase_0.5_only": (r["gsec"], 0 * all_l)}
        med = {}
        for c, (gm, lm) in conds.items():
            sc = np.mean([per_seed(models, noisy(sub, gm.astype(float), lm.astype(float), np.random.default_rng(s))).mean(0) for s in range(REPEATS)], axis=0)
            med[c] = round(float(np.median(sc)), 3)
        res["groups"][name] = {"n": int(len(idx)), "median_score": med}
    d = res["groups"]["deep_confirmed_gt_10000ppm"]["median_score"]
    rise = d["everywhere"] - d["none"]
    res["rise_everywhere"] = round(rise, 3)
    res["share_of_rise"] = {k: round((d[k] - d["none"]) / rise, 2) for k in ("in_transit_only", "out_of_transit_only", "phase_0.5_only")} if rise > 0 else None
    s = res["share_of_rise"] or {}
    res["reading"] = ("S/N" if s.get("out_of_transit_only", 0) >= 2 / 3 else
                      "shape" if max(s.get("in_transit_only", 0), s.get("phase_0.5_only", 0)) >= 2 / 3 else "not separated")
    (OUT / "noise_vs_shape.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
