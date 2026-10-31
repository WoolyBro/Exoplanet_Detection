"""
retention_model_comparison.py
=============================

eda/POST_REVIEW_PROTOCOL.md, analysis K (post hoc): which terms of the retention model do the data need, and how far do the labels,
the magma threshold and the margin definition matter? Neutral priors throughout.

    python stage_c_priority_fusion/retention_model_comparison.py
Output: outputs/post_review/retention_model_comparison.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import retention_model as rm  # noqa: E402

OUT = rm.RESEARCH / "outputs" / "post_review"
N = 60_000
NEUTRAL = {"alpha": (0.0, 1.5), "beta": (0.0, 2.0), "gamma": (0.0, 2.0)}


def run(y, margin, magma, use_beta, use_gamma, seed=rm.SEED) -> dict:
    rng = np.random.default_rng(seed)
    th = np.column_stack([rng.normal(*NEUTRAL["alpha"], N), rng.normal(*NEUTRAL["beta"], N) * use_beta, rng.normal(*NEUTRAL["gamma"], N) * use_gamma])
    p = np.clip(rm.prob(th, margin, magma), 1e-9, 1 - 1e-9)
    ll = y * np.log(p) + (1 - y) * np.log(1 - p)
    lw = ll.sum(1)
    w = np.exp(lw - lw.max())
    w /= w.sum()

    def loo_i(i):
        v = np.exp((lw - ll[:, i]) - (lw - ll[:, i]).max())
        return float(np.sum(v / v.sum() * p[:, i]))

    q = np.clip([loo_i(i) for i in range(len(y))], 1e-6, 1 - 1e-6)
    return {"log_marginal": float(logsumexp(lw) - np.log(N)), "loo_log_score": float(np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))),
            "p_beta_negative": float(np.sum(w * (th[:, 1] < 0))) if use_beta else None, "p_gamma_positive": float(np.sum(w * (th[:, 2] > 0))) if use_gamma else None}


def table(y, margin, magma) -> dict:
    res = {"intercept": run(y, margin, magma, 0, 0), "slope": run(y, margin, magma, 1, 0), "magma": run(y, margin, magma, 0, 1), "both": run(y, margin, magma, 1, 1)}
    base = res["intercept"]["log_marginal"]
    for r in res.values():
        r["ln_bf_vs_intercept"] = round(r["log_marginal"] - base, 2)
    return res


def main() -> int:
    obs = pd.concat([pd.read_csv(HERE / "rocky_atmosphere_observations.csv"), pd.read_csv(HERE / "rocky_atmosphere_observations_new.csv")], ignore_index=True)
    _, draws = rm.planets_with_draws(obs.planet.tolist())
    teq = np.nanmedian(draws["teq"], axis=1)
    mar, emp = draws["margin_primary"], draws["margin_empirical"]
    base = dict(rm.LABEL)
    y = np.array([base[(a, b)] for a, b in zip(obs.status, obs.strength)])
    out = {"protocol": "eda/POST_REVIEW_PROTOCOL.md (K), post hoc", "n": int(len(y)), "main": table(y, mar, (teq >= 1200).astype(float))}
    alt = {"hard labels": {("atmosphere_evidence", "tentative"): 1.0, ("no_thick_atmosphere", "tentative"): 0.0},
           "softer labels": {("atmosphere_evidence", "tentative"): 0.5, ("no_thick_atmosphere", "tentative"): 0.4}}
    out["sensitivity"] = {}
    for name, ch in alt.items():
        m = {**base, **ch}
        yy = np.array([m[(a, b)] for a, b in zip(obs.status, obs.strength)])
        out["sensitivity"][name] = run(yy, mar, (teq >= 1200).astype(float), 1, 1)
    keep = np.array([b != "tentative" for b in obs.strength])
    out["sensitivity"]["tentative planets dropped"] = {**run(y[keep], mar[keep], (teq[keep] >= 1200).astype(float), 1, 1), "n": int(keep.sum())}
    for t in (1000, 1500):
        out["sensitivity"][f"magma threshold {t} K"] = run(y, mar, (teq >= t).astype(float), 1, 1)
    out["sensitivity"]["empirical shoreline margin"] = run(y, emp, (teq >= 1200).astype(float), 1, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "retention_model_comparison.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: {"bf": v["ln_bf_vs_intercept"], "loo": round(v["loo_log_score"], 3)} for k, v in out["main"].items()}, indent=1))
    print(json.dumps({k: {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != "log_marginal"} for k, v in out["sensitivity"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
