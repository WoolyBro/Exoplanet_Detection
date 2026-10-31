"""
retention_prior_sensitivity.py
==============================

eda/POST_REVIEW_PROTOCOL.md, analysis E (post hoc): how much of the retention result comes from the prior on beta? The same model,
likelihood and importance sampling as retention_model.py, with neutral priors.

    python stage_c_priority_fusion/retention_prior_sensitivity.py
Output: outputs/post_review/retention_prior_sensitivity.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import retention_model as rm  # noqa: E402

OUT = rm.RESEARCH / "outputs" / "post_review"


def fit(obs: pd.DataFrame, beta_prior: tuple) -> dict:
    rng = np.random.default_rng(rm.SEED)
    _, draws = rm.planets_with_draws(obs.planet.tolist())
    margin, magma = draws["margin_primary"], (np.nanmedian(draws["teq"], axis=1) >= rm.MAGMA_TEQ).astype(float)
    y = obs.y.to_numpy()
    th = np.column_stack([rng.normal(*rm.PRIOR["alpha"], rm.N_PRIOR), rng.normal(*beta_prior, rm.N_PRIOR), rng.normal(*rm.PRIOR["gamma"], rm.N_PRIOR)])
    p = np.clip(rm.prob(th, margin, magma), 1e-9, 1 - 1e-9)
    ll = y * np.log(p) + (1 - y) * np.log(1 - p)
    lw = ll.sum(1)
    w = np.exp(lw - lw.max())
    w /= w.sum()

    def wq(x, q):
        o = np.argsort(x)
        return float(np.interp(q, np.cumsum(w[o]), x[o]))

    def loo_i(i):
        v = np.exp((lw - ll[:, i]) - (lw - ll[:, i]).max())
        return float(np.sum(v / v.sum() * p[:, i]))

    pb, prior_pb = float(np.sum(w * (th[:, 1] < 0))), float(np.mean(th[:, 1] < 0))
    loo = np.array([loo_i(i) for i in range(len(y))])
    s = lambda q: float(np.mean(y * np.log(np.clip(q, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - q, 1e-6, 1))))
    base = np.array([np.mean(np.delete(y, i)) for i in range(len(y))])
    return {"beta_prior": list(beta_prior), "n": int(len(y)), "prior_p_beta_negative": round(prior_pb, 3), "p_beta_negative": round(pb, 3),
            "beta_median": round(wq(th[:, 1], 0.5), 2), "beta_ci95": [round(wq(th[:, 1], 0.025), 2), round(wq(th[:, 1], 0.975), 2)],
            "p_gamma_positive": round(float(np.sum(w * (th[:, 2] > 0))), 3), "loo_log_score": round(s(loo), 3), "base_rate_log_score": round(s(base), 3)}


def main() -> int:
    obs = pd.read_csv(HERE / "rocky_atmosphere_observations.csv")
    new = pd.read_csv(HERE / "rocky_atmosphere_observations_new.csv")
    for d in (obs, new):
        d["y"] = [rm.LABEL[(a, b)] for a, b in zip(d.status, d.strength)]
    all23 = pd.concat([obs, new], ignore_index=True)
    rows = [fit(obs, p) for p in [(-2, 1), (0, 1), (0, 1.5), (0, 2), (0, 3)]] + [fit(all23, p) for p in [(-2, 1), (0, 2)]]
    assert rows[0]["p_beta_negative"] == 0.986 and rows[0]["beta_median"] == -1.27
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "retention_prior_sensitivity.json").write_text(json.dumps({"protocol": "eda/POST_REVIEW_PROTOCOL.md (E), post hoc", "rows": rows}, indent=2))
    print(pd.DataFrame(rows).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
