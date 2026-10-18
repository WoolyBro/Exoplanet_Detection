"""
retention_update.py
===================

Addendum 2 of HABITABILITY_FACTOR_PROTOCOL.md (commit 2f3582c): an out-of-sample test of the fitted
retention model on three rocky planets published after the fit, then the refit on all 23.

  1. The frozen 20-planet posterior (outputs/stage_c/retention_posterior.npz) predicts HD 3167 b, TOI-431 b
     and GJ 357 b; log score against the base rate and the fixed shoreline (A5).
  2. The same importance-sampling fit as retention_model.py on 20 + 3 planets: posterior, leave-one-out
     log scores (A6, A7), and the Stage C targets' retention under the refit.

    python stage_c_priority_fusion/retention_update.py
Output: outputs/stage_c/retention_update.json (the 20-planet outputs are left untouched)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(HERE))
import retention_model as rm  # noqa: E402

OUT = RESEARCH / "outputs" / "stage_c"


def labelled(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    return d.assign(y=[rm.LABEL[(s, st)] for s, st in zip(d.status, d.strength)])


def inputs(obs: pd.DataFrame):
    _, draws = rm.planets_with_draws(obs.planet)
    teq = np.nanmedian(draws["teq"], axis=1)
    return draws["margin_primary"], (teq >= rm.MAGMA_TEQ).astype(float), teq


def log_score(y, q):
    q = np.clip(np.asarray(q, float), 1e-6, 1 - 1e-6)
    return float(np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def fit(y, margin, magma, seed=rm.SEED):
    rng = np.random.default_rng(seed)
    theta = np.column_stack([rng.normal(*rm.PRIOR[k], rm.N_PRIOR) for k in ("alpha", "beta", "gamma")])
    p = np.clip(rm.prob(theta, margin, magma), 1e-9, 1 - 1e-9)
    ll_i = y * np.log(p) + (1 - y) * np.log(1 - p)
    logw = ll_i.sum(axis=1)
    w = np.exp(logw - logw.max())
    w /= w.sum()
    loo = np.zeros(len(y))
    for i in range(len(y)):
        lw = logw - ll_i[:, i]
        wi = np.exp(lw - lw.max())
        loo[i] = float(np.sum(wi / wi.sum() * p[:, i]))
    return theta, w, loo


def wq(x, w, q):
    o = np.argsort(x)
    return float(np.interp(q, np.cumsum(w[o]), x[o]))


def main() -> int:
    from sklearn.metrics import roc_auc_score

    old, new = labelled(HERE / "rocky_atmosphere_observations.csv"), labelled(HERE / "rocky_atmosphere_observations_new.csv")
    m_new, g_new, teq_new = inputs(new)
    theta_frozen = np.load(OUT / "retention_posterior.npz")["theta"]
    p_frozen = rm.prob(theta_frozen, m_new, g_new).mean(axis=0)
    p_shore = (m_new < 0).mean(axis=1)
    base = float(old.y.mean())
    yn = new.y.to_numpy()
    a5 = {"planets": {p: {"label": float(y), "magma": bool(g), "teq_K": round(float(t)), "margin_median_dex": round(float(np.median(m)), 2),
                          "p_fitted_frozen": round(float(pf), 3), "p_fixed_shoreline": round(float(ps), 3)}
                      for p, y, g, t, m, pf, ps in zip(new.planet, yn, g_new, teq_new, m_new, p_frozen, p_shore)},
          "log_score": {"fitted_frozen": round(log_score(yn, p_frozen), 3), "base_rate": round(log_score(yn, np.full(len(yn), base)), 3),
                        "fixed_shoreline": round(log_score(yn, p_shore), 3)}}

    allp = pd.concat([old, new], ignore_index=True)
    m_all, g_all, _ = inputs(allp)
    y = allp.y.to_numpy()
    theta, w, loo = fit(y, m_all, g_all)
    shore_all = (m_all < 0).mean(axis=1)
    base_loo = np.array([np.mean(np.delete(y, i)) for i in range(len(y))])
    yb = (y >= 0.5).astype(int)
    post = {k: {"median": round(wq(theta[:, i], w, 0.5), 3), "ci95": [round(wq(theta[:, i], w, 0.025), 3), round(wq(theta[:, i], w, 0.975), 3)]}
            for i, k in enumerate(["alpha", "beta", "gamma"])}
    refit = {"n_planets": int(len(y)), "posterior": post, "effective_sample_size": round(float(1 / np.sum(w ** 2))),
             "p_beta_negative": round(float(np.sum(w * (theta[:, 1] < 0))), 3),
             "p_gamma_positive": round(float(np.sum(w * (theta[:, 2] > 0))), 3),
             "loo_log_score": {"fitted": round(log_score(y, loo), 4), "fixed_shoreline": round(log_score(y, shore_all), 4),
                               "base_rate": round(log_score(y, base_loo), 4)},
             "loo_auc": {"fitted": round(float(roc_auc_score(yb, loo)), 3), "fixed_shoreline": round(float(roc_auc_score(yb, shore_all)), 3)},
             "loo_auc_non_magma_only": round(float(roc_auc_score(yb[g_all == 0], loo[g_all == 0])), 3) if len(set(yb[g_all == 0])) == 2 else None}

    # Stage C targets under the refit (all temperate: magma = 0)
    stagec = pd.read_csv(OUT / "final_priority_table.csv")
    tgt = stagec[stagec.eligible].pl_name
    _, tdraws = rm.planets_with_draws(tgt)
    idx = np.random.default_rng(rm.SEED).choice(len(theta), size=2000, p=w)
    pt = rm.prob(theta[idx], tdraws["margin_primary"], np.zeros(len(tgt))).mean(axis=0)
    prev = pd.read_csv(OUT / "retention_fitted_targets.csv").set_index("pl_name").p_retain_fitted
    targets = pd.DataFrame({"pl_name": tgt.values, "p_retain_refit23": pt, "p_retain_fit20": prev.reindex(tgt).values})

    res = {"addendum": "HABITABILITY_FACTOR_PROTOCOL.md addendum 2 (commit 2f3582c)", "A5_out_of_sample": a5, "refit_23": refit,
           "targets_max_abs_change": round(float(np.nanmax(np.abs(targets.p_retain_refit23 - targets.p_retain_fit20))), 3),
           "targets": targets.round(3).to_dict(orient="records")}
    ls = a5["log_score"]
    res["expectations"] = {"A5 frozen model beats base rate and fixed shoreline on the new planets":
                               bool(ls["fitted_frozen"] > ls["base_rate"] and ls["fitted_frozen"] > ls["fixed_shoreline"]),
                           "A6 refit P(beta<0) >= 0.90 and P(gamma>0) >= 0.95":
                               bool(refit["p_beta_negative"] >= 0.90 and refit["p_gamma_positive"] >= 0.95),
                           "A7 refit LOO beats fixed shoreline and base rate":
                               bool(refit["loo_log_score"]["fitted"] > refit["loo_log_score"]["fixed_shoreline"]
                                    and refit["loo_log_score"]["fitted"] > refit["loo_log_score"]["base_rate"])}
    (OUT / "retention_update.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in ["A5_out_of_sample", "refit_23", "targets_max_abs_change", "expectations"]}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
