"""
retention_model.py
==================

Part (a) of HABITABILITY_FACTOR_PROTOCOL.md: atmosphere retention fitted to the 20 observed rocky planets
instead of assumed. logit P(atmosphere) = alpha + beta * shoreline_margin + gamma * magma (T_eq >= 1,200 K),
theory priors, soft labels from the evidence strength, each planet's margin uncertainty averaged over.
Importance sampling from the prior; exact leave-one-out by reweighting.

    python stage_c_priority_fusion/retention_model.py
Output: outputs/stage_c/retention_model.json, retention_posterior.npz, eda/retention_model.png
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
import target_physics as tp  # noqa: E402

OUT = RESEARCH / "outputs" / "stage_c"
PSCOMP = RESEARCH / "exoplanet_research_data" / "01_candidate_catalogs" / "pscomppars_confirmed_planets.csv"
N_PRIOR, N_DRAWS, SEED = 200_000, 4000, 20261019
MAGMA_TEQ = 1200.0
LABEL = {("atmosphere_evidence", "strong"): 1.0, ("atmosphere_evidence", "moderate"): 1.0,
         ("atmosphere_evidence", "tentative"): 0.6, ("no_thick_atmosphere", "tentative"): 0.25,
         ("no_thick_atmosphere", "moderate"): 0.0, ("no_thick_atmosphere", "strong"): 0.0}
PRIOR = {"alpha": (0.0, 1.5), "beta": (-2.0, 1.0), "gamma": (0.0, 2.0)}


def planets_with_draws(names) -> tuple[pd.DataFrame, dict]:
    ps = pd.read_csv(PSCOMP, low_memory=False)
    rows = (ps[ps.pl_name.isin(names)].assign(_n=lambda d: d.notna().sum(axis=1))
            .sort_values("_n", ascending=False).drop_duplicates("pl_name").drop(columns="_n"))
    rows = rows.set_index("pl_name").loc[list(names)].reset_index()
    phys, draws = tp.monte_carlo(rows, n=N_DRAWS, seed=SEED, return_draws=True)
    return phys, draws


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def prob(theta: np.ndarray, margin_draws: np.ndarray, magma: np.ndarray, thin: int = 50) -> np.ndarray:
    """P(atmosphere) per (theta sample, planet), averaged over each planet's margin draws (thinned)."""
    m = margin_draws[:, ::thin]                                         # (planets, draws)
    out = np.empty((len(theta), len(m)))
    for i in range(0, len(theta), 10_000):                             # chunked: the full array would be ~2.5 GB
        t = theta[i:i + 10_000]
        z = t[:, 0, None, None] + t[:, 1, None, None] * m[None] + t[:, 2, None, None] * magma[None, :, None]
        out[i:i + 10_000] = sigmoid(z).mean(axis=2)
    return out                                                          # (samples, planets)


def main() -> int:
    rng = np.random.default_rng(SEED)
    obs = pd.read_csv(HERE / "rocky_atmosphere_observations.csv")
    obs["y"] = [LABEL[(s, st)] for s, st in zip(obs.status, obs.strength)]
    phys, draws = planets_with_draws(obs.planet)
    margin = draws["margin_primary"]
    teq = np.nanmedian(draws["teq"], axis=1)
    magma = (teq >= MAGMA_TEQ).astype(float)
    y = obs.y.to_numpy()

    theta = np.column_stack([rng.normal(*PRIOR[k], N_PRIOR) for k in ("alpha", "beta", "gamma")])
    p = np.clip(prob(theta, margin, magma), 1e-9, 1 - 1e-9)
    ll_i = y * np.log(p) + (1 - y) * np.log(1 - p)                      # (samples, planets)
    logw = ll_i.sum(axis=1)
    w = np.exp(logw - logw.max())
    w /= w.sum()
    ess = 1 / np.sum(w ** 2)

    def wq(x, q):
        o = np.argsort(x)
        c = np.cumsum(w[o])
        return float(np.interp(q, c, x[o]))
    post = {k: {"median": round(wq(theta[:, i], 0.5), 3), "ci68": [round(wq(theta[:, i], 0.16), 3), round(wq(theta[:, i], 0.84), 3)],
                "ci95": [round(wq(theta[:, i], 0.025), 3), round(wq(theta[:, i], 0.975), 3)]}
            for i, k in enumerate(["alpha", "beta", "gamma"])}
    p_beta_neg = float(np.sum(w * (theta[:, 1] < 0)))
    p_gamma_pos = float(np.sum(w * (theta[:, 2] > 0)))

    # exact leave-one-out: drop planet i from the likelihood, predict it
    loo_p = np.zeros(len(y))
    for i in range(len(y)):
        lw = logw - ll_i[:, i]
        wi = np.exp(lw - lw.max())
        wi /= wi.sum()
        loo_p[i] = float(np.sum(wi * p[:, i]))
    score = lambda q: float(np.mean(y * np.log(np.clip(q, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - q, 1e-6, 1))))
    sv = pd.read_csv(OUT / "shoreline_validation.csv").set_index("planet").reindex(obs.planet)
    shore = sv.p_retain_primary.to_numpy()
    base = np.array([np.mean(np.delete(y, i)) for i in range(len(y))])
    from sklearn.metrics import roc_auc_score
    yb = (y >= 0.5).astype(int)
    loo = {"fitted": round(score(loo_p), 4), "fixed_shoreline": round(score(shore), 4), "base_rate": round(score(base), 4),
           "auc_fitted_loo": round(float(roc_auc_score(yb, loo_p)), 3), "auc_fixed_shoreline": round(float(roc_auc_score(yb, shore)), 3)}

    # posterior predictive for the Stage C targets (all temperate -> magma = 0)
    stagec = pd.read_csv(OUT / "final_priority_table.csv")
    tgt = stagec[stagec.eligible].pl_name
    tphys, tdraws = planets_with_draws(tgt)
    idx = rng.choice(N_PRIOR, size=2000, p=w)
    pt = prob(theta[idx], tdraws["margin_primary"], np.zeros(len(tgt)))     # (2000, targets)
    pred = pd.DataFrame({"pl_name": tgt.values, "p_retain_fitted": pt.mean(axis=0),
                         "p_retain_fitted_p5": np.percentile(pt, 5, axis=0), "p_retain_fitted_p95": np.percentile(pt, 95, axis=0),
                         "p_retain_shoreline": stagec[stagec.eligible].set_index("pl_name").reindex(tgt).c_retention.values})
    np.savez_compressed(OUT / "retention_posterior.npz", theta=theta[idx], names=np.array(["alpha", "beta", "gamma"]))
    pred.to_csv(OUT / "retention_fitted_targets.csv", index=False)

    res = {"n_planets": int(len(y)), "labels": dict(zip(obs.planet, y)), "magma_planets": list(obs.planet[magma == 1]),
           "effective_sample_size": round(float(ess)), "posterior": post, "p_beta_negative": round(p_beta_neg, 3),
           "p_gamma_positive": round(p_gamma_pos, 3), "loo_mean_log_score": loo,
           "loo_predictions": dict(zip(obs.planet, np.round(loo_p, 3))),
           "expectations": {"A1 P(beta<0) >= 0.90": p_beta_neg >= 0.90,
                            "A2 LOO beats fixed shoreline and base rate": loo["fitted"] > loo["fixed_shoreline"] and loo["fitted"] > loo["base_rate"],
                            "A3 P(gamma>0) >= 0.80": p_gamma_pos >= 0.80}}
    (OUT / "retention_model.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in ["effective_sample_size", "posterior", "p_beta_negative", "p_gamma_positive",
                                           "loo_mean_log_score", "expectations", "magma_planets"]}, indent=2, default=float))
    print(pred.sort_values("p_retain_fitted", ascending=False).round(3).head(12).to_string(index=False))
    figure(obs, loo_p, shore, margin, magma, theta[idx])
    return 0


def figure(obs, loo_p, shore, margin, magma, th):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    m = np.median(margin, axis=1)
    xs = np.linspace(-3, 3.5, 200)
    for mg, c, lab in [(0, "#2a78d6", "T_eq < 1,200 K"), (1, "#eb6834", "T_eq >= 1,200 K (magma dayside)")]:
        curves = sigmoid(th[:, 0, None] + th[:, 1, None] * xs[None] + th[:, 2, None] * mg)
        ax.fill_between(xs, *np.percentile(curves, [5, 95], axis=0), color=c, alpha=0.18, lw=0)
        ax.plot(xs, np.median(curves, axis=0), color=c, lw=2, label=f"fitted, {lab}")
        sel = magma == mg
        ax.scatter(m[sel], obs.y[sel], color=c, edgecolor="k", s=40, zorder=3)
    for xi, yi, n in zip(m, obs.y, obs.planet):
        if yi >= 0.6:
            ax.annotate(n, (xi, yi), xytext=(4, -10), textcoords="offset points", fontsize=7)
    ax.axvline(0, color="0.5", ls=":", lw=1)
    ax.set(xlabel="cosmic-shoreline margin (dex; > 0 = airless side)", ylabel="P(atmosphere) / soft label",
           title="Retention fitted to 20 observed rocky planets (90 % band)")
    ax.legend(fontsize=8)
    ax = axes[1]
    order = np.argsort(loo_p)
    ax.scatter(np.arange(len(order)), obs.y.to_numpy()[order], color="k", label="soft label", zorder=3)
    ax.plot(np.arange(len(order)), loo_p[order], "o-", color="#2a78d6", label="fitted model, leave-one-out")
    ax.plot(np.arange(len(order)), shore[order], "s--", color="#8a8984", label="fixed shoreline")
    ax.set(xticks=np.arange(len(order)), xticklabels=obs.planet.to_numpy()[order], ylabel="P(atmosphere)",
           title="Held-out predictions")
    ax.tick_params(axis="x", rotation=80, labelsize=7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESEARCH / "eda" / "retention_model.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())
