"""
sensitivity.py
==============

Part (b) of HABITABILITY_FACTOR_PROTOCOL.md: how much the priority ranking depends on each of its choices,
as Sobol indices (Saltelli sampling, Jansen estimators), plus the robust core of targets that stay in the
top 10 whatever the choices. Also expectation A4: the ranking with the fitted retention model.

Inputs (independent):
  x1-x3  weights of HZ, terrestrial-and-retained, observability: log2 multipliers U(-1, 1) on 0.40 / 0.30 / 0.30
  x4     retention treatment: shoreline primary | empirical | bolometric | fitted (retention_model.py)
  x5     credit for the optimistic-only HZ band, U(0, 1) (registered: 0.5)

    python stage_c_priority_fusion/sensitivity.py
Output: outputs/stage_c/sensitivity.json, eda/stage_c_sensitivity.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(HERE))
import build_priority_table as bpt  # noqa: E402
import target_physics as tp  # noqa: E402

OUT = RESEARCH / "outputs" / "stage_c"
N_BASE, SEED = 4096, 20261019
INPUTS = ["w_hz", "w_terrestrial_atm", "w_observable", "retention_treatment", "optimistic_hz_credit"]
TREATMENTS = ["primary", "empirical", "bolometric", "fitted"]


def components() -> tuple[pd.Series, dict]:
    stagec = pd.read_csv(OUT / "final_priority_table.csv")
    names = stagec[stagec.eligible].pl_name
    ps = pd.read_csv(bpt._PACK, low_memory=False)
    cat = (ps[ps.pl_name.isin(names)].assign(_n=lambda d: d.notna().sum(axis=1)).sort_values("_n", ascending=False)
           .drop_duplicates("pl_name").drop(columns="_n").set_index("pl_name").loc[list(names)].reset_index())
    phys, d = tp.monte_carlo(cat, n=bpt.N_DRAWS, return_draws=True)
    small = d["radius"] < 1.5
    th = np.load(OUT / "retention_posterior.npz")["theta"][::10]                      # 200 posterior samples
    fitted = np.mean([1 / (1 + np.exp(-(a + b * d["margin_primary"]))) for a, b, _ in th], axis=0)
    c = {"hz_cons": (d["hz"] == 1).mean(axis=1), "hz_opt_only": (d["hz"] == 0.5).mean(axis=1),
         "observable": d["observable"].mean(axis=1),
         "ta": {"primary": (small & (d["margin_primary"] < 0)).mean(axis=1),
                "empirical": (small & (d["margin_empirical"] < 0)).mean(axis=1),
                "bolometric": (small & (d["margin_bolometric"] < 0)).mean(axis=1),
                "fitted": (small * fitted).mean(axis=1)}}
    return names.reset_index(drop=True), c


def scores(x: np.ndarray, c: dict) -> np.ndarray:
    """x: (n, 5) -> scores (n, planets)."""
    w = 0.4 * 2.0 ** x[:, 0], 0.3 * 2.0 ** x[:, 1], 0.3 * 2.0 ** x[:, 2]
    t = np.clip(np.floor(x[:, 3]).astype(int), 0, 3)
    ta = np.stack([c["ta"][TREATMENTS[i]] for i in t])
    hz = c["hz_cons"][None] + x[:, 4, None] * c["hz_opt_only"][None]
    return (w[0][:, None] * hz + w[1][:, None] * ta + w[2][:, None] * c["observable"][None]) / (w[0] + w[1] + w[2])[:, None]


def sample(n, rng) -> np.ndarray:
    return np.column_stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), rng.uniform(-1, 1, n),
                            rng.uniform(0, 4, n), rng.uniform(0, 1, n)])


def sobol(fA, fB, fAB) -> tuple[np.ndarray, np.ndarray]:
    var = np.var(np.r_[fA, fB])
    s1 = np.array([np.mean(fB * (fABi - fA)) / var for fABi in fAB])
    st = np.array([0.5 * np.mean((fA - fABi) ** 2) / var for fABi in fAB])
    return s1, st


def main() -> int:
    rng = np.random.default_rng(SEED)
    names, c = components()
    reg = pd.read_csv(OUT / "final_priority_table.csv").set_index("pl_name").loc[names]
    reg_rank = rankdata(-reg.priority_score.to_numpy(), method="min")
    reg_top = set(np.argsort(-reg.priority_score.to_numpy())[:10])

    A, B = sample(N_BASE, rng), sample(N_BASE, rng)
    AB = []
    for i in range(len(INPUTS)):
        m = A.copy()
        m[:, i] = B[:, i]
        AB.append(m)
    rho = lambda S: np.array([spearmanr(s, -reg_rank).statistic for s in S])
    top = lambda S: np.argsort(-S, axis=1)[:, :10]
    SA, SB, SAB = scores(A, c), scores(B, c), [scores(m, c) for m in AB]
    s1, st = sobol(rho(SA), rho(SB), [rho(s) for s in SAB])
    member = np.zeros(len(names))
    for t in np.r_[top(SA), top(SB)]:
        member[t] += 1
    member /= 2 * N_BASE
    # per-planet total indices for top-10 membership (the planets whose fate is uncertain)
    per = {}
    for j in np.flatnonzero((member > 0.05) & (member < 0.95)):
        f = lambda S: np.isin(j, top(S), assume_unique=False) if False else (np.argsort(-S, axis=1)[:, :10] == j).any(axis=1).astype(float)
        a1, at = sobol(f(SA), f(SB), [f(s) for s in SAB])
        per[names[j]] = {"p_top10": round(float(member[j]), 3), "total_index": dict(zip(INPUTS, np.round(at, 3)))}

    # A4: the registered weights, fitted retention
    xr = np.array([[0, 0, 0, 3.5, 0.5]])
    fitted_rank = rankdata(-scores(xr, c)[0], method="min")
    a4 = float(spearmanr(fitted_rank, reg_rank).statistic)
    top10_fitted = set(np.argsort(-scores(xr, c)[0])[:10])
    core = names[member >= 0.9].tolist()
    res = {"n_targets": int(len(names)), "n_base_samples": N_BASE,
           "spearman_vs_registered": {"first_order": dict(zip(INPUTS, np.round(s1, 3))), "total": dict(zip(INPUTS, np.round(st, 3))),
                                      "median": round(float(np.median(rho(SA))), 3), "p5": round(float(np.percentile(rho(SA), 5)), 3)},
           "p_top10": dict(zip(names, np.round(member, 3))), "robust_core_ge_0.9": core,
           "uncertain_targets_total_indices": per,
           "fitted_retention_ranking": {"spearman_vs_registered": round(a4, 3), "top10_overlap": len(top10_fitted & reg_top),
                                        "top10": [names[i] for i in sorted(top10_fitted, key=lambda i: fitted_rank[i])]},
           "expectations": {"A4 fitted-retention ranking Spearman >= 0.8": a4 >= 0.8,
                            "B1 HZ weight has the largest total index": int(np.argmax(st)) == 0,
                            "B2 >= 5 targets in the top 10 in >= 90 % of samples": len(core) >= 5}}
    (OUT / "sensitivity.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in ["spearman_vs_registered", "robust_core_ge_0.9", "fitted_retention_ranking", "expectations"]}, indent=2, default=float))
    figure(res, names, member)
    return 0


def figure(res, names, member):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), width_ratios=[1, 1.3])
    ax = axes[0]
    s = res["spearman_vs_registered"]
    x = np.arange(len(INPUTS))
    ax.bar(x - 0.18, [s["first_order"][k] for k in INPUTS], 0.36, color="#9ec5f4", label="first order")
    ax.bar(x + 0.18, [s["total"][k] for k in INPUTS], 0.36, color="#2a78d6", label="total")
    ax.set(xticks=x, xticklabels=["HZ weight", "rocky+retained\nweight", "observability\nweight", "retention\ntreatment", "optimistic-HZ\ncredit"],
           ylabel="Sobol index (Spearman with the registered ranking)", title="What the ranking depends on")
    ax.legend(fontsize=8)
    ax = axes[1]
    o = np.argsort(-member)[:20]
    ax.barh(np.arange(len(o)), member[o], color=["#2a78d6" if member[i] >= 0.9 else "#9ec5f4" for i in o])
    ax.set(yticks=np.arange(len(o)), yticklabels=[names[i] for i in o], xlabel="fraction of samples in the top 10", xlim=(0, 1))
    ax.invert_yaxis()
    ax.axvline(0.9, color="k", ls=":", lw=1)
    ax.set_title(f"Robust core (>= 90 %): {len(res['robust_core_ge_0.9'])} targets")
    fig.tight_layout()
    fig.savefig(RESEARCH / "eda" / "stage_c_sensitivity.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())
