"""
sensitivity_extended.py
=======================

Addendum to HABITABILITY_FACTOR_PROTOCOL.md (commit 225b269): the Sobol analysis of sensitivity.py with two
more uncertain assumptions, the shoreline zero point (x6, a shift of U(-1, +1) dex on every margin) and the
mass-radius relation for planets without a measured mass (x7: Chen & Kipping 2017 or Otegi et al. 2020
rocky). Expectations B3-B5. Plus the consistency check the protocol asks for (not pre-registered): the
fitted retention model on 8 Solar System bodies that were not in the fit.

    python stage_c_priority_fusion/sensitivity_extended.py
Output: outputs/stage_c/sensitivity_extended.json, eda/stage_c_sensitivity_extended.png
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
import cosmic_shoreline as cs  # noqa: E402
import sensitivity as sv  # noqa: E402
import target_physics as tp  # noqa: E402
import validate_shoreline as vs  # noqa: E402

OUT = RESEARCH / "outputs" / "stage_c"
N_BASE, SEED = 4096, 20261019
INPUTS = sv.INPUTS + ["shoreline_zero_point", "mass_radius_relation"]
RELATIONS = ["chen_kipping", "otegi_rocky"]
DELTAS = np.linspace(-1.0, 1.0, 41)
THIN = 4                                                   # every 4th Monte-Carlo draw for the fitted model


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def components(theta: np.ndarray) -> tuple[pd.Series, dict]:
    reg = pd.read_csv(OUT / "final_priority_table.csv")
    names = reg[reg.eligible].pl_name
    ps = pd.read_csv(bpt._PACK, low_memory=False)
    cat = (ps[ps.pl_name.isin(names)].assign(_n=lambda d: d.notna().sum(axis=1)).sort_values("_n", ascending=False)
           .drop_duplicates("pl_name").drop(columns="_n").set_index("pl_name").loc[list(names)].reset_index())
    c = {}
    for mr in RELATIONS:
        _, d = tp.monte_carlo(cat, n=bpt.N_DRAWS, return_draws=True, mass_relation=mr)
        small = d["radius"] < 1.5
        ta = {}
        for t in ["primary", "empirical", "bolometric"]:
            ta[t] = np.stack([(small & (d[f"margin_{t}"] + dl < 0)).mean(axis=1) for dl in DELTAS])
        m, sm = d["margin_primary"][:, ::THIN], small[:, ::THIN]
        ta["fitted"] = np.stack([(sm * np.mean([sigmoid(a + b * (m + dl)) for a, b, _ in theta], axis=0)).mean(axis=1)
                                 for dl in DELTAS])
        c[mr] = {"hz_cons": (d["hz"] == 1).mean(axis=1), "hz_opt_only": (d["hz"] == 0.5).mean(axis=1),
                 "observable": d["observable"].mean(axis=1), "ta": ta}
    return names.reset_index(drop=True), c


def scores(x: np.ndarray, c: dict) -> np.ndarray:
    w = 0.4 * 2.0 ** x[:, 0], 0.3 * 2.0 ** x[:, 1], 0.3 * 2.0 ** x[:, 2]
    t = np.clip(np.floor(x[:, 3]).astype(int), 0, 3)
    j = np.clip(np.rint((x[:, 5] + 1) / 2 * (len(DELTAS) - 1)).astype(int), 0, len(DELTAS) - 1)
    mr = np.clip(np.floor(x[:, 6]).astype(int), 0, 1)
    out = np.empty((len(x), len(c[RELATIONS[0]]["observable"])))
    for i in range(len(x)):
        cc = c[RELATIONS[mr[i]]]
        hz = cc["hz_cons"] + x[i, 4] * cc["hz_opt_only"]
        ta = cc["ta"][sv.TREATMENTS[t[i]]][j[i]]
        out[i] = (w[0][i] * hz + w[1][i] * ta + w[2][i] * cc["observable"]) / (w[0][i] + w[1][i] + w[2][i])
    return out


def sample(n, rng) -> np.ndarray:
    return np.column_stack([sv.sample(n, rng), rng.uniform(-1, 1, n), rng.uniform(0, 2, n)])


def solar_system(theta: np.ndarray, n: int = 4000) -> list[dict]:
    """Fitted retention on Solar System bodies (L = 1, so cumulative XUV = insolation), with the same
    zero-point and XUV-history draws as the exoplanet margins."""
    rng = np.random.default_rng(SEED)
    rows = []
    for name, (v, insol, atm) in vs.SOLAR_SYSTEM.items():
        log_i0 = rng.uniform(*cs.LOG_I0_RANGE, n)
        xuv = rng.normal(0.0, cs.XUV_SCALING_SIGMA_DEX, n)
        m = cs.margins(np.full(n, v), np.full(n, insol), np.ones(n), log_i0, xuv)["primary"]
        p = np.mean([sigmoid(a + b * m[::THIN]).mean() for a, b, _ in theta])
        rows.append({"body": name, "has_atmosphere": atm, "margin_primary_median": round(float(np.median(m)), 2),
                     "p_retain_fitted": round(float(p), 3), "p_retain_shoreline": round(float((m < 0).mean()), 3),
                     "fitted_correct_at_0.5": bool((p >= 0.5) == atm), "shoreline_correct_at_0.5": bool(((m < 0).mean() >= 0.5) == atm)})
    return rows


def main() -> int:
    rng = np.random.default_rng(SEED)
    theta = np.load(OUT / "retention_posterior.npz")["theta"][::10]                # 200 posterior samples
    names, c = components(theta)
    reg = pd.read_csv(OUT / "final_priority_table.csv").set_index("pl_name").loc[names]
    reg_rank = rankdata(-reg.priority_score.to_numpy(), method="min")
    A, B = sample(N_BASE, rng), sample(N_BASE, rng)
    AB = []
    for i in range(len(INPUTS)):
        m = A.copy()
        m[:, i] = B[:, i]
        AB.append(m)
    rho = lambda S: np.array([spearmanr(s, -reg_rank).statistic for s in S])
    top = lambda S: np.argsort(-S, axis=1)[:, :10]
    SA, SB, SAB = scores(A, c), scores(B, c), [scores(m, c) for m in AB]
    s1, st = sv.sobol(rho(SA), rho(SB), [rho(s) for s in SAB])
    member = np.zeros(len(names))
    for t in np.r_[top(SA), top(SB)]:
        member[t] += 1
    member /= 2 * N_BASE
    core = names[member >= 0.9].tolist()
    tot = dict(zip(INPUTS, np.round(st, 3)))
    ss = solar_system(theta)
    res = {"addendum": "HABITABILITY_FACTOR_PROTOCOL.md (commit 225b269)", "n_targets": int(len(names)), "n_base_samples": N_BASE,
           "spearman_vs_registered": {"first_order": dict(zip(INPUTS, np.round(s1, 3))), "total": tot,
                                      "median": round(float(np.median(rho(SA))), 3), "p5": round(float(np.percentile(rho(SA), 5)), 3)},
           "p_top10": dict(zip(names, np.round(member, 3))), "robust_core_ge_0.9": core,
           "expectations": {"B3 >= 5 targets in the top 10 in >= 90 % of samples": len(core) >= 5,
                            "B4 zero point total index < observability weight's": tot["shoreline_zero_point"] < tot["w_observable"],
                            "B5 mass-radius total index < 0.05": tot["mass_radius_relation"] < 0.05},
           "solar_system_consistency_not_preregistered": ss,
           "solar_system_correct": {"fitted": f"{sum(r['fitted_correct_at_0.5'] for r in ss)}/{len(ss)}",
                                    "shoreline": f"{sum(r['shoreline_correct_at_0.5'] for r in ss)}/{len(ss)}"}}
    (OUT / "sensitivity_extended.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in ["spearman_vs_registered", "robust_core_ge_0.9", "expectations", "solar_system_correct"]}, indent=2, default=float))
    print(pd.DataFrame(ss).to_string(index=False))
    figure(res, names, member)
    return 0


def figure(res, names, member):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), width_ratios=[1.2, 1])
    ax = axes[0]
    s = res["spearman_vs_registered"]
    x = np.arange(len(INPUTS))
    ax.bar(x - 0.18, [s["first_order"][k] for k in INPUTS], 0.36, color="#9ec5f4", label="first order")
    ax.bar(x + 0.18, [s["total"][k] for k in INPUTS], 0.36, color="#2a78d6", label="total")
    ax.set(xticks=x, xticklabels=["HZ\nweight", "rocky+ret.\nweight", "observ.\nweight", "retention\ntreatment", "opt.-HZ\ncredit",
                                  "shoreline\nzero point", "mass-radius\nrelation"],
           ylabel="Sobol index (Spearman with the registered ranking)", title="Seven assumptions, varied together")
    ax.legend(fontsize=8)
    ax = axes[1]
    o = np.argsort(-member)[:15]
    ax.barh(np.arange(len(o)), member[o], color=["#2a78d6" if member[i] >= 0.9 else "#9ec5f4" for i in o])
    ax.set(yticks=np.arange(len(o)), yticklabels=[names[i] for i in o], xlabel="fraction of samples in the top 10", xlim=(0, 1))
    ax.invert_yaxis()
    ax.axvline(0.9, color="k", ls=":", lw=1)
    ax.set_title(f"Robust core (>= 90 %): {len(res['robust_core_ge_0.9'])} targets")
    fig.tight_layout()
    fig.savefig(RESEARCH / "eda" / "stage_c_sensitivity_extended.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())
