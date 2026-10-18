"""
contamination_residuals.py
==========================

How much stellar contamination does the linear two-template model leave behind, visit by visit?
Measured on every public NIRSpec PRISM transit of TRAPPIST-1 in the research pack: b (1 visit), c (1),
d (2) and e (4), i.e. 8 visits of one star. The result calibrates the "visit-varying contamination"
term of detectability_forecast.py.

Runs in the petitRADTRANS environment (PHOENIX spectra for the templates):
    ~/micromamba/envs/prt/bin/python stage_b_atmospheres/contamination_residuals.py

Model, for visit v of planet p (y in ppm, on the published bins):

    y_v = D_p + B_v theta_v + g_v + n_v

  D_p        the planet's depth, flat: the weighted mean over its visits. None of these planets shows an
             atmosphere at PRISM precision (b and c from thermal emission, Greene et al. 2023, Zieba et al.
             2023; d, Piaulet-Ghorayeb et al. 2025; e, retrieval_trappist1e_real.py, FORECAST_RESULTS.md 4d),
             and a common planetary signal would be the same in every visit, unlike what is measured here.
  B_v theta  the retrieval's linear contamination, exactly as in retrieval_trappist1e_real.py:
             B = [D t_spot, D t_facula, 1], theta ~ N(0, diag(0.5^2, 0.5^2, 500^2)), marginalised.
  n_v        the published per-bin uncertainties.
  g_v        what the linear model misses: a Gaussian process in ln(wavelength),
                 cov_ij = A_v^2 D_p^2 w_i w_j exp(-(ln l_i - ln l_j)^2 / (2 ell^2)),
             with w = t_spot ("template-weighted": the unmodelled part follows the spot contrast, which
             falls toward the red) or w = 1 ("stationary"). A_v is a fraction of the transit depth per
             unit template; ell is shared by all 8 visits.

Fit: ell by profile likelihood on a grid; A_v per visit by maximum likelihood, with a 68 % interval
where the profile drops by 0.5, and a likelihood-ratio test against A_v = 0. The two kernel forms are
compared by their maximised likelihoods (same number of parameters).

Output: outputs/forecast/contamination_residuals.json, eda/contamination_residuals.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(HERE))
import retrieval_trappist1e_real as rr  # noqa: E402

OUT = RESEARCH / "outputs" / "forecast"
PLANETS = ["TRAPPIST-1 b", "TRAPPIST-1 c", "TRAPPIST-1 d", "TRAPPIST-1 e"]
ELL_GRID = np.geomspace(0.01, 1.5, 40)
LOGA_BOUNDS = (-4.5, 0.0)


def load_all() -> list[dict]:
    """Every PRISM visit with its bins, templates and the planet's flat depth."""
    visits = []
    for p in PLANETS:
        rr.PLANET_NAME = p
        rr._DATA.clear(); rr._PHX.clear(); rr._TPL.clear()           # the module caches one planet
        d, t = rr.load_visits(), rr.templates()
        w = 1 / d["sigma"] ** 2
        D = float(np.sum(d["depth"] * w) / np.sum(w))
        for v in range(len(d["depth"])):
            visits.append({"planet": p, "visit": v + 1, "wl": d["wl"].copy(), "y": d["depth"][v].copy(),
                           "sigma": d["sigma"][v].copy(), "spot": t["spot"].copy(), "facula": t["facula"].copy(), "D": D})
    return visits


def base_cov(v: dict) -> np.ndarray:
    """Published noise plus the retrieval's marginalised linear contamination."""
    B = np.column_stack([v["D"] * v["spot"], v["D"] * v["facula"], np.ones(len(v["y"]))])
    lam = np.array([rr.PRIOR_SD["spot"], rr.PRIOR_SD["facula"], rr.PRIOR_SD["offset_ppm"]]) ** 2
    return np.diag(v["sigma"] ** 2) + (B * lam) @ B.T


def gp_shape(v: dict, ell: float, kernel: str) -> np.ndarray:
    x = np.log(v["wl"])
    w = v["D"] * (v["spot"] if kernel == "template" else np.ones_like(x))
    return np.outer(w, w) * np.exp(-0.5 * ((x[:, None] - x[None, :]) / ell) ** 2)


def loglike(r: np.ndarray, C: np.ndarray) -> float:
    L = np.linalg.cholesky(C)
    z = np.linalg.solve(L, r)
    return float(-0.5 * (z @ z) - np.sum(np.log(np.diag(L))) - 0.5 * len(r) * np.log(2 * np.pi))


def fit_visit(v: dict, ell: float, kernel: str) -> tuple[float, float, float]:
    """(best log10 A, lnL at best, lnL at A = 0)."""
    r = v["y"] - v["D"]
    C0, K = base_cov(v), gp_shape(v, ell, kernel)
    l0 = loglike(r, C0)
    f = lambda la: -loglike(r, C0 + 10 ** (2 * la) * K + 1e-9 * np.eye(len(r)))
    res = minimize_scalar(f, bounds=LOGA_BOUNDS, method="bounded", options={"xatol": 1e-3})
    return (float(res.x), -float(res.fun), l0) if -res.fun > l0 else (-np.inf, l0, l0)


def interval(v: dict, ell: float, kernel: str, best_la: float, best_l: float) -> list[float]:
    r = v["y"] - v["D"]
    C0, K = base_cov(v), gp_shape(v, ell, kernel)
    grid = np.linspace(LOGA_BOUNDS[0], LOGA_BOUNDS[1], 361)
    prof = np.array([loglike(r, C0 + 10 ** (2 * la) * K + 1e-9 * np.eye(len(r))) for la in grid])
    inside = grid[prof >= best_l - 0.5]
    lo = -np.inf if inside.min() <= LOGA_BOUNDS[0] + 1e-9 else float(inside.min())
    return [lo, float(inside.max())]


def main() -> int:
    visits = load_all()
    print(f"{len(visits)} visits: " + ", ".join(f"{v['planet'][-1]}{v['visit']} ({len(v['y'])} bins)" for v in visits))
    res = {"visits_used": [f"{v['planet']} visit {v['visit']}" for v in visits], "kernels": {}}
    for kernel in ("template", "stationary"):
        prof = []
        for ell in ELL_GRID:
            prof.append(sum(fit_visit(v, ell, kernel)[1] for v in visits))
        prof = np.array(prof)
        ell = float(ELL_GRID[np.argmax(prof)])
        ell_ci = [float(ELL_GRID[prof >= prof.max() - 0.5].min()), float(ELL_GRID[prof >= prof.max() - 0.5].max())]
        rows = []
        for v in visits:
            la, lb, l0 = fit_visit(v, ell, kernel)
            dl = lb - l0
            r = v["y"] - v["D"]
            rows.append({"planet": v["planet"], "visit": v["visit"], "n_bins": len(r), "depth_ppm": round(v["D"], 1),
                         "chi2_red_linear_model": round(float(r @ np.linalg.solve(base_cov(v), r)) / len(r), 3),
                         "A": 0.0 if not np.isfinite(la) else round(10 ** la, 5),
                         "A_68": [0.0 if not np.isfinite(x) else round(10 ** x, 5) for x in interval(v, ell, kernel, la, lb)]
                         if np.isfinite(la) else [0.0, 0.0],
                         "delta_lnL_vs_A0": round(dl, 2),
                         # one parameter on the boundary: the LR statistic is a 50:50 mixture of chi2_0 and chi2_1
                         "p_value": round(float(0.5 * chi2.sf(2 * max(dl, 0), 1)) if dl > 0 else 1.0, 5)})
        res["kernels"][kernel] = {"ell": ell, "ell_68": ell_ci, "lnL_total": round(float(prof.max()), 2),
                                  "lnL_total_A0": round(float(sum(fit_visit(v, ell, kernel)[2] for v in visits)), 2),
                                  "profile": {"ell": [round(float(e), 4) for e in ELL_GRID], "lnL": [round(float(x), 2) for x in prof]},
                                  "visits": rows}
    k = res["kernels"]
    res["preferred_kernel"] = max(k, key=lambda n: k[n]["lnL_total"])
    pref = k[res["preferred_kernel"]]
    A = np.array([r["A"] for r in pref["visits"]])
    res["forecast_input"] = {"kernel": res["preferred_kernel"], "ell": pref["ell"], "A_per_visit": A.tolist(),
                             "note": "each forecast visit draws its unmodelled-contamination amplitude from these 8 values "
                                     "(the empirical visit-to-visit distribution of one very active M8 star)"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "contamination_residuals.json").write_text(json.dumps(res, indent=2))
    for n, kk in k.items():
        print(f"\n{n}: ell = {kk['ell']:.3f} [{kk['ell_68'][0]:.3f}, {kk['ell_68'][1]:.3f}], lnL {kk['lnL_total']} "
              f"(A = 0: {kk['lnL_total_A0']})")
        for r in kk["visits"]:
            print(f"  {r['planet']} v{r['visit']}: n={r['n_bins']:>3}  chi2_red(linear) {r['chi2_red_linear_model']:.2f}  "
                  f"A = {r['A']:.4f} {r['A_68']}  dlnL {r['delta_lnL_vs_A0']:+.2f}  p {r['p_value']:.3g}")
    print(f"\npreferred kernel: {res['preferred_kernel']}")
    make_figure(visits, res)
    return 0


def make_figure(visits, res):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kern = res["preferred_kernel"]
    kk = res["kernels"][kern]
    ell = kk["ell"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 10.5))
    for ax, v, row in zip(axes.flat, visits, kk["visits"]):
        r = v["y"] - v["D"]
        C0, K = base_cov(v), gp_shape(v, ell, kern)
        A2 = row["A"] ** 2
        # residual after the linear model's conditional best fit (what the retrieval leaves behind)
        Bm = np.column_stack([v["D"] * v["spot"], v["D"] * v["facula"], np.ones(len(r))])
        lam = np.array([rr.PRIOR_SD["spot"], rr.PRIOR_SD["facula"], rr.PRIOR_SD["offset_ppm"]]) ** 2
        Ctot = C0 + A2 * K + 1e-9 * np.eye(len(r))
        lin = (Bm * lam) @ Bm.T @ np.linalg.solve(Ctot, r)
        gp_mean = A2 * K @ np.linalg.solve(Ctot, r)
        ax.errorbar(v["wl"], r - lin, v["sigma"], fmt="o", ms=2, color="0.45", elinewidth=0.6, label="data - linear contamination fit")
        ax.plot(v["wl"], gp_mean, color="#d1495b", lw=1.6, label="unmodelled part (GP mean)")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{v['planet']} visit {v['visit']}: A = {row['A']:.3f}, dlnL {row['delta_lnL_vs_A0']:+.1f}", fontsize=9)
        ax.set(xlim=(0.6, 5.3), ylim=(-700, 700))
        ax.tick_params(labelsize=8)
    ax = axes.flat[-1]
    for n, c in [("template", "#2a6fdb"), ("stationary", "#2e933c")]:
        p = res["kernels"][n]["profile"]
        ax.plot(p["ell"], np.array(p["lnL"]) - res["kernels"][n]["lnL_total_A0"], color=c, label=f"{n} kernel")
    ax.set(xscale="log", xlabel="correlation length ell (in ln wavelength)", ylabel="lnL gain over no GP (all 8 visits)",
           title="Profile likelihood of the shared length scale", ylim=(0, None))
    ax.legend(fontsize=8)
    axes.flat[0].legend(fontsize=7, loc="lower left")
    for ax in axes[-1, :2]:
        ax.set_xlabel("wavelength (um)")
    for ax in axes[:, 0]:
        ax.set_ylabel("ppm")
    fig.suptitle("TRAPPIST-1, all 8 NIRSpec PRISM transits: stellar contamination left after the linear spot + facula model",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(RESEARCH / "eda" / "contamination_residuals.png", dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())
