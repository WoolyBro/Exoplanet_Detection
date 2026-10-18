"""
real_retrieval_report.py
========================

Summarises the retrievals on the real TRAPPIST-1 e JWST spectra (retrieval_trappist1e_real.py).
Runs in the petitRADTRANS environment (it re-evaluates best-fit spectra):
    ~/micromamba/envs/prt/bin/python stage_b_atmospheres/real_retrieval_report.py

Reports:
  * ln B and the Benneke & Seager sigma for: any atmosphere vs a bare rock; CH4; CO2; and the
    Archean scenario vs a bare rock (negative = the data DISFAVOUR it);
  * one-sided 95 % upper limits on log10 CH4 and CO2 from the full model's posterior;
  * per-visit chi^2 at the best fit, with the contamination and offset set to their conditional
    best values given that planet spectrum.

Output: outputs/retrieval_real/summary.json, eda/retrieval_trappist1e_real.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(HERE))
import retrieval_trappist1e as base  # noqa: E402
import retrieval_trappist1e_real as rr  # noqa: E402

OUT = RESEARCH / "outputs" / "retrieval_real"


def weighted_quantile(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(np.interp(q, c / c[-1], x[o]))


def conditional_fit(depth, y, sigma):
    """Best (a, b, c) for one visit given the planet depth (prior-regularised least squares), and chi^2."""
    t = rr.templates()
    ok = np.isfinite(depth) & np.isfinite(y)
    B = np.column_stack([depth[ok] * t["spot"][ok], depth[ok] * t["facula"][ok], np.ones(ok.sum())])
    lam = np.array([rr.PRIOR_SD["spot"], rr.PRIOR_SD["facula"], rr.PRIOR_SD["offset_ppm"]]) ** 2
    W = 1 / sigma[ok] ** 2
    A = (B.T * W) @ B + np.diag(1 / lam)
    theta = np.linalg.solve(A, (B.T * W) @ (y[ok] - depth[ok]))
    model = np.full(len(y), np.nan)
    model[ok] = depth[ok] + B @ theta
    return theta, model, float(np.sum(((y[ok] - model[ok]) / sigma[ok]) ** 2))


def main() -> int:
    res = {r["model"]: r for r in json.loads((OUT / "results.json").read_text())["runs"]}
    lnz = {k: v["logz"] for k, v in res.items()}
    err = {k: v["logz_err"] for k, v in res.items()}

    def comp(a, b):
        if a not in lnz or b not in lnz:
            return None
        lb = lnz[a] - lnz[b]
        return {"lnB": round(lb, 2), "lnB_err": round(float(np.hypot(err[a], err[b])), 2),
                "sigma": round(base.bayes_sigma(lb), 2) if lb > 0 else -round(base.bayes_sigma(-lb), 2)}

    summary = {"comparisons": {
        "atmosphere (full) vs bare rock": comp("full", "bare_rock"),
        "CH4 (full vs no_CH4)": comp("full", "no_CH4"),
        "CO2 (full vs no_CO2)": comp("full", "no_CO2"),
        "Archean scenario vs bare rock": comp("archean", "bare_rock"),
        "Archean scenario vs free atmosphere": comp("archean", "full")},
        "sign_convention": "positive sigma favours the first model; negative sigma favours the second"}

    # upper limits from the full model
    if (OUT / "posteriors" / "full.npz").is_file():
        z = np.load(OUT / "posteriors" / "full.npz", allow_pickle=True)
        names = list(z["names"].astype(str))
        lim = {}
        for g in ("CH4", "CO2", "H2O"):
            i = names.index(f"log_{g}")
            lim[g] = {"95pc_upper_log10_vmr": round(weighted_quantile(z["samples"][:, i], z["weights"], 0.95), 2),
                      "median": round(weighted_quantile(z["samples"][:, i], z["weights"], 0.5), 2),
                      "archean_value": float(np.log10(rr.ARCHEAN[g]))}
        summary["full_model_limits"] = lim

    # goodness of fit at the maximum-likelihood point of each model
    d = rr.load_visits()
    fits, curves = {}, {}
    for m, r in res.items():
        bf = r["best_fit"]
        gases = rr.MODELS[m]
        if gases == "fixed":
            vmr = dict(rr.ARCHEAN)
        else:
            vmr = {g: 10 ** bf[f"log_{g}"] for g in gases}
        depth = rr.planet_depth(vmr if m != "bare_rock" else {}, bf.get("T", 0.0), bf.get("logP", 0.0), bf["Rp"])
        chis, models, thetas = [], [], []
        for v in range(len(d["depth"])):
            th, mod, chi = conditional_fit(depth, d["depth"][v], d["sigma"][v])
            chis.append(round(chi, 1))
            models.append(mod)
            thetas.append([round(float(x), 4) for x in th])
        fits[m] = {"chi2_per_visit": chis, "chi2_total": round(sum(chis), 1), "n_points": int(d["depth"].size),
                   "contamination_theta_per_visit [spot, facula, offset_ppm]": thetas}
        curves[m] = (depth, models)
    summary["fits"] = fits
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    make_figure(d, curves)
    return 0


def make_figure(d, curves):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(13, 8.5))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.2, 1])
    wl = d["wl"]
    nv = len(d["depth"])
    colours = ["#2a6fdb", "#d1495b", "#2e933c", "#8f5bd6"]
    ax = fig.add_subplot(gs[0, :])
    for v in range(nv):
        ax.errorbar(wl, d["depth"][v], d["sigma"][v], fmt="o", ms=2.5, color=colours[v], alpha=0.7, label=f"visit {v + 1}")
        if "bare_rock" in curves:
            ax.plot(wl, curves["bare_rock"][1][v], color=colours[v], lw=0.8)
    ax.set(ylabel="transit depth (ppm)", ylim=(3800, 6000), xlim=(0.6, 5.3),
           title="TRAPPIST-1 e, JWST NIRSpec PRISM (Espinoza et al. 2025): 4 visits, each with its own contamination fit (lines: bare rock)")
    ax.legend(fontsize=7, ncol=4)
    ax = fig.add_subplot(gs[1, :2])
    if "bare_rock" in curves:
        depth0, mods0 = curves["bare_rock"]
        corr = np.array([d["depth"][v] - (mods0[v] - depth0) for v in range(nv)])
        w = 1 / d["sigma"] ** 2
        mean = np.sum(corr * w, axis=0) / np.sum(w, axis=0)
        sd = 1 / np.sqrt(np.sum(w, axis=0))
        ax.errorbar(wl, mean - np.nanmedian(depth0), sd, fmt="o", ms=3, color="k", label="4 visits combined, contamination removed")
        for m, c, lab in [("archean", "#d1495b", "Archean CO2 + CH4 (forecast scenario), best fit"), ("full", "#2a6fdb", "free atmosphere, best fit")]:
            if m in curves:
                ax.plot(wl, curves[m][0] - np.nanmedian(curves[m][0]), color=c, lw=1.5, label=lab)
        ax.axhline(0, color="0.6", lw=0.8)
    ax.set(xlabel="wavelength (um)", ylabel="depth - median (ppm)", ylim=(-400, 400), xlim=(0.6, 5.3))
    ax.legend(fontsize=7, loc="lower left")
    ax = fig.add_subplot(gs[1, 2])
    summ = json.loads((OUT / "summary.json").read_text())["fits"]
    if "bare_rock" in summ and "full" in summ:
        gain = np.array(summ["bare_rock"]["chi2_per_visit"]) - np.array(summ["full"]["chi2_per_visit"])
        ax.bar(np.arange(1, nv + 1), gain, color=colours[:nv])
        ax.axhline(0, color="k", lw=0.8)
        ax.set(xlabel="visit", ylabel="chi^2 improvement: bare rock -> free atmosphere", xticks=np.arange(1, nv + 1),
               title="A planet would improve every visit;\nthis 'signal' lives in visits 3 and 4")
    fig.tight_layout()
    fig.savefig(RESEARCH / "eda" / "retrieval_trappist1e_real.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())
