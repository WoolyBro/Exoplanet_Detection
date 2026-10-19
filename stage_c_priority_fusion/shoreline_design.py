"""
shoreline_design.py
===================

SHORELINE_DESIGN_PROTOCOL.md (commit d1ecaab): rank rocky planets without a JWST atmosphere verdict by the
expected information a MIRI 15 um eclipse programme would give about the cosmic shoreline, per JWST hour.

  prior      retention posterior refitted on 23 planets (retention_update.fit), 2,000 resampled draws
  data       each candidate's shoreline-margin draws and magma flag (target_physics, retention_model)
  noise      sigma per visit = s0 * 10^(0.2 (K - 8)) * sqrt(2 h / T14), s0 fitted on 8 published eclipses
  programme  visits for a 3-sigma bare-rock vs thick-atmosphere separation (cap 10), 3*T14 + 1 h each
  value      mutual information (bits) between the noisy verdict and the parameters, per JWST hour

    python stage_c_priority_fusion/shoreline_design.py
Output: outputs/stage_c/shoreline_design.{csv,json}, eda/shoreline_design.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(HERE))
import retention_model as rm  # noqa: E402
import retention_update as ru  # noqa: E402

OUT = RESEARCH / "outputs" / "stage_c"
# per-visit 15 um eclipse uncertainty (ppm): combined sigma x sqrt(visits)
CALIB = {"GJ 357 b": (12.7, 1), "GJ 3473 b": (45, 4), "LHS 1478 b": (56, 1), "TRAPPIST-1 b": (99, 5), "TRAPPIST-1 c": (94, 4),
         "GJ 3929 b": (21, 3), "TOI-1468 b": (31, 3), "LTT 3780 b": (38, 2)}
DDT = ["LTT 1445 A c", "TOI-198 b", "TOI-1452 b", "TOI-771 b", "HD 260655 c", "TOI-244 b"]
H, C, KB, WL = 6.626e-34, 2.998e8, 1.381e-23, 15e-6
RSUN_REARTH, AU_RSUN = 109.1, 215.03


def planck(T):
    return 1.0 / np.expm1(H * C / (WL * KB * np.asarray(T, float)))


def depths(rp_re, rs, teff, a_au):
    k2 = (rp_re / (rs * RSUN_REARTH)) ** 2
    tirr = teff * np.sqrt(rs / (a_au * AU_RSUN))
    bare = k2 * planck(tirr * (2 / 3) ** 0.25) / planck(teff) * 1e6
    atm = k2 * planck(tirr * 0.25 ** 0.25) / planck(teff) * 1e6
    return bare, atm


def bits(q):
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))


def main() -> int:
    ps = pd.read_csv(rm.PSCOMP, low_memory=False)
    # ---- noise model (S1) ----
    cal = ps[ps.pl_name.isin(CALIB)].drop_duplicates("pl_name").set_index("pl_name").loc[list(CALIB)]
    per_visit = np.array([s * np.sqrt(v) for s, v in CALIB.values()])
    shape = 10 ** (0.2 * (cal.sy_kmag.to_numpy() - 8)) * np.sqrt(2.0 / cal.pl_trandur.to_numpy())
    log_s0 = float(np.mean(np.log10(per_visit / shape)))
    resid = np.log10(per_visit / shape) - log_s0
    s0, rms = 10 ** log_s0, float(np.sqrt(np.mean(resid ** 2)))

    # ---- prior: the 23-planet retention posterior ----
    old = ru.labelled(HERE / "rocky_atmosphere_observations.csv")
    new = ru.labelled(HERE / "rocky_atmosphere_observations_new.csv")
    allp = pd.concat([old, new], ignore_index=True)
    m_all, g_all, _ = ru.inputs(allp)
    theta, w, _ = ru.fit(allp.y.to_numpy(), m_all, g_all)
    th = theta[np.random.default_rng(rm.SEED).choice(len(theta), size=2000, p=w)]

    # ---- pool ----
    need = ["sy_kmag", "pl_trandur", "pl_eqt", "pl_rade", "st_rad", "st_teff", "pl_insol", "pl_orbsmax"]
    pool = ps[(ps.tran_flag == 1) & (ps.pl_rade <= 1.8)].dropna(subset=need)
    pool = pool[~pool.pl_name.isin(allp.planet)].drop_duplicates("pl_name").reset_index(drop=True)
    _, draws = rm.planets_with_draws(pool.pl_name)
    margin = draws["margin_primary"]
    teq = np.nanmedian(draws["teq"], axis=1)
    magma = (teq >= rm.MAGMA_TEQ).astype(float)
    p = np.concatenate([rm.prob(th, margin[i:i + 100], magma[i:i + 100]) for i in range(0, len(pool), 100)], axis=1)  # (2000, n)

    bare, atm = depths(pool.pl_rade.to_numpy(), pool.st_rad.to_numpy(), pool.st_teff.to_numpy(), pool.pl_orbsmax.to_numpy())
    delta = bare - atm
    sigma = s0 * 10 ** (0.2 * (pool.sy_kmag.to_numpy() - 8)) * np.sqrt(2.0 / pool.pl_trandur.to_numpy())
    n_vis = np.clip(np.ceil((3 * sigma / np.maximum(delta, 1e-9)) ** 2), 1, 10)
    snr = delta * np.sqrt(n_vis) / sigma
    eps = norm.cdf(-snr / 2)
    hours = n_vis * (3 * pool.pl_trandur.to_numpy() + 1)
    q = eps + (1 - 2 * eps) * p                                               # P(verdict "atmosphere" | theta)
    mi = bits(q.mean(axis=0)) - bits(q).mean(axis=0)
    beta = th[:, 1]
    var0 = beta.var()
    w1, w0 = q / q.sum(axis=0), (1 - q) / (1 - q).sum(axis=0)
    ev = lambda ww: (ww * beta[:, None] ** 2).sum(axis=0) - (ww * beta[:, None]).sum(axis=0) ** 2
    pv1 = q.mean(axis=0)
    dvar_beta = var0 - (pv1 * ev(w1) + (1 - pv1) * ev(w0))
    t = pool[["pl_name", "pl_rade", "pl_eqt", "st_teff", "sy_kmag", "pl_trandur"]].assign(
        magma=magma.astype(bool), p_retain_fitted=p.mean(axis=0), margin_median=np.median(margin, axis=1),
        depth_bare_ppm=bare, depth_atm_ppm=atm, sigma_visit_ppm=sigma, visits=n_vis.astype(int), snr_separation=snr,
        verdict_error=eps, jwst_hours=hours, info_bits=mi, beta_var_reduction=dvar_beta, bits_per_hour=mi / hours,
        hours_range_lo=hours * 10 ** (-2 * rms), hours_range_hi=hours * np.minimum(10 ** (2 * rms), 10 / n_vis))
    t = t.sort_values("bits_per_hour", ascending=False).reset_index(drop=True)
    t["rank"] = np.arange(1, len(t) + 1)
    t.to_csv(OUT / "shoreline_design.csv", index=False)
    top10 = t.head(10)
    ddt = t[t.pl_name.isin(DDT)][["pl_name", "rank", "bits_per_hour", "p_retain_fitted", "visits", "jwst_hours"]]
    res = {"protocol": "stage_c_priority_fusion/SHORELINE_DESIGN_PROTOCOL.md (commit d1ecaab)", "pool_size": int(len(t)),
           "noise_model": {"s0_ppm_at_K8_T14_2h": round(s0, 1), "log_rms_dex": round(rms, 3),
                           "calibration": {k: {"per_visit_ppm": round(float(v), 1), "model_ppm": round(float(s0 * sh), 1)}
                                           for k, v, sh in zip(CALIB, per_visit, shape)}},
           "prior": {"n_planets": int(len(allp)), "beta_sd": round(float(np.sqrt(var0)), 3)},
           "top15": t.head(15)[["rank", "pl_name", "pl_rade", "pl_eqt", "magma", "p_retain_fitted", "visits", "jwst_hours", "info_bits",
                                "beta_var_reduction", "bits_per_hour"]].round(4).to_dict(orient="records"),
           "ddt_targets": ddt.round(4).to_dict(orient="records"), "ddt_median_rank": float(ddt["rank"].median()) if len(ddt) else None,
           "expectations": {"S1 noise model log-rms <= 0.2 dex": bool(rms <= 0.2),
                            "S2 >= 6 of the top 10 have 0.2 <= P(retain) <= 0.8": bool(top10.p_retain_fitted.between(0.2, 0.8).sum() >= 6),
                            "S3 DDT median rank in the top quarter": bool(len(ddt) and ddt["rank"].median() <= 0.25 * len(t))}}
    (OUT / "shoreline_design.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in ["pool_size", "noise_model", "prior", "ddt_targets", "ddt_median_rank", "expectations"]}, indent=2, default=float))
    pd.set_option("display.width", 220)
    print(t.head(20)[["rank", "pl_name", "pl_rade", "pl_eqt", "magma", "p_retain_fitted", "visits", "jwst_hours", "info_bits", "bits_per_hour"]].round(3).to_string(index=False))
    figure(t, res)
    return 0


def figure(t, res):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), width_ratios=[1.25, 1])
    ax = axes[0]
    ok = t.bits_per_hour > 0
    ax.scatter(t[ok].p_retain_fitted, t[ok].bits_per_hour, s=8, color="#b0b0b0", label="rocky planets without a JWST verdict")
    d = t[t.pl_name.isin(DDT)]
    ax.scatter(d.p_retain_fitted, d.bits_per_hour, s=60, marker="D", color="#e8743b", edgecolor="k", label="Rocky Worlds DDT targets")
    top = t.head(10)
    ax.scatter(top.p_retain_fitted, top.bits_per_hour, s=40, color="#2a78d6", edgecolor="k", label="top 10 by bits per hour")
    for _, r in pd.concat([top.head(6), d]).drop_duplicates("pl_name").iterrows():
        ax.annotate(r.pl_name, (r.p_retain_fitted, r.bits_per_hour), fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax.set(yscale="log", xlabel="fitted P(retains an atmosphere)", ylabel="expected information per JWST hour (bits/h)",
           title="Where one more 15 um eclipse programme teaches the most about the shoreline")
    ax.legend(fontsize=8, loc="lower left")
    ax = axes[1]
    h = t.head(15).iloc[::-1]
    ax.barh(h.pl_name, h.bits_per_hour, color=["#e8743b" if n in DDT else "#2a78d6" for n in h.pl_name])
    ax.set(xlabel="bits per JWST hour", title=f"Top 15 of {len(t)} (orange: DDT targets)")
    fig.tight_layout()
    fig.savefig(RESEARCH / "eda" / "shoreline_design.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())
