"""
retrieval_trappist1e_real.py
============================

Bayesian retrievals on the REAL JWST transmission spectra of TRAPPIST-1 e. These are the 4 NIRSpec
PRISM visits of Espinoza et al. 2025, as published in the NASA Exoplanet Archive. The question
is whether the data already constrain the habitability-relevant gases that the forecast asks about.

Runs in the petitRADTRANS environment:
    ~/micromamba/envs/prt/bin/python stage_b_atmospheres/retrieval_trappist1e_real.py --workers 4
Progress: python tools/retrieval_progress.py

Data
  * The 4 visits as published (67 points each, 0.63-5.21 um), with their own uncertainties. The
    model is binned onto each point's wavelength bin (bin width from the point spacing).
  * Points below 0.6 um are dropped (the forward model's lower bound).

Model, shared by all visits: an isothermal N2 atmosphere (forward_model, R = 300 opacities)
  * Free: log10 VMR of CO2, CH4 and H2O ~ U(-10, -0.5); T ~ U(150, 400) K;
    log10 P_surface ~ U(-3, 1) bar; R_p ~ U(0.85, 0.99) R_earth.
  * Mass and stellar radius are fixed (pscomppars).

Stellar contamination, separate for EACH visit, because TRAPPIST-1's contamination changes between
visits (Espinoza et al. 2025; Piaulet-Ghorayeb et al. 2025)
  * depth x 1 / (1 - f_i (1 - F(T_i) / F(T_phot))), with PHOENIX spectra; T_phot = 2566 K.
  * f_i ~ U(0, 1), T_i ~ U(2002, 3600) K. (f <= 0.5 pinned visit 3 at the bound in a first attempt.) A single population per visit can be spots or
    faculae. That is a simplification of reality, where both are present at once.

Contamination as fitted (default, LINEAR = True): for each visit, depth x (1 + a S_spot + b S_facula) + c,
  where S = 1 - F_het/F_phot are PHOENIX templates for cool spots (2200 K, about 0.86 T_phot) and hot
  faculae (T_phot + 150 K), and c is a flux offset. Priors are a, b ~ N(0, 0.5^2) and c ~ N(0, 500 ppm^2).
  These per-visit nuisances enter linearly, so they are integrated out ANALYTICALLY (a Gaussian
  marginal likelihood). The sampler explores only the planet. It is the first-order version of the
  per-visit (f_i, T_i) model below, and the same treatment as the forecast's starspot term. The full
  non-linear 14-parameter version was abandoned: at ~16 likelihood calls/s it needed tens of hours.
  A bare-rock fit with it is kept in outputs/retrieval_real_nonlinear_partial/.

Models compared (dynesty, Bayesian evidence)
  full        atmosphere with CO2, CH4 and H2O, plus contamination                         (6 sampled)
  no_CH4      the same without CH4                                                         (5)
  no_CO2      the same without CO2                                                         (5)
  bare_rock   a flat planetary spectrum (no atmosphere), plus contamination                 (1)
  archean     CO2 1e-2, CH4 1e-3 and H2O 1e-3 fixed (the forecast's scenario); T, P, R_p free  (3)
Read-outs:
  * atmosphere vs bare rock;
  * CH4 and CO2 Bayes factors and 95 % upper limits;
  * whether an Archean-like atmosphere is DISFAVOURED relative to a bare rock (Bayes factor < 1),
    which is the direct test of the forecast.

Output: outputs/retrieval_real/{results.json, posteriors/*.npz}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(HERE))
import retrieval_trappist1e as base  # noqa: E402

# Which TRAPPIST-1 planet. Read from the environment so the sampler's worker processes inherit it.
# TRAPPIST-1 e is the science target. b, c and d are EMPIRICAL NULL TESTS: same star, same instrument
# mode, and JWST finds no thick atmosphere on b and c (thermal emission) and a flat spectrum on d.
# Planet parameters are pscomppars values.
import os  # noqa: E402

# Shortest wavelength used (um). The default 0.6 is PRISM's blue end. RETRIEVAL_WL_MIN=3.0 keeps only
# the 3-5.2 um range (CH4 3.3, CO2 4.3), where unocculted spots and faculae matter least: the null
# test on airless TRAPPIST-1 b showed the two-template contamination model fails at short wavelengths.
WL_MIN = float(os.environ.get("RETRIEVAL_WL_MIN", "0.6"))
# Optional subset of visits, 1-based, e.g. RETRIEVAL_VISITS=1,2: a real atmosphere is common to every
# transit, whereas stellar contamination changes from visit to visit.
VISITS = [int(v) for v in os.environ.get("RETRIEVAL_VISITS", "").split(",") if v.strip()]

PLANETS = {"TRAPPIST-1 b": {"rp": 1.116, "mp": 1.374, "teq": 397.6},
           "TRAPPIST-1 c": {"rp": 1.097, "mp": 1.308, "teq": 339.7},
           "TRAPPIST-1 d": {"rp": 0.788, "mp": 0.388, "teq": 286.2},
           "TRAPPIST-1 e": {"rp": 0.920, "mp": 0.692, "teq": 249.7}}
PLANET_NAME = os.environ.get("RETRIEVAL_PLANET", "TRAPPIST-1 e")
PL = PLANETS[PLANET_NAME]
if PLANET_NAME == "TRAPPIST-1 e":
    RP_RANGE, T_RANGE = base.RP_RANGE, base.T_RANGE
else:                                   # the same relative widths as for e
    RP_RANGE = (PL["rp"] * base.RP_RANGE[0] / 0.92, PL["rp"] * base.RP_RANGE[1] / 0.92)
    T_RANGE = (base.T_RANGE[0], max(base.T_RANGE[1], 1.5 * PL["teq"]))
OUT = RESEARCH / "outputs" / (("retrieval_real" if PLANET_NAME == "TRAPPIST-1 e"
                               else "retrieval_real_" + PLANET_NAME.lower().replace(" ", "").replace("-", ""))
                              + ("" if WL_MIN == 0.6 else f"_wl{WL_MIN:g}")
                              + ("" if not VISITS else "_v" + "".join(map(str, VISITS))))
PROG = OUT / "progress.json"
N_VISITS = 4
ARCHEAN = {"CO2": 1e-2, "CH4": 1e-3, "H2O": 1e-3}
MODELS = {"full": ("CO2", "CH4", "H2O"), "no_CH4": ("CO2", "H2O"), "no_CO2": ("CH4", "H2O"),
          "bare_rock": (), "archean": "fixed"}
NLIVE, DLOGZ = 400, 0.3
# Covering fraction may take its full physical range here: a first run with f <= 0.5 pinned visit 3
# at the bound, and a nuisance that cannot absorb the contamination could fake planetary features.
F_RANGE = (0.0, 1.0)
_DATA = {}


# ------------------------------------------------------------------------------------------------ #
def load_visits() -> dict:
    """The 4 published visits on a common wavelength grid: wl, edges, depth (4, n), sigma (4, n)."""
    if not _DATA:
        import stage_b_pipeline as sbp

        sel = sbp.select_spectra([PLANET_NAME])
        sel = sel[sel.instrument.astype(str).str.contains("PRISM")].sort_values("file")
        if VISITS:
            sel = sel.iloc[[v - 1 for v in VISITS]]
        depths, sigmas, wl = [], [], None
        for r in sel.itertuples():
            df, _ = sbp.parse_tbl(sbp.SPEC_DIR / "spectra_tbl" / r.file)
            df = df[df.wl >= WL_MIN].sort_values("wl").reset_index(drop=True)
            if wl is None:
                wl = df.wl.to_numpy()
            assert np.allclose(df.wl.to_numpy(), wl), "visits are not on a common wavelength grid"
            depths.append(df.depth_ppm.to_numpy())
            sigmas.append(df.err_ppm.to_numpy())
        mid = 0.5 * (wl[1:] + wl[:-1])
        edges = np.r_[wl[0] - (mid[0] - wl[0]), mid, wl[-1] + (wl[-1] - mid[-1])]
        _DATA.update(wl=wl, edges=edges, depth=np.array(depths), sigma=np.array(sigmas))
    return _DATA


_PHX = {}


def phoenix_on_data():
    if not _PHX:
        import os
        from petitRADTRANS.config import petitradtrans_config_parser as cfg
        cfg.set_input_data_path(os.path.expanduser("~/petitRADTRANS/input_data"))
        from petitRADTRANS.stellar_spectra.phoenix import PhoenixStarTable
        t = PhoenixStarTable()
        t.load()
        e = load_visits()["edges"]
        grid = np.arange(base.THET_RANGE[0], base.THET_RANGE[1] + 1, 20.0)
        rows = []
        for T in list(grid) + [base.T_PHOT]:
            spec, _ = t.compute_spectrum(T)
            idx = np.digitize(spec[:, 0] * 1e4, e) - 1
            rows.append([spec[idx == b, 1].mean() for b in range(len(e) - 1)])
        _PHX.update(grid=grid, flux=np.array(rows[:-1]), phot=np.array(rows[-1]))
    return _PHX


def tls(f, t_het):
    p = phoenix_on_data()
    i = np.clip(np.searchsorted(p["grid"], t_het) - 1, 0, len(p["grid"]) - 2)
    w = (t_het - p["grid"][i]) / (p["grid"][i + 1] - p["grid"][i])
    return 1.0 / (1.0 - f * (1.0 - ((1 - w) * p["flux"][i] + w * p["flux"][i + 1]) / p["phot"]))


# ---- linear, analytically marginalised contamination (the default; see the module docstring) ----
SPOT_T, FACULA_DT = 2200.0, 150.0          # ~0.86 T_phot (Rackham et al. 2018), and T_phot + 150 K
PRIOR_SD = {"spot": 0.5, "facula": 0.5, "offset_ppm": 500.0}
_TPL = {}


def templates():
    """(1 - F_het/F_phot) for a cool-spot and a hot-facula population on the data bins."""
    if not _TPL:
        p = phoenix_on_data()
        def flux(T):
            i = np.clip(np.searchsorted(p["grid"], T) - 1, 0, len(p["grid"]) - 2)
            w = (T - p["grid"][i]) / (p["grid"][i + 1] - p["grid"][i])
            return (1 - w) * p["flux"][i] + w * p["flux"][i + 1]
        _TPL["spot"] = 1 - flux(SPOT_T) / p["phot"]
        _TPL["facula"] = 1 - flux(base.T_PHOT + FACULA_DT) / p["phot"]
    return _TPL


def marginal_loglike(depth: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> float:
    """log N(y | depth + B theta, diag sigma^2) integrated over theta ~ N(0, diag lam), where
    B = [depth * spot, depth * facula, 1]: first-order contamination plus a flux offset."""
    t = templates()
    ok = np.isfinite(depth) & np.isfinite(y)
    B = np.column_stack([depth[ok] * t["spot"][ok], depth[ok] * t["facula"][ok], np.ones(ok.sum())])
    lam = np.array([PRIOR_SD["spot"], PRIOR_SD["facula"], PRIOR_SD["offset_ppm"]]) ** 2
    C = np.diag(sigma[ok] ** 2) + (B * lam) @ B.T
    r = y[ok] - depth[ok]
    sign, logdet = np.linalg.slogdet(C)
    return float(-0.5 * (r @ np.linalg.solve(C, r) + logdet + ok.sum() * np.log(2 * np.pi)))


def planet_depth(gases: dict, T: float, logp: float, rp: float) -> np.ndarray:
    import forward_model as fm
    d = load_visits()
    if not gases:                                     # bare rock: an opaque disc, no atmosphere
        return np.full(len(d["wl"]), (rp * fm.R_EARTH_CM / (base.PLANET["rs"] * fm.R_SUN_CM)) ** 2 * 1e6)
    wl, dep = fm.transmission_spectrum(gases, "N2", T, rp, PL["mp"], base.PLANET["rs"], surface_pressure=10 ** logp,
                                       wl_range=(WL_MIN - 0.05, 5.3), species=base.SPECIES, resolution=base.OPACITY_R)
    binned = fm.bin_spectrum(wl, dep, d["edges"])
    # Data bins narrower than the model's own sampling (TRAPPIST-1 b/c at the red end) receive no model
    # point. They must get the interpolated model, NOT be dropped: dropping them silently removed 68
    # of 362 points from the atmosphere models only, and inflated ln B by ~500 (bug found 2026-10-18).
    empty = ~np.isfinite(binned)
    if empty.any():
        binned[empty] = np.interp(d["wl"][empty], wl, dep)
    return binned


# --- parameter layout: [gas log VMRs] [T, logP] [R_p] [f_1, T_1, ..., f_4, T_4] (T, logP absent for bare rock) ---
def layout(model: str):
    gases = MODELS[model]
    n_gas = 0 if gases == "fixed" else len(gases)
    has_atm = model != "bare_rock"
    return n_gas, has_atm


def prior_transform(u, model):
    x = np.array(u, float)
    n_gas, has_atm = layout(model)
    x[:n_gas] = base.LOGX[0] + (base.LOGX[1] - base.LOGX[0]) * u[:n_gas]
    k = n_gas
    if has_atm:
        x[k] = T_RANGE[0] + (T_RANGE[1] - T_RANGE[0]) * u[k]
        x[k + 1] = base.LOGP_RANGE[0] + (base.LOGP_RANGE[1] - base.LOGP_RANGE[0]) * u[k + 1]
        k += 2
    x[k] = RP_RANGE[0] + (RP_RANGE[1] - RP_RANGE[0]) * u[k]
    k += 1
    for v in range(0 if LINEAR else N_VISITS):
        x[k + 2 * v] = F_RANGE[0] + (F_RANGE[1] - F_RANGE[0]) * u[k + 2 * v]
        x[k + 2 * v + 1] = base.THET_RANGE[0] + (base.THET_RANGE[1] - base.THET_RANGE[0]) * u[k + 2 * v + 1]
    return x


def log_likelihood(x, model):
    d = load_visits()
    n_gas, has_atm = layout(model)
    gases = MODELS[model]
    if gases == "fixed":
        vmr = dict(ARCHEAN)
    else:
        vmr = {g: 10 ** x[i] for i, g in enumerate(gases)}
        if sum(vmr.values()) >= 0.99:
            return -1e100
    k = n_gas
    T, logp = (x[k], x[k + 1]) if has_atm else (0.0, 0.0)
    k += 2 if has_atm else 0
    rp = x[k]
    k += 1
    try:
        depth = planet_depth(vmr if has_atm else {}, T, logp, rp)
    except Exception:
        return -1e100
    if not np.all(np.isfinite(depth)):
        raise ValueError("planet model is undefined at some data points; every model must see the same data")
    if LINEAR:
        return sum(marginal_loglike(depth, d["depth"][v], d["sigma"][v]) for v in range(len(d["depth"])))
    ll = 0.0
    for v in range(N_VISITS):
        m = depth * tls(x[k + 2 * v], x[k + 2 * v + 1])
        ok = np.isfinite(m)
        ll += -0.5 * np.sum(((m[ok] - d["depth"][v][ok]) / d["sigma"][v][ok]) ** 2)
    return float(ll)


LINEAR = True        # set False for the full non-linear per-visit (f, T) parameters (14-D; very slow)


def ndim(model):
    n_gas, has_atm = layout(model)
    return n_gas + (2 if has_atm else 0) + 1 + (0 if LINEAR else 2 * N_VISITS)


def names(model):
    n_gas, has_atm = layout(model)
    g = [] if MODELS[model] == "fixed" else [f"log_{x}" for x in MODELS[model]]
    return g + (["T", "logP"] if has_atm else []) + ["Rp"] + ([] if LINEAR else [f"{p}_{v + 1}" for v in range(N_VISITS) for p in ("f", "Thet")])


def write_progress(**kw):
    p = json.loads(PROG.read_text()) if PROG.is_file() else {}
    p.update(kw)
    p["updated"] = datetime.now().isoformat(timespec="seconds")
    PROG.write_text(json.dumps(p, indent=1))


def run(model, workers, nlive):
    import dynesty
    from dynesty.pool import Pool

    t0, last = time.time(), [0.0]

    def print_fn(results, niter, ncall, *a, **k):
        if time.time() - last[0] > 20:
            last[0] = time.time()
            try:
                dz = getattr(results, "delta_logz", None)
                dz = results[-1] if dz is None else dz
                write_progress(current=model, current_ncall=int(ncall), current_seconds=round(time.time() - t0),
                               current_dlogz=None if dz is None else round(float(dz), 3))
            except Exception:
                pass

    with Pool(workers, log_likelihood, prior_transform, logl_args=(model,), ptform_args=(model,)) as pool:
        s = dynesty.NestedSampler(pool.loglike, pool.prior_transform, ndim(model), nlive=nlive, pool=pool,
                                  queue_size=workers, sample="unif" if LINEAR else "rslice", rstate=np.random.default_rng(3))
        s.run_nested(dlogz=DLOGZ, print_progress=True, print_func=print_fn)
    r = s.results
    w = np.exp(r.logwt - r.logz[-1])
    nm = names(model)
    np.savez_compressed(OUT / "posteriors" / f"{model}.npz", samples=r.samples, weights=w, logz=r.logz[-1], names=np.array(nm))
    post = {}
    for i, n in enumerate(nm):
        o = np.argsort(r.samples[:, i])
        c = np.cumsum(w[o])
        post[n] = [float(np.interp(q, c, r.samples[o, i])) for q in (0.025, 0.16, 0.5, 0.84, 0.975)]
    best = r.samples[np.argmax(r.logl)]
    return {"model": model, "logz": float(r.logz[-1]), "logz_err": float(r.logzerr[-1]), "ncall": int(np.sum(r.ncall)),
            "seconds": round(time.time() - t0), "max_loglike": float(np.max(r.logl)),
            "posterior_2.5_16_50_84_97.5": post, "best_fit": dict(zip(nm, map(float, best)))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nlive", type=int, default=NLIVE)
    ap.add_argument("--models", nargs="+", default=["bare_rock", "full", "archean", "no_CH4", "no_CO2"])
    a = ap.parse_args()
    (OUT / "posteriors").mkdir(parents=True, exist_ok=True)
    d = load_visits()
    res_path = OUT / "results.json"
    done = json.loads(res_path.read_text())["runs"] if res_path.is_file() else []
    have = {x["model"] for x in done}
    write_progress(total=len(a.models), completed=len(have), plan=a.models, planet=PLANET_NAME,
                   data={"visits": int(len(d["depth"])), "points_per_visit": int(len(d["wl"])), "wl_um": [float(d["wl"][0]), float(d["wl"][-1])],
                         "median_sigma_ppm": float(np.median(d["sigma"]))})
    for m in a.models:
        if m in have:
            continue
        print(f"=== {m} ({ndim(m)} parameters) ===", flush=True)
        out = run(m, a.workers, a.nlive)
        done.append(out)
        res_path.write_text(json.dumps({"runs": done, "n_points": int(d["depth"].size)}, indent=1))
        write_progress(completed=len(done), last_finished=m)
        print(f"    lnZ = {out['logz']:.2f} +- {out['logz_err']:.2f}; max lnL {out['max_loglike']:.1f}; {out['seconds']} s", flush=True)
    write_progress(current="done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
