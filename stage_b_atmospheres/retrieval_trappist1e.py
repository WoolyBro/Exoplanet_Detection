"""
retrieval_trappist1e.py
=======================

Full Bayesian retrievals for TRAPPIST-1 e, replacing the forecast's known-template (matched-filter)
detection with what an observer would actually do: fit the spectrum with free parameters and test
each gas by Bayesian model comparison.

Runs in the petitRADTRANS environment (needs dynesty):
    ~/micromamba/envs/prt/bin/python stage_b_atmospheres/retrieval_trappist1e.py --workers 3
Progress: python tools/retrieval_progress.py

Simulated observation
  * Truth: the forecast's Archean scenario on TRAPPIST-1 e. N2 background with CO2 1e-2, CH4 1e-3 and
    H2O 1e-3, isothermal at T_eq = 250 K, opaque surface at 1 bar, R_p = 0.920 R_earth,
    M_p = 0.692 M_earth, R_s = 0.1192 R_sun.
  * Instrument: NIRSpec PRISM, 0.95-5.3 um, R = 100 bins. Per-bin noise for N transits comes from
    noise_model.py (calibrated on real single-transit JWST spectra, median bias removed) / sqrt(N).
  * Data are the noise-free expected spectrum with those error bars. This is the standard way to
    forecast an *expected* Bayes factor, without one random draw deciding the answer. Three
    noise-realisation retrievals at one N show the scatter around it.
  * Forward model: forward_model.transmission_spectrum, with correlated-k opacities binned to R = 300
    (exo_k), the same for the simulated data and the retrieval. Against R = 1000 this changes the
    gas signals by < 0.2 % and the spectrum by <= 6 ppm (rms 1.3), at 2.3x the speed. R = 100 was
    rejected: it differed by up to 20 ppm.

Retrieval
  * Free parameters:
      - log10 VMR of CO2, CH4 and H2O, each U(-10, -0.5), with the sum required to be < 1;
      - T ~ U(150, 400) K;
      - log10 P_surface ~ U(-3, 1) bar;
      - R_p ~ U(0.85, 0.99) R_earth, the radius at the surface, which also absorbs any absolute offset.
  * The background (N2) and the mass are fixed. A real analysis of TRAPPIST-1 e would also free the
    background, and add stellar contamination; both would raise the transit counts (see the write-up).
  * Sampler: dynesty static nested sampling, multi-ellipsoid uniform sampling, 250 live points, dlogz = 0.1.
  * Detection of gas X: lnB = lnZ(full) - lnZ(model without X), converted to a "sigma" with the
    Benneke & Seager (2013, ApJ 778, 153) bound, B = -1/(e p ln p).

With --tls (stellar contamination, the transit light-source effect, Rackham et al. 2018):
  * TRAPPIST-1's JWST spectra are dominated by unocculted spots/faculae (200-300 ppm, varying between
    visits; Piaulet-Ghorayeb et al. 2025). The data are multiplied by
        eps(lambda) = 1 / (1 - f (1 - F_het(lambda) / F_phot(lambda)))
    with PHOENIX spectra (Husser et al. 2013, petitRADTRANS's table) binned to the data bins.
  * Injected: one cool-spot population with f = 0.08 and T_het = 2300 K (photosphere 2566 K). That is a
    ~250 ppm signal, matching the observed amplitude.
  * Retrieved: two extra free parameters, f ~ U(0, 0.5) and T_het ~ U(2002, 3600) K, the latter covering
    both spots and faculae.
  * The contamination is held fixed across the stacked visits. It is not in reality, so this is still
    optimistic.
  * Output goes to outputs/retrieval_tls/.

Output: outputs/retrieval/{results.json, retrievals.csv, posteriors/*.npz}, eda/retrieval_trappist1e.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(HERE))

OUT = RESEARCH / "outputs" / "retrieval"
PROG = OUT / "progress.json"
TRUTH = {"CO2": 1e-2, "CH4": 1e-3, "H2O": 1e-3}
PLANET = {"T": 249.7, "rp": 0.920, "mp": 0.692, "rs": 0.1192}       # Stage C table, pscomppars
STAR = {"jmag": 11.354, "teff": 2566.0, "t14_h": 0.9293}
SPECIES = ("CH4", "CO2", "H2O")
WL = (0.95, 5.3)
LOGX = (-10.0, -0.5)
T_RANGE, LOGP_RANGE, RP_RANGE = (150.0, 400.0), (-3.0, 1.0), (0.85, 0.99)
N_TRANSITS = (2, 4, 8, 16)
MODELS = {"full": ("CO2", "CH4", "H2O"), "no_CH4": ("CO2", "H2O"), "no_CO2": ("CH4", "H2O")}
NLIVE, DLOGZ = 250, 0.1
TLS_TRUTH = {"f": 0.08, "T_het": 2300.0}
T_PHOT, F_RANGE, THET_RANGE = 2566.0, (0.0, 0.5), (2002.0, 3600.0)
OPACITY_R = 300            # correlated-k resolving power for truth AND retrieval (3x the R = 100 data bins)


# ----------------------------------------------------------------------------------------------- #
def edges():
    import detectability_forecast as df
    return df.r_edges(*WL)


def model_binned(gases: dict, T: float, logp: float, rp: float) -> np.ndarray:
    import forward_model as fm
    wl, dep = fm.transmission_spectrum(gases, "N2", T, rp, PLANET["mp"], PLANET["rs"], surface_pressure=10 ** logp,
                                       wl_range=WL, species=SPECIES, resolution=OPACITY_R)
    return fm.bin_spectrum(wl, dep, edges())


_PHX = {}


def phoenix_binned():
    """PHOENIX F_nu on the data bins, on a 20 K grid (for fast interpolation inside the likelihood)."""
    if not _PHX:
        import os
        from petitRADTRANS.config import petitradtrans_config_parser as cfg
        cfg.set_input_data_path(os.path.expanduser("~/petitRADTRANS/input_data"))
        from petitRADTRANS.stellar_spectra.phoenix import PhoenixStarTable
        t = PhoenixStarTable()
        t.load()
        e = edges()
        grid = np.arange(THET_RANGE[0], THET_RANGE[1] + 1, 20.0)
        rows = []
        for T in list(grid) + [T_PHOT]:
            spec, _ = t.compute_spectrum(T)
            idx = np.digitize(spec[:, 0] * 1e4, e) - 1
            rows.append([spec[idx == b, 1].mean() for b in range(len(e) - 1)])
        _PHX.update(grid=grid, flux=np.array(rows[:-1]), phot=np.array(rows[-1]))
    return _PHX


def tls_factor(f: float, t_het: float) -> np.ndarray:
    p = phoenix_binned()
    i = np.clip(np.searchsorted(p["grid"], t_het) - 1, 0, len(p["grid"]) - 2)
    w = (t_het - p["grid"][i]) / (p["grid"][i + 1] - p["grid"][i])
    het = (1 - w) * p["flux"][i] + w * p["flux"][i + 1]
    return 1.0 / (1.0 - f * (1.0 - het / p["phot"]))


def sigma_per_transit() -> np.ndarray:
    import noise_model as nm
    calib = json.loads((RESEARCH / "outputs" / "forecast" / "noise_calibration.json").read_text())["modes"]["NIRSpec_PRISM"]
    e = edges()
    c, w = 0.5 * (e[1:] + e[:-1]), np.diff(e)
    return nm.predict_sigma_ppm(calib, c, w, STAR["jmag"], STAR["teff"], STAR["t14_h"]) / 10 ** calib["loso"]["median_log10_ratio"]


# --- likelihood and prior: module-level so the dynesty pool can pickle them (macOS spawns workers) ---
def prior_transform(u, gases, tls=False):
    x = np.array(u, float)
    k = len(gases)
    if tls:
        x[k + 3] = F_RANGE[0] + (F_RANGE[1] - F_RANGE[0]) * u[k + 3]
        x[k + 4] = THET_RANGE[0] + (THET_RANGE[1] - THET_RANGE[0]) * u[k + 4]
    x[:k] = LOGX[0] + (LOGX[1] - LOGX[0]) * u[:k]
    x[k] = T_RANGE[0] + (T_RANGE[1] - T_RANGE[0]) * u[k]
    x[k + 1] = LOGP_RANGE[0] + (LOGP_RANGE[1] - LOGP_RANGE[0]) * u[k + 1]
    x[k + 2] = RP_RANGE[0] + (RP_RANGE[1] - RP_RANGE[0]) * u[k + 2]
    return x


def log_likelihood(x, gases, data, sigma, tls=False):
    k = len(gases)
    vmr = {g: 10 ** x[i] for i, g in enumerate(gases)}
    if sum(vmr.values()) >= 0.99:
        return -1e100
    try:
        m = model_binned(vmr, x[k], x[k + 1], x[k + 2])
        if tls:
            m = m * tls_factor(x[k + 3], x[k + 4])
    except Exception:
        return -1e100
    ok = np.isfinite(m) & np.isfinite(data)
    return float(-0.5 * np.sum(((m[ok] - data[ok]) / sigma[ok]) ** 2))


# ----------------------------------------------------------------------------------------------- #
def bayes_sigma(ln_b: float) -> float:
    """Benneke & Seager 2013: the frequentist significance equivalent to a Bayes factor B,
    from B = -1 / (e p ln p), valid for p < 1/e. Returns 0 for B <= 1."""
    from scipy.optimize import brentq
    from scipy.special import erfcinv

    if ln_b <= 0:
        return 0.0
    f = lambda lp: -1.0 / (np.e * np.exp(lp) * lp) - np.exp(ln_b)      # lp = ln p
    if f(-1.0) > 0:          # B below the bound's floor (~2.5 sigma at most 'weak')
        return float(np.sqrt(2) * erfcinv(np.exp(-1.0)))
    lp = brentq(f, -700, -1.0)
    return float(np.sqrt(2) * erfcinv(np.exp(lp)))


def write_progress(**kw):
    p = json.loads(PROG.read_text()) if PROG.is_file() else {}
    p.update(kw)
    p["updated"] = datetime.now().isoformat(timespec="seconds")
    PROG.write_text(json.dumps(p, indent=1))


def run_one(name: str, gases: tuple, data: np.ndarray, sigma: np.ndarray, workers: int, seed: int, tag: str,
            nlive: int = NLIVE, tls: bool = False) -> dict:
    import dynesty
    from dynesty.pool import Pool

    ndim = len(gases) + 3 + (2 if tls else 0)
    names = [f"log_{g}" for g in gases] + ["T", "logP", "Rp"] + (["f_het", "T_het"] if tls else [])
    t0 = time.time()
    last = [0.0]

    def print_fn(results, niter, ncall, add_live_it=None, dlogz=None, stop_val=None, nbatch=None, logl_min=-np.inf,
                 logl_max=np.inf, **kwargs):
        if time.time() - last[0] > 20:                   # progress file only; must never stop the run
            last[0] = time.time()
            try:
                dz = getattr(results, "delta_logz", None)
                dz = results[-1] if dz is None else dz
                write_progress(current=tag, current_ncall=int(ncall), current_seconds=round(time.time() - t0),
                               current_dlogz=None if dz is None else round(float(dz), 3))
            except Exception:
                pass

    with Pool(workers, log_likelihood, prior_transform, logl_args=(gases, data, sigma, tls), ptform_args=(gases, tls)) as pool:
        s = dynesty.NestedSampler(pool.loglike, pool.prior_transform, ndim, nlive=nlive, pool=pool,
                                  queue_size=workers, rstate=np.random.default_rng(seed), sample="unif")
        s.run_nested(dlogz=DLOGZ, print_progress=True, print_func=print_fn)
    r = s.results
    w = np.exp(r.logwt - r.logz[-1])
    np.savez_compressed(OUT / "posteriors" / f"{tag}.npz", samples=r.samples, weights=w, logz=r.logz[-1],
                        names=np.array(names))
    post = {}
    for i, n in enumerate(names):
        order = np.argsort(r.samples[:, i])
        cdf = np.cumsum(w[order])
        post[n] = [float(np.interp(q, cdf, r.samples[order, i])) for q in (0.16, 0.5, 0.84)]
    return {"model": name, "logz": float(r.logz[-1]), "logz_err": float(r.logzerr[-1]), "ncall": int(np.sum(r.ncall)),
            "seconds": round(time.time() - t0), "posterior_16_50_84": post}


def matched_filter_snr1(sigma: np.ndarray) -> dict:
    """The forecast's statistic on the same data grid, for direct comparison."""
    full = model_binned(TRUTH, PLANET["T"], 0.0, PLANET["rp"])
    out = {}
    for g in ("CO2", "CH4"):
        s = full - model_binned({k: v for k, v in TRUTH.items() if k != g}, PLANET["T"], 0.0, PLANET["rp"])
        ok = np.isfinite(s)
        w = 1 / sigma[ok] ** 2
        s = s[ok] - np.sum(s[ok] * w) / np.sum(w)
        out[g] = float(np.sqrt(np.sum((s / sigma[ok]) ** 2)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--realisations", type=int, default=3, help="noisy retrievals at N = --noisy-n")
    ap.add_argument("--noisy-n", type=int, default=8)
    ap.add_argument("--nlive", type=int, default=NLIVE)
    ap.add_argument("--tls", action="store_true", help="inject and retrieve stellar contamination")
    ap.add_argument("--n-transits", type=int, nargs="+", default=list(N_TRANSITS))
    a = ap.parse_args()
    global OUT, PROG
    if a.tls:
        OUT, PROG = OUT.parent / "retrieval_tls", OUT.parent / "retrieval_tls" / "progress.json"
    (OUT / "posteriors").mkdir(parents=True, exist_ok=True)
    sigma1 = sigma_per_transit()
    truth = model_binned(TRUTH, PLANET["T"], 0.0, PLANET["rp"])
    if a.tls:
        truth = truth * tls_factor(TLS_TRUTH["f"], TLS_TRUTH["T_het"])
    mf = matched_filter_snr1(sigma1)
    plan = [(n, m, None) for n in a.n_transits for m in MODELS] + \
           [(a.noisy_n, m, k) for k in range(a.realisations) for m in MODELS]
    res_path = OUT / "results.json"
    done = json.loads(res_path.read_text())["runs"] if res_path.is_file() else []
    have = {(d["n_transits"], d["model"], d.get("realisation")) for d in done}
    write_progress(total=len(plan), completed=len(have), started=datetime.now().isoformat(timespec="seconds"),
                   plan=[f"N={n} {m}" + (f" noise#{k}" if k is not None else "") for n, m, k in plan])
    for n, m, k in plan:
        if (n, m, k) in have:
            continue
        sigma = sigma1 / np.sqrt(n)
        data = truth.copy()
        if k is not None:
            data = truth + np.random.default_rng(1000 + k).normal(0, 1, len(truth)) * sigma
        tag = f"N{n}_{m}" + (f"_noise{k}" if k is not None else "")
        print(f"=== {tag} ===", flush=True)
        out = run_one(m, MODELS[m], data, sigma, a.workers, seed=7 + n, tag=tag, nlive=a.nlive, tls=a.tls)
        out.update(n_transits=n, realisation=k)
        done.append(out)
        res_path.write_text(json.dumps({"runs": done, "matched_filter_snr_1_transit": mf, "tls": a.tls,
                                        "sigma_per_transit_median_ppm": float(np.nanmedian(sigma1))}, indent=1))
        write_progress(completed=len(done), last_finished=tag, last_seconds=out["seconds"])
        print(f"    lnZ = {out['logz']:.2f} +- {out['logz_err']:.2f}, {out['ncall']} calls, {out['seconds']} s", flush=True)
    summarise(done, mf, OUT)
    write_progress(current="done")
    return 0


def summarise(done: list[dict], mf: dict, out: Path = OUT):
    import csv

    rows = []
    groups = {}
    for d in done:
        groups.setdefault((d["n_transits"], d.get("realisation")), {})[d["model"]] = d
    for (n, k), g in sorted(groups.items(), key=lambda t: (t[0][0], -1 if t[0][1] is None else t[0][1])):
        if "full" not in g:
            continue
        for gas, red in (("CH4", "no_CH4"), ("CO2", "no_CO2")):
            if red not in g:
                continue
            ln_b = g["full"]["logz"] - g[red]["logz"]
            err = float(np.hypot(g["full"]["logz_err"], g[red]["logz_err"]))
            rows.append({"n_transits": n, "realisation": "expected" if k is None else k, "gas": gas,
                         "lnB": round(ln_b, 2), "lnB_err": round(err, 2), "sigma_retrieval": round(bayes_sigma(ln_b), 2),
                         "sigma_matched_filter": round(mf[gas] * np.sqrt(n), 2)})
    with (out / "retrievals.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    raise SystemExit(main())
