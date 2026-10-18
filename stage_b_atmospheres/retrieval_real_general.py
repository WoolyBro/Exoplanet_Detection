"""
retrieval_real_general.py
=========================

Contamination-aware Bayesian retrievals of real multi-visit JWST transmission spectra for any target
(RETRIEVAL_HZ_PROTOCOL.md). It generalises retrieval_trappist1e_real.py: each spectrum may have its own
wavelength grid and instrument, and gets its own linear stellar-contamination terms (PHOENIX spot and facula
templates at the host's T_eff) and offset, marginalised analytically.

    TARGET="TRAPPIST-1 f" ~/micromamba/envs/prt/bin/python stage_b_atmospheres/retrieval_real_general.py --workers 3
    TARGET="LHS 1140 b"   ~/micromamba/envs/prt/bin/python stage_b_atmospheres/retrieval_real_general.py --workers 3
    ... --report        # summary (ln B, per-spectrum chi^2) and figure, after the runs
Output: outputs/retrieval_<target>/{results.json, summary.json, posteriors/}, eda/retrieval_<target>.png
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

TARGET = os.environ.get("TARGET", "TRAPPIST-1 f")
PACK = RESEARCH / "exoplanet_research_data" / "04_atmospheric_spectra" / "spectra_tbl"
REFRESH = sorted((RESEARCH / "external_data").glob("archive_spectra_*"))[-1] / "spectra_tbl"
CONFIG = {
    "TRAPPIST-1 f": {"files": [REFRESH / f"TRAPPIST-1_f_Transmission_TRAPPIST_1_f_3.11991_6157_{i}.tbl" for i in range(1, 6)],
                     "labels": [f"NIRISS visit {i}" for i in range(1, 6)]},
    "LHS 1140 b": {"files": [PACK / "LHS_1140_b_Transmission_LHS_1140_b_3.12019_5422_1.tbl",
                             PACK / "LHS_1140_b_Transmission_LHS_1140_b_3.12019_5435_1.tbl",
                             PACK / "LHS_1140_b_Transmission_LHS_1140_b_3.12019_5435_2.tbl"],
                   "labels": ["NIRISS SOSS (Cadieux 2024)", "NIRSpec G235H (Damiano 2024)", "NIRSpec G395H (Damiano 2024)"]},
}[TARGET]
SLUG = TARGET.lower().replace(" ", "").replace("-", "")
OUT = RESEARCH / "outputs" / f"retrieval_{SLUG}"
WL_MIN = 0.6
SPECIES = ("CH4", "CO2", "H2O")
LOGX, T_RANGE, LOGP_RANGE = (-10.0, -0.5), (150.0, 400.0), (-3.0, 1.0)
OPACITY_R = 300
PRIOR_SD = {"spot": 0.5, "facula": 0.5, "offset_ppm": 500.0}
MODELS = {"bare_rock": None, "n2": "N2", "h2": "H2He"}
NLIVE, DLOGZ = 400, 0.3
_DATA, _TPL, _PL = {}, {}, {}


def planet() -> dict:
    if not _PL:
        import pandas as pd
        ps = pd.read_csv(RESEARCH / "exoplanet_research_data" / "01_candidate_catalogs" / "pscomppars_confirmed_planets.csv", low_memory=False)
        r = ps[ps.pl_name == TARGET].iloc[0]
        _PL.update(rp=float(r.pl_rade), mp=float(r.pl_bmasse), rs=float(r.st_rad), teff=float(r.st_teff), teq=float(r.pl_eqt))
    return _PL


def load() -> list[dict]:
    if not _DATA:
        import stage_b_pipeline as sbp
        spectra = []
        for f, lab in zip(CONFIG["files"], CONFIG["labels"]):
            df, _ = sbp.parse_tbl(f)
            df = df[(df.wl >= WL_MIN) & np.isfinite(df.depth_ppm) & np.isfinite(df.err_ppm)].sort_values("wl").reset_index(drop=True)
            wl = df.wl.to_numpy()
            mid = 0.5 * (wl[1:] + wl[:-1])
            edges = np.r_[wl[0] - (mid[0] - wl[0]), mid, wl[-1] + (wl[-1] - mid[-1])]
            spectra.append({"label": lab, "file": f.name, "wl": wl, "edges": edges, "depth": df.depth_ppm.to_numpy(float),
                            "sigma": df.err_ppm.to_numpy(float)})
        _DATA["spectra"] = spectra
    return _DATA["spectra"]


def templates() -> list[dict]:
    """(1 - F_het / F_phot) for spots (0.86 T_eff) and faculae (T_eff + 150 K) on each spectrum's bins."""
    if not _TPL:
        from petitRADTRANS.config import petitradtrans_config_parser as cfg
        cfg.set_input_data_path(os.path.expanduser("~/petitRADTRANS/input_data"))
        from petitRADTRANS.stellar_spectra.phoenix import PhoenixStarTable
        t = PhoenixStarTable()
        t.load()
        teff = planet()["teff"]
        out = []
        for s in load():
            def binned(T):
                spec, _ = t.compute_spectrum(T)
                idx = np.digitize(spec[:, 0] * 1e4, s["edges"]) - 1
                return np.array([spec[idx == b, 1].mean() if (idx == b).any() else np.nan for b in range(len(s["edges"]) - 1)])
            phot = binned(teff)
            sp, fa = 1 - binned(0.86 * teff) / phot, 1 - binned(teff + 150.0) / phot
            for arr in (sp, fa):                                       # bins narrower than the PHOENIX sampling
                bad = ~np.isfinite(arr)
                if bad.any():
                    arr[bad] = np.interp(s["wl"][bad], s["wl"][~bad], arr[~bad])
            out.append({"spot": sp, "facula": fa})
        _TPL["t"] = out
    return _TPL["t"]


def planet_depths(gases: dict, background: str | None, T: float, logp: float, rp: float) -> list[np.ndarray]:
    import forward_model as fm
    pl = planet()
    spectra = load()
    if background is None:                                            # bare rock: an opaque disc
        flat = (rp * fm.R_EARTH_CM / (pl["rs"] * fm.R_SUN_CM)) ** 2 * 1e6
        return [np.full(len(s["wl"]), flat) for s in spectra]
    lo = min(s["edges"][0] for s in spectra) - 0.05
    hi = max(s["edges"][-1] for s in spectra) + 0.05
    wl, dep = fm.transmission_spectrum(gases, background, T, rp, pl["mp"], pl["rs"], surface_pressure=10 ** logp,
                                       wl_range=(max(lo, 0.55), min(hi, 5.3)), species=SPECIES, resolution=OPACITY_R)
    out = []
    for s in spectra:
        b = fm.bin_spectrum(wl, dep, s["edges"])
        empty = ~np.isfinite(b)
        if empty.any():
            b[empty] = np.interp(s["wl"][empty], wl, dep)
        if not np.all(np.isfinite(b)):
            raise ValueError("planet model undefined at some data points")
        out.append(b)
    return out


def marginal_loglike(depth, y, sigma, tpl) -> float:
    B = np.column_stack([depth * tpl["spot"], depth * tpl["facula"], np.ones(len(y))])
    lam = np.array([PRIOR_SD["spot"], PRIOR_SD["facula"], PRIOR_SD["offset_ppm"]]) ** 2
    C = np.diag(sigma ** 2) + (B * lam) @ B.T
    r = y - depth
    sign, logdet = np.linalg.slogdet(C)
    return float(-0.5 * (r @ np.linalg.solve(C, r) + logdet + len(y) * np.log(2 * np.pi)))


def rp_range() -> tuple[float, float]:
    return 0.85 * planet()["rp"], 1.10 * planet()["rp"]


def ndim(model):
    return 1 if MODELS[model] is None else len(SPECIES) + 3


def names(model):
    return ["Rp"] if MODELS[model] is None else [f"log_{g}" for g in SPECIES] + ["T", "logP", "Rp"]


def prior_transform(u, model):
    x = np.array(u, float)
    lo, hi = rp_range()
    if MODELS[model] is None:
        x[0] = lo + (hi - lo) * u[0]
        return x
    n = len(SPECIES)
    x[:n] = LOGX[0] + (LOGX[1] - LOGX[0]) * u[:n]
    x[n] = T_RANGE[0] + (T_RANGE[1] - T_RANGE[0]) * u[n]
    x[n + 1] = LOGP_RANGE[0] + (LOGP_RANGE[1] - LOGP_RANGE[0]) * u[n + 1]
    x[n + 2] = lo + (hi - lo) * u[n + 2]
    return x


def unpack(x, model):
    if MODELS[model] is None:
        return {}, None, 0.0, 0.0, x[0]
    n = len(SPECIES)
    return {g: 10 ** x[i] for i, g in enumerate(SPECIES)}, MODELS[model], x[n], x[n + 1], x[n + 2]


def log_likelihood(x, model):
    gases, bg, T, logp, rp = unpack(x, model)
    if gases and sum(gases.values()) >= 0.99:
        return -1e100
    try:
        depths = planet_depths(gases, bg, T, logp, rp)
    except ValueError:
        raise
    except Exception:
        return -1e100
    return sum(marginal_loglike(dp, s["depth"], s["sigma"], tp) for dp, s, tp in zip(depths, load(), templates()))


def run(model, workers, nlive):
    import dynesty
    from dynesty.pool import Pool

    t0 = time.time()
    with Pool(workers, log_likelihood, prior_transform, logl_args=(model,), ptform_args=(model,)) as pool:
        s = dynesty.NestedSampler(pool.loglike, pool.prior_transform, ndim(model), nlive=nlive, pool=pool,
                                  queue_size=workers, sample="unif", rstate=np.random.default_rng(3))
        s.run_nested(dlogz=DLOGZ, print_progress=False)
    r = s.results
    w = np.exp(r.logwt - r.logz[-1])
    nm = names(model)
    np.savez_compressed(OUT / "posteriors" / f"{model}.npz", samples=r.samples, weights=w, logz=r.logz[-1], names=np.array(nm))
    post = {}
    for i, n in enumerate(nm):
        o = np.argsort(r.samples[:, i])
        c = np.cumsum(w[o])
        post[n] = [float(np.interp(q, c / c[-1], r.samples[o, i])) for q in (0.025, 0.16, 0.5, 0.84, 0.975)]
    best = r.samples[np.argmax(r.logl)]
    return {"model": model, "logz": float(r.logz[-1]), "logz_err": float(r.logzerr[-1]), "ncall": int(np.sum(r.ncall)),
            "seconds": round(time.time() - t0), "max_loglike": float(np.max(r.logl)),
            "posterior_2.5_16_50_84_97.5": post, "best_fit": dict(zip(nm, map(float, best)))}


def conditional_chi2(depth, s, tp) -> float:
    """chi^2 of one spectrum at the planet model, with its contamination and offset at their conditional best values."""
    B = np.column_stack([depth * tp["spot"], depth * tp["facula"], np.ones(len(depth))])
    lam = np.array([PRIOR_SD["spot"], PRIOR_SD["facula"], PRIOR_SD["offset_ppm"]]) ** 2
    W = 1 / s["sigma"] ** 2
    theta = np.linalg.solve((B.T * W) @ B + np.diag(1 / lam), (B.T * W) @ (s["depth"] - depth))
    return float(np.sum(((s["depth"] - depth - B @ theta) / s["sigma"]) ** 2))


def report() -> int:
    import retrieval_trappist1e as base
    res = {r["model"]: r for r in json.loads((OUT / "results.json").read_text())["runs"]}
    lz = {k: v["logz"] for k, v in res.items()}
    er = {k: v["logz_err"] for k, v in res.items()}
    comp = lambda a, b: {"lnB": round(lz[a] - lz[b], 2), "lnB_err": round(float(np.hypot(er[a], er[b])), 2),
                         "sigma": (1 if lz[a] >= lz[b] else -1) * round(base.bayes_sigma(abs(lz[a] - lz[b])), 2)}
    atm = max(["n2", "h2"], key=lambda m: lz[m])
    summ = {"target": TARGET, "spectra": [s["label"] for s in load()],
            "atmosphere_vs_bare": {"best_atmosphere": atm, **comp(atm, "bare_rock")},
            "n2_vs_h2": comp("n2", "h2"), "logz": lz}
    chis = {}
    for m, r in res.items():
        gases, bg, T, logp, rp = unpack(np.array([r["best_fit"][n] for n in names(m)]), m)
        dps = planet_depths(gases, bg, T, logp, rp)
        chis[m] = [round(conditional_chi2(dp, s, tp), 1) for dp, s, tp in zip(dps, load(), templates())]
    gain = [round(b - a, 1) for a, b in zip(chis[atm], chis["bare_rock"])]
    n_improve = int(sum(g > 0 for g in gain))
    need = int(np.ceil(2 * len(gain) / 3))
    credited = bool(summ["atmosphere_vs_bare"]["lnB"] >= 3 and n_improve >= need)
    summ.update({"chi2_per_spectrum": chis, "chi2_gain_bare_to_atm": gain, "n_spectra_improved": n_improve, "n_needed": need,
                 "R1_atmosphere_credited": credited,
                 "n_points": [int(len(s["wl"])) for s in load()]})
    if TARGET == "LHS 1140 b":
        summ["R2_n2_preferred_over_h2"] = bool(lz["n2"] > lz["h2"])
        summ["R3_atmosphere_vs_bare_inconclusive"] = bool(abs(summ["atmosphere_vs_bare"]["lnB"]) < 3)
    else:
        summ["R4_no_atmosphere_credited"] = not credited
    z = np.load(OUT / "posteriors" / f"{atm}.npz", allow_pickle=True)
    nm = list(z["names"].astype(str))
    lim = {}
    for g in SPECIES:
        i = nm.index(f"log_{g}")
        o = np.argsort(z["samples"][:, i])
        c = np.cumsum(z["weights"][o])
        lim[g] = round(float(np.interp(0.95, c / c[-1], z["samples"][o, i])), 2)
    summ[f"95pc_upper_log10_vmr_{atm}"] = lim
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))
    figure(res, atm, chis, gain, summ)
    return 0


def figure(res, atm, chis, gain, summ):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spectra = load()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), width_ratios=[2, 1])
    ax = axes[0]
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(spectra)))
    for s, c in zip(spectra, cols):
        ax.errorbar(s["wl"], s["depth"], s["sigma"], fmt="o", ms=2.5, color=c, alpha=0.7, label=s["label"])
    ax.set(xlabel="wavelength (um)", ylabel="transit depth (ppm)", title=f"{TARGET}: the spectra as published")
    ax.legend(fontsize=7)
    ax = axes[1]
    ax.bar(np.arange(1, len(gain) + 1), gain, color=cols)
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xticks=np.arange(1, len(gain) + 1), xlabel="spectrum", ylabel=f"chi^2 improvement, bare rock -> {atm}",
           title=f"ln B({atm} vs bare) = {summ['atmosphere_vs_bare']['lnB']:+.2f}: "
                 + ("atmosphere credited" if summ["R1_atmosphere_credited"] else "no atmosphere credited")
                 + f"\n(R1 needs ln B >= 3 and a gain in >= {summ['n_needed']} of {len(gain)} spectra)")
    fig.tight_layout()
    fig.savefig(RESEARCH / "eda" / f"retrieval_{SLUG}.png", dpi=150, bbox_inches="tight")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--nlive", type=int, default=NLIVE)
    ap.add_argument("--models", nargs="+", default=["bare_rock", "n2", "h2"])
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        return report()
    (OUT / "posteriors").mkdir(parents=True, exist_ok=True)
    res_path = OUT / "results.json"
    done = json.loads(res_path.read_text())["runs"] if res_path.is_file() else []
    for m in a.models:
        if m in {x["model"] for x in done}:
            continue
        print(f"=== {TARGET}: {m} ({ndim(m)} parameters) ===", flush=True)
        done.append(run(m, a.workers, a.nlive))
        res_path.write_text(json.dumps({"target": TARGET, "runs": done}, indent=2))
        print(f"  ln Z = {done[-1]['logz']:.2f} +/- {done[-1]['logz_err']:.2f} in {done[-1]['seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
