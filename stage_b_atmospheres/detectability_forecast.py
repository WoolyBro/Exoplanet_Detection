"""
detectability_forecast.py
=========================

How many JWST transits would it take to detect CO2, CH4, O3 (and H2O, O2) in the atmospheres of the
habitable-zone targets that Stage C ranks, if those atmospheres resemble stated scenarios?

This is the gas-level "habitability" question the project can answer honestly. The sample of rocky
planets with JWST spectra is too small, and their spectra are featureless, so a classifier cannot
learn biosignature gases from real data. A forecast, by contrast, rests only on physics and on JWST's
measured noise.

Runs in the petitRADTRANS environment:
    ~/micromamba/envs/prt/bin/python stage_b_atmospheres/detectability_forecast.py

Method
  Targets    Transiting planets of the Stage C table with P(optimistic HZ) >= 0.5.
  Spectra    forward_model.transmission_spectrum: isothermal at the Kempton T_eq, opaque surface at
             1 bar ("clear") or an opaque cloud deck at 0.1 bar ("cloudy").
  Scenarios  (volume mixing ratios; N2 fills the rest)
               modern_earth  O2 0.21, CO2 4.2e-4, H2O 1e-3, CH4 1.8e-6, O3 1e-6
               archean       CO2 1e-2, CH4 1e-3, H2O 1e-3   (anoxic, the CO2+CH4 disequilibrium pair)
               co2_rich      CO2 0.9, H2O 1e-4              (abiotic, Venus/Mars-like but 1 bar)
               h2_rich       H2/He background with CH4 1e-2, CO2 1e-2, H2O 1e-4 (only for R_p >= 1.8)
  Signal     For each gas, spectrum(all gases) - spectrum(without that gas), binned at R = 100. Each
             mode's own best-fit constant offset is removed, since a retrieval absorbs an offset freely.
  Noise      noise_model.py: photon noise times the empirical scale factor JWST achieved on 117 single-transit
             spectra of 30 stars, per mode, validated leave-one-star-out (x1.2-1.7 at 68 %).
  Modes      NIRSpec PRISM if J >= 10.5 (brighter stars saturate it, approximately), otherwise NIRSpec
             G395 and NIRISS SOSS. MIRI LRS for every target (O3 9.6 um; NIRSpec's ~4.7 um O3 bands often do as well). Each gas is
             reported for its best mode.
  Detection  Matched-filter S/N = sqrt(sum (signal / sigma)^2), sigma for N transits = sigma_1 / sqrt(N),
             plus an optional per-bin noise floor (sensitivity case). N for 5 sigma (and 3 sigma) is
             solved for each Monte Carlo draw.
  Uncertainty 4000 draws: planet radius, mass, stellar radius and T_eq from their catalogue errors
             (signal scales as R_p^3 T / (M_p R_s^2) to first order), times the noise model's validated
             log-normal error.
  Time       Hours = N x (3 x T14 + 1 h overhead). Years = N / (transits per year visible), taking 27 %
             of transits as observable (JWST field of regard, near-ecliptic target), over a 20-year mission.

What this is NOT
  * The matched filter assumes the template is known. A free retrieval needs more data, typically
    of order 1.5-2x the transits; the numbers here are lower bounds in that sense.
  * Stellar contamination is handled only to first order: fixed spot and facula templates are
    projected out (columns n5_tls_*, for hosts with T_eff < 4000 K). Visit-to-visit changes in the
    contamination, which dominated the TRAPPIST-1 b/c spectra, are not modelled.
  * The atmospheres are isothermal with no cold trap. Real H2O above the tropopause is far lower
    than 1e-3.
  * The scenarios are assumptions, not predictions of what these planets have.

Output: outputs/forecast/{forecast.csv, forecast_best_mode.csv, forecast_summary.json} (spectra are kept in
        memory only: they regenerate in minutes and would add 5 MB)
        eda/forecast_transits_needed.png
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
sys.path.insert(0, str(RESEARCH / "stage_c_priority_fusion"))
import noise_model as nm  # noqa: E402
import target_physics as tp  # noqa: E402

OUT = RESEARCH / "outputs" / "forecast"
N_DRAWS, SEED = 4000, 20261017
R_BIN = 100
PRISM_BRIGHT_LIMIT_J = 10.5
VISIBLE_FRACTION, MISSION_YEARS, OVERHEAD_H = 0.27, 20.0, 1.0
# Sensitivity case: per-bin systematic floors (ppm) of Greene et al. 2016, ApJ 817, 17 ("reasonably
# optimistic": 20 NIRISS SOSS, 30 NIRCam grism 2.5-5 um, 50 MIRI LRS). They give no NIRSpec value; the
# 2.5-5 um value is used for NIRSpec. Applied to each R = 100 bin.
FLOOR_PPM = {"NIRISS_SOSS": 20.0, "NIRSpec_PRISM": 30.0, "NIRSpec_G395": 30.0, "MIRI_LRS": 50.0}
MODE_RANGE = {"NIRSpec_PRISM": (0.95, 5.3), "NIRSpec_G395": (2.87, 5.18), "NIRISS_SOSS": (0.95, 2.8),
              "MIRI_LRS": (5.0, 12.0)}

# Stellar contamination (transit light-source effect, Rackham et al. 2018, ApJ 853, 122) for M-dwarf
# hosts. Unocculted spots (T = 0.86 T_eff) and faculae (T = T_eff + 100 K), Rackham's M-dwarf
# conventions, multiply the depth by 1 / (1 - f (1 - F_het/F_phot)). To first order in f, each adds
# a template depth x (1 - F_het/F_phot) with an unknown amplitude. Those templates, plus the
# constant offset, are fitted out of the data. Only the part of a gas signal ORTHOGONAL to them (in
# the noise-weighted sense) can count as detection; the rest is degenerate with contamination. Spectra
# are PHOENIX (Husser et al. 2013) from petitRADTRANS's table, whose Teff grid near M dwarfs is
# coarse (2002, 2314, 2642, 2814 K, ...), so they are interpolated.
TLS_HOST_TEFF_MAX, TLS_SPOT_RATIO, TLS_FACULA_DT = 4000.0, 0.86, 100.0
_PHOENIX = {}


def tls_templates(edges_by_mode: dict, teff: float) -> dict:
    """(1 - F_spot/F_phot, 1 - F_fac/F_phot) on each mode's bins, for a host of this T_eff."""
    if "table" not in _PHOENIX:
        import os
        from petitRADTRANS.config import petitradtrans_config_parser as cfg
        cfg.set_input_data_path(os.path.expanduser("~/petitRADTRANS/input_data"))
        from petitRADTRANS.stellar_spectra.phoenix import PhoenixStarTable
        _PHOENIX["table"] = PhoenixStarTable()
        _PHOENIX["table"].load()
    t = _PHOENIX["table"]
    out = {}
    for m, e in edges_by_mode.items():
        def binned(T):
            spec, _ = t.compute_spectrum(T)
            idx = np.digitize(spec[:, 0] * 1e4, e) - 1
            return np.array([spec[idx == b, 1].mean() if (idx == b).any() else np.nan for b in range(len(e) - 1)])
        phot = binned(teff)
        out[m] = (1 - binned(TLS_SPOT_RATIO * teff) / phot, 1 - binned(teff + TLS_FACULA_DT) / phot)
    return out


def project_out(s: np.ndarray, basis: np.ndarray, w: np.ndarray) -> np.ndarray:
    """s minus its weighted least-squares fit by the columns of basis (weights w = 1/sigma^2)."""
    a = basis * np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(a, s * np.sqrt(w), rcond=None)
    return s - basis @ coef


SCENARIOS = {
    "modern_earth": ("N2", {"O2": 0.21, "CO2": 4.2e-4, "H2O": 1e-3, "CH4": 1.8e-6, "O3": 1e-6}),
    "archean": ("N2", {"CO2": 1e-2, "CH4": 1e-3, "H2O": 1e-3}),
    "co2_rich": ("N2", {"CO2": 0.9, "H2O": 1e-4}),
    "h2_rich": ("H2He", {"CH4": 1e-2, "CO2": 1e-2, "H2O": 1e-4}),
}
CLOUDS = {"clear": 1.0, "cloudy": 0.1}       # opaque pressure (bar)


def r_edges(lo, hi, R=R_BIN):
    n = int(np.floor(np.log(hi / lo) * R))
    return lo * np.exp(np.arange(n + 1) / R)


def targets() -> pd.DataFrame:
    t = pd.read_csv(RESEARCH / "outputs" / "stage_c" / "final_priority_table.csv")
    t = t[t.transiting & (t.p_hz_optimistic >= 0.5)]
    ps = pd.read_csv(nm.PSCOMP, low_memory=False)
    cols = ["pl_name", "pl_rade", "pl_radeerr1", "pl_radeerr2", "pl_bmasse", "pl_bmasseerr1", "pl_bmasseerr2",
            "pl_bmassprov", "pl_bmasselim", "st_rad", "st_raderr1", "st_raderr2", "st_teff", "st_tefferr1",
            "st_tefferr2", "pl_orbsmax", "pl_orbsmaxerr1", "pl_orbsmaxerr2", "pl_orbper", "pl_trandur", "sy_jmag"]
    return ps[ps.pl_name.isin(t.pl_name)][cols].merge(t[["pl_name", "rank_median", "teq_kempton_K"]], on="pl_name") \
        .sort_values("rank_median").reset_index(drop=True)


def amplitude_draws(r: pd.Series, rng) -> tuple[np.ndarray, float]:
    """Signal-amplitude multiplier per draw relative to the nominal planet, ~ R_p^3 T / (M_p R_s^2), and
    the nominal mass. A mass from the Chen & Kipping relation also carries its intrinsic scatter: the
    paper's radius scatter at fixed mass (0.0403 dex Terran, 0.146 dex Neptunian) divided by the
    power-law slope (0.279, 0.589), i.e. 0.144 / 0.248 dex in mass."""
    one = lambda v, e1, e2: tp.split_normal(pd.Series([v]), pd.Series([e1]), pd.Series([e2]), N_DRAWS, rng)[0]
    R = one(r.pl_rade, r.pl_radeerr1, r.pl_radeerr2)
    measured = r.pl_bmassprov == "Mass" and not (r.pl_bmasselim == 1) and np.isfinite(r.pl_bmasse)
    if measured:
        M = one(r.pl_bmasse, r.pl_bmasseerr1, r.pl_bmasseerr2)
    else:
        M = tp.chen_kipping_mass(R) * 10 ** (rng.normal(0, 1, N_DRAWS) * np.where(R < 1.23, 0.144, 0.248))
    M0 = r.pl_bmasse if measured else float(tp.chen_kipping_mass([r.pl_rade])[0])
    Rs = one(r.st_rad, r.st_raderr1, r.st_raderr2)
    T = tp.teq_kempton(one(r.st_teff, r.st_tefferr1, r.st_tefferr2), Rs, one(r.pl_orbsmax, r.pl_orbsmaxerr1, r.pl_orbsmaxerr2))
    return (R / r.pl_rade) ** 3 * (M0 / M) * (r.st_rad / Rs) ** 2 * (T / r.teq_kempton_K), M0


def solve_n(signal2: np.ndarray, sig1_2: np.ndarray, floor2: float, z: float) -> np.ndarray:
    """Transits N giving S/N = z, per draw. signal2, sig1_2: (draws, bins). inf where a floor forbids it."""
    def snr2(n):
        return (signal2 / (sig1_2 / n[:, None] + floor2)).sum(axis=1)
    lo, hi = np.full(len(signal2), 1e-4), np.full(len(signal2), 1e7)
    reach = snr2(hi) >= z ** 2
    for _ in range(80):                                  # bisection in log N
        mid = np.sqrt(lo * hi)
        ok = snr2(mid) >= z ** 2
        hi, lo = np.where(ok, mid, hi), np.where(ok, lo, mid)
    return np.where(reach, hi, np.inf)


def gas_signal(fm, vmr, background, gas, planet, p_surf, edges_by_mode, full):
    without = {g: v for g, v in vmr.items() if g != gas} or {"H2O": 1e-12}
    wl, dep = fm.transmission_spectrum(without, background, planet["T"], planet["rp"], planet["mp"], planet["rs"],
                                       surface_pressure=p_surf)
    return {m: full[m] - fm.bin_spectrum(wl, dep, e) for m, e in edges_by_mode.items()}, (wl, dep)


def main() -> int:
    import forward_model as fm

    OUT.mkdir(parents=True, exist_ok=True)
    calib = json.loads((OUT / "noise_calibration.json").read_text())["modes"]
    tg = targets()
    rng = np.random.default_rng(SEED)
    edges = {m: r_edges(*MODE_RANGE[m]) for m in MODE_RANGE}
    rows, spectra = [], {}
    for _, r in tg.iterrows():
        modes = (["NIRSpec_PRISM"] if r.sy_jmag >= PRISM_BRIGHT_LIMIT_J else ["NIRSpec_G395", "NIRISS_SOSS"]) + ["MIRI_LRS"]
        amp, m0 = amplitude_draws(r, rng)
        planet = {"T": float(r.teq_kempton_K), "rp": float(r.pl_rade), "mp": float(m0), "rs": float(r.st_rad)}
        tls = tls_templates({m: edges[m] for m in modes}, float(r.st_teff)) if r.st_teff < TLS_HOST_TEFF_MAX else None
        sig1, noise_k = {}, {}
        for m in modes:
            e = edges[m]
            c, w = 0.5 * (e[1:] + e[:-1]), np.diff(e)
            lo = calib[m]["loso"]
            bias = lo["median_log10_ratio"]
            sd = np.sqrt(max(lo["rms_log10_ratio"] ** 2 - bias ** 2, 0.0))
            sig1[m] = nm.predict_sigma_ppm(calib[m], c, w, r.sy_jmag, r.st_teff, r.pl_trandur) / 10 ** bias
            noise_k[m] = 10 ** rng.normal(0, sd, N_DRAWS)
        for scen, (bg, vmr) in SCENARIOS.items():
            if scen == "h2_rich" and r.pl_rade < 1.8:
                continue
            for cloud, p_surf in CLOUDS.items():
                wl, dep = fm.transmission_spectrum(vmr, bg, planet["T"], planet["rp"], planet["mp"], planet["rs"],
                                                   surface_pressure=p_surf)
                full = {m: fm.bin_spectrum(wl, dep, edges[m]) for m in modes}
                spectra[f"{r.pl_name}|{scen}|{cloud}|all"] = np.stack([wl, dep]).astype(np.float32)
                for gas in vmr:
                    sigs, (wl2, dep2) = gas_signal(fm, vmr, bg, gas, planet, p_surf, {m: edges[m] for m in modes}, full)
                    spectra[f"{r.pl_name}|{scen}|{cloud}|no_{gas}"] = np.stack([wl2, dep2]).astype(np.float32)
                    for m in modes:
                        s, sg = sigs[m], sig1[m]
                        ok = np.isfinite(s) & np.isfinite(sg) & np.isfinite(full[m])
                        if tls is not None:
                            ok &= np.isfinite(tls[m][0]) & np.isfinite(tls[m][1])
                        s, sg = s[ok], sg[ok]
                        wts = 1 / sg ** 2
                        n5_tls = snr1_tls = None
                        if tls is not None:
                            basis = np.column_stack([np.ones(ok.sum()), full[m][ok] * tls[m][0][ok], full[m][ok] * tls[m][1][ok]])
                            s_tls = project_out(s, basis, wts)
                            snr1_tls = float(np.sqrt(np.sum((s_tls / sg) ** 2)))
                            n5_tls = solve_n((amp[:, None] * s_tls[None, :]) ** 2, (noise_k[m][:, None] * sg[None, :]) ** 2, 0.0, 5.0)
                        s = s - np.sum(s * wts) / np.sum(wts)          # remove the free offset
                        snr1 = float(np.sqrt(np.sum((s / sg) ** 2)))
                        sig2 = (amp[:, None] * s[None, :]) ** 2
                        n2 = (noise_k[m][:, None] * sg[None, :]) ** 2
                        n5 = solve_n(sig2, n2, 0.0, 5.0)
                        n3 = solve_n(sig2, n2, 0.0, 3.0)
                        n5f = solve_n(sig2, n2, FLOOR_PPM[m] ** 2, 5.0)
                        per_year = 365.25 / r.pl_orbper * VISIBLE_FRACTION
                        q = lambda x, p: float(np.percentile(x, p, method="nearest"))   # inf = unreachable
                        rows.append({
                            "pl_name": r.pl_name, "rank_stage_c": r.rank_median, "scenario": scen, "cloud": cloud,
                            "gas": gas, "mode": m, "jmag": r.sy_jmag, "t14_h": r.pl_trandur, "period_d": r.pl_orbper,
                            "peak_signal_ppm": float(np.max(np.abs(s))), "snr_1_transit": snr1,
                            "n5_median": q(n5, 50), "n5_p16": q(n5, 16), "n5_p84": q(n5, 84), "n3_median": q(n3, 50),
                            "n5_floor_median": q(n5f, 50), "p_n5_floor_reachable": float(np.isfinite(n5f).mean()),
                            "hours_5sigma": q(n5, 50) * (3 * r.pl_trandur + OVERHEAD_H),
                            "years_5sigma": q(n5, 50) / per_year,
                            "transits_in_mission": per_year * MISSION_YEARS,
                            "snr_1_transit_tls": snr1_tls,
                            "tls_signal_retained": None if snr1_tls is None else snr1_tls / max(snr1, 1e-12),
                            "n5_tls_median": None if n5_tls is None else q(n5_tls, 50),
                            "n5_tls_p16": None if n5_tls is None else q(n5_tls, 16),
                            "n5_tls_p84": None if n5_tls is None else q(n5_tls, 84),
                        })
        print(f"  {r.pl_name}: done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "forecast.csv", index=False)
    best = (df.sort_values("n5_median").groupby(["pl_name", "scenario", "cloud", "gas"], sort=False).head(1)
            .sort_values(["rank_stage_c", "pl_name", "scenario", "cloud", "gas"]))
    best.to_csv(OUT / "forecast_best_mode.csv", index=False)
    summary = summarise(best)
    val, real_val = validate_k2_18b(df), real_noise_check(spectra)
    for gas, v in real_val.items():
        val[gas] = {**val.get(gas, {}), **v} if isinstance(v, dict) else v
    val["published"] = {"CH4": "5 sigma", "CO2": "3 sigma", "H2O": "not detected (95 % upper limit)",
                        "source": "Madhusudhan et al. 2023, ApJL 956, L13"}
    summary["validation_k2_18b"] = val
    (OUT / "forecast_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary["validation_k2_18b"], indent=2, default=float))
    make_figure(best, spectra, tg)
    return 0


def summarise(best: pd.DataFrame) -> dict:
    out = {}
    for (p, scen, cloud), g in best.groupby(["pl_name", "scenario", "cloud"], sort=False):
        out.setdefault(p, {}).setdefault(scen, {})[cloud] = {
            r.gas: {"mode": r["mode"], "n5": round(r.n5_median, 2), "n5_68": [round(r.n5_p16, 2), round(r.n5_p84, 2)],
                    "years": round(r.years_5sigma, 2), "in_mission": bool(r.n5_median <= r.transits_in_mission)}
            for _, r in g.iterrows()}
    return {"by_planet": out}


def validate_k2_18b(df: pd.DataFrame) -> dict:
    """K2-18 b, where JWST did detect CH4 (and, more tentatively, CO2) from one NIRISS SOSS plus one
    NIRSpec G395H transit. Two checks: (a) the forecast's S/N for 1 + 1 transits under the h2_rich
    scenario; (b) the same model signal scored against the REAL published uncertainties of those two
    spectra, which removes the noise model from the comparison."""
    d = df[(df.pl_name == "K2-18 b") & (df.scenario == "h2_rich") & (df.cloud == "clear")]
    if d.empty:
        return {}
    res = {}
    for gas in ["CH4", "CO2", "H2O"]:
        g = d[d.gas == gas].set_index("mode").snr_1_transit
        res[gas] = {"forecast_snr_1_soss_plus_1_g395": round(float(np.sqrt(g.get("NIRISS_SOSS", 0) ** 2
                                                                         + g.get("NIRSpec_G395", 0) ** 2)), 2)}
    return res


def real_noise_check(spectra: dict) -> dict:
    """Model signals scored against the published per-bin uncertainties of the K2-18 b spectra
    (Madhusudhan et al. 2023: one NIRISS SOSS and one NIRSpec G395H transit), each with its own offset."""
    import forward_model as fm

    meta = pd.read_csv(nm.SPEC / "metadata" / "spectra_metadata.csv", low_memory=False)
    meta = meta[(meta.pl_name == "K2-18 b") & meta.authors.astype(str).str.startswith("Madhusudhan et al. 2023")]
    files = {p.name: p for p in (nm.SPEC / "spectra_tbl").glob("*.tbl")}
    real = []
    for sp in meta.spec_path:
        base = sp.split("/")[-1]
        hit = [p for n, p in files.items() if n == base or n.endswith("_" + base)]
        if len(hit) == 1:
            real.append(nm.read_spectrum(hit[0]))
    key = "K2-18 b|h2_rich|clear|"
    if not real or key + "all" not in spectra:
        return {}
    out = {"real_spectra_used": len(real)}
    for gas in ["CH4", "CO2", "H2O"]:
        total = 0.0
        for d in real:
            d = d[(d.wl > 0.96) & (d.wl < 11.9)]
            edges = np.r_[d.wl.to_numpy() - d.dwl.to_numpy() / 2, d.wl.to_numpy()[-1] + d.dwl.to_numpy()[-1] / 2]
            ok = np.diff(edges) > 0
            wl_a, dep_a = spectra[key + "all"]
            wl_b, dep_b = spectra[key + f"no_{gas}"]
            s = fm.bin_spectrum(wl_a, dep_a, edges) - fm.bin_spectrum(wl_b, dep_b, edges)
            sg = d.sigma.to_numpy()
            m = ok & np.isfinite(s)
            w = 1 / sg[m] ** 2
            s = s[m] - np.sum(s[m] * w) / np.sum(w)
            total += float(np.sum((s / sg[m]) ** 2))
        out.setdefault(gas, {})["snr_with_real_published_noise"] = round(np.sqrt(total), 2)
    return out


def make_figure(best, spectra, tg):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.1, 1]})
    # (a) TRAPPIST-1 e spectra under two scenarios
    ax = axes[0]
    for scen, c in [("modern_earth", "#2a6fdb"), ("archean", "#d1495b")]:
        key = f"TRAPPIST-1 e|{scen}|clear|all"
        if key in spectra:
            wl, dep = spectra[key]
            e = r_edges(0.95, 12.0)
            import forward_model as fm
            b = fm.bin_spectrum(wl, dep, e)
            ax.plot(0.5 * (e[1:] + e[:-1]), b - np.nanmedian(b), color=c, label=scen.replace("_", " "))
    top = ax.get_ylim()[1]
    for x, lab, dy in [(4.3, "CO2", 0), (3.3, "CH4", 0), (9.6, "O3", 0), (7.7, "CH4", 0), (2.7, "H2O+CO2", -0.07)]:
        ax.axvline(x, color="0.8", lw=0.8, zorder=0)
        ax.text(x, top * (1.0 + dy), lab, fontsize=7, ha="center", va="bottom", color="0.4")
    ax.set(xscale="log", xlabel="wavelength (um)", ylabel="transit depth - median (ppm)",
           title="TRAPPIST-1 e, clear 1-bar N2 atmosphere (R = 100)")
    ax.set_xticks([1, 2, 3, 5, 8, 12]); ax.set_xticklabels(["1", "2", "3", "5", "8", "12"])
    ax.legend(fontsize=8)
    # (b) transits needed, archean + modern Earth, clear, per target
    ax = axes[1]
    sel = best[(best.cloud == "clear") & (((best.scenario == "archean") & best.gas.isin(["CO2", "CH4"]))
                                          | ((best.scenario == "modern_earth") & best.gas.isin(["O3"])))]
    names = list(dict.fromkeys(tg.pl_name))
    colours = {("archean", "CO2"): "#d1495b", ("archean", "CH4"): "#2e933c", ("modern_earth", "O3"): "#8f5bd6"}
    for j, ((scen, gas), c) in enumerate(colours.items()):
        s = sel[(sel.scenario == scen) & (sel.gas == gas)].set_index("pl_name").reindex(names)
        y = np.arange(len(names)) + (j - 1) * 0.25
        x = s.n5_median.clip(upper=1e5)
        ax.errorbar(x, y, xerr=[x - s.n5_p16.clip(upper=1e5), s.n5_p84.clip(upper=1e5) - x], fmt="o", color=c, ms=4,
                    label=f"{gas} ({scen.replace('_', ' ')})", capsize=2)
    lim = tg.set_index("pl_name").reindex(names)
    ax.scatter(365.25 / lim.pl_orbper * VISIBLE_FRACTION * MISSION_YEARS, np.arange(len(names)), marker="|", s=200,
               color="k", label="all observable transits in 20 yr")
    ax.set(xscale="log", yticks=np.arange(len(names)), yticklabels=names, xlabel="JWST transits for 5 sigma (median, 68 %)",
           title="Transits needed (clear sky, best mode)")
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(RESEARCH / "eda" / "forecast_transits_needed.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())
