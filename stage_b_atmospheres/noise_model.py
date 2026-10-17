"""
noise_model.py
==============

How precisely does ONE JWST transit measure a transmission spectrum, for a given star? Calibrated on
what JWST has actually delivered, not on assumed instrument throughputs.

Method
  1. Ideal photon noise. The star is a blackbody at T_eff scaled to its 2MASS J magnitude (zero point
     3.129e-10 W m^-2 um^-1 at 1.235 um, Cohen et al. 2003). Collecting area is 25.4 m^2. N_in is the
     number of photons in one spectral bin during the transit (duration T14). Assuming an
     out-of-transit baseline of 2 x T14, sigma_ideal = sqrt(1/N_in + 1/(2 N_in)) = sqrt(1.5/N_in).
  2. Measured precision. Every JWST transmission spectrum in the NASA Exoplanet Archive whose
     metadata says it comes from a SINGLE transit (mintranmid == maxtranmid). Noise depends on the
     star rather than the planet, so hot Jupiters calibrate as well as rocky planets do. The per-bin
     uncertainty is the mean of the upper and lower errors, and the bin width is the spacing of
     adjacent bins.
  3. Empirical scale factor eta(lambda) = (sigma_ideal / sigma_measured)^2. It absorbs instrument
     throughput, duty cycle, detector noise, systematics left by the reduction, longer real baselines,
     and the blackbody's error for real (redder) stellar spectra. It is NOT a pure throughput: for
     NIRSpec G395 it exceeds 1, because a J-normalised blackbody underestimates M- and K-dwarf flux
     at 3-5 um. That does not matter for prediction, which the validation below tests directly. It is fitted per observing mode as a smooth function of wavelength (the median
     over stars of each star's median curve, so no single star dominates).
  4. Leave-one-star-out validation. Each host star's measured precision is predicted from the
     other stars only. The spread of log10(sigma_pred / sigma_meas) across stars is the noise
     model's own uncertainty, and the forecast propagates it.

Run (project .venv): python stage_b_atmospheres/noise_model.py
Output: outputs/forecast/noise_calibration.json, noise_validation.csv, figure eda/forecast_noise_model.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
SPEC = RESEARCH / "exoplanet_research_data" / "04_atmospheric_spectra"
PSCOMP = RESEARCH / "exoplanet_research_data" / "01_candidate_catalogs" / "pscomppars_confirmed_planets.csv"
OUT = RESEARCH / "outputs" / "forecast"

H, C, KB = 6.62607e-34, 2.99792458e8, 1.380649e-23
AREA_M2 = 25.4
J_LAMBDA_UM, J_ZERO_W_M2_UM = 1.235, 3.129e-10
BASELINE_RATIO = 2.0                       # out-of-transit time / T14

# Wavelength range (um) and fitting grid of each observing mode.
MODES = {
    "NIRSpec_PRISM": (0.6, 5.3),
    "NIRSpec_G395": (2.87, 5.18),
    "NIRISS_SOSS": (0.85, 2.8),
    "MIRI_LRS": (5.0, 12.0),
}
N_FIT_BINS = 16


def infer_mode(instrument: str, lo: float, hi: float) -> str | None:
    if "MIRI" in instrument:
        return "MIRI_LRS" if hi > 9 else None
    if "NIRISS" in instrument:
        return "NIRISS_SOSS"
    if "NIRSpec" in instrument:
        if lo < 1.2 and hi > 4.5:
            return "NIRSpec_PRISM"
        if lo > 2.5 and hi > 4.5:
            return "NIRSpec_G395"
    return None


def host_of(pl_name: str) -> str:
    return pl_name.rsplit(" ", 1)[0]


def photon_rate(wl_um, jmag, teff):
    """Stellar photons per second per micron collected by JWST at wavelength wl_um."""
    wl = np.asarray(wl_um, float)

    def planck(w_um):
        w = w_um * 1e-6
        return 1.0 / (w ** 5 * np.expm1(H * C / (w * KB * teff)))

    f_lambda = J_ZERO_W_M2_UM * 10 ** (-0.4 * jmag) * planck(wl) / planck(J_LAMBDA_UM)   # W m^-2 um^-1
    return f_lambda * AREA_M2 * wl * 1e-6 / (H * C)


def ideal_sigma_ppm(wl_um, dwl_um, jmag, teff, t14_hours):
    n_in = photon_rate(wl_um, jmag, teff) * np.asarray(dwl_um, float) * t14_hours * 3600.0
    return np.sqrt((1 + 1 / BASELINE_RATIO) / n_in) * 1e6


def read_spectrum(path: Path) -> pd.DataFrame:
    from astropy.io import ascii as astro_ascii

    t = astro_ascii.read(path, format="ipac")

    def col(name):
        if name not in t.colnames:
            return np.full(len(t), np.nan)
        c = t[name]
        return np.ma.filled(np.ma.asarray(c, dtype=float), np.nan) if hasattr(c, "mask") else np.asarray(c, float)

    wl, dep, e1, e2 = col("CENTRALWAVELNG"), col("PL_TRANDEP"), col("PL_TRANDEPERR1"), col("PL_TRANDEPERR2")
    if np.all(~np.isfinite(dep)):                       # radius-ratio spectra: depth = (Rp/Rs)^2
        r, r1, r2 = col("PL_RATROR"), col("PL_RATRORERR1"), col("PL_RATRORERR2")
        dep, e1, e2 = 100 * r ** 2, 100 * 2 * r * np.abs(r1), 100 * 2 * r * np.abs(r2)
    sig = 1e4 * 0.5 * (np.abs(e1) + np.abs(e2))          # % -> ppm
    d = pd.DataFrame({"wl": wl, "sigma": sig}).dropna()
    d = d[(d.sigma > 0) & (d.wl > 0)].sort_values("wl").drop_duplicates("wl").reset_index(drop=True)
    gaps = np.diff(d.wl.to_numpy())
    left = np.r_[gaps[0], gaps] if len(gaps) else np.array([np.nan])
    right = np.r_[gaps, gaps[-1]] if len(gaps) else np.array([np.nan])
    d["dwl"] = np.minimum(left, right)                  # min, so a detector gap does not inflate a bin
    return d[d.dwl > 0]


def calibration_set() -> pd.DataFrame:
    """One row per spectral point of every single-transit JWST transmission spectrum with known host."""
    meta = pd.read_csv(SPEC / "metadata" / "spectra_metadata.csv", low_memory=False)
    meta = meta[(meta.spec_type == "Transmission") & meta.facility.astype(str).str.contains("James Webb")
                & (meta.mintranmid == meta.maxtranmid)].copy()
    meta["mode"] = [infer_mode(str(i), lo, hi) for i, lo, hi in zip(meta.instrument, meta.minwavelng, meta.maxwavelng)]
    meta = meta[meta["mode"].notna()]
    ps = pd.read_csv(PSCOMP, low_memory=False)[["pl_name", "sy_jmag", "st_teff", "pl_trandur"]]
    meta = meta.merge(ps, on="pl_name", how="inner").dropna(subset=["sy_jmag", "st_teff", "pl_trandur"])
    files = {p.name: p for p in (SPEC / "spectra_tbl").glob("*.tbl")}
    rows = []
    for k, r in meta.reset_index(drop=True).iterrows():
        base = r.spec_path.split("/")[-1]
        hits = [p for n, p in files.items() if n == base or n.endswith("_" + base)]
        if len(hits) != 1:
            continue
        d = read_spectrum(hits[0])
        lo, hi = MODES[r["mode"]]
        d = d[(d.wl >= lo) & (d.wl <= hi)]
        if len(d) < 5:
            continue
        d = d.assign(spectrum=k, pl_name=r.pl_name, host=host_of(r.pl_name), mode=r["mode"], authors=r.authors,
                     jmag=r.sy_jmag, teff=r.st_teff, t14=r.pl_trandur)
        d["sigma_ideal"] = ideal_sigma_ppm(d.wl, d.dwl, r.sy_jmag, r.st_teff, r.pl_trandur)
        d["log_eta"] = 2 * np.log10(d.sigma_ideal / d.sigma)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def fit_eta(points: pd.DataFrame, mode: str) -> dict:
    """log10 eta on a fixed wavelength grid: median over stars of each star's binned median."""
    lo, hi = MODES[mode]
    edges = np.linspace(lo, hi, N_FIT_BINS + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    p = points.assign(b=np.digitize(points.wl, edges) - 1)
    p = p[(p.b >= 0) & (p.b < N_FIT_BINS)]
    per_spec = p.groupby(["host", "spectrum", "b"]).log_eta.median().reset_index()
    per_star = per_spec.groupby(["host", "b"]).log_eta.median().reset_index()
    curve = per_star.groupby("b").log_eta.median().reindex(range(N_FIT_BINS))
    good = curve.notna().to_numpy()
    filled = np.interp(centres, centres[good], curve.to_numpy()[good])   # edge bins without data: nearest
    return {"wl": centres.round(4).tolist(), "log10_eta": filled.round(4).tolist(),
            "n_stars": int(per_star.host.nunique()), "n_spectra": int(per_spec.spectrum.nunique())}


def eta_at(curve: dict, wl) -> np.ndarray:
    return 10 ** np.interp(wl, curve["wl"], curve["log10_eta"])


def predict_sigma_ppm(curve: dict, wl, dwl, jmag, teff, t14_hours) -> np.ndarray:
    """Per-bin transit-depth uncertainty (ppm) for ONE transit."""
    return ideal_sigma_ppm(wl, dwl, jmag, teff, t14_hours) / np.sqrt(eta_at(curve, wl))


def leave_one_star_out(points: pd.DataFrame, mode: str) -> pd.DataFrame:
    p = points[points["mode"] == mode]
    rows = []
    for host in sorted(p.host.unique()):
        train, test = p[p.host != host], p[p.host == host]
        if train.host.nunique() < 2:
            continue
        curve = fit_eta(train, mode)
        pred = predict_sigma_ppm(curve, test.wl, test.dwl, test.jmag, test.teff, test.t14)
        lr = np.log10(pred / test.sigma.to_numpy())
        per_spec = pd.Series(lr).groupby(test.spectrum.to_numpy()).median()
        rows.append({"mode": mode, "host": host, "n_spectra": len(per_spec), "jmag": float(test.jmag.iloc[0]),
                     "teff": float(test.teff.iloc[0]), "log10_pred_over_meas": float(per_spec.median())})
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pts = calibration_set()
    calib, val = {}, []
    for mode in MODES:
        sub = pts[pts["mode"] == mode]
        if sub.empty:
            continue
        calib[mode] = fit_eta(sub, mode)
        v = leave_one_star_out(pts, mode)
        val.append(v)
        lr = v.log10_pred_over_meas
        calib[mode]["loso"] = {"n_hosts": len(v), "median_log10_ratio": round(float(lr.median()), 3),
                               "rms_log10_ratio": round(float(np.sqrt(np.mean(lr ** 2))), 3),
                               "factor_68": round(float(10 ** np.sqrt(np.mean(lr ** 2))), 2),
                               "spearman_vs_jmag": round(float(pd.Series(lr).corr(v.jmag, method="spearman")), 2)
                               if len(v) > 2 else None}
        print(f"{mode:15s} {calib[mode]['n_spectra']:3d} spectra / {calib[mode]['n_stars']:2d} stars | "
              f"LOSO median log ratio {calib[mode]['loso']['median_log10_ratio']:+.3f}, "
              f"rms {calib[mode]['loso']['rms_log10_ratio']:.3f} (x{calib[mode]['loso']['factor_68']})")
    val = pd.concat(val, ignore_index=True)
    val.to_csv(OUT / "noise_validation.csv", index=False)
    meta = {"assumptions": {"collecting_area_m2": AREA_M2, "baseline_over_t14": BASELINE_RATIO,
                            "stellar_sed": "blackbody at T_eff scaled to 2MASS J", "J_zero_point_W_m2_um": J_ZERO_W_M2_UM,
                            "calibration_spectra": "JWST archive transmission spectra with mintranmid == maxtranmid"},
            "modes": calib}
    (OUT / "noise_calibration.json").write_text(json.dumps(meta, indent=2))
    make_figure(pts, calib, val)
    return 0


def make_figure(pts, calib, val):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colours = {"NIRSpec_PRISM": "#2a6fdb", "NIRSpec_G395": "#d1495b", "NIRISS_SOSS": "#2e933c", "MIRI_LRS": "#8f5bd6"}
    import matplotlib.patheffects as pe

    for mode in calib:
        sub = pts[pts["mode"] == mode]
        axes[0].scatter(sub.wl, sub.log_eta, s=2, alpha=0.06, color=colours[mode], rasterized=True)
    for mode, c in calib.items():                       # curves on top, outlined, so none hides in its cloud
        axes[0].plot(c["wl"], c["log10_eta"], color=colours[mode], lw=2.2, label=f"{mode} ({c['n_stars']} stars)",
                     path_effects=[pe.Stroke(linewidth=4, foreground="white"), pe.Normal()])
    axes[0].set(xlabel="wavelength (um)", ylabel="log10 scale factor  (sigma_ideal / sigma_measured)^2",
                title="Single JWST transits vs. blackbody photon noise", ylim=(-2, 0.8))
    axes[0].legend(fontsize=8)
    for mode, c in colours.items():
        v = val[val["mode"] == mode]
        axes[1].scatter(v.jmag, v.log10_pred_over_meas, color=c, label=mode, s=25)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set(xlabel="host J magnitude", ylabel="log10 (predicted / measured sigma)",
                title="Leave-one-star-out check of the noise model")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESEARCH / "eda" / "forecast_noise_model.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())
