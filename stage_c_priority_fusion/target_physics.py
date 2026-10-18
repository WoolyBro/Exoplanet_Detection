"""
target_physics.py
=================

The physical quantities Stage C ranks on, each from a published, cited definition, with measurement
uncertainty propagated by Monte Carlo. Nothing here is fitted to this project's data.

1. Habitable-zone membership - Kopparapu et al. 2014, ApJL 787, L29 (doi:10.1088/2041-8205/787/2/L29),
   the update of Kopparapu et al. 2013, ApJ 765, 131. Effective-flux boundaries
       S_eff(T_eff) = S_sun + a T + b T^2 + c T^3 + d T^4,   T = T_eff - 5780 K,
   valid for 2600 K <= T_eff <= 7200 K. Coefficients are the 1 Earth-mass values distributed with the
   authors' HZ calculator (personal.ems.psu.edu/~jfk4/ruk15/planets/hzcalc.py), checked against the
   published table on 2026-10-17.
       conservative HZ: runaway greenhouse (inner) .. maximum greenhouse (outer)
       optimistic HZ:   recent Venus (inner)      .. early Mars (outer)
   A planet is IN a zone if S_outer <= S_planet <= S_inner, where S_planet is the incident flux in
   Earth units (pl_insol). Being in the HZ is a statement about received flux. It is NOT
   habitability: atmosphere, water inventory, tidal state and stellar activity are not modelled.

2. Transmission Spectroscopy Metric - Kempton et al. 2018, PASP (doi:10.1088/1538-3873/aadf6f),
   equations 1-3 and Table 1, read from the paper on 2026-10-17:
       TSM = scale(R_p) * R_p^3 * T_eq / (M_p * R_s^2) * 10^(-m_J / 5)
       T_eq = T_s * sqrt(R_s / a) * (1/4)^(1/4)      (zero albedo, full redistribution)
       scale = 0.190 (R_p < 1.5), 1.26 (1.5-2.75), 1.28 (2.75-4.0), 1.15 (4.0-10) Earth radii
       suggested follow-up cuts: TSM > 12 (R_p < 1.5), TSM > 92 (1.5 < R_p < 10)
   R_p, M_p in Earth units, R_s in solar radii, a in the same units as R_s. Unmeasured masses use
   Chen & Kipping 2017 as given in Kempton eq. 2. The metric is only defined for TRANSITING planets,
   and its scale factors only up to 10 R_earth: larger planets use the 4-10 bin and are flagged.

Uncertainty: every quantity with a catalogue error is drawn from a split normal (err1 above the value,
|err2| below); quantities without one are held fixed and the row is flagged, so a narrow interval can
always be traced to missing errors rather than read as precision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---- Kopparapu et al. 2014 (1 M_earth) ------------------------------------------------------- #
HZ_LIMITS = {                      # S_sun,   a,         b,         c,          d
    "recent_venus":      (1.776, 2.136e-4, 2.533e-8, -1.332e-11, -3.097e-15),
    "runaway_greenhouse": (1.107, 1.332e-4, 1.580e-8, -8.308e-12, -1.931e-15),
    "maximum_greenhouse": (0.356, 6.171e-5, 1.698e-9, -3.198e-12, -5.575e-16),
    "early_mars":        (0.320, 5.547e-5, 1.526e-9, -2.874e-12, -5.011e-16),
}
HZ_TEFF_RANGE = (2600.0, 7200.0)

# ---- Kempton et al. 2018 ---------------------------------------------------------------------- #
TSM_BINS = [(0.0, 1.5, 0.190, 12.0), (1.5, 2.75, 1.26, 92.0), (2.75, 4.0, 1.28, 92.0), (4.0, np.inf, 1.15, 92.0)]
R_SUN_AU = 0.00465047              # solar radius in AU


def hz_flux_limits(teff) -> dict[str, np.ndarray]:
    """Effective-flux limits (Earth units) at each T_eff, evaluated inside the fit's validity range."""
    t = np.clip(np.asarray(teff, float), *HZ_TEFF_RANGE) - 5780.0
    return {k: s + a * t + b * t ** 2 + c * t ** 3 + d * t ** 4 for k, (s, a, b, c, d) in HZ_LIMITS.items()}


def in_hz(insol, teff) -> tuple[np.ndarray, np.ndarray]:
    """(in conservative HZ, in optimistic HZ) as boolean arrays."""
    lim = hz_flux_limits(teff)
    s = np.asarray(insol, float)
    cons = (s <= lim["runaway_greenhouse"]) & (s >= lim["maximum_greenhouse"])
    opt = (s <= lim["recent_venus"]) & (s >= lim["early_mars"])
    return cons, opt


def chen_kipping_mass(rade) -> np.ndarray:
    """Kempton et al. 2018 eq. 2 (Chen & Kipping 2017), Earth units; valid below 14.26 R_earth."""
    r = np.asarray(rade, float)
    return np.where(r < 1.23, 0.9718 * r ** 3.58, 1.436 * r ** 1.70)


def teq_kempton(teff, st_rad, a_au) -> np.ndarray:
    """Equilibrium temperature, zero albedo, full redistribution (Kempton eq. 3)."""
    a_rs = np.asarray(a_au, float) / (np.asarray(st_rad, float) * R_SUN_AU)
    return np.asarray(teff, float) * np.sqrt(1.0 / a_rs) * 0.25 ** 0.25


def tsm(rade, masse, st_rad, teq, jmag) -> tuple[np.ndarray, np.ndarray]:
    """(TSM, Kempton follow-up threshold for that radius bin)."""
    r = np.asarray(rade, float)
    scale = np.select([(r >= lo) & (r < hi) for lo, hi, _, _ in TSM_BINS], [s for *_, s, _ in TSM_BINS], np.nan)
    cut = np.select([(r >= lo) & (r < hi) for lo, hi, _, _ in TSM_BINS], [c for *_, c in TSM_BINS], np.nan)
    value = scale * r ** 3 * np.asarray(teq, float) / (np.asarray(masse, float) * np.asarray(st_rad, float) ** 2) \
        * 10 ** (-np.asarray(jmag, float) / 5.0)
    return value, cut


# ---- Monte Carlo -------------------------------------------------------------------------------- #
def split_normal(value, err_hi, err_lo, n: int, rng: np.random.Generator, positive: bool = True) -> np.ndarray:
    """n draws per row: value + |N(0,1)| * err_hi above, value - |N(0,1)| * |err_lo| below (50/50).
    Rows with no error stay at their value. Returns shape (rows, n)."""
    v = np.asarray(value, float)[:, None]
    hi = np.nan_to_num(np.abs(np.asarray(err_hi, float)), nan=0.0)[:, None]
    lo = np.nan_to_num(np.abs(np.asarray(err_lo, float)), nan=0.0)[:, None]
    z = rng.standard_normal((len(v), n))
    out = np.where(z >= 0, v + z * hi, v + z * lo)
    return np.clip(out, 1e-9, None) if positive else out


def monte_carlo(df: pd.DataFrame, n: int = 4000, seed: int = 20261017, return_draws: bool = False):
    """HZ membership probabilities, TSM and their uncertainty for each planet in `df` (pscomppars rows).

    Insolation: pl_insol with its own errors; where those are missing, its fractional uncertainty is
    taken from S = L / a^2 (st_lum is log10 L/L_sun; error propagated from st_lum and pl_orbsmax), and
    the row says so. Mass: measured where pl_bmassprov is "Mass" and pl_bmasselim is 0; anything else
    (a limit, Msini, a catalogue M-R estimate, missing) -> Chen & Kipping from radius, with its
    intrinsic scatter, flagged. Atmosphere retention: position relative to the cosmic shoreline
    (cosmic_shoreline.py), with luminosity from st_lum (or R_s and T_eff when missing).
    """
    rng = np.random.default_rng(seed)
    d = df.reset_index(drop=True)
    notes = [[] for _ in range(len(d))]

    # ---- insolation ----
    has_s_err = d.pl_insolerr1.notna() & d.pl_insolerr2.notna()
    frac_l = np.log(10) * d.st_lumerr1.abs()                     # d(ln L) from a dex error
    frac_a = 2 * (d.pl_orbsmaxerr1.abs() / d.pl_orbsmax)
    frac_s = np.sqrt(frac_l.fillna(0) ** 2 + frac_a.fillna(0) ** 2)
    s_hi = np.where(has_s_err, d.pl_insolerr1, d.pl_insol * frac_s)
    s_lo = np.where(has_s_err, d.pl_insolerr2, d.pl_insol * frac_s)
    for i in np.flatnonzero(~has_s_err):
        notes[i].append("insolation error derived from L and a")
    S = split_normal(d.pl_insol, s_hi, s_lo, n, rng)
    T = split_normal(d.st_teff, d.st_tefferr1, d.st_tefferr2, n, rng)
    for i in np.flatnonzero((d.st_teff < HZ_TEFF_RANGE[0]) | (d.st_teff > HZ_TEFF_RANGE[1])):
        notes[i].append(f"T_eff {d.st_teff[i]:.0f} K outside the HZ fit range 2600-7200 K: evaluated at the edge")
    cons, opt = in_hz(S, T)

    # ---- size ----
    R = split_normal(d.pl_rade, d.pl_radeerr1, d.pl_radeerr2, n, rng)
    p_small = (R < 1.5).mean(axis=1)                              # Kempton's terrestrial bin
    transiting = (d.tran_flag == 1).to_numpy()
    for i in np.flatnonzero(~transiting):
        notes[i].append(f"not transiting ({d.discoverymethod[i]}): radius is an estimate; TSM not applicable")

    # ---- TSM (transiting planets only) ----
    prov = d.pl_bmassprov.astype(str).to_numpy()
    # pl_bmasselim = 1 marks an UPPER LIMIT (e.g. Kepler-62 f "< 35 M_earth"), not a measurement; used as
    # a mass it understates the TSM by the ratio of the limit to the true mass (about 10x for these).
    limit = d.pl_bmasselim.fillna(0).to_numpy() != 0 if "pl_bmasselim" in d else np.zeros(len(d), bool)
    measured = (prov == "Mass") & d.pl_bmasse.notna().to_numpy() & ~limit
    M = split_normal(d.pl_bmasse, d.pl_bmasseerr1, d.pl_bmasseerr2, n, rng)
    # Chen & Kipping masses carry the relation's intrinsic scatter: radius scatter at fixed mass
    # (0.0403 / 0.146 dex, Terran / Neptunian) over the slope (0.279 / 0.589) = 0.144 / 0.248 dex in mass.
    ck_scatter = 10 ** (rng.normal(0.0, 1.0, R.shape) * np.where(R < 1.23, 0.144, 0.248))
    M = np.where(measured[:, None], M, chen_kipping_mass(R) * ck_scatter)
    for i in np.flatnonzero(~measured):
        why = f"catalogue gives only a limit of {d.pl_bmasse[i]:.3g}" if limit[i] else f"catalogue: {prov[i]}"
        notes[i].append(f"mass from Chen & Kipping ({why})")
    Rs = split_normal(d.st_rad, d.st_raderr1, d.st_raderr2, n, rng)
    A = split_normal(d.pl_orbsmax, d.pl_orbsmaxerr1, d.pl_orbsmaxerr2, n, rng)
    teq = teq_kempton(T, Rs, A)
    tsm_draw, cut = tsm(R, M, Rs, teq, d.sy_jmag.to_numpy(float)[:, None])
    tsm_draw = np.where(transiting[:, None], tsm_draw, np.nan)
    for i in np.flatnonzero(transiting & (d.pl_rade > 10).to_numpy()):
        notes[i].append("R_p > 10 R_earth: TSM uses the 4-10 R_earth scale factor (extrapolated)")

    obs_draw = np.where(transiting[:, None], np.clip(np.nan_to_num(tsm_draw / cut, nan=0.0), 0.0, 1.0), 0.0)
    # ---- atmosphere retention: the cosmic shoreline (cosmic_shoreline.py) ----
    import cosmic_shoreline as cs
    nan_col = pd.Series(np.nan, index=d.index)
    st_lum = d["st_lum"] if "st_lum" in d else nan_col
    has_lum = st_lum.notna().to_numpy()
    log_l = split_normal(st_lum.fillna(0.0), d.get("st_lumerr1", nan_col), d.get("st_lumerr2", nan_col), n, rng,
                         positive=False)
    log_l_sb = np.log10(Rs ** 2 * (T / 5772.0) ** 4)                         # Stefan-Boltzmann fallback
    L = 10 ** np.where(has_lum[:, None], log_l, log_l_sb)
    for i in np.flatnonzero(~has_lum):
        notes[i].append("stellar luminosity from R_s and T_eff (catalogue st_lum missing)")
    shore = cs.draws(M, R, S, L, n, rng)
    shore_summary = cs.summarise(shore)
    for i in np.flatnonzero(~transiting):
        notes[i].append("shoreline uses an estimated radius")

    draws = {"hz": cons.astype(float) + 0.5 * (opt & ~cons), "terrestrial": (R < 1.5).astype(float),
             "observable": obs_draw, "retention": (shore["primary"] < 0).astype(float),
             # a terrestrial-size planet that can also keep an atmosphere: the shoreline is a statement about
             # rocky bodies, so on its own it would reward giants for trivially keeping H/He envelopes
             "terrestrial_atm": ((R < 1.5) & (shore["primary"] < 0)).astype(float),
             # raw draws for the fitted retention model (retention_model.py); not part of the score
             "margin_primary": shore["primary"], "radius": R, "teq": teq}

    q = lambda x, p: np.nanpercentile(x, p, axis=1) if np.isfinite(x).any() else np.full(len(x), np.nan)
    with np.errstate(all="ignore"):
        out = pd.DataFrame({
            "pl_name": d.pl_name,
            "transiting": transiting,
            "p_hz_conservative": cons.mean(axis=1),
            "p_hz_optimistic": opt.mean(axis=1),
            "insol_earth": d.pl_insol,
            "hz_inner_runaway": hz_flux_limits(d.st_teff)["runaway_greenhouse"],
            "hz_outer_maxgreen": hz_flux_limits(d.st_teff)["maximum_greenhouse"],
            "p_radius_lt_1p5": p_small,
            "teq_kempton_K": np.nanmedian(teq, axis=1),
            "tsm_median": q(tsm_draw, 50), "tsm_p16": q(tsm_draw, 16), "tsm_p84": q(tsm_draw, 84),
            "tsm_followup_cut": np.nanmedian(cut, axis=1),
            "p_tsm_above_cut": np.where(transiting, np.nanmean(tsm_draw > cut, axis=1), np.nan),
            "v_esc_kms": np.median(cs.v_esc_kms(M, R), axis=1),
            "cumulative_xuv_earth": np.median(cs.cumulative_xuv(S, L), axis=1),
            **shore_summary,
            "physics_notes": ["; ".join(x) for x in notes],
        })
    return (out, draws) if return_draws else out
