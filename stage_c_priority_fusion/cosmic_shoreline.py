"""
cosmic_shoreline.py
===================

Can a planet keep an atmosphere at all? The "cosmic shoreline" (Zahnle & Catling 2017, ApJ 843, 122)
is the empirical boundary, in escape-velocity vs irradiation space, between Solar System bodies that
have atmospheres and those that do not. Energy-limited escape predicts it as I proportional to v_esc^4.

Definitions used here
  v_esc        = 11.186 km/s x sqrt(M_p / R_p), with M_p and R_p in Earth units.
  I_XUV        = cumulative XUV fluence relative to Earth's, from the present instellation I via
                 Zahnle & Catling's eq. 27: I_XUV = I (L_star / L_sun)^(-0.6). It encodes that
                 low-luminosity stars stay XUV-active for longer relative to their bolometric output.
  shoreline    (primary) I_XUV = I0 v_esc^4, slope 4, with log10 I0 ~ U(log10 7e-4, log10 1e-3).
                 Mars lies on the 7e-4 line; 1e-3 covers its upper uncertainty. Both values are as
                 used by the statistical M-dwarf shoreline study, arXiv:2508.08253.
  sensitivity  (a) the EMPIRICAL shoreline of arXiv:2508.12865 (MNRAS 2026),
                   log10 I_XUV = 5.89 log10 v_esc - 4.49, anchored on Mars and 55 Cnc e;
               (b) the bolometric shoreline, I = 1e-4 v_esc^4 (arXiv:2508.08253).
  margin       = log10 I_XUV - log10 I_shoreline (dex). Negative means the atmosphere-retaining side.

Uncertainty (Monte Carlo)
  * Planet mass and radius from their catalogue errors. A mass from the Chen & Kipping relation also
    carries its intrinsic scatter (0.144 / 0.248 dex, as in the forecast).
  * Instellation and stellar luminosity from their errors.
  * The XUV scaling itself: a log-normal factor with sigma = 0.48 dex. arXiv:2508.08253 finds this
    scaling agrees with age-based estimates "to within a factor of ~3"; log10 3 = 0.48.
  * The shoreline zero point, as above.

What this is NOT: a probability that a planet has an atmosphere. It is where a planet sits relative
to an empirical boundary, which JWST results for M-dwarf planets suggest may itself be optimistic.
See validate_shoreline.py.
"""

from __future__ import annotations

import numpy as np

V_ESC_EARTH = 11.186            # km/s
LOG_I0_RANGE = (np.log10(7e-4), np.log10(1e-3))
XUV_SCALING_SIGMA_DEX = np.log10(3.0)
ECS_SLOPE, ECS_INTERCEPT = 5.89, -4.49
BOL_I0 = 1e-4


def v_esc_kms(mass_earth, radius_earth):
    return V_ESC_EARTH * np.sqrt(np.asarray(mass_earth, float) / np.asarray(radius_earth, float))


def cumulative_xuv(insol_earth, lum_solar):
    """Zahnle & Catling 2017 eq. 27, relative to Earth."""
    return np.asarray(insol_earth, float) * np.asarray(lum_solar, float) ** -0.6


def margins(v_esc, insol, lum, log_i0, xuv_factor_dex):
    """Shoreline margins (dex; negative = atmosphere-retaining side) for the three definitions."""
    lv = np.log10(v_esc)
    l_xuv = np.log10(cumulative_xuv(insol, lum)) + xuv_factor_dex
    return {"primary": l_xuv - (log_i0 + 4 * lv),
            "empirical": l_xuv - (ECS_INTERCEPT + ECS_SLOPE * lv),
            "bolometric": np.log10(insol) - (np.log10(BOL_I0) + 4 * lv)}


def draws(mass, radius, insol, lum, n: int, rng: np.random.Generator) -> dict:
    """mass, radius, insol, lum: arrays (planets, n) of Monte-Carlo draws. Returns margin draws."""
    shape = np.shape(mass)
    log_i0 = rng.uniform(*LOG_I0_RANGE, size=shape)
    xuv = rng.normal(0.0, XUV_SCALING_SIGMA_DEX, size=shape)
    return margins(v_esc_kms(mass, radius), insol, lum, log_i0, xuv)


def summarise(m: dict) -> dict:
    """Per-planet P(atmosphere side) and median margin for each definition."""
    out = {}
    for k, v in m.items():
        out[f"p_retain_{k}"] = np.mean(v < 0, axis=1)
        out[f"shoreline_margin_{k}_dex"] = np.median(v, axis=1)
    return out
