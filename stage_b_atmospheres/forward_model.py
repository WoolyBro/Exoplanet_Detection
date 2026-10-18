"""
forward_model.py
================

Transmission spectra from first principles with petitRADTRANS 3.4 (Mollière et al. 2019), shared by
the gas-detectability forecast and the synthetic-spectra classifier.

RUNS IN ITS OWN ENVIRONMENT, not the project .venv: petitRADTRANS compiles Fortran. Set up once with
micromamba (no admin rights needed) - see stage_b_atmospheres/README_forward_model.md:
    ~/micromamba/envs/prt/bin/python <script that imports this module>

Opacities (downloaded once to ~/petitRADTRANS/input_data, 1.7 GB), each chosen explicitly so no
default is picked silently:
    H2O  POKAZATEL (ExoMol)      CO2  UCL-4000 (ExoMol)    CH4  HITEMP
    CO   HITEMP                  SO2  ExoAmes (ExoMol)     O3   HITRAN      O2  default
    correlated-k, R = 1000.
Collision-induced absorption (CIA): N2-N2, O2-O2, CO2-CO2 for secondary atmospheres; H2-H2 and
H2-He for primary ones. The CIA tables each cover only part of 0.95-12 um and petitRADTRANS refuses
a table outside its range, so the spectrum is computed in wavelength SEGMENTS, each with every CIA
table valid there, and joined. Rayleigh scattering from N2, O2, CO2, H2O, H2, He.

Atmosphere: isothermal at T, hydrostatic, constant volume mixing ratios (VMR) with altitude, on top
of an N2 or H2/He background. A surface (rocky planets) or a grey cloud deck is represented as an
opaque layer at that pressure. These are the standard simplifications of exoplanet transmission
forecasts; they set feature amplitudes to within factors of order unity, not to precision.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from petitRADTRANS.config import petitradtrans_config_parser as _cfg

_cfg.set_input_data_path(os.path.expanduser("~/petitRADTRANS/input_data"))
from petitRADTRANS.radtrans import Radtrans  # noqa: E402

# gas -> (petitRADTRANS line-species name, molar mass g/mol)
SPECIES = {
    "H2O": ("H2O__POKAZATEL.R1000_0.3-50mu", 18.015),
    "CO2": ("CO2__UCL-4000.R1000_0.3-50mu", 44.009),
    "CH4": ("CH4__HITEMP.R1000_0.1-250mu", 16.043),
    "CO": ("CO__HITEMP.R1000_0.1-250mu", 28.010),
    "SO2": ("SO2__ExoAmes.R1000_0.3-50mu", 64.066),
    "O3": ("O3__HITRAN.R1000_0.1-250mu", 47.998),
    "O2": ("O2", 31.998),
}
M_N2, M_H2, M_HE = 28.013, 2.016, 4.0026
H2_FRACTION = 0.85          # H2 : He by number in an H2/He background (solar-like)

# Wavelength segments (um) with the CIA tables valid inside each (callers choose wl_range; the
# default still starts at 0.95 um). Boundaries follow the tables'
# own ranges: N2-N2 2-100 um, CO2-CO2 3.08-100 um, O2-O2 0.34-8.7 um, H2-H2 0.607-250 um.
SEGMENTS = {
    # O2-O2 is valid from 0.34 um, so the first N2 segment reaches PRISM's blue end at 0.6 um
    "N2":   [(0.60, 2.0, ("O2--O2",)), (2.0, 3.08, ("N2--N2", "O2--O2")),
             (3.08, 8.7, ("N2--N2", "CO2--CO2", "O2--O2")), (8.7, 12.0, ("N2--N2", "CO2--CO2"))],
    "H2He": [(0.95, 12.0, ("H2--H2-NatAbund__BoRi.R831_0.6-250mu", "H2--He-NatAbund__BoRi.DeltaWavenumber2_0.5-500mu"))],
}
PRESSURES = np.logspace(-8, 2, 90)        # bar; surfaces and clouds are opaque layers inside this grid

G_CGS, R_EARTH_CM, M_EARTH_G, R_SUN_CM = 6.674e-8, 6.371e8, 5.972e27, 6.957e10


# ONE Radtrans object per (background, wavelength segment, line-species set): every gas in the set
# is always loaded, and gases a sample lacks get zero abundance. Keying the cache on each sample's own
# gas combination instead built a new ~0.5 GB opacity object per combination and exhausted 16 GB of
# RAM within minutes (2026-10-17); the cache is now small and bounded.
def line_name(gas: str, resolution: int = 1000) -> str:
    """petitRADTRANS species name at a correlated-k resolving power. Lower powers are binned down
    from the R = 1000 tables once by exo_k and cached in the input-data folder."""
    name = SPECIES[gas][0]
    return name if resolution == 1000 or ".R1000_" not in name else name.replace(".R1000_", f".R{resolution}_")


@lru_cache(maxsize=8)
def _radtrans(background: str, lo: float, hi: float, cia: tuple, gases: tuple, resolution: int = 1000) -> Radtrans:
    rayleigh = ["N2", "O2", "CO2", "H2O"] if background == "N2" else ["H2", "He", "H2O"]
    return Radtrans(pressures=PRESSURES, line_species=[line_name(g, resolution) for g in gases],
                    rayleigh_species=rayleigh, gas_continuum_contributors=list(cia),
                    wavelength_boundaries=[lo, hi], line_opacity_mode="c-k")


def mean_molar_mass(vmr: dict[str, float], background: str) -> float:
    trace = sum(vmr.values())
    bg = M_N2 if background == "N2" else H2_FRACTION * M_H2 + (1 - H2_FRACTION) * M_HE
    return sum(v * SPECIES[g][1] for g, v in vmr.items()) + (1 - trace) * bg


def transmission_spectrum(vmr: dict[str, float], background: str, temperature: float, rp_earth: float,
                          mp_earth: float, rs_sun: float, surface_pressure: float = 1.0,
                          wl_range: tuple[float, float] = (0.95, 12.0),
                          species: tuple[str, ...] | None = None,
                          resolution: int = 1000, haze_kappa_350: float | None = None,
                          haze_gamma: float = -4.0) -> tuple[np.ndarray, np.ndarray]:
    """(wavelength um, transit depth ppm) for an isothermal atmosphere.

    vmr: volume mixing ratios of trace gases (keys from SPECIES). The background (N2, or H2/He)
    fills the rest. surface_pressure (bar) is where the atmosphere becomes opaque - a rocky surface,
    or a cloud deck for a giant - and is also the reference pressure at which the radius is rp_earth.
    species: the line species to load (default: all of SPECIES). Keep it the same across calls so the
    cached opacity objects are reused; gases in it but not in vmr get zero abundance.
    haze_kappa_350, haze_gamma: optional haze, an opacity of haze_kappa_350 (cm^2/g) at 0.35 um scaling
    as (lambda / 0.35 um)^haze_gamma (gamma = -4 is Rayleigh-like; petitRADTRANS power-law opacity).
    """
    if not 0 < sum(vmr.values()) < 1:
        raise ValueError("trace-gas VMRs must sum to between 0 and 1")
    gases = tuple(sorted(species or SPECIES))
    if set(vmr) - set(gases):
        raise ValueError(f"gases {sorted(set(vmr) - set(gases))} are not in species={gases}")
    vmr = {g: float(vmr.get(g, 0.0)) for g in gases}
    mu = mean_molar_mass(vmr, background)
    n = len(PRESSURES)
    mass = {line_name(g, resolution): np.full(n, vmr[g] * SPECIES[g][1] / mu) for g in gases}
    rest = 1 - sum(vmr.values())
    if background == "N2":
        mass["N2"] = np.full(n, rest * M_N2 / mu)
    else:
        mass["H2"] = np.full(n, rest * H2_FRACTION * M_H2 / mu)
        mass["He"] = np.full(n, rest * (1 - H2_FRACTION) * M_HE / mu)
    for g in ("O2", "CO2", "H2O"):          # CIA / Rayleigh partners read these keys too
        mass[g] = np.full(n, vmr.get(g, 0.0) * SPECIES[g][1] / mu)
    for g in (["N2"] if background == "N2" else ["H2", "He"]):
        mass.setdefault(g, np.zeros(n))
    gravity = G_CGS * mp_earth * M_EARTH_G / (rp_earth * R_EARTH_CM) ** 2

    wl_all, depth_all = [], []
    for lo, hi, cia in SEGMENTS[background]:
        lo2, hi2 = max(lo, wl_range[0]), min(hi, wl_range[1])
        if hi2 <= lo2:
            continue
        rt = _radtrans(background, lo2, hi2, cia, gases, resolution)   # absent CIA partners have zero abundance
        wl_cm, r_cm, _ = rt.calculate_transit_radii(
            temperatures=np.full(n, float(temperature)), mass_fractions=mass,
            mean_molar_masses=np.full(n, mu), reference_gravity=gravity,
            reference_pressure=surface_pressure, planet_radius=rp_earth * R_EARTH_CM,
            opaque_cloud_top_pressure=surface_pressure,
            **({} if haze_kappa_350 is None else {"power_law_opacity_350nm": haze_kappa_350,
                                                  "power_law_opacity_coefficient": haze_gamma}))
        wl = wl_cm * 1e4
        keep = (wl >= lo2) & (wl < hi2)
        wl_all.append(wl[keep])
        depth_all.append((r_cm[keep] / (rs_sun * R_SUN_CM)) ** 2 * 1e6)
    wl = np.concatenate(wl_all)
    order = np.argsort(wl)
    return wl[order], np.concatenate(depth_all)[order]


def bin_spectrum(wl: np.ndarray, depth: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Mean depth in each wavelength bin (NaN where a bin has no model point)."""
    idx = np.digitize(wl, edges) - 1
    out = np.full(len(edges) - 1, np.nan)
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.any():
            out[b] = depth[sel].mean()
    return out
