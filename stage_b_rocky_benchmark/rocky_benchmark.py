"""Rocky-planet benchmark: characterising current detection limits for rocky-exoplanet atmospheres.

NOT a classifier and NOT a biosignature search. No rocky-planet atmosphere has a confirmed detection, so
there is nothing to train on. This measures how large a molecular band the existing JWST spectra could
have seen, in ppm and in atmospheric scale heights.

Data: the Kreidberg & Stevenson (2025) Zenodo compilation only (10.5281/zenodo.15084225), one curated
spectrum per planet (two instruments for LHS 1140 b). TRAPPIST-1 b and TRAPPIST-1 h are excluded pending
provenance (see EXCLUDED). The local .tbl files are used only for the reduction-sensitivity check, never as
benchmark data. Nothing here writes to stage_b/, labels/verified_gas_labels.csv or any Stage B output.

Steps (numbering follows the task):
  12  scale height H = k T / (mu m_u g) and feature amplitude per scale height A_H = 2 Rp H / Rs^2
  13  band statistic: flat local continuum + a box offset delta inside the band;
      Delta chi^2 = (delta / sigma_delta)^2, signed z = delta / sigma_delta (positive = extra absorption)
  14  literature comparison + >3 sigma bug check (injection-recovery, error-scaling, reduction sensitivity)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import zenodo_io as z

sb = z.sb
OUT = z.OUT_DIR

# Molecule bands as specified for this benchmark (they differ from Stage B's BANDS; not reused).
# CH4 overlaps H2O at 2.6-2.9 um and CO2 overlaps CO at 4.35-4.55 um, so those pairs are not independent.
BANDS = {"H2O": [(2.5, 2.9)], "CH4": [(2.2, 2.4), (2.6, 3.5)], "SO2": [(3.85, 4.05)],
         "CO2": [(4.05, 4.55)], "CO": [(4.35, 4.85)]}
MOLECULES = list(BANDS)
CONT_WINDOW_UM = 0.5   # continuum = points outside every band, within this distance of the tested band
MIN_POINTS = 3         # in the band and in the continuum, else "not evaluable"

# Mean molecular weight. Primary 28 (N2-dominated, Earth-like): the value Kreidberg & Stevenson use to put
# these same spectra in scale heights (their Fig. 3), so our numbers are directly comparable. 18 (steam) and
# 44 (CO2, Venus-like) bracket high-mu secondary atmospheres; H2-dominated mu~2.3 is already ruled out for
# most of these planets and is not a rocky-planet assumption.
MU_PRIMARY = 28.0
MU_BRACKET = {"H2O (steam)": 18.0, "N2 (Earth-like)": 28.0, "CO2 (Venus-like)": 44.0}

EXCLUDED = {
    "TRAPPIST1b.txt": ("provenance unresolved: the Zenodo record names no source; the paper (arXiv:2507.00933) "
                       "states only that its TRAPPIST-1 b spectrum is contamination-corrected and cites Lim et al. "
                       "2023 (NIRISS) for stellar contamination, not as the data source; the file matches no local "
                       "JWST spectrum (-150 ppm offset, corr -0.18 vs Rathcke 2025 PRISM)"),
    "TRAPPIST1h.txt": ("provenance unresolved: the Zenodo record names no source; the paper's only TRAPPIST-1 h "
                       "reference is Garcia et al. 2022 (HST/WFC3, 1.1-1.7 um), which cannot have produced a "
                       "0.69-5.38 um spectrum; no local JWST spectrum exists"),
}

K_B, M_U, G_N = 1.380649e-23, 1.66053907e-27, 6.674e-11
R_EARTH, M_EARTH, R_SUN = 6.371e6, 5.972e24, 6.957e8


# --------------------------------------------------------------------------- #
# Step 12: scale heights
# --------------------------------------------------------------------------- #
def planet_mass_earth(p: pd.Series) -> tuple[float, str]:
    """Mass in Earth masses and how it was obtained."""
    if p.get("pl_bmasselim", 0) == 1 or not np.isfinite(p.get("pl_bmasse", np.nan)):
        # GJ 341 b: 4.0 M_E is an upper limit (DiTomasso+2025); at 0.88 R_E it would mean ~32 g/cm3.
        # Use an Earth-like rocky mass-radius relation M = R^3.7 (R = M^0.27), flagged.
        m = float(p.pl_rade) ** 3.7
        return m, f"estimated M=R^3.7 (catalogue {p.pl_bmasse} M_E is an upper limit)"
    prov = str(p.get("pl_bmassprov", ""))
    return float(p.pl_bmasse), ("M-R relation (catalogue)" if "M-R" in prov else "measured (catalogue)")


def scale_height_row(planet: str, p: pd.Series) -> dict:
    mass, mass_note = planet_mass_earth(p)
    rp = float(p.pl_rade) * R_EARTH
    g = G_N * mass * M_EARTH / rp ** 2
    rs = float(p.st_rad) * R_SUN
    row = {"planet": planet, "Rp_Rearth": float(p.pl_rade), "M_Mearth": round(mass, 3), "mass_source": mass_note,
           "Teq_K": float(p.pl_eqt), "g_ms2": round(g, 2), "Rs_Rsun": float(p.st_rad), "st_spectype": p.get("st_spectype")}
    for mu in MU_BRACKET.values():
        h = K_B * float(p.pl_eqt) / (mu * M_U * g)
        row[f"H_km_mu{int(mu)}"] = round(h / 1e3, 1)
        row[f"A_H_ppm_mu{int(mu)}"] = round(2 * rp * h / rs ** 2 * 1e6, 2)
    return row


# --------------------------------------------------------------------------- #
# Step 13: band statistic
# --------------------------------------------------------------------------- #
def _in(wl: np.ndarray, intervals) -> np.ndarray:
    m = np.zeros(len(wl), bool)
    for lo, hi in intervals:
        m |= (wl >= lo) & (wl <= hi)
    return m


def box_means(d: np.ndarray, e: np.ndarray, band: np.ndarray, cont: np.ndarray) -> tuple[float, float, float]:
    """Inverse-variance weighted mean depth in the band and in the continuum, and the uncertainty of their
    difference: (mean_band, mean_cont, sigma_delta)."""
    wb, wc = 1 / e[band] ** 2, 1 / e[cont] ** 2
    mb, mc = np.sum(wb * d[band]) / wb.sum(), np.sum(wc * d[cont]) / wc.sum()
    return mb, mc, np.sqrt(1 / wb.sum() + 1 / wc.sum())


def band_test(wl: np.ndarray, d: np.ndarray, e: np.ndarray, mol: str) -> dict:
    band = _in(wl, BANDS[mol])
    any_band = _in(wl, [iv for ivs in BANDS.values() for iv in ivs])
    near = _in(wl, [(lo - CONT_WINDOW_UM, hi + CONT_WINDOW_UM) for lo, hi in BANDS[mol]])
    cont = near & ~any_band
    out = {"n_band": int(band.sum()), "n_cont": int(cont.sum())}
    if band.sum() < MIN_POINTS or cont.sum() < MIN_POINTS:
        out["evaluable"] = False
        return out
    mb, mc, sig = box_means(d, e, band, cont)
    delta = mb - mc
    # Scatter check on the points actually used: chi2/dof of the best (flat + box) model.
    model = np.where(band, mb, mc)
    used = band | cont
    chi2_red = float(np.sum(((d - model)[used] / e[used]) ** 2) / max(used.sum() - 2, 1))
    infl = np.sqrt(max(chi2_red, 1.0))  # only inflate, never shrink, the quoted errors
    return {**out, "evaluable": True, "delta_ppm": round(float(delta), 1), "sigma_delta_ppm": round(float(sig), 1),
            "z": round(float(delta / sig), 2), "dchi2": round(float((delta / sig) ** 2), 2),
            "chi2_red_local": round(chi2_red, 2), "z_inflated": round(float(delta / (sig * infl)), 2),
            "sigma_delta_inflated_ppm": round(float(sig * infl), 1),
            "ul3_ppm": round(float(max(delta, 0) + 3 * sig * infl), 1)}


def injection_recovery(wl, d, e, mol, n_sigma=5.0) -> float | None:
    """Add a box of n_sigma * sigma_delta inside the band; the statistic should return z ~ z0 + n_sigma."""
    base = band_test(wl, d, e, mol)
    if not base.get("evaluable"):
        return None
    d2 = d + np.where(_in(wl, BANDS[mol]), n_sigma * base["sigma_delta_ppm"], 0.0)
    return round(band_test(wl, d2, e, mol)["z"] - base["z"], 2)


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def load_benchmark() -> tuple[dict, pd.DataFrame]:
    params = z.planet_params()
    spectra = {f: z.parse_zenodo(z.ZEN_DIR / f, params)[0] for f in sorted(z.FILES) if f not in EXCLUDED}
    return spectra, params


def reduction_sensitivity() -> pd.DataFrame:
    """Band z on every local .tbl reduction/visit of each benchmark planet (robustness only, not benchmark data)."""
    rows = []
    for planet in sorted({z.FILES[f] for f in z.FILES if f not in EXCLUDED}):
        sel = sb.select_spectra([planet])
        for r in sel.itertuples():
            df = z.parse_local(r.file)
            df = df[np.isfinite(df.err_ppm) & (df.err_ppm > 0)]
            wl, d, e = df.wl.to_numpy(), df.depth_ppm.to_numpy(), df.err_ppm.to_numpy()
            for mol in MOLECULES:
                t = band_test(wl, d, e, mol)
                if t.get("evaluable"):
                    rows.append({"planet": planet, "local_file": r.file, "authors": r.authors,
                                 "instrument": r.instrument, "molecule": mol, "z": t["z"],
                                 "z_inflated": t["z_inflated"], "delta_ppm": t["delta_ppm"],
                                 "sigma_delta_ppm": t["sigma_delta_ppm"]})
    return pd.DataFrame(rows)


def main() -> None:
    spectra, params = load_benchmark()
    print(f"benchmark spectra: {len(spectra)} files, planets: {sorted({z.FILES[f] for f in spectra})}")

    # Step 12
    sh = pd.DataFrame([scale_height_row(pl, params.loc[pl]) for pl in sorted({z.FILES[f] for f in spectra})])
    per_spec = []
    for f, df in spectra.items():
        row = sh.set_index("planet").loc[z.FILES[f]]
        e = df.err_ppm.to_numpy()
        per_spec.append({"file": f, "planet": z.FILES[f], "n_points": len(df),
                         "median_point_err_ppm": round(float(np.median(e)), 1),
                         "A_H_ppm_mu28": row.A_H_ppm_mu28,
                         "point_err_in_H_mu28": round(float(np.median(e)) / row.A_H_ppm_mu28, 2),
                         "point_err_in_H_mu18": round(float(np.median(e)) / row.A_H_ppm_mu18, 2),
                         "point_err_in_H_mu44": round(float(np.median(e)) / row.A_H_ppm_mu44, 2)})
    sh.to_csv(OUT / "rocky_scale_heights.csv", index=False)
    ps = pd.DataFrame(per_spec)
    ps.to_csv(OUT / "rocky_noise_vs_scale_height.csv", index=False)

    # Step 13 (+ injection-recovery)
    stats = []
    for f, df in spectra.items():
        wl, d, e = df.wl.to_numpy(), df.depth_ppm.to_numpy(), df.err_ppm.to_numpy()
        a_h = sh.set_index("planet").loc[z.FILES[f]]
        for mol in MOLECULES:
            t = band_test(wl, d, e, mol)
            row = {"file": f, "planet": z.FILES[f], "molecule": mol, **t}
            if t.get("evaluable"):
                row["injection_recovered_of_5sigma"] = injection_recovery(wl, d, e, mol)
                for mu in (18, 28, 44):
                    row[f"ul3_in_H_mu{mu}"] = round(t["ul3_ppm"] / a_h[f"A_H_ppm_mu{mu}"], 1)
                row["significance_sigma"] = round(abs(t["z_inflated"]), 2)
            stats.append(row)
    st = pd.DataFrame(stats)
    st.to_csv(OUT / "rocky_band_statistics.csv", index=False)

    # Step 14 robustness: reduction sensitivity from the local .tbl files
    rs = reduction_sensitivity()
    rs.to_csv(OUT / "rocky_reduction_sensitivity.csv", index=False)

    with open(OUT / "rocky_benchmark_config.json", "w") as fh:
        json.dump({"excluded": EXCLUDED, "bands": BANDS, "mu_primary": MU_PRIMARY, "mu_bracket": MU_BRACKET,
                   "cont_window_um": CONT_WINDOW_UM, "min_points": MIN_POINTS}, fh, indent=2)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("\n=== Step 12: scale heights ===")
    print(sh.to_string(index=False))
    print(ps.to_string(index=False))
    print("\n=== Step 13: band statistic ===")
    cols = ["file", "molecule", "n_band", "n_cont", "delta_ppm", "sigma_delta_ppm", "z", "chi2_red_local",
            "z_inflated", "ul3_ppm", "ul3_in_H_mu28", "injection_recovered_of_5sigma"]
    print(st[[c for c in cols if c in st.columns]].to_string(index=False))
    print("\n=== reduction sensitivity (local files; z per molecule) ===")
    if len(rs):
        print(rs.groupby(["planet", "molecule"]).z.agg(["count", "min", "median", "max", "std"]).round(2).to_string())


if __name__ == "__main__":
    main()
