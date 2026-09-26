"""Step 14 bug check: investigate every |z| > 3 before reporting it.

Hypotheses tested for the two >3 sigma "CH4" results (GJ 1132 b, GJ 486 b):
  H1  the excess sits in the blue end of the CH4 box (2.6-3.0 um), i.e. the 2.7 um water / unocculted-starspot
      region, not the CH4 3.3 um band core (3.2-3.45 um)
  H2  it is a wavelength slope: with G395H the CH4 continuum lies only red-ward (3.5-3.85 um), so any slope
      falling toward the red reads as band absorption
  H3  it is reduction-dependent (other reductions of the same observations do not show it)
Plus: identify the local-file outliers (LHS 475 b CO z=9.5, LHS 1140 b CH4 3.7, GJ 341 b CO2 3.1).
Read-only; writes rocky_benchmark/rocky_high_sigma_diagnostics.txt.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import rocky_benchmark as rb
import zenodo_io as z

LINES: list[str] = []


def say(s: str = "") -> None:
    print(s)
    LINES.append(s)


def sub_band(wl, d, e, band, cont_mask):
    b = (wl >= band[0]) & (wl <= band[1])
    if b.sum() < 3 or cont_mask.sum() < 3:
        return None
    mb, mc, sig = rb.box_means(d, e, b, cont_mask)
    delta = mb - mc
    return b.sum(), delta, sig, delta / sig


def slope_test(wl, d, e):
    """Weighted linear fit to all points OUTSIDE every band; report slope and the band excess vs that line."""
    any_band = rb._in(wl, [iv for ivs in rb.BANDS.values() for iv in ivs])
    c = ~any_band
    if c.sum() < 4:
        return None
    A = np.vstack([np.ones(c.sum()), wl[c]]).T
    W = 1 / e[c] ** 2
    cov = np.linalg.inv(A.T @ (A * W[:, None]))
    beta = cov @ (A.T @ (W * d[c]))
    return beta, cov, c


def main() -> None:
    spectra, _ = rb.load_benchmark()
    for f in ("GJ1132b.txt", "GJ486b.txt"):
        df = spectra[f]
        wl, d, e = df.wl.to_numpy(), df.depth_ppm.to_numpy(), df.err_ppm.to_numpy()
        any_band = rb._in(wl, [iv for ivs in rb.BANDS.values() for iv in ivs])
        cont = rb._in(wl, [(2.2 - 0.5, 3.5 + 0.5)]) & ~any_band
        say(f"=== {f}: spectrum starts at {wl.min():.3f} um; CH4-test continuum points at "
            f"{np.round(wl[cont], 3).tolist()} (red side only)")
        for band in [(2.6, 3.5), (2.6, 3.0), (3.0, 3.5), (3.2, 3.45)]:
            r = sub_band(wl, d, e, band, cont)
            if r:
                say(f"  H1 sub-band {band}: n={r[0]:3d} delta={r[1]:6.1f} +/- {r[2]:5.1f} ppm  z={r[3]:5.2f}")
        s = slope_test(wl, d, e)
        if s:
            beta, cov, c = s
            say(f"  H2 linear fit to the {c.sum()} out-of-band points: slope {beta[1]:.1f} +/- {np.sqrt(cov[1, 1]):.1f} "
                f"ppm/um (continuum only at {wl[c].min():.2f}-{wl[c].max():.2f} um)")
            band = rb._in(wl, rb.BANDS["CH4"])
            resid = d - (beta[0] + beta[1] * wl)
            w = 1 / e[band] ** 2
            m = np.sum(w * resid[band]) / w.sum()
            # propagate the line's uncertainty at the band's weighted-mean wavelength
            x0 = np.sum(w * wl[band]) / w.sum()
            var_line = cov[0, 0] + 2 * x0 * cov[0, 1] + x0 ** 2 * cov[1, 1]
            sig = np.sqrt(1 / w.sum() + var_line)
            say(f"  H2 CH4-band excess over the fitted SLOPED continuum: {m:.1f} +/- {sig:.1f} ppm  z={m / sig:.2f}")
        # blue-end profile, binned to 0.1 um
        say("  depth profile (weighted mean in 0.1 um bins, ppm relative to 3.5-3.85 um continuum):")
        wc = 1 / e[cont] ** 2
        c0 = np.sum(wc * d[cont]) / wc.sum()
        for lo in np.arange(2.7, 3.9, 0.1):
            m = (wl >= lo) & (wl < lo + 0.1)
            if m.sum():
                w = 1 / e[m] ** 2
                say(f"    {lo:.1f}-{lo + 0.1:.1f}: {np.sum(w * d[m]) / w.sum() - c0:7.1f} +/- {1 / np.sqrt(w.sum()):5.1f}  (n={m.sum()})")
        say()

    say("=== H3: reduction sensitivity for the same planets (local .tbl reductions/visits) ===")
    rs = pd.read_csv(z.OUT_DIR / "rocky_reduction_sensitivity.csv")
    for planet, mol in [("GJ 1132 b", "CH4"), ("GJ 486 b", "CH4"), ("LHS 475 b", "CO"), ("LHS 1140 b", "CH4"),
                        ("GJ 341 b", "CO2"), ("GJ 1132 b", "SO2")]:
        sub = rs[(rs.planet == planet) & (rs.molecule == mol)].sort_values("z")
        say(f"{planet} {mol}: z across {len(sub)} local files = {sub.z.round(2).tolist()}")
        top = sub.iloc[-1] if sub.z.iloc[-1] >= abs(sub.z.iloc[0]) else sub.iloc[0]
        say(f"   extreme: {top.local_file} ({top.authors}, {top.instrument}) z={top.z}")

    # LHS 475 b CO outlier: look at the points
    say("\n=== LHS 475 b CO z=9.5 outlier: points in band and continuum ===")
    sub = rs[(rs.planet == "LHS 475 b") & (rs.molecule == "CO")].sort_values("z")
    worst = sub.iloc[-1].local_file
    df = z.parse_local(worst)
    df = df[np.isfinite(df.err_ppm) & (df.err_ppm > 0)]
    sel = df[(df.wl >= 4.3) & (df.wl <= 5.4)]
    say(f"{worst}: {len(df)} points, {df.wl.min():.3f}-{df.wl.max():.3f} um; median err {df.err_ppm.median():.1f} ppm")
    say(f"  points 4.30-5.40 um: n={len(sel)}, median err {sel.err_ppm.median():.1f} ppm, "
        f"min err {sel.err_ppm.min():.2f} ppm, depth range {sel.depth_ppm.min():.0f}-{sel.depth_ppm.max():.0f}")
    say(f"  whole-file depth percentiles 1/50/99: {np.percentile(df.depth_ppm, [1, 50, 99]).round(0).tolist()}")
    say(f"  whole-file error percentiles 1/50/99: {np.percentile(df.err_ppm, [1, 50, 99]).round(1).tolist()}")
    rb_t = rb.band_test(df.wl.to_numpy(), df.depth_ppm.to_numpy(), df.err_ppm.to_numpy(), "CO")
    say(f"  band test: {rb_t}")

    (z.OUT_DIR / "rocky_high_sigma_diagnostics.txt").write_text("\n".join(LINES), encoding="utf-8")


if __name__ == "__main__":
    main()
