"""Step 16: eda/rocky_planet_detection_limits.png

Each benchmark spectrum in units of atmospheric scale heights (mu = 28, the Kreidberg & Stevenson Fig. 3
convention), on ONE shared y-scale so planets compare directly: the blue envelope is the per-point +/-1 sigma
noise; the orange strip is the 2-5 H size of a strong band in a clear high-mu atmosphere (an assumption, stated
in the report). Molecule bands are labelled bars in a neutral track (identity by text, not colour).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import rocky_benchmark as rb  # noqa: E402
import zenodo_io as z  # noqa: E402

SURFACE, INK, INK2, GRID, BAND_BG = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#efeee9"
SERIES, EXPECTED = "#2a78d6", "#eb6834"  # validated pair (categorical slots 1-2)
YLIM = 32.0  # scale heights, shared by every panel (data area)
TRACK_TOP = 46.0  # axis top; the band track lives in YLIM..TRACK_TOP, above the data
TRACK_ROW = {"H2O": 0, "SO2": 0, "CH4": 1, "CO2": 1, "CO": 2}  # overlapping bands on different rows
# From the Step 6 cross-match to local .tbl files (instrument of the matched local spectrum)
INSTRUMENT = {"GJ1132b.txt": "NIRSpec G395H/M", "GJ341b.txt": "NIRCam", "GJ486b.txt": "NIRSpec G395H",
              "L98-59c.txt": "NIRSpec G395H", "L98-59d.txt": "NIRSpec G395H", "LHS1140b-NIRISS.txt": "NIRISS SOSS",
              "LHS1140b-NIRSpec.txt": "NIRSpec G235H+G395H", "LHS1140c.txt": "NIRISS SOSS",
              "LHS475b.txt": "NIRSpec G395H", "TOI836b.txt": "NIRSpec G395H"}


def main() -> None:
    spectra, params = rb.load_benchmark()
    sh = pd.DataFrame([rb.scale_height_row(pl, params.loc[pl]) for pl in sorted({z.FILES[f] for f in spectra})]).set_index("planet")
    order = sorted(spectra, key=lambda f: np.median(spectra[f].err_ppm) / sh.loc[z.FILES[f], "A_H_ppm_mu28"])

    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 8})
    fig, axes = plt.subplots(5, 2, figsize=(12, 15), sharex=True, sharey=True)
    for ax, f in zip(axes.ravel(), order):
        df = spectra[f]
        planet = z.FILES[f]
        a_h = sh.loc[planet, "A_H_ppm_mu28"]
        wl, d, e = df.wl.to_numpy(), df.depth_ppm.to_numpy(), df.err_ppm.to_numpy()
        w = 1 / e ** 2
        y = (d - np.sum(w * d) / w.sum()) / a_h
        ey = e / a_h

        # molecule band track: neutral shading + labelled bars along the top
        for mol, ivs in rb.BANDS.items():
            for lo, hi in ivs:
                ax.axvspan(lo, hi, color=BAND_BG, zorder=0, lw=0, ymax=(YLIM + YLIM) / (YLIM + TRACK_TOP))
                ytrack = YLIM + 2.0 + 4.0 * TRACK_ROW[mol]
                ax.plot([lo + 0.01, hi - 0.01], [ytrack, ytrack], color=INK2, lw=2, solid_capstyle="round", zorder=3)
                ax.text(hi + 0.04, ytrack, mol, ha="left", va="center", fontsize=6.5, color=INK2, zorder=3)
        ax.axhline(YLIM, color=GRID, lw=0.8, zorder=1)  # separates the band track from the data area

        ax.axhspan(2, 5, color=EXPECTED, alpha=0.30, lw=0, zorder=1)
        ax.axhline(0, color=INK2, lw=0.6, zorder=1)
        # clip to the data area so points beyond +/-YLIM never run into the band track
        lo_env, hi_env = np.clip(y - ey, -YLIM, YLIM), np.clip(y + ey, -YLIM, YLIM)
        ax.fill_between(wl, lo_env, hi_env, color=SERIES, alpha=0.22, lw=0, zorder=2, step="mid")
        ax.plot(wl, np.clip(y, -YLIM, YLIM), color=SERIES, lw=1.4, zorder=4, drawstyle="steps-mid")

        med_h = np.median(e) / a_h
        inst = INSTRUMENT[f]
        ax.set_title(f"{planet}  ·  {inst}  ·  1 H = {a_h:.1f} ppm  ·  median 1σ = {med_h:.1f} H ({np.median(e):.0f} ppm)",
                     fontsize=8, loc="left", color=INK)
        ax.set_ylim(-YLIM, TRACK_TOP)
        ax.set_yticks(np.arange(-30, 31, 10))
        ax.set_xlim(0.6, 5.4)
    for ax in axes[:, 0]:
        ax.set_ylabel("depth − mean, in scale heights (μ = 28)")
    for ax in axes[-1, :]:
        ax.set_xlabel("wavelength (µm)")

    handles = [plt.Line2D([], [], color=SERIES, lw=1.4, label="JWST spectrum (Kreidberg & Stevenson 2025)"),
               plt.Rectangle((0, 0), 1, 1, color=SERIES, alpha=0.22, label="±1σ per-point noise"),
               plt.Rectangle((0, 0), 1, 1, color=EXPECTED, alpha=0.30, label="strong band in a clear N2 atmosphere (2–5 H, assumed)"),
               plt.Rectangle((0, 0), 1, 1, color=BAND_BG, label="molecule bands tested (labelled)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.965), fontsize=8)
    fig.suptitle("Rocky-planet JWST transmission spectra vs. the size of an atmospheric signal\n"
                 "panels ordered by noise in scale heights (best first); TRAPPIST-1 b and h excluded (provenance unresolved)",
                 fontsize=10, y=0.995, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    out = z.RESEARCH / "eda" / "rocky_planet_detection_limits.png"
    fig.savefig(out, dpi=150)
    print("saved", out, "| panel order:", [z.FILES[f] + (" (NIRISS)" if "NIRISS" in f else " (NIRSpec)" if "NIRSpec" in f else "") for f in order])


if __name__ == "__main__":
    main()
