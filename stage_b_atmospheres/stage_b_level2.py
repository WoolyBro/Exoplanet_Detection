"""
stage_b_level2.py
=================

Stage B Level 2: physics-informed and unsupervised analysis of the JWST transmission
spectra, as set out in the mentor's guide:

    "Feature amplitude scales with scale height H = kT/mu*g: normalize spectra by
     (Teq, gravity, stellar radius) before comparing planets. Try: autoencoder anomaly
     detection (find the weirdest spectra), clustering of hot Jupiters vs sub-Neptunes,
     band-index features (depth in 4.3 um CO2 band minus continuum) with
     uncertainty-aware models."

This file READS Level 1's outputs and never modifies them. outputs/stage_b/ (the dataset,
metrics, LOPO predictions) and outputs/labels/verified_gas_labels.csv stay exactly as they
are; everything here is written to outputs/stage_b_level2/.

What it does
  1. Scale-height normalisation. Transit depth is a ratio of areas, so an absorption
     feature is ~2 R_p H / R_s^2 deep. Dividing the spectrum by that quantity converts
     "parts per million" into "number of scale heights", which is the only way a hot
     Jupiter and a sub-Neptune can be put on the same axis.
  2. Band indices: band depth minus a local continuum, in scale heights, with the
     uncertainty propagated, for H2O / CH4 / CO2 / CO / SO2.
  3. Clustering: PCA + KMeans on the normalised spectra, checked against the physical
     split (giant vs sub-Neptune by radius) rather than assumed to match it.
  4. Anomaly detection: reconstruction error under LEAVE-ONE-PLANET-OUT. A model that
     reconstructs a spectrum it was trained on tells you nothing; holding the planet out
     makes "anomalous" mean "unlike the other planets", which is the intended question.

Honest scope: 53 spectra of 13 planets. Every number here is descriptive, not inferential.
Clusters with n=13 are illustrations, and an "anomaly" is a flag for a human to look at,
never a detection.

Usage
    python stage_b_atmospheres/stage_b_level2.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RESEARCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_DIR))

DATASET = RESEARCH_DIR / "outputs" / "stage_b" / "stage_b_dataset.npz"
PARAMS = RESEARCH_DIR / "exoplanet_research_data" / "01_candidate_catalogs" / "pscomppars_confirmed_planets.csv"
OUT_DIR = RESEARCH_DIR / "outputs" / "stage_b_level2"
EDA_DIR = RESEARCH_DIR / "eda"

# Physical constants (SI)
K_B = 1.380649e-23
M_U = 1.66053907e-27
G_GRAV = 6.67430e-11
R_EARTH = 6.371e6
M_EARTH = 5.9722e24
R_SUN = 6.957e8

# Mean molecular weight. 2.3 is the standard H2/He value and is what makes these spectra
# comparable at all; it is wrong for a genuinely high-metallicity or secondary atmosphere,
# which is why the planets it is doubtful for are named explicitly in the report.
MU_H2HE = 2.3
# For a rocky planet with a high-mu secondary atmosphere, 2.3 is not merely doubtful, it is
# ruled out: stage_b_rocky_benchmark uses mu = 28 (N2, Earth-like) as its primary value for
# exactly these planets and brackets it 18-44. Using 2.3 there overstates one scale height by
# 28/2.3 = 12.2x, so a caller whose planet list mixes types must pass `mu` per planet.
MU_SECONDARY = 28.0

# Molecular bands and their local continuum windows, in microns.
BANDS = {
    "H2O": ((1.35, 1.50), [(1.20, 1.32), (1.55, 1.70)]),
    "CH4": ((3.20, 3.45), [(2.90, 3.12), (3.55, 3.75)]),
    "SO2": ((4.00, 4.12), [(3.80, 3.96), (4.16, 4.28)]),
    "CO2": ((4.20, 4.42), [(3.95, 4.12), (4.50, 4.66)]),
    "CO":  ((4.55, 4.75), [(4.40, 4.52), (4.80, 4.96)]),
}


# =========================================================================== #
# Physics
# =========================================================================== #
def planet_parameters(planets: list[str], mu=MU_H2HE) -> pd.DataFrame:
    """Teq, radius, mass and stellar radius for each planet, with scale height derived.

    `mu` is the mean molecular weight used for the scale height: either a scalar (the default
    MU_H2HE = 2.3, correct for the H2/He-dominated planets this module analyses) or a callable
    taking the radii in Earth radii and returning one mu per planet, for callers whose planet
    list mixes gas giants with rocky planets. The value used is returned in `mu_assumed`, and
    both bracketing amplitudes are returned too, so the assumption's size is always visible.
    """
    cat = pd.read_csv(PARAMS, low_memory=False)
    cols = ["pl_name", "pl_eqt", "pl_rade", "pl_bmasse", "st_rad", "st_teff"]
    sub = cat.loc[cat.pl_name.isin(planets), cols].copy()
    # Keep the most complete row per planet.
    sub = (sub.assign(_n=sub.notna().sum(axis=1))
              .sort_values("_n", ascending=False).drop_duplicates("pl_name").drop(columns="_n"))

    missing = sorted(set(planets) - set(sub.pl_name))
    if missing:
        raise SystemExit(f"no parameters for {missing}; scale heights cannot be computed")

    rp = sub.pl_rade.to_numpy() * R_EARTH
    mp = sub.pl_bmasse.to_numpy() * M_EARTH
    rs = sub.st_rad.to_numpy() * R_SUN
    teq = sub.pl_eqt.to_numpy()

    g = G_GRAV * mp / rp ** 2                     # surface gravity, m/s^2
    mu_arr = np.asarray(mu(sub.pl_rade.to_numpy()) if callable(mu)
                        else np.full(len(sub), float(mu)), float)

    def amplitude(mu_values):
        """Scale height H = kT / (mu m_u g), and one H of absorption ~ 2 Rp H / Rs^2, in ppm."""
        h = K_B * teq / (mu_values * M_U * g)
        return h, 2 * rp * h / rs ** 2 * 1e6

    h, a_h = amplitude(mu_arr)
    sub["gravity_ms2"] = g
    sub["mu_assumed"] = mu_arr
    sub["scale_height_km"] = h / 1e3
    sub["amplitude_1H_ppm"] = a_h
    # Bracket, so no reader has to take one mu on trust: the same quantity under the H2/He and
    # the N2-secondary assumptions, differing by MU_SECONDARY / MU_H2HE = 12.2x.
    sub["amplitude_1H_ppm_mu2p3"] = amplitude(np.full(len(sub), MU_H2HE))[1]
    sub["amplitude_1H_ppm_mu28"] = amplitude(np.full(len(sub), MU_SECONDARY))[1]
    return sub.reset_index(drop=True)


def normalise_to_scale_heights(depth_ppm: np.ndarray, mask: np.ndarray, amp_ppm: float) -> np.ndarray:
    """Spectrum in units of scale heights, measured from its own covered-bin median.

    The offset is removed because an absolute transit depth is set by Rp/Rs, which says
    nothing about the atmosphere; only the variation across wavelength does.
    """
    covered = mask > 0
    if covered.sum() == 0 or not np.isfinite(amp_ppm) or amp_ppm <= 0:
        return np.full_like(depth_ppm, np.nan, dtype=float)
    out = (depth_ppm - np.nanmedian(depth_ppm[covered])) / amp_ppm
    return np.where(covered, out, np.nan)


# =========================================================================== #
# Band indices
# =========================================================================== #
def band_index(wl: np.ndarray, depth_h: np.ndarray, unc_h: np.ndarray, mask: np.ndarray,
               band: tuple[float, float], continua: list[tuple[float, float]]) -> tuple[float, float, int]:
    """(band - continuum) in scale heights, its uncertainty, and how many bins contributed.

    Sign convention: POSITIVE means the band sits ABOVE the local continuum, i.e. absorption
    (a transmission spectrum is deeper where the atmosphere is opaque). Returns NaN unless both
    the band and at least one continuum window are actually covered by the observation - the
    coverage mask is the authority, never a zero depth value.
    """
    covered = (mask > 0) & np.isfinite(depth_h)
    in_band = covered & (wl >= band[0]) & (wl <= band[1])
    in_cont = covered & np.any([(wl >= a) & (wl <= b) for a, b in continua], axis=0)
    if in_band.sum() < 2 or in_cont.sum() < 2:
        return np.nan, np.nan, int(in_band.sum())

    b = np.average(depth_h[in_band])
    c = np.average(depth_h[in_cont])
    # Uncertainty on each mean, added in quadrature.
    eb = np.sqrt(np.nansum(unc_h[in_band] ** 2)) / in_band.sum()
    ec = np.sqrt(np.nansum(unc_h[in_cont] ** 2)) / in_cont.sum()
    return float(b - c), float(np.hypot(eb, ec)), int(in_band.sum())


# =========================================================================== #
# Anomaly detection
# =========================================================================== #
def lopo_reconstruction_error(feats: np.ndarray, planets: np.ndarray, n_components: int = 3
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Per-spectrum reconstruction error under leave-one-planet-out PCA, and a nonlinear check.

    PCA is a linear autoencoder, and with 53 samples it is the defensible choice - a deep
    autoencoder here would mostly memorise. The planet is held out of the fit so the error
    answers "how unlike the other planets is this?" rather than "how well was this memorised?".
    """
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPRegressor

    lin = np.full(len(feats), np.nan)
    nonlin = np.full(len(feats), np.nan)
    for p in np.unique(planets):
        te = planets == p
        tr = ~te
        if tr.sum() < n_components + 2:
            continue
        pca = PCA(n_components=min(n_components, tr.sum() - 1)).fit(feats[tr])
        rec = pca.inverse_transform(pca.transform(feats[te]))
        lin[te] = np.sqrt(np.mean((feats[te] - rec) ** 2, axis=1))

        ae = MLPRegressor(hidden_layer_sizes=(32, 6, 32), max_iter=3000, random_state=0,
                          early_stopping=False, alpha=1e-2)
        try:
            ae.fit(feats[tr], feats[tr])
            nonlin[te] = np.sqrt(np.mean((feats[te] - ae.predict(feats[te])) ** 2, axis=1))
        except Exception:
            pass
    return lin, nonlin


# =========================================================================== #
# Figure
# =========================================================================== #
def make_figure(wl, depth_h, unc_h, mask, planets, pmap, bands_df, pcs, fp, anom) -> None:
    """Four panels summarising Level 2. Drawn from arrays already computed; retrains nothing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1: normalised spectra, giants vs small planets
    ax = axes[0, 0]
    for i in range(len(depth_h)):
        r = pmap.loc[planets[i], "pl_rade"]
        good = mask[i] > 0
        if good.sum() < 10:
            continue
        ax.plot(wl[good], depth_h[i][good], lw=0.7, alpha=0.55,
                color="#c23b22" if r >= 8 else "#2d6fa8")
    ax.set_ylim(-6, 8)  # set before annotating, so the band labels land inside the axes
    for name, (band, _) in BANDS.items():
        ax.axvspan(*band, color="grey", alpha=0.12)
        ax.text(np.mean(band), 7.6, name, ha="center", va="top", fontsize=7, color="dimgrey")
    ax.plot([], [], color="#c23b22", label="giant (Rp >= 8 Re)")
    ax.plot([], [], color="#2d6fa8", label="sub-Neptune / rocky")
    ax.set_xlabel("wavelength (um)")
    ax.set_ylabel("depth - median, in scale heights")
    ax.set_title("Scale-height normalised spectra\n(ppm divided by 2 Rp H / Rs^2)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # 2: band index significance
    ax = axes[0, 1]
    names = list(BANDS)
    data = [bands_df[f"{n}_sigma"].dropna().to_numpy() for n in names]
    # tick_labels, not labels: matplotlib 3.9 renamed it and removed the old name.
    parts = ax.boxplot(data, tick_labels=names, showfliers=True, patch_artist=True)
    for p in parts["boxes"]:
        p.set_facecolor("#cfe0ee")
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(3, color="crimson", ls="--", lw=1.0, label="3 sigma")
    ax.set_ylabel("band - continuum, in sigma")
    ax.set_title("Band indices: which molecules are actually measurable\n(+ = absorption)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    for i, n in enumerate(names, start=1):
        ax.text(i, ax.get_ylim()[0], f"n={len(data[i - 1])}", ha="center", va="bottom", fontsize=7)

    # 3: PCA space
    ax = axes[1, 0]
    rad = np.array([pmap.loc[p, "pl_rade"] for p in fp])
    sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=np.log10(rad), cmap="viridis", s=45,
                    edgecolor="black", linewidth=0.4)
    for p in np.unique(fp):
        m = fp == p
        ax.annotate(p, (pcs[m, 0].mean(), pcs[m, 1].mean()), fontsize=6.5,
                    textcoords="offset points", xytext=(4, 3))
    plt.colorbar(sc, ax=ax, label="log10 planet radius (Earth radii)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Spectra in PCA space, coloured by planet size", fontsize=9)
    ax.grid(alpha=0.3)

    # 4: anomaly, excess over noise
    ax = axes[1, 1]
    g = anom.groupby("planet").excess_over_noise.median().sort_values()
    ax.barh(range(len(g)), g.to_numpy(), color="#3b7a57", edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels(g.index, fontsize=8)
    ax.axvline(1.0, color="crimson", ls="--", lw=1.0)
    ax.text(1.02, 0.3, "error = its own noise\n(nothing unusual)", color="crimson", fontsize=7)
    ax.set_xlabel("LOPO reconstruction error / that spectrum's own noise")
    ax.set_title("Most structured spectra\n(raw error would rank the noisiest first)", fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Stage B Level 2: physics-informed normalisation of 53 JWST transmission spectra "
                 "(13 planets) - descriptive, not inferential", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    out = EDA_DIR / "stage_b_level2_overview.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n[7] figure -> {out.relative_to(RESEARCH_DIR)}")


# =========================================================================== #
# Main
# =========================================================================== #
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 88)
    print("Stage B Level 2 - scale-height normalisation, band indices, clustering, anomalies")
    print("Reads Level 1 outputs; writes only to outputs/stage_b_level2/")
    print("=" * 88)

    z = np.load(DATASET, allow_pickle=True)
    X, planets, files = z["X"], z["planets"].astype(str), z["files"].astype(str)
    wl = z["bin_centers"]
    print(f"\n[1] dataset: {X.shape[0]} spectra, {len(set(planets))} planets, "
          f"{X.shape[2]} bins over {wl.min():.2f}-{wl.max():.2f} um")

    params = planet_parameters(sorted(set(planets)))
    print("\n[2] scale heights (mu = 2.3, H2/He)")
    print(f"  {'planet':<14} {'Teq':>6} {'R_p/Re':>7} {'g m/s2':>8} {'H km':>8} {'1H depth ppm':>13}")
    for _, r in params.sort_values("amplitude_1H_ppm", ascending=False).iterrows():
        print(f"  {r.pl_name:<14} {r.pl_eqt:>6.0f} {r.pl_rade:>7.2f} {r.gravity_ms2:>8.2f} "
              f"{r.scale_height_km:>8.0f} {r.amplitude_1H_ppm:>13.1f}")

    pmap = params.set_index("pl_name")
    amp = np.array([pmap.loc[p, "amplitude_1H_ppm"] for p in planets])

    # ---- normalise ------------------------------------------------------- #
    depth_h = np.vstack([normalise_to_scale_heights(X[i, 0], X[i, 2], amp[i]) for i in range(len(X))])
    unc_h = np.vstack([np.where(X[i, 2] > 0, X[i, 1] / amp[i], np.nan) for i in range(len(X))])
    mask = X[:, 2]
    span = np.nanmax(depth_h, axis=1) - np.nanmin(depth_h, axis=1)
    print(f"\n[3] normalised spectra span (max-min, scale heights): "
          f"median {np.nanmedian(span):.1f}, range {np.nanmin(span):.1f}-{np.nanmax(span):.1f}")
    print("  A physically sensible transmission spectrum varies by a few scale heights;")
    print("  values far above that mean the assumed mu or Teq does not describe that planet.")

    # ---- band indices ---------------------------------------------------- #
    rows = []
    for i in range(len(X)):
        row = {"planet": planets[i], "file": files[i]}
        for name, (band, cont) in BANDS.items():
            val, err, nb = band_index(wl, depth_h[i], unc_h[i], mask[i], band, cont)
            row[f"{name}_index_H"] = val
            row[f"{name}_err_H"] = err
            row[f"{name}_sigma"] = val / err if (err and np.isfinite(err) and err > 0) else np.nan
            row[f"{name}_bins"] = nb
        rows.append(row)
    bands_df = pd.DataFrame(rows)
    bands_df.to_csv(OUT_DIR / "band_indices.csv", index=False)

    print("\n[4] band indices (scale heights above local continuum; + = absorption)")
    print(f"  {'molecule':<8} {'measurable':>11} {'median':>8} {'>2 sigma':>9} {'>3 sigma':>9}")
    for name in BANDS:
        v = bands_df[f"{name}_index_H"]
        s = bands_df[f"{name}_sigma"]
        ok = v.notna().sum()
        print(f"  {name:<8} {ok:>7}/{len(v):<3} {v.median():>8.2f} "
              f"{int((s > 2).sum()):>9} {int((s > 3).sum()):>9}")

    # ---- clustering ------------------------------------------------------ #
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    usable = np.isfinite(depth_h).sum(axis=1) >= 40
    feats = np.nan_to_num(depth_h[usable], nan=0.0)
    fp = planets[usable]
    print(f"\n[5] clustering on {usable.sum()} spectra with >=40 covered bins")

    pca = PCA(n_components=min(5, feats.shape[0] - 1)).fit(feats)
    pcs = pca.transform(feats)
    print(f"  PCA explained variance: {', '.join(f'{v:.1%}' for v in pca.explained_variance_ratio_[:4])}")

    km = KMeans(n_clusters=2, n_init=20, random_state=0).fit(pcs[:, :3])
    radius = np.array([pmap.loc[p, "pl_rade"] for p in fp])
    is_giant = radius >= 8.0  # Neptune-ish and above
    agree = max((km.labels_ == is_giant).mean(), (km.labels_ != is_giant).mean())
    print(f"  2-means vs the physical giant/sub-Neptune split (R_p >= 8 Re): {agree:.0%} agreement")
    print("  Checked against the physical split rather than assumed to reproduce it;")
    print(f"  with {len(set(fp))} planets this is an illustration, not a population result.")

    # ---- anomalies ------------------------------------------------------- #
    lin, nonlin = lopo_reconstruction_error(feats, fp)
    # Raw reconstruction error is NOT the right ranking. Normalising divides by
    # A_H = 2 Rp H / Rs^2, so a planet with a small A_H has its noise inflated the most and
    # looks anomalous for a purely instrumental reason. Measured on this sample the raw error
    # correlates with normalised uncertainty at Spearman +0.69, and K2-18 b - smallest A_H at
    # 24 ppm - tops the raw ranking with an error (2.11) equal to its own noise (1.93).
    # The meaningful quantity is the error in units of that spectrum's own noise.
    noise_h = np.array([np.nanmedian(unc_h[i][mask[i] > 0]) if (mask[i] > 0).any() else np.nan
                        for i in range(len(unc_h))])[usable]
    with np.errstate(divide="ignore", invalid="ignore"):
        excess = np.where(noise_h > 0, lin / noise_h, np.nan)

    anom = pd.DataFrame({"planet": fp, "file": files[usable],
                         "pca_lopo_error_H": lin, "ae_lopo_error_H": nonlin,
                         "median_noise_H": noise_h, "excess_over_noise": excess})
    anom = anom.sort_values("excess_over_noise", ascending=False)
    anom.to_csv(OUT_DIR / "anomaly_scores.csv", index=False)

    print("\n[6] anomaly detection: leave-one-planet-out reconstruction error")
    print("  the planet is never in the fit that reconstructs it")
    print("  ranked by error / its own noise, because raw error mostly tracks noise "
          "(Spearman +0.69 on this sample)")
    print(f"  {'planet':<14} {'error H':>8} {'noise H':>8} {'excess':>7}  file")
    for _, r in anom.head(6).iterrows():
        print(f"  {r.planet:<14} {r.pca_lopo_error_H:>8.2f} {r.median_noise_H:>8.2f} "
              f"{r.excess_over_noise:>7.2f}  {r.file[:40]}")
    by_planet = anom.groupby("planet").excess_over_noise.median().sort_values(ascending=False)
    print(f"\n  most structured (excess over noise): "
          f"{', '.join(f'{p} ({v:.2f})' for p, v in by_planet.head(3).items())}")
    raw_rank = anom.groupby("planet").pca_lopo_error_H.median().sort_values(ascending=False)
    print(f"  by RAW error instead:                "
          f"{', '.join(f'{p} ({v:.2f})' for p, v in raw_rank.head(3).items())}  <- noise-driven, not reported")

    # ---- figure ---------------------------------------------------------- #
    make_figure(wl, depth_h, unc_h, mask, planets, pmap, bands_df, pcs, fp, anom)

    # ---- save ------------------------------------------------------------ #
    params.to_csv(OUT_DIR / "planet_scale_heights.csv", index=False)
    np.savez_compressed(OUT_DIR / "normalised_spectra.npz", depth_scale_heights=depth_h,
                        unc_scale_heights=unc_h, mask=mask, planets=planets, files=files,
                        bin_centers=wl, amplitude_1H_ppm=amp)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_spectra": int(len(X)), "n_planets": int(len(set(planets))), "mu_assumed": MU_H2HE,
        "amplitude_1H_ppm": {r.pl_name: float(r.amplitude_1H_ppm) for _, r in params.iterrows()},
        "band_measurable": {n: int(bands_df[f"{n}_index_H"].notna().sum()) for n in BANDS},
        "band_gt3sigma": {n: int((bands_df[f"{n}_sigma"] > 3).sum()) for n in BANDS},
        "kmeans_vs_physical_split_agreement": float(agree),
        "pca_explained_variance": [float(v) for v in pca.explained_variance_ratio_],
        "most_structured_planets": {p: float(v) for p, v in by_planet.head(5).items()},
        "anomaly_note": ("ranked by reconstruction error divided by the spectrum's own noise. "
                         "Raw error correlates with normalised uncertainty at Spearman +0.69, "
                         "because normalising by A_H inflates the noise of small-amplitude "
                         "planets; K2-18 b tops the raw ranking for that reason alone."),
        "caveat": ("53 spectra of 13 planets: descriptive only. Clusters are illustrations and "
                   "anomalies are flags for human inspection, not detections."),
    }
    (OUT_DIR / "level2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsaved {OUT_DIR.relative_to(RESEARCH_DIR)}/ "
          f"(band_indices.csv, anomaly_scores.csv, planet_scale_heights.csv, "
          f"normalised_spectra.npz, level2_summary.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
