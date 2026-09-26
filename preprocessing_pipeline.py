"""
preprocessing_pipeline.py
=========================

Turns a (star, KOI/TOI) pair into model-ready input for the Stage 1 CNN+LSTM:
a GLOBAL view (whole phase-folded orbit) and a LOCAL view (zoom on the transit),
following the dual-view design of Astronet (Shallue & Vanderburg 2018).

Labels are PER KOI, not per star: one light curve can contain several transit
signals with different dispositions, so a star's light curve is flattened once
(with every known transit on that star protected) and then folded separately for
each KOI using that KOI's own period, epoch and duration from the split files.

Functions (each importable and testable on its own):
    load_and_flatten(path, ...)               Step 1  load CSV, quality-mask, detrend
    phase_fold(flat, period, epoch)           Step 2  fold on ONE KOI's ephemeris
    build_global_view(folded, n_bins)         Step 3  2001-bin global view
    build_local_view(folded, duration, n_bins)Step 3  201-bin local view
    fetch_lightcurve_via_api(id, mission)     Step 5  lightkurve/MAST pull -> same CSV format
    build_dataset(split_df, output_dir, ...)  Step 6  one .npz per KOI + index.csv

Command line:
    python preprocessing_pipeline.py sample      # Steps 1-4 on the 36 mentor light curves
    python preprocessing_pipeline.py kepler90    # per-KOI folding check on Kepler-90 (API)
    python preprocessing_pipeline.py api-test    # Step 5: 5 Kepler train targets via API
    python preprocessing_pipeline.py build-test  # Step 6: first 20 rows of Kepler train

Reads splits via data_manifest.load_stage1_catalogs() and light curves from the
mentor pack (read-only). Writes views to detection_views/ and figures to eda/;
downloaded light curves are cached outside the project (see LK_CACHE); with
build-train --discard-raw each star's raw download is deleted once its views are saved.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import sys
import time as _time
import warnings
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*tpfmodel submodule is not available.*")
import lightkurve as lk  # noqa: E402
from lightkurve.utils import KeplerQualityFlags, TessQualityFlags  # noqa: E402

import data_manifest as dm  # noqa: E402

RESEARCH_DIR = dm.RESEARCH_DIR
LC_DIR = RESEARCH_DIR / "exoplanet_research_data" / "02_lightcurves"
# Light-curve cache: on D: while that drive was attached (user decision 2026-09-17: a full Kepler build
# keeps ~60 GB of raw downloads). On a laptop without D: (2026-09-21) it falls back to the home folder,
# and build-train --discard-raw keeps it to the stars in flight. Override with EXO_LIGHTKURVE_CACHE.
def _default_cache() -> Path:
    """EXO_LIGHTKURVE_CACHE if set; else D:/lightkurve_cache when a D: drive is attached (the original
    setup); else ~/lightkurve_cache. The fallback is outside the OneDrive-synced project on purpose."""
    env = os.environ.get("EXO_LIGHTKURVE_CACHE")
    if env:
        return Path(env)
    if Path("D:/").exists():
        return Path("D:/lightkurve_cache")
    return Path.home() / "lightkurve_cache"


LK_CACHE = _default_cache()
EDA_DIR = RESEARCH_DIR / "eda"
VIEWS_DIR = RESEARCH_DIR / "detection_views"
LOG_PATH = RESEARCH_DIR / "run_log.txt"
# Curated, manually reviewed exclusion lists (inputs shared by every split's build). Each maps a
# category name to a CSV with at least object_id, reason and note columns. Unlike unreliable-
# ephemeris exclusions (computed during each build), these are decisions made from evidence review.
EXCLUSIONS_DIR = RESEARCH_DIR / "exclusions"
MANUAL_EXCLUSIONS = {"circumbinary planet": EXCLUSIONS_DIR / "excluded_circumbinary_planet.csv"}

# ---- Quality masking ------------------------------------------------------- #
# lightkurve's HARD bitmask: DEFAULT (attitude tweaks, safe mode, coarse/earth point,
# desaturations, manual exclude, ...) plus cosmic rays, stray light, sensitivity
# dropouts and possible thruster firings - flags that corrupt flux even when flux is
# not NaN. DEFAULT alone removes nothing extra in these files; HARDEST would also drop
# rolling-band cadences (56% of Kepler-22's quarter), which detrending handles.
# TESS additionally drops "Planet Search Exclude" (8192): SPOC itself excluded those
# cadences from its transit search (scattered light).
QUALITY_BITMASK = {
    "Kepler": KeplerQualityFlags.HARD_BITMASK,
    "TESS": TessQualityFlags.HARD_BITMASK | 8192,
}

# ---- Detrending ------------------------------------------------------------ #
MIN_WINDOW_DAYS = 1.0          # Savitzky-Golay window never shorter than this
WINDOW_DURATION_FACTOR = 3.0   # ... and at least 3x the longest transit on the star
TRANSIT_MASK_FACTOR = 1.5      # protect 1.5x each transit duration from the trend fit
SG_POLYORDER = 2
BREAK_TOLERANCE = 5            # gaps > 5 cadences start a new detrending segment
UPPER_OUTLIER_SIGMA = 3.0      # clip only upward outliers; dips are kept
SHORT_CADENCE_MINUTES = 1.0    # below this (TESS 20 s) data is binned before detrending
BIN_TO_MINUTES = 2.0

# ---- Views (Astronet defaults) --------------------------------------------- #
GLOBAL_BINS = 2001
LOCAL_BINS = 201
LOCAL_BIN_WIDTH_FACTOR = 0.16  # local bin width = 0.16 x duration
LOCAL_NUM_DURATIONS = 4        # local window = +/- 4 durations

LABEL_INDEX = {"FALSE POSITIVE": 0, "CANDIDATE": 1, "CONFIRMED": 2}
TESS_BJD_OFFSET = 2457000.0    # TOI epochs are BJD; TESS light curves use BTJD
MAX_DRIFT_FRACTION = 0.5       # ephemeris unreliable if propagated timing error > 0.5x duration

K2_OUT_OF_SCOPE = (
    "K2 light-curve modeling is out of scope for this project: K2 light curves carry large "
    "pointing-drift systematics (from the spacecraft's two-wheel operation) that need dedicated "
    "correction (e.g. K2SFF or EVEREST) before transits can be folded, which this pipeline does not "
    "do. K2 catalog data (splits/k2_planets_candidates_split.csv) remains usable for EDA and "
    "statistics only.")


# =========================================================================== #
# Data containers
# =========================================================================== #
@dataclass
class FlattenedLightCurve:
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    mission: str
    meta: dict = field(default_factory=dict)


@dataclass
class FoldedLightCurve:
    phase: np.ndarray   # days from transit centre, in [-period/2, period/2), sorted
    flux: np.ndarray
    period: float
    epoch: float


@dataclass
class View:
    values: np.ndarray  # normalised: median 0, deepest bin -1 (Astronet convention)
    coverage: float     # fraction of bins containing at least one cadence
    scale: float        # |min| of the median-subtracted view before normalising
    n_points: int


@dataclass
class Ephemeris:
    object_id: str      # e.g. K00351.01 or TOI 700.01
    star_id: int
    mission: str
    period: float       # days
    epoch: float        # in the light curve's time system (BKJD / BTJD)
    duration: float     # days
    label: str | None
    period_err: float = np.nan  # days (catalogue +1 sigma)
    epoch_err: float = np.nan   # days


# =========================================================================== #
# Catalog helpers
# =========================================================================== #
def mission_of(df: pd.DataFrame) -> str:
    if "kepid" in df.columns:
        return "Kepler"
    if "tid" in df.columns:
        return "TESS"
    if "epic_hostname" in df.columns:
        return "K2"
    raise ValueError("cannot tell the mission from the split columns")


def ephemeris_from_row(row: pd.Series | dict, mission: str) -> Ephemeris:
    """One KOI/TOI's own period, epoch and duration, converted to days and the LC time system."""
    row = dict(row)
    if mission == "Kepler":
        return Ephemeris(object_id=str(row["kepoi_name"]), star_id=int(row["kepid"]), mission=mission,
                         period=float(row["koi_period"]), epoch=float(row["koi_time0bk"]),  # BKJD already
                         duration=float(row["koi_duration"]) / 24.0, label=row.get("label_harmonized"),
                         period_err=abs(float(row.get("koi_period_err1", np.nan))),
                         epoch_err=abs(float(row.get("koi_time0bk_err1", np.nan))))
    if mission == "TESS":
        # object_name lets a caller supply a real planet name (e.g. "WASP-39 b") for targets that
        # are not TOIs; without it the TOI number is used exactly as before.
        oid = str(row["object_name"]) if row.get("object_name") else f"TOI {row['toi']}"
        return Ephemeris(object_id=oid, star_id=int(row["tid"]), mission=mission,
                         period=float(row["pl_orbper"]), epoch=float(row["pl_tranmid"]) - TESS_BJD_OFFSET,
                         duration=float(row["pl_trandurh"]) / 24.0, label=row.get("label_harmonized"),
                         period_err=abs(float(row.get("pl_orbpererr1", np.nan))),
                         epoch_err=abs(float(row.get("pl_tranmiderr1", np.nan))))
    if mission == "K2":
        raise NotImplementedError(K2_OUT_OF_SCOPE)
    raise ValueError(f"unknown mission {mission!r}")


def validate_ephemeris(eph: Ephemeris) -> None:
    for name in ("period", "epoch", "duration"):
        value = getattr(eph, name)
        if not np.isfinite(value) or (name != "epoch" and value <= 0):
            raise ValueError(f"{eph.object_id}: invalid {name} ({value})")


@lru_cache(maxsize=1)
def stage1_catalogs() -> dict[str, pd.DataFrame]:
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return dm.load_stage1_catalogs()


def catalog_for(mission: str) -> pd.DataFrame:
    name = {"Kepler": "koi_cumulative_split.csv", "TESS": "toi_tess_candidates_split.csv",
            "K2": "k2_planets_candidates_split.csv"}[mission]
    return stage1_catalogs()[name]


def star_ephemerides(star_id: int, mission: str) -> list[Ephemeris]:
    """Every KOI/TOI on a star from the FULL catalog (not just the slice being processed)."""
    cat = catalog_for(mission)
    col = "kepid" if mission == "Kepler" else "tid"
    return [ephemeris_from_row(r, mission) for r in cat[cat[col] == star_id].to_dict("records")]


def local_lightcurve_index() -> dict[tuple[str, int], Path]:
    """(mission, star id) -> mentor CSV. TESS manifest rows lack file names, so those
    are resolved from the file-naming pattern tess_TIC<id>_<target>.csv."""
    man = pd.read_csv(LC_DIR / "lightcurve_manifest.csv")
    index = {}
    for r in man.to_dict("records"):
        mission, star = r["mission"], int(str(r["id"]).split()[1])
        if isinstance(r.get("file_csv"), str):
            path = LC_DIR / mission.lower() / r["file_csv"]
        else:
            path = LC_DIR / mission.lower() / f"{mission.lower()}_TIC{star}_{r['target']}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"manifest row {r['id']} -> {path} not found")
        index[(mission, star)] = path
    return index


# =========================================================================== #
# Step 1: load + quality mask + detrend
# =========================================================================== #
def _infer_mission(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("kepler"):
        return "Kepler"
    if name.startswith("tess"):
        return "TESS"
    raise ValueError(f"cannot infer mission from file name {path.name}; pass mission=")


def transit_mask(time: np.ndarray, ephemerides: list[Ephemeris], factor: float = TRANSIT_MASK_FACTOR) -> np.ndarray:
    """True for cadences within factor x duration/2 of any listed transit centre."""
    mask = np.zeros(len(time), dtype=bool)
    for e in ephemerides:
        if not (np.isfinite(e.period) and e.period > 0 and np.isfinite(e.epoch) and np.isfinite(e.duration)):
            continue
        phase = (time - e.epoch + 0.5 * e.period) % e.period - 0.5 * e.period
        mask |= np.abs(phase) < factor * e.duration / 2
    return mask


def load_and_flatten(light_curve_path: str | Path, mission: str | None = None,
                     ephemerides: list[Ephemeris] | None = None,
                     window_days: float | None = None) -> FlattenedLightCurve:
    """Load a light-curve CSV (time, flux, flux_err, quality), quality-mask and detrend it.

    - Cadences are removed if their quality flag hits the mission bitmask, whether
      or not flux is NaN; then remaining NaN time/flux cadences are dropped.
    - Cadences shorter than 1 minute (TESS 20 s) are binned to 2 minutes first.
    - Detrending: Savitzky-Golay (lightkurve.flatten), polyorder 2, window =
      max(1 d, 3x the longest transit on the star). Every known transit in
      `ephemerides` is masked out of the trend fit so the filter cannot eat it.
    - Upward outliers > 3 sigma are clipped; downward points (transits) are kept.
    """
    path = Path(light_curve_path)
    mission = mission or _infer_mission(path)
    raw = pd.read_csv(path)
    missing = {"time", "flux", "flux_err", "quality"} - set(raw.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    meta = {"source_file": path.name, "mission": mission, "n_raw": len(raw)}

    raw = raw[raw["time"].notna()]
    q = raw["quality"].fillna(0).astype(np.int64).to_numpy()
    flagged = (q & QUALITY_BITMASK[mission]) != 0
    meta["n_quality_masked"] = int(flagged.sum())
    meta["n_quality_masked_with_flux"] = int((flagged & raw["flux"].notna().to_numpy()).sum())
    kept = raw[~flagged]
    good = kept["flux"].notna() & np.isfinite(kept["flux"])
    meta["n_nan_flux_dropped"] = int((~good).sum())
    kept = kept[good]
    if len(kept) < 50:
        raise ValueError(f"{path.name}: only {len(kept)} usable cadences after masking")

    t = kept["time"].to_numpy(float)
    f = kept["flux"].to_numpy(float)
    e = kept["flux_err"].to_numpy(float)
    e = np.where(np.isfinite(e), e, np.nanmedian(e))
    cadence_min = float(np.median(np.diff(t)) * 1440)
    meta["cadence_minutes_raw"] = round(cadence_min, 3)

    lc = lk.LightCurve(time=t, flux=f, flux_err=e)
    meta["binned_to_2min"] = False
    if cadence_min < SHORT_CADENCE_MINUTES:
        # 20 s TESS data: ~6x the cadences, more cosmic-ray hits and NaNs. Binning to 2 min
        # gives the same cadence as the other TESS files, so one detrending setup applies.
        lc = lc.bin(time_bin_size=BIN_TO_MINUTES / 1440)
        lc = lc[np.isfinite(lc.flux.value)]
        meta["binned_to_2min"] = True
    cadence_days = float(np.median(np.diff(lc.time.value)))

    ephemerides = ephemerides or []
    longest = max([x.duration for x in ephemerides if np.isfinite(x.duration)], default=0.0)
    window_days = window_days or max(MIN_WINDOW_DAYS, WINDOW_DURATION_FACTOR * longest)
    window = int(round(window_days / cadence_days)) | 1  # odd number of cadences
    window = max(window, SG_POLYORDER + 3)
    mask = transit_mask(lc.time.value, ephemerides)
    meta.update({"window_days": round(window_days, 4), "window_cadences": window,
                 "n_transit_protected": int(mask.sum())})

    flat = lc.flatten(window_length=window, polyorder=SG_POLYORDER, break_tolerance=BREAK_TOLERANCE,
                      mask=mask, sigma=3, niters=3)
    flat = flat[np.isfinite(flat.flux.value)]
    flat, outliers = flat.remove_outliers(sigma_lower=np.inf, sigma_upper=UPPER_OUTLIER_SIGMA, return_mask=True)
    meta["n_upper_outliers_clipped"] = int(outliers.sum())
    meta["n_final"] = len(flat)
    return FlattenedLightCurve(time=flat.time.value.astype(float), flux=flat.flux.value.astype(float),
                               flux_err=flat.flux_err.value.astype(float), mission=mission, meta=meta)


# =========================================================================== #
# Step 2: per-KOI folding
# =========================================================================== #
def phase_fold(flattened_lightcurve: FlattenedLightCurve, period: float, epoch: float) -> FoldedLightCurve:
    """Fold on ONE KOI's period and epoch. Phase is in days from that KOI's transit centre."""
    if not (np.isfinite(period) and period > 0 and np.isfinite(epoch)):
        raise ValueError(f"invalid ephemeris: period={period}, epoch={epoch}")
    lc = flattened_lightcurve
    phase = (lc.time - epoch + 0.5 * period) % period - 0.5 * period
    order = np.argsort(phase)
    return FoldedLightCurve(phase=phase[order], flux=lc.flux[order], period=float(period), epoch=float(epoch))


def remove_other_transits(flat: FlattenedLightCurve, target: Ephemeris,
                          others: list[Ephemeris]) -> tuple[FlattenedLightCurve, list[str]]:
    """Drop cadences inside OTHER KOIs' transits before folding `target`.

    A sibling whose transit windows would remove more than half of the target's own
    in-transit cadences (same or aliased ephemeris) is left in, so a KOI is never
    erased by its own duplicate.
    """
    own = transit_mask(flat.time, [target], factor=1.0)
    keep = np.ones(len(flat.time), dtype=bool)
    removed = []
    for o in others:
        if o.object_id == target.object_id:
            continue
        m = transit_mask(flat.time, [o])
        if own.any() and (m & own).sum() > 0.5 * own.sum():
            continue
        if m.any():
            keep &= ~m
            removed.append(o.object_id)
    out = replace(flat, time=flat.time[keep], flux=flat.flux[keep], flux_err=flat.flux_err[keep],
                  meta={**flat.meta, "n_other_transit_cadences_removed": int((~keep).sum())})
    return out, removed


# =========================================================================== #
# Step 3: views
# =========================================================================== #
def _median_binned_view(x: np.ndarray, y: np.ndarray, n_bins: int, bin_width: float,
                        x_min: float, x_max: float) -> View:
    """Astronet-style median binning: n_bins bins of width bin_width, evenly spaced over
    [x_min, x_max]; empty bins take the median of all points in range. Then centre at 0
    and scale so the deepest bin is -1."""
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    in_range = (x >= x_min) & (x <= x_max)
    xs, ys = x[in_range], y[in_range]
    fill = float(np.median(ys)) if len(ys) else 1.0
    spacing = (x_max - x_min - bin_width) / (n_bins - 1)
    starts = x_min + spacing * np.arange(n_bins)
    lo = np.searchsorted(xs, starts, side="left")
    hi = np.searchsorted(xs, starts + bin_width, side="left")
    values = np.full(n_bins, fill)
    filled = hi > lo
    for i in np.flatnonzero(filled):
        values[i] = np.median(ys[lo[i]:hi[i]])
    values = values - np.median(values)
    scale = float(abs(values.min()))
    if scale > 0:
        values = values / scale
    return View(values=values, coverage=float(filled.mean()), scale=scale, n_points=int(len(xs)))


def build_global_view(folded_lc: FoldedLightCurve, n_bins: int = GLOBAL_BINS, duration: float | None = None) -> View:
    """Whole orbit, [-P/2, P/2), in n_bins bins. Bin width P/n_bins, widened to 0.16x
    duration when that is larger (Astronet), so short transits are not split across bins."""
    p = folded_lc.period
    width = p / n_bins
    if duration is not None:
        width = max(width, LOCAL_BIN_WIDTH_FACTOR * duration)
    return _median_binned_view(folded_lc.phase, folded_lc.flux, n_bins, width, -p / 2, p / 2)


def build_local_view(folded_lc: FoldedLightCurve, duration: float, n_bins: int = LOCAL_BINS) -> View:
    """Zoom on the transit: +/- 4 durations (capped at half a period), bin width 0.16x duration."""
    if not (np.isfinite(duration) and duration > 0):
        raise ValueError(f"invalid duration {duration}")
    p = folded_lc.period
    half = min(p / 2, LOCAL_NUM_DURATIONS * duration)
    return _median_binned_view(folded_lc.phase, folded_lc.flux, n_bins, LOCAL_BIN_WIDTH_FACTOR * duration, -half, half)


# =========================================================================== #
# One KOI end to end (after the star's light curve is flattened)
# =========================================================================== #
def transit_diagnostics(folded: FoldedLightCurve, duration: float) -> dict:
    """In-transit cadence count and measured depth (ppm) with its significance."""
    inside = np.abs(folded.phase) < duration / 2
    outside = (np.abs(folded.phase) > duration) & (np.abs(folded.phase) < max(3 * duration, 0.5))
    if not outside.any():
        outside = ~inside
    n_in = int(inside.sum())
    if n_in == 0:
        return {"n_in_transit": 0, "depth_ppm": np.nan, "depth_snr": np.nan, "transit_observed": False}
    base = np.median(folded.flux[outside])
    scatter = 1.4826 * np.median(np.abs(folded.flux[outside] - base))
    depth = base - np.median(folded.flux[inside])
    snr = depth / (scatter / np.sqrt(n_in)) if scatter > 0 else np.nan
    return {"n_in_transit": n_in, "depth_ppm": float(depth * 1e6), "depth_snr": float(snr), "transit_observed": True}


def ephemeris_drift_hours(eph: Ephemeris, time: np.ndarray) -> float:
    """1-sigma transit-time uncertainty (hours) propagated to the light curve's farthest cadence:
    sqrt(epoch_err^2 + (n_cycles * period_err)^2). NaN when the catalogue gives no errors."""
    if not (np.isfinite(eph.period_err) and np.isfinite(eph.epoch_err)) or len(time) == 0:
        return np.nan
    n_cycles = np.max(np.abs(time - eph.epoch)) / eph.period
    return float(np.hypot(eph.epoch_err, n_cycles * eph.period_err) * 24)


def process_koi(flat: FlattenedLightCurve, eph: Ephemeris, siblings: list[Ephemeris]) -> dict:
    validate_ephemeris(eph)
    cleaned, removed = remove_other_transits(flat, eph, siblings)
    folded = phase_fold(cleaned, eph.period, eph.epoch)
    gv = build_global_view(folded, GLOBAL_BINS, duration=eph.duration)
    lv = build_local_view(folded, eph.duration, LOCAL_BINS)
    diag = transit_diagnostics(folded, eph.duration)
    drift = ephemeris_drift_hours(eph, cleaned.time)
    # A catalogue ephemeris propagated far from its epoch (e.g. a 2018 TOI folded on 2025 data)
    # can put the transit hours away from phase 0. The fold is NOT re-centred on the data (that
    # would manufacture a dip for false positives); the row is flagged instead.
    reliable = bool(drift <= MAX_DRIFT_FRACTION * eph.duration * 24) if np.isfinite(drift) else None
    return {"global": gv, "local": lv, "folded": folded, "other_kois_removed": removed,
            "ephemeris_drift_h": drift, "ephemeris_reliable": reliable, **diag}


# =========================================================================== #
# Step 5: API fetch
# =========================================================================== #
def invalid_cached_fits(star: int, mission: str, cache_dir: Path | None = None) -> list[Path]:
    """Cached FITS files for this star (in the default download folder) that are empty,
    truncated (size not a multiple of the 2880-byte FITS block) or lack a FITS header."""
    base = Path(cache_dir or LK_CACHE) / "mast" / "mastDownload"
    pattern = f"Kepler/kplr{star:09d}_*/*.fits" if mission == "Kepler" else f"TESS/*{star:016d}*/*.fits"
    bad = []
    for f in base.glob(pattern):
        size = f.stat().st_size
        if size == 0 or size % 2880:
            bad.append(f)
            continue
        with f.open("rb") as fh:
            if not fh.read(9).startswith(b"SIMPLE  ="):
                bad.append(f)
    return bad


def raw_cache_paths(star: int, mission: str, cache_dir: Path | None = None) -> list[Path]:
    """Everything fetch_lightcurve_via_api() stored for one star: its MAST FITS files, any
    mast_redownload/ retry folders, the S3 folder and the stitched CSV. Nothing belonging to
    another star matches."""
    cache_dir = Path(cache_dir or LK_CACHE)
    prefix = {"Kepler": "kepler_KIC", "TESS": "tess_TIC"}[mission]
    fits = f"Kepler/kplr{star:09d}_*/*.fits" if mission == "Kepler" else f"TESS/*{star:016d}*/*.fits"
    paths = list((cache_dir / "mast" / "mastDownload").glob(fits))
    paths += [p for p in (cache_dir / "mast_redownload").glob(f"{prefix}{star}_*") if p.is_dir()]
    paths += [p for p in (cache_dir / "csv").glob(f"{prefix}{star}_api.*") if p.is_file()]
    # The S3 download folder, whose name differs per mission. Both must be listed: when only the
    # Kepler form was here, --discard-raw silently never removed any TESS download and the cache
    # grew to 22.7 GB over 1,485 stars before it was noticed (2026-09-25).
    s3_dir = cache_dir / "s3" / (f"kplr{star:09d}" if mission == "Kepler" else f"tic{star:016d}")
    if s3_dir.is_dir():
        paths.append(s3_dir)
    return paths


def discard_raw_files(star: int, mission: str, cache_dir: Path | None = None) -> int:
    """Delete one star's raw download (see raw_cache_paths) once its views are saved; returns bytes freed.

    Only used by build_dataset(discard_raw=True), after the star's records are in progress.jsonl, so a
    crash can never lose a result. The raw data stays available from MAST: reprocessing a star just
    downloads it again. Emptied per-quarter FITS folders are removed too; files that cannot be deleted
    (e.g. still open) are left in place.
    """
    freed = 0
    folders = set()
    for p in raw_cache_paths(star, mission, cache_dir):
        try:
            if p.is_dir():
                size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                shutil.rmtree(p)
            else:
                size = p.stat().st_size
                p.unlink()
                if p.suffix == ".fits":
                    folders.add(p.parent)
            freed += size
        except OSError:
            continue
    for folder in folders:
        try:
            folder.rmdir()  # only succeeds if empty
        except OSError:
            pass
    return freed


def _write_stitched_csv(collection, out: Path, star: int) -> Path:
    """Stitch downloaded quarters/sectors into the mentor CSV format (time, flux, flux_err, quality).

    Each quarter is divided by its own median first, because absolute flux levels jump between them.
    Shared by the MAST and S3 paths so both produce exactly the same file.
    """
    frames = []
    for lc in collection:
        f = np.asarray(lc.flux.value, float)
        med = np.nanmedian(f)
        if not np.isfinite(med) or med == 0:
            continue
        frames.append(pd.DataFrame({
            "time": np.asarray(lc.time.value, float), "flux": f / med,
            "flux_err": np.asarray(lc.flux_err.value, float) / med,
            "quality": np.asarray(lc.quality.value, np.int64)}))
    if not frames:
        raise LookupError(f"all downloaded light curves for {star} were empty")
    data = pd.concat(frames).sort_values("time")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".tmp{os.getpid()}")
    data.to_csv(tmp, index=False)
    os.replace(tmp, out)
    return out


S3_BUCKET = "stpubdata"  # STScI's public mirror of the MAST archive on AWS (no credentials needed)


def _s3_client(pool: int = 10):
    """Anonymous S3 client for the public STScI bucket.

    Explicit connect/read timeouts matter: this is the whole reason the S3 route is safe for an
    unattended build. astroquery's own HTTP calls have no effective timeout, so a degraded
    archive hangs a worker indefinitely (observed 2026-09-25: stars still running after 16 min,
    which is what the stall watchdog then had to kill). Here a stuck read fails in 120 s and is
    retried, so progress is bounded.

    `pool` raises the connection-pool size for the threaded TESS scan, which would otherwise warn
    and serialise once more than 10 requests are in flight.
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client("s3", config=Config(signature_version=UNSIGNED, connect_timeout=30, read_timeout=120,
                                            retries={"max_attempts": 3}, max_pool_connections=pool))


def fetch_kepler_fits_via_s3(star: int, cache_dir: Path) -> list[Path]:
    """Download one Kepler star's long-cadence FITS from the public AWS mirror; returns the local paths.

    Same files MAST serves, from s3://stpubdata/kepler/public/lightcurves/<kkkk>/<kepid>/ (STScI publish
    them as AWS Open Data). Added 2026-09-24 because MAST itself degraded to ~10 KB/s: measured on this
    machine, one 17-quarter star takes ~390 s through MAST, 190 s through astroquery's cloud option, and
    ~23 s fetching from S3 directly. Raises LookupError when the star has no long-cadence data, matching
    the API path's "no light curve found" case.
    """
    s3 = _s3_client()
    kepid = f"{star:09d}"
    prefix = f"kepler/public/lightcurves/{kepid[:4]}/{kepid}/"  # e.g. .../0124/012470954/
    keys, token = [], None
    while True:
        kw = {"Bucket": S3_BUCKET, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        page = s3.list_objects_v2(**kw)
        keys += [o["Key"] for o in page.get("Contents", []) if o["Key"].endswith("_llc.fits")]
        token = page.get("NextContinuationToken")
        if not token:
            break
    if not keys:
        raise LookupError(f"no Kepler long-cadence light curves on S3 for {star}")
    out_dir = Path(cache_dir) / "s3" / f"kplr{star:09d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for key in sorted(keys):
        dest = out_dir / key.rsplit("/", 1)[-1]
        if not dest.is_file() or dest.stat().st_size == 0:
            tmp = dest.with_suffix(f".tmp{os.getpid()}")
            s3.download_file(S3_BUCKET, key, str(tmp))
            os.replace(tmp, dest)
        paths.append(dest)
    return paths


def _enable_cloud_once() -> None:
    """Route astroquery's downloads through the AWS mirror, keeping MAST for the search.

    Best option for TESS: the metadata search tells us which sectors exist (fast, ~7 s),
    which is exactly what S3 cannot do without scanning ~70 sector prefixes per star.
    Failure here is not fatal - astroquery just keeps downloading from MAST.
    """
    global _CLOUD_ENABLED
    if _CLOUD_ENABLED:
        return
    try:
        from astroquery.mast import Observations

        Observations.enable_cloud_dataset(provider="AWS")
        _CLOUD_ENABLED = True
    except Exception as exc:
        print(f"  could not enable the AWS cloud dataset ({type(exc).__name__}: {exc}); using MAST", flush=True)
        _CLOUD_ENABLED = True  # do not retry per star


_CLOUD_ENABLED = False


_TESS_SECTORS: list[str] = []


def _tess_sectors(s3) -> list[str]:
    """Sector directory names under tess/public/tid/, listed once and reused.

    Listed, never guessed: as of 2026-09-25 there are 108 of them running past s0106, so the
    obvious "scan sectors 1..70" would silently miss every recent sector and quietly build
    views from partial photometry.
    """
    global _TESS_SECTORS
    if _TESS_SECTORS:
        return _TESS_SECTORS
    names, token = [], None
    while True:
        kw = {"Bucket": S3_BUCKET, "Prefix": "tess/public/tid/", "Delimiter": "/"}
        if token:
            kw["ContinuationToken"] = token
        page = s3.list_objects_v2(**kw)
        names += [p["Prefix"].rstrip("/").rsplit("/", 1)[-1] for p in page.get("CommonPrefixes", [])]
        token = page.get("NextContinuationToken")
        if not token:
            break
    _TESS_SECTORS = [n for n in names if n.startswith("s")]
    return _TESS_SECTORS


def fetch_tess_fits_via_s3(star: int, cache_dir: Path) -> list[Path]:
    """Download one TIC's SPOC light-curve FITS from the AWS mirror; returns the local paths.

    TESS has no per-star index on S3 - a star's files are spread across sector directories - so
    the sectors are scanned in parallel. Sequentially that costs ~38 s per star; with 32 threads
    it is ~2 s, which is what makes this route viable at all. Compare 152 s per star through
    MAST while the archive is degraded (measured 2026-09-25), with some stars never returning.

    Raises LookupError when the star has no SPOC light curves, matching the API path.
    """
    from concurrent.futures import ThreadPoolExecutor

    s3 = _s3_client(pool=40)
    pad = f"{star:016d}"
    groups = "/".join(pad[i:i + 4] for i in range(0, 16, 4))

    def scan(sector: str) -> list[str]:
        try:
            page = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=f"tess/public/tid/{sector}/{groups}/",
                                      MaxKeys=50)
        except Exception:
            return []
        return [o["Key"] for o in page.get("Contents", []) if o["Key"].endswith("_lc.fits")]

    sectors = _tess_sectors(s3)
    with ThreadPoolExecutor(max_workers=32) as ex:
        keys = sorted(k for batch in ex.map(scan, sectors) for k in batch)
    if not keys:
        raise LookupError(f"no TESS SPOC light curves on S3 for {star}")

    out_dir = Path(cache_dir) / "s3" / f"tic{star:016d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def grab(key: str) -> Path:
        dest = out_dir / key.rsplit("/", 1)[-1]
        if not dest.is_file() or dest.stat().st_size == 0:
            tmp = dest.with_suffix(f".tmp{os.getpid()}")
            s3.download_file(S3_BUCKET, key, str(tmp))
            os.replace(tmp, dest)
        return dest

    with ThreadPoolExecutor(max_workers=8) as ex:
        return list(ex.map(grab, keys))


def fetch_lightcurve_via_api(kepid_or_tic_id: int | str, mission: str, cache_dir: Path | None = None,
                             overwrite: bool = False, attempts: int = 3, use_s3: bool = False,
                             use_cloud: bool = False) -> Path:
    """Download every available PDCSAP light curve for a star via lightkurve and write it in
    the mentor CSV format (time, flux, flux_err, quality). Returns the CSV path.

    Kepler: author "Kepler", long cadence (1800 s). TESS: author "SPOC", 120 s preferred,
    falling back to 20 s. Downloads use quality_bitmask="none" so load_and_flatten applies
    the same quality masking as for the local files. Each quarter/sector is divided by its
    own median before stitching, because absolute flux levels jump between them.

    Network errors are retried (attempts, with 30 s / 90 s back-off). "No data" (LookupError)
    is not retried. The CSV is written to a temporary name and renamed, so an interrupted run
    never leaves a partial CSV that a later run would treat as cached.

    Truncated downloads: lightkurve reuses any FITS already in its download folder, so a file
    cut off by a dropped connection would fail on every retry. If the star has an invalid
    cached FITS (empty, size not a multiple of 2880 bytes, or no FITS header), and on every
    retry, files are downloaded into a fresh folder under mast_redownload/ instead. Broken
    files are never deleted here; invalid_cached_fits() lists them.
    """
    cache_dir = Path(cache_dir or LK_CACHE)
    star = int(str(kepid_or_tic_id).split()[-1])
    if mission == "K2":
        raise NotImplementedError(K2_OUT_OF_SCOPE)
    prefix = {"Kepler": "kepler_KIC", "TESS": "tess_TIC"}.get(mission)
    if prefix is None:
        raise ValueError(f"unknown mission {mission!r}")
    out = cache_dir / "csv" / f"{prefix}{star}_api.csv"
    if out.is_file() and not overwrite:
        return out

    # Fast path: read the same files straight from the AWS mirror. Any failure - no boto3, nothing
    # in the bucket, an S3 error - falls through to the MAST path below, so this can only help.
    if use_s3:
        fetch = {"Kepler": fetch_kepler_fits_via_s3, "TESS": fetch_tess_fits_via_s3}.get(mission)
        if fetch is not None:
            try:
                files = fetch(star, cache_dir)
                # A generator, not a list: a TESS star can have 40+ sectors, and holding every
                # LightCurve object open at once is what would put a worker back into memory trouble.
                return _write_stitched_csv((lk.read(str(f), quality_bitmask="none") for f in files), out, star)
            except LookupError:
                raise
            except Exception as exc:
                print(f"  S3 fetch failed for {star} ({type(exc).__name__}: {exc}); falling back to MAST",
                      flush=True)

    if use_cloud:
        _enable_cloud_once()

    last_exc = None
    broken = invalid_cached_fits(star, mission, cache_dir)
    for attempt in range(1, attempts + 1):
        try:
            if broken or attempt > 1:
                download_dir = cache_dir / "mast_redownload" / f"{prefix}{star}_{datetime.now():%Y%m%d%H%M%S}_a{attempt}"
            else:
                download_dir = cache_dir / "mast"
            if mission == "Kepler":
                results = lk.search_lightcurve(f"KIC {star}", mission="Kepler", author="Kepler", exptime=1800)
            else:
                results = lk.search_lightcurve(f"TIC {star}", mission="TESS", author="SPOC", exptime=120)
                if len(results) == 0:
                    results = lk.search_lightcurve(f"TIC {star}", mission="TESS", author="SPOC", exptime=20)
            if len(results) == 0:
                raise LookupError(f"no {mission} PDCSAP light curves found for {star}")
            collection = results.download_all(quality_bitmask="none", download_dir=str(download_dir))
            if collection is None or len(collection) == 0:
                raise LookupError(f"download returned nothing for {mission} {star}")
            return _write_stitched_csv(collection, out, star)
        except LookupError:
            raise
        except Exception as exc:  # network / archive / corrupt-file errors: retry
            last_exc = exc
            if attempt < attempts:
                _time.sleep(30 * 3 ** (attempt - 1))
    raise ApiFetchError(f"{type(last_exc).__name__}: {last_exc}") from last_exc


# =========================================================================== #
# Step 6: dataset builder
# =========================================================================== #
class ApiFetchError(RuntimeError):
    """Light-curve download failed after all retries (network/archive problem)."""


NETWORK_TIMEOUT_S = 120  # per socket operation


def _set_network_timeouts(seconds: float = NETWORK_TIMEOUT_S) -> None:
    """Put a time limit on every HTTP read the downloads make (added 2026-09-23).

    lightkurve/astroquery issue requests with no timeout, so a connection that dies mid-download (a MAST
    hiccup, or the machine sleeping) blocks the worker for ever: only the stall watchdog notices, 20 minutes
    later. With a socket timeout the read raises instead, fetch_lightcurve_via_api retries it normally, and
    the watchdog goes back to being the last resort rather than the first line of defence."""
    socket.setdefaulttimeout(seconds)
    try:
        from astroquery.mast import Observations

        Observations.TIMEOUT = seconds
    except Exception:
        pass


def _kill_pool_workers() -> int:
    """Kill this process's child processes (the pool workers) and return how many.

    Used by the stall watchdog: a download whose connection died (typically while the machine slept) can
    block a worker for ever, because lightkurve's HTTP read has no timeout. Killing the worker makes its
    future fail with BrokenProcessPool, which the build already handles by requeuing the star and
    rebuilding the pool. The build process spawns nothing else, so only pool workers are killed."""
    try:
        import psutil
    except ImportError:
        return 0
    killed = 0
    for child in psutil.Process().children(recursive=True):
        try:
            child.kill()
            killed += 1
        except psutil.Error:
            continue
    return killed


def _star_col(mission: str) -> str:
    return {"Kepler": "kepid", "TESS": "tid"}[mission]


def _npz_name(object_id: str) -> str:
    return f"{re.sub(r'[^A-Za-z0-9.]+', '_', object_id)}.npz"


def classify_failure(stage: str, exc: Exception) -> str:
    """Failure category used in the build report."""
    if isinstance(exc, NotImplementedError):
        return "unsupported mission"
    if stage == "fetch":
        return "api: no light curve found" if isinstance(exc, LookupError) else "api fetch failure"
    if isinstance(exc, ValueError) and re.search(r"invalid (period|epoch|duration)", str(exc)):
        return "missing/invalid ephemeris"
    if stage == "flatten":
        return "insufficient data after masking" if "usable cadences" in str(exc) else f"flatten failure ({type(exc).__name__})"
    return f"other ({type(exc).__name__})"


def _process_star(job: dict) -> dict:
    """Download/load, flatten and build views for every requested KOI on one star.

    Runs in a worker process. Writes one .npz per successful KOI and returns the
    per-KOI records (never raises: failures are returned as records)."""
    mission, star, output_dir = job["mission"], job["star"], Path(job["output_dir"])
    _set_network_timeouts()  # worker processes start fresh: set the download timeout here, not just in the parent
    started = _time.time()
    siblings = star_ephemerides(star, mission)
    todo = [ephemeris_from_row(r, mission) for r in job["rows"]]
    records = []

    def fail_all(stage, exc):
        category = classify_failure(stage, exc)
        return [{"object_id": e.object_id, "star_id": star, "label": e.label, "status": "failed",
                 "category": category, "error": f"{type(exc).__name__}: {exc}"[:500]} for e in todo]

    local = job["local_path"]
    source = "local mentor file" if local else "lightkurve API"
    try:
        path = Path(local) if local else fetch_lightcurve_via_api(star, mission, cache_dir=job["cache_dir"],
                                                                 use_s3=job.get("use_s3", False),
                                                                 use_cloud=job.get("use_cloud", False))
    except Exception as exc:
        return {"star": star, "records": fail_all("fetch", exc), "seconds": _time.time() - started}
    try:
        flat = load_and_flatten(path, mission=mission, ephemerides=siblings)
    except Exception as exc:
        return {"star": star, "records": fail_all("flatten", exc), "seconds": _time.time() - started}

    for eph in todo:
        npz = output_dir / _npz_name(eph.object_id)
        base = {"object_id": eph.object_id, "star_id": star, "label": eph.label, "source": source}
        try:
            res = process_koi(flat, eph, siblings)
            drift = res["ephemeris_drift_h"]
            if res["ephemeris_reliable"] is False:
                reason = (f"ephemeris unreliable: propagated transit-time uncertainty {drift:.2f} h > "
                          f"{MAX_DRIFT_FRACTION} x duration {eph.duration * 24:.2f} h over this light curve's baseline")
                if npz.is_file():
                    npz.unlink()  # never leave stale views for an excluded row
                records.append({**base, "status": "excluded", "category": "unreliable ephemeris", "error": reason,
                                "ephemeris_drift_h": round(drift, 3), "duration_h": round(eph.duration * 24, 3)})
                continue
            meta = {"object_id": eph.object_id, "star_id": star, "mission": mission, "source": source,
                    "source_file": flat.meta["source_file"], "period_days": eph.period, "epoch": eph.epoch,
                    "duration_days": eph.duration, "label": eph.label,
                    "global_coverage": res["global"].coverage, "local_coverage": res["local"].coverage,
                    "global_scale": res["global"].scale, "local_scale": res["local"].scale,
                    "n_in_transit": res["n_in_transit"], "depth_ppm": res["depth_ppm"],
                    "depth_snr": res["depth_snr"], "transit_observed": res["transit_observed"],
                    "ephemeris_drift_h": drift, "ephemeris_reliable": res["ephemeris_reliable"],
                    "other_kois_removed": res["other_kois_removed"], "flatten": flat.meta}
            tmp = npz.with_name(npz.stem + f".tmp{os.getpid()}.npz")
            np.savez_compressed(tmp, global_view=res["global"].values.astype(np.float32),
                                local_view=res["local"].values.astype(np.float32),
                                label=np.array(eph.label if eph.label is not None else ""),
                                label_index=np.int8(LABEL_INDEX.get(eph.label, -1)),
                                metadata=np.array(json.dumps(meta, default=float)))
            os.replace(tmp, npz)  # atomic: a killed run never leaves a half-written view file
            records.append({**base, "status": "ok", "category": "", "file": npz.name,
                            "transit_observed": res["transit_observed"],
                            "ephemeris_drift_h": round(drift, 3) if np.isfinite(drift) else np.nan,
                            "ephemeris_reliable": res["ephemeris_reliable"],
                            "local_coverage": round(res["local"].coverage, 3),
                            "global_coverage": round(res["global"].coverage, 3),
                            "depth_snr": round(res["depth_snr"], 1) if np.isfinite(res["depth_snr"]) else np.nan})
        except Exception as exc:
            records.append({**base, "status": "failed", "category": classify_failure("koi", exc),
                            "error": f"{type(exc).__name__}: {exc}"[:500]})
    return {"star": star, "records": records, "seconds": _time.time() - started}


def _read_progress(progress_path: Path) -> dict[str, dict]:
    """Latest record per object_id from the append-only progress log."""
    latest = {}
    if progress_path.is_file():
        with progress_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:  # a line cut off by a hard kill
                    continue
                if "object_id" in rec:
                    latest[rec["object_id"]] = rec
    return latest


def load_manual_exclusions() -> dict[str, dict]:
    """object_id -> {category, reason, note, ...} from every curated list in MANUAL_EXCLUSIONS."""
    out = {}
    for category, path in MANUAL_EXCLUSIONS.items():
        if not Path(path).is_file():
            continue
        table = pd.read_csv(path, dtype={"object_id": str}, keep_default_na=False)
        for rec in table.to_dict("records"):
            if rec["object_id"] in out:
                raise ValueError(f"{rec['object_id']} appears in more than one manual exclusion list")
            out[rec["object_id"]] = {**rec, "category": category}
    return out


def _free_gb(path: Path) -> float:
    probe = Path(path)
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1e9


def _free_ram_gb() -> float | None:
    """Available system RAM in GB, or None if it cannot be measured (guard then disabled).

    "available" (not "free") is what psutil reports as usable without swapping, which is the
    number that matters here: each worker holds a full light curve plus its flattened copy.
    """
    try:
        import psutil  # optional; the build runs without it, just without the memory guard

        return psutil.virtual_memory().available / 1024 ** 3
    except Exception:
        return None


# Recycle a worker after this many stars so any leaked memory (lightkurve/astropy caches) is
# returned to the OS. Python 3.11+ only; ignored on older versions.
TASKS_PER_CHILD = 25


# use_local_files_first defaults to False: the mentor files cover ONE quarter/sector (~97 d Kepler,
# ~27 d TESS) while the API returns the full mission (~4 yr Kepler). Mixing 97-day local baselines
# with 4-year API baselines would make data source a confound between the 36 mentor-sample KOIs and
# everyone else (different depth SNR, transits missing from short windows). Local files stay
# available for quick tests and offline checks by passing True explicitly.
def build_dataset(split_df: pd.DataFrame, output_dir: str | Path, use_local_files_first: bool = False,
                  overwrite: bool = False, verbose: bool = True, n_workers: int = 1,
                  cache_dir: str | Path | None = None, min_free_gb: float = 3.0,
                  progress_every: int = 25, dry_run: bool = False, min_free_ram_gb: float = 2.0,
                  ram_wait_s: float = 20.0, ram_wait_max_s: float = 900.0,
                  net_wait_s: float = 30.0, net_wait_max_s: float = 7200.0,
                  discard_raw: bool = False, isolate_workers: bool = False,
                  stall_timeout_s: float = 1200.0, use_s3: bool = False,
                  use_cloud: bool = False) -> pd.DataFrame:
    """Build one .npz per KOI/TOI row: global_view, local_view, label, label_index, metadata.

    Output format: one file per KOI plus index.csv (status / failure category per row).
    Chosen over a single consolidated array because a full build makes thousands of
    API calls: per-KOI files make the run resumable, isolate failures to one row, and
    can be concatenated into one array later.

    Checkpointing: every finished star appends its per-KOI records to progress.jsonl in
    output_dir immediately. A rerun skips KOIs that already have a view file or were
    excluded for an unreliable ephemeris, and retries failures. index.csv and
    excluded_unreliable_ephemeris.csv are rebuilt from progress.jsonl at the end.

    Each star is downloaded/flattened once, protecting ALL of its known transits (from
    the full catalog, not just this slice); each KOI is then folded on its own ephemeris
    with the other KOIs' transits removed. Rows whose ephemeris is unreliable are not
    saved (see excluded_unreliable_ephemeris.csv); unknown reliability is kept and flagged.

    n_workers > 1 processes stars in parallel worker processes. Before each new star the
    free space where light curves are cached is checked; below min_free_gb the build stops
    starting new stars, finishes those in flight and returns (rerun to resume).

    Memory guard (added 2026-09-18 after a BrokenProcessPool crash with 4 workers): before each
    new star, available RAM is checked. Below min_free_ram_gb no new star is started; the build
    waits for stars already in flight to finish and free memory, re-checking every ram_wait_s.
    If memory stays low for ram_wait_max_s with nothing in flight, the build stops cleanly
    (resumable) instead of being killed by the OS. Needs psutil; without it the guard is skipped
    and the build logs that once. Workers are also recycled every TASKS_PER_CHILD stars, and if a
    worker is killed anyway the pool is rebuilt and the interrupted stars are requeued.

    Manual exclusions: KOIs listed in the curated files of MANUAL_EXCLUSIONS (e.g.
    exclusions/excluded_circumbinary_planet.csv) are never processed. They are recorded as
    status "excluded" with the list's reason and note, any stale view file is removed, and each
    build writes excluded_circumbinary_planet.csv in output_dir. The lists are re-read on every
    run, so adding or removing an entry takes effect on the next build or resume.

    dry_run=True only plans: it returns what would be processed, skipped or excluded and writes nothing.

    discard_raw=True (added 2026-09-21 for a laptop without room for the ~60 GB raw cache): once a
    star's records are in progress.jsonl, its raw download (FITS, retry folders, stitched CSV) is
    deleted, so the cache only ever holds the stars in flight. Views, the checkpoint and resuming are
    unaffected; reprocessing a star later re-downloads it from MAST. Stars read from local mentor files
    are never touched.

    isolate_workers=True runs stars in a worker pool even with n_workers=1, so a single-worker build
    keeps worker recycling and crash recovery (the plain n_workers=1 path runs in-process).

    Stall watchdog (added 2026-09-23 after the machine slept mid-build and every worker hung on a dead
    connection for 6.5 h): if no star finishes within stall_timeout_s, the workers are killed, their stars
    are requeued and the pool is rebuilt. A normal star takes well under a minute, so the default 20 min
    only fires on a hang.

    Raises NotImplementedError for K2 split rows (see K2_OUT_OF_SCOPE).
    """
    mission = mission_of(split_df)
    if mission == "K2":
        raise NotImplementedError(K2_OUT_OF_SCOPE)
    output_dir = Path(output_dir)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir or LK_CACHE)
    progress_path = output_dir / "progress.jsonl"
    log_path = output_dir / "build_progress.txt"
    local_index = local_lightcurve_index() if use_local_files_first else {}
    previous = {} if overwrite else _read_progress(progress_path)

    manual = load_manual_exclusions()

    def log(msg: str) -> None:
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        if not dry_run:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        if verbose:
            print(line, flush=True)

    # ---- plan: which KOIs still need work ---------------------------------- #
    jobs, n_rows, n_skipped = [], 0, 0
    plan, manual_records = [], []
    for star, rows in split_df.groupby(_star_col(mission), sort=False):
        star = int(star)
        pending = []
        for row in rows.to_dict("records"):
            n_rows += 1
            eph = ephemeris_from_row(row, mission)
            npz = output_dir / _npz_name(eph.object_id)
            if eph.object_id in manual:  # curated exclusion: checked before anything else, never processed
                m = manual[eph.object_id]
                manual_records.append({"object_id": eph.object_id, "star_id": star, "label": eph.label,
                                       "status": "excluded", "category": m["category"], "error": m["reason"],
                                       "note": m.get("note", ""), "source": m.get("source", "manual review"),
                                       "duration_h": round(eph.duration * 24, 3)})
                plan.append({"object_id": eph.object_id, "star_id": star, "action": f"excluded: {m['category']}"})
                if npz.is_file() and not dry_run:
                    npz.unlink()  # never leave a stale view for a manually excluded row
                continue
            prev = previous.get(eph.object_id, {})
            # Only computed ephemeris exclusions count as done; manual ones are re-read from the lists.
            done = npz.is_file() or (prev.get("status") == "excluded" and prev.get("category") == "unreliable ephemeris")
            if done and not overwrite:
                n_skipped += 1
                plan.append({"object_id": eph.object_id, "star_id": star, "action": "skip (already done)"})
            else:
                pending.append(row)
                plan.append({"object_id": eph.object_id, "star_id": star, "action": "process"})
        if pending:
            local = local_index.get((mission, star))
            jobs.append({"mission": mission, "star": star, "rows": pending, "output_dir": str(output_dir),
                         "cache_dir": str(cache_dir), "local_path": str(local) if local else None,
                         "use_s3": use_s3, "use_cloud": use_cloud})

    if dry_run:
        plan = pd.DataFrame(plan)
        log(f"DRY RUN: {n_rows} rows | {dict(plan.action.value_counts())} | {len(jobs)} stars would be processed")
        return plan

    run_started = _time.time()
    if manual_records:
        with progress_path.open("a", encoding="utf-8") as fh:
            for rec in manual_records:
                fh.write(json.dumps({**rec, "time": datetime.now().isoformat(timespec="seconds")}, default=str) + "\n")
    with progress_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "session_start", "time": datetime.now().isoformat(timespec="seconds"),
                             "rows": n_rows, "already_done": n_skipped, "stars_to_process": len(jobs),
                             "n_workers": n_workers, "cache_dir": str(cache_dir),
                             "discard_raw": discard_raw}) + "\n")
    log(f"build start: {n_rows} rows, {n_skipped} already done, {len(jobs)} stars to process, "
        f"{n_workers} worker(s), cache {cache_dir}, output {output_dir}"
        + (" | raw downloads discarded after each star" if discard_raw else "")
        + (f" | manually excluded: {[r['object_id'] for r in manual_records]}" if manual_records else ""))

    counts: dict[str, int] = {}
    stars_done, stop_reason = 0, None
    local_stars = {j["star"] for j in jobs if j["local_path"]}
    discarded_bytes = 0

    def record(result: dict) -> None:
        nonlocal stars_done, discarded_bytes
        stars_done += 1
        with progress_path.open("a", encoding="utf-8") as fh:
            for rec in result["records"]:
                rec = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in rec.items()}
                fh.write(json.dumps({**rec, "seconds_for_star": round(result["seconds"], 1),
                                     "time": datetime.now().isoformat(timespec="seconds")}, default=str) + "\n")
                key = rec["status"] if rec["status"] == "ok" else f"{rec['status']}: {rec['category']}"
                counts[key] = counts.get(key, 0) + 1
        # Records are on disk: only now is the star's raw download safe to drop.
        if discard_raw and result["star"] not in local_stars:
            discarded_bytes += discard_raw_files(result["star"], mission, cache_dir)
        if stars_done % progress_every == 0 or stars_done == len(jobs):
            elapsed = _time.time() - run_started
            eta_h = elapsed / stars_done * (len(jobs) - stars_done) / 3600
            log(f"stars {stars_done}/{len(jobs)} | KOIs {counts} | elapsed {elapsed / 3600:.2f} h | "
                f"ETA {eta_h:.1f} h | free on cache drive {_free_gb(cache_dir):.1f} GB"
                + (f" | raw discarded {discarded_bytes / 1e9:.2f} GB" if discard_raw else ""))

    def low_disk() -> bool:
        free = _free_gb(cache_dir)
        return free < min_free_gb

    ram_state = {"waited": 0.0, "warned": False, "pauses": 0}
    net_state = {"waited": 0.0, "pauses": 0, "stars_saved": 0}
    stalls = 0  # set by the stall watchdog in the worker-pool path; read in the summary below

    def net_ok() -> bool:
        """True if the archive is reachable. Waits, rather than spending stars, while it is not.

        Without this a brief DNS outage is catastrophic to a long build: every star dispatched
        during the outage fails with NameResolutionError, is recorded as 'api fetch failure',
        and is consumed from the queue. That is exactly how 137 Kepler KOIs were lost on
        2026-09-24 and another 26 TESS stars on 2026-09-25 - in both cases the data was fine
        and the laptop's connection had simply dropped for a few minutes.

        A failed lookup is cheap (sub-second) and is only consulted before starting a star, so
        this costs nothing when the network is healthy.
        """
        nonlocal stop_reason
        try:
            socket.getaddrinfo("mast.stsci.edu", 443)
            net_state["waited"] = 0.0
            return True
        except OSError:
            pass
        net_state["pauses"] += 1
        log(f"network down (cannot resolve mast.stsci.edu): waiting {net_wait_s:.0f} s rather than "
            f"failing stars (waited {net_state['waited']:.0f}/{net_wait_max_s:.0f} s)")
        _time.sleep(net_wait_s)
        net_state["waited"] += net_wait_s
        if net_state["waited"] >= net_wait_max_s:
            stop_reason = f"network unreachable for {net_wait_max_s / 60:.0f} min"
        return False

    def ram_ok(busy: bool) -> bool:
        """True if a new star may start. Waits (or stops the build) while memory is tight.

        busy says whether stars are already in flight: if so this returns False immediately so the
        caller can collect a finished star, which is what actually frees memory.
        """
        nonlocal stop_reason
        free = _free_ram_gb()
        if free is None:
            if not ram_state["warned"]:
                log("memory guard disabled: psutil not available")
                ram_state["warned"] = True
            return True
        if free >= min_free_ram_gb:
            ram_state["waited"] = 0.0
            return True
        if busy:
            return False  # let an in-flight star finish and release its light curve first
        ram_state["pauses"] += 1
        log(f"memory guard: {free:.1f} GB available < {min_free_ram_gb} GB and nothing in flight; "
            f"waiting {ram_wait_s:.0f} s (waited {ram_state['waited']:.0f}/{ram_wait_max_s:.0f} s)")
        _time.sleep(ram_wait_s)
        ram_state["waited"] += ram_wait_s
        if ram_state["waited"] >= ram_wait_max_s:
            stop_reason = f"available RAM stayed below {min_free_ram_gb} GB for {ram_wait_max_s / 60:.0f} min"
        return False

    if n_workers <= 1 and not isolate_workers:
        for job in jobs:
            if low_disk():
                stop_reason = f"free space on cache drive below {min_free_gb} GB"
                break
            while (not ram_ok(busy=False) or not net_ok()) and stop_reason is None:
                pass
            if stop_reason:
                break
            record(_process_star(job))
    else:
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
        from concurrent.futures.process import BrokenProcessPool
        queue = list(jobs)
        pool_restarts, max_pool_restarts = 0, 5
        max_stalls = 10
        try:
            while (queue or pool_restarts) and stop_reason is None:
                in_flight: dict = {}  # future -> job, so an interrupted star can be requeued
                try:
                    with ProcessPoolExecutor(max_workers=n_workers, max_tasks_per_child=TASKS_PER_CHILD) as pool:
                        while queue or in_flight:
                            while queue and len(in_flight) < n_workers * 2 and stop_reason is None:
                                if low_disk():
                                    stop_reason = f"free space on cache drive below {min_free_gb} GB"
                                    break
                                if not ram_ok(busy=bool(in_flight)):
                                    break
                                # Only when nothing is in flight: with stars running, their own
                                # results arrive first and the check is repeated next pass anyway.
                                if not in_flight and not net_ok():
                                    break
                                job = queue.pop(0)
                                in_flight[pool.submit(_process_star, job)] = job
                            if not in_flight:
                                if not queue or stop_reason:
                                    break
                                continue  # memory guard blocked the start; it already waited, retry
                            finished, _ = wait(list(in_flight), return_when=FIRST_COMPLETED,
                                               timeout=stall_timeout_s)
                            if not finished:  # nothing finished in stall_timeout_s: a hung download
                                stalls += 1
                                killed = _kill_pool_workers()
                                log(f"no star finished in {stall_timeout_s / 60:.0f} min with "
                                    f"{len(in_flight)} in flight (hung download?); killed {killed} worker(s), "
                                    f"stall {stalls}/{max_stalls}. Their stars are requeued.")
                                if stalls > max_stalls:
                                    stop_reason = f"workers stalled {stalls} times in a row"
                                    break
                                continue  # the killed workers' futures now fail -> requeue + pool restart
                            for fut in finished:
                                job = in_flight.pop(fut)
                                try:
                                    record(fut.result())
                                    # A star came through: the pool is healthy again, so the restart and
                                    # stall budgets below count CONSECUTIVE failures, not lifetime ones.
                                    pool_restarts = stalls = 0
                                except BrokenProcessPool:
                                    queue.insert(0, job)  # worker killed mid-star: redo this star
                                    raise
                                except Exception as exc:  # this one star failed; keep going
                                    log(f"star {job['star']} raised {type(exc).__name__}: {exc}")
                    break  # pool closed with the queue drained
                except BrokenProcessPool as exc:
                    for job in in_flight.values():
                        queue.insert(0, job)
                    pool_restarts += 1
                    free = _free_ram_gb()
                    log(f"worker pool died ({type(exc).__name__}: {exc}) with {len(in_flight)} star(s) in flight, "
                        f"{free if free is None else round(free, 1)} GB RAM available; restart "
                        f"{pool_restarts}/{max_pool_restarts}, {len(queue)} stars still queued")
                    if pool_restarts > max_pool_restarts:
                        stop_reason = f"worker pool died {pool_restarts} times in a row without finishing a star"
                        break
                    _time.sleep(30)  # give the OS a moment to reclaim the dead workers' memory
        except KeyboardInterrupt:
            stop_reason = "interrupted"
            raise
        finally:
            if stop_reason:
                log(f"stopping early: {stop_reason}; rerun the same command to resume")
            if ram_state["pauses"]:
                log(f"memory guard paused the build {ram_state['pauses']} time(s)")
            if net_state["pauses"]:
                log(f"network guard paused the build {net_state['pauses']} time(s) "
                    f"({net_state['pauses'] * net_wait_s / 60:.0f} min total), rather than failing those stars")
            if stalls:
                log(f"stall watchdog restarted the workers {stalls} time(s)")

    # ---- rebuild index + exclusions from the checkpoint log ----------------- #
    latest = _read_progress(progress_path)
    wanted = [ephemeris_from_row(r, mission).object_id for r in split_df.to_dict("records")]
    rows = []
    for oid in wanted:
        rec = latest.get(oid)
        if rec is None:
            has_file = (output_dir / _npz_name(oid)).is_file()
            rec = {"object_id": oid, "status": "ok" if has_file else "not processed",
                   "category": "" if has_file else "pending", "file": _npz_name(oid) if has_file else None}
        rows.append(rec)
    index = pd.DataFrame(rows)
    index.to_csv(output_dir / "index.csv", index=False)
    cols = ["object_id", "star_id", "label", "ephemeris_drift_h", "duration_h", "source", "error"]
    category = index["category"] if "category" in index.columns else pd.Series("", index=index.index)
    eph_ex = index[(index.status == "excluded") & (category == "unreliable ephemeris")]
    eph_ex.reindex(columns=cols).rename(columns={"error": "reason"}).to_csv(
        output_dir / "excluded_unreliable_ephemeris.csv", index=False)
    for cat, path in MANUAL_EXCLUSIONS.items():
        man_ex = index[(index.status == "excluded") & (category == cat)]
        man_ex.reindex(columns=cols + ["note"]).rename(columns={"error": "reason"}).to_csv(
            output_dir / Path(path).name, index=False)
    log(f"build end: {dict(index.status.value_counts())} | session wall-clock {(_time.time() - run_started) / 3600:.2f} h"
        + (f" | stopped early: {stop_reason}" if stop_reason else ""))
    return index


def load_view_file(path: str | Path) -> dict:
    """Read one saved .npz back (views, label, metadata dict)."""
    with np.load(path, allow_pickle=False) as z:
        return {"global_view": z["global_view"], "local_view": z["local_view"], "label": str(z["label"]),
                "label_index": int(z["label_index"]), "metadata": json.loads(str(z["metadata"]))}


# =========================================================================== #
# Verification runs (Steps 2, 4, 5, 6)
# =========================================================================== #
def verify_per_koi_folding(flat: FlattenedLightCurve, ephemerides: list[Ephemeris]) -> pd.DataFrame:
    """For every KOI on one star: fold on its own ephemeris and check the result.

    - distinct: its local view differs from every sibling's (max |corr| < 0.99)
    - dip_centred: the local-view minimum lies within +/- half a duration of phase 0
    - wrong_epoch_snr: control fold at epoch + P/2; a real transit's SNR should collapse
    """
    rows, local_views = [], {}
    for eph in ephemerides:
        try:
            res = process_koi(flat, eph, ephemerides)
        except ValueError as exc:
            rows.append({"object_id": eph.object_id, "label": eph.label, "status": f"skipped: {exc}"})
            continue
        lv = res["local"]
        half = min(eph.period / 2, LOCAL_NUM_DURATIONS * eph.duration)
        centre_bins = int(np.ceil((LOCAL_BINS - 1) * (eph.duration / 2) / (2 * half)))
        argmin = int(np.argmin(lv.values))
        cleaned = remove_other_transits(flat, eph, ephemerides)[0]
        wrong = transit_diagnostics(phase_fold(cleaned, eph.period, eph.epoch + eph.period / 2), eph.duration)
        local_views[eph.object_id] = lv.values
        rows.append({"object_id": eph.object_id, "label": eph.label, "period_d": round(eph.period, 4),
                     "status": "ok", "n_in_transit": res["n_in_transit"],
                     "depth_snr": round(res["depth_snr"], 1) if res["transit_observed"] else np.nan,
                     "dip_centred": bool(abs(argmin - LOCAL_BINS // 2) <= centre_bins),
                     "wrong_epoch_snr": round(wrong["depth_snr"], 1) if wrong["transit_observed"] else np.nan,
                     "others_removed": len(res["other_kois_removed"]),
                     "drift_h": round(res["ephemeris_drift_h"], 2) if np.isfinite(res["ephemeris_drift_h"]) else np.nan})
    table = pd.DataFrame(rows)
    ids = list(local_views)
    max_corr = {i: max([abs(np.corrcoef(local_views[i], local_views[j])[0, 1]) for j in ids if j != i] or [np.nan])
                for i in ids}
    table["max_corr_vs_siblings"] = table.object_id.map(max_corr).round(3)
    table["distinct"] = table["max_corr_vs_siblings"] < 0.99
    return table


def run_sample() -> pd.DataFrame:
    """Steps 1-4 on the 36 mentor light curves: every KOI/TOI on each star, ephemerides from the splits."""
    rows = []
    for (mission, star), path in local_lightcurve_index().items():
        siblings = star_ephemerides(star, mission)
        try:
            flat = load_and_flatten(path, mission=mission, ephemerides=siblings)
        except Exception as exc:
            rows.append({"mission": mission, "star_id": star, "object_id": "-", "status": f"load failed: {exc}"})
            continue
        for eph in siblings:
            base = {"mission": mission, "star_id": star, "object_id": eph.object_id, "label": eph.label,
                    "n_kois_on_star": len(siblings), "file": path.name,
                    "cadence_min": flat.meta["cadence_minutes_raw"], "binned_20s": flat.meta["binned_to_2min"],
                    "masked_with_flux": flat.meta["n_quality_masked_with_flux"],
                    "pct_masked": round(100 * flat.meta["n_quality_masked"] / flat.meta["n_raw"], 1)}
            try:
                res = process_koi(flat, eph, siblings)
                rows.append({**base, "status": "ok", "period_d": round(eph.period, 4),
                             "duration_h": round(eph.duration * 24, 2), "n_in_transit": res["n_in_transit"],
                             "transit_observed": res["transit_observed"],
                             "depth_snr": round(res["depth_snr"], 1) if res["transit_observed"] else np.nan,
                             "global_coverage": round(res["global"].coverage, 3),
                             "local_coverage": round(res["local"].coverage, 3),
                             "drift_h": round(res["ephemeris_drift_h"], 2) if np.isfinite(res["ephemeris_drift_h"]) else np.nan,
                             "ephemeris_reliable": res["ephemeris_reliable"]})
            except Exception as exc:
                rows.append({**base, "status": f"failed: {type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)


def plot_sample_views(sample: pd.DataFrame, out_path: Path, per_class: int = 2) -> list[str]:
    """Global + local view pairs for a few KOIs per disposition (best-covered, clearest first)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid, series = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    plt.rcParams.update({"figure.facecolor": surface, "axes.facecolor": surface, "savefig.facecolor": surface,
                         "axes.edgecolor": grid, "axes.labelcolor": ink2, "xtick.color": ink2, "ytick.color": ink2,
                         "text.color": ink, "axes.grid": True, "grid.color": grid, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 8})
    ok = sample[(sample.status == "ok") & (sample.transit_observed == True)  # noqa: E712
                & (sample.ephemeris_reliable != False)].copy()  # noqa: E712
    chosen = []
    for label in ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"]:
        pool = ok[ok.label == label].sort_values(["local_coverage", "depth_snr"], ascending=False)
        picks = pd.concat([pool[pool.mission == m].head(1) for m in ["Kepler", "TESS"]])  # one per mission if possible
        picks = pd.concat([picks, pool[~pool.index.isin(picks.index)]]).head(per_class)
        chosen += picks.to_dict("records")

    index = local_lightcurve_index()
    fig, axes = plt.subplots(len(chosen), 2, figsize=(11, 1.9 * len(chosen)),
                             gridspec_kw={"width_ratios": [2.2, 1]}, squeeze=False)
    for (ax_g, ax_l), r in zip(axes, chosen):
        mission, star = r["mission"], int(r["star_id"])
        siblings = star_ephemerides(star, mission)
        eph = next(e for e in siblings if e.object_id == r["object_id"])
        flat = load_and_flatten(index[(mission, star)], mission=mission, ephemerides=siblings)
        res = process_koi(flat, eph, siblings)
        half = min(eph.period / 2, LOCAL_NUM_DURATIONS * eph.duration)
        ax_g.plot(np.linspace(-0.5, 0.5, GLOBAL_BINS), res["global"].values, ".", color=series, markersize=1.5)
        ax_l.plot(np.linspace(-half, half, LOCAL_BINS) / eph.duration, res["local"].values, ".-",
                  color=series, markersize=2.5, linewidth=0.6)
        ax_g.set_ylabel("norm. flux")
        ax_g.set_title(f"{r['object_id']} ({mission} {star}) | {r['label']} | P = {eph.period:.3f} d | "
                       f"global view, coverage {res['global'].coverage:.0%}", loc="left", fontsize=8)
        ax_l.set_title(f"local view, depth SNR {res['depth_snr']:.0f}", loc="left", fontsize=8)
        for ax in (ax_g, ax_l):
            ax.axvline(0, color=ink2, linewidth=0.6, linestyle=":")
    axes[-1, 0].set_xlabel("phase (fraction of period)")
    axes[-1, 1].set_xlabel("time from transit centre (transit durations)")
    fig.suptitle("Per-KOI global (2001 bins) and local (201 bins) views from the mentor light curves",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [r["object_id"] for r in chosen]


def summarize_build(split_df: pd.DataFrame, output_dir: str | Path, figure_path: Path | None = None,
                    n_plot: int = 10, seed: int = 42) -> str:
    """Final build report: counts by outcome/category, wall-clock, class balance of what was
    saved vs the split, disk use, and a spot-check figure of random successful KOIs."""
    output_dir = Path(output_dir)
    index = pd.read_csv(output_dir / "index.csv")
    lines = []
    add = lines.append

    sessions, first, last = [], None, None
    with (output_dir / "progress.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = datetime.fromisoformat(rec["time"])
            if rec.get("event") == "session_start":
                sessions.append([t, t])
            elif sessions:
                sessions[-1][1] = t
    wall_h = sum((b - a).total_seconds() for a, b in sessions) / 3600

    add(f"KOIs attempted: {len(index)} (split rows) in {len(sessions)} session(s), wall-clock {wall_h:.2f} h")
    add(f"  saved (ok): {int((index.status == 'ok').sum())}")
    other = index[index.status != "ok"]
    for (status, category), n in other.groupby(["status", other.category.fillna("")]).size().items():
        add(f"  {status}: {category}: {n}")
    if (other.status == "failed").any():
        add("  example failures: " + "; ".join(
            f"{r.object_id}: {str(r.error)[:120]}" for r in other[other.status == "failed"].head(5).itertuples()))

    saved = index[index.status == "ok"]
    labels = ["FALSE POSITIVE", "CONFIRMED", "CANDIDATE"]
    split_share = split_df.label_harmonized.value_counts(normalize=True).reindex(labels, fill_value=0) * 100
    split_n = split_df.label_harmonized.value_counts().reindex(labels, fill_value=0)
    saved_labels = saved.object_id.map(dict(zip(
        [ephemeris_from_row(r, mission_of(split_df)).object_id for r in split_df.to_dict("records")],
        split_df.label_harmonized)))
    saved_n = saved_labels.value_counts().reindex(labels, fill_value=0)
    saved_share = saved_labels.value_counts(normalize=True).reindex(labels, fill_value=0) * 100
    add("class balance, split vs saved:")
    skewed = []
    for lab in labels:
        diff = saved_share[lab] - split_share[lab]
        kept = 100 * saved_n[lab] / split_n[lab] if split_n[lab] else float("nan")
        add(f"  {lab:<15} split {split_n[lab]:>5} ({split_share[lab]:5.1f}%) | saved {saved_n[lab]:>5} "
            f"({saved_share[lab]:5.1f}%) | kept {kept:5.1f}% of class | shift {diff:+.1f} pp")
        if abs(diff) > 2.0:
            skewed.append(f"{lab} {diff:+.1f} pp")
    add("  FLAG: exclusions/failures shifted class balance by more than 2 pp: " + ", ".join(skewed)
        if skewed else "  class balance preserved within 2 pp")

    npz_bytes = sum(p.stat().st_size for p in output_dir.glob("*.npz"))
    other_bytes = sum(p.stat().st_size for p in output_dir.iterdir() if p.is_file() and p.suffix != ".npz")
    cache_bytes = sum(p.stat().st_size for p in Path(LK_CACHE).rglob("*") if p.is_file()) if Path(LK_CACHE).exists() else 0
    add(f"disk: views {npz_bytes / 1e6:.1f} MB ({len(list(output_dir.glob('*.npz')))} .npz) + logs/index "
        f"{other_bytes / 1e6:.1f} MB in {output_dir}; light-curve cache {cache_bytes / 1e9:.2f} GB in {LK_CACHE}")
    add(f"saved views with transit_observed False: {int((saved.transit_observed == False).sum())}; "  # noqa: E712
        f"ephemeris reliability unknown: {int(saved.ephemeris_reliable.isna().sum())}")

    if figure_path is not None and len(saved):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(seed)
        per = {lab: n_plot // 3 + (1 if i < n_plot % 3 else 0) for i, lab in enumerate(labels)}
        picks = []
        for lab in labels:
            pool = saved[saved_labels == lab]
            if len(pool):
                picks += list(pool.iloc[rng.choice(len(pool), size=min(per[lab], len(pool)), replace=False)].object_id)
        surface, ink, ink2, grid, series = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
        plt.rcParams.update({"figure.facecolor": surface, "axes.facecolor": surface, "savefig.facecolor": surface,
                             "axes.edgecolor": grid, "axes.labelcolor": ink2, "xtick.color": ink2,
                             "ytick.color": ink2, "text.color": ink, "axes.grid": True, "grid.color": grid,
                             "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                             "font.size": 8})
        fig, axes = plt.subplots(len(picks), 2, figsize=(11, 1.75 * len(picks)),
                                 gridspec_kw={"width_ratios": [2.2, 1]}, squeeze=False)
        for (ax_g, ax_l), oid in zip(axes, picks):
            v = load_view_file(output_dir / _npz_name(oid))
            m = v["metadata"]
            ax_g.plot(np.linspace(-0.5, 0.5, len(v["global_view"])), v["global_view"], ".", color=series, markersize=1.5)
            half = min(m["period_days"] / 2, LOCAL_NUM_DURATIONS * m["duration_days"]) / m["duration_days"]
            ax_l.plot(np.linspace(-half, half, len(v["local_view"])), v["local_view"],
                      ".-", color=series, markersize=2.5, linewidth=0.6)
            ax_g.set_title(f"{oid} (KIC {m['star_id']}) | {m['label']} | P = {m['period_days']:.3f} d | global view",
                           loc="left", fontsize=8)
            ax_l.set_title(f"local view, depth SNR {m['depth_snr']:.0f}", loc="left", fontsize=8)
            ax_g.set_ylabel("norm. flux")
            for ax in (ax_g, ax_l):
                ax.axvline(0, color=ink2, linewidth=0.6, linestyle=":")
        axes[-1, 0].set_xlabel("phase (fraction of period)")
        axes[-1, 1].set_xlabel("time from transit centre (transit durations)")
        split_name = str(split_df["split"].iloc[0]) if "split" in split_df.columns and len(split_df) else "split"
        fig.suptitle(f"{mission_of(split_df)} {split_name} build spot-check: {len(picks)} random saved KOIs (seed {seed})",
                     x=0.01, ha="left", fontsize=10, fontweight="bold")
        fig.tight_layout()
        fig.savefig(figure_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        add(f"spot-check figure: {figure_path} ({', '.join(picks)})")
    return "\n".join(lines)


def _print_table(df: pd.DataFrame) -> None:
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60):
        print(df.to_string(index=False))


def main(argv: list[str]) -> None:
    command = argv[1] if len(argv) > 1 else "sample"
    if command == "sample":
        sample = run_sample()
        print("\nPER-KOI RESULTS ON THE 36 MENTOR LIGHT CURVES")
        _print_table(sample.drop(columns=["file"]))
        EDA_DIR.mkdir(exist_ok=True)
        sample.to_csv(EDA_DIR / "preprocessing_sample_summary.csv", index=False)
        print("\nMULTI-KOI STAR CHECKS (local files, stars with >= 3 KOIs)")
        for (mission, star), path in local_lightcurve_index().items():
            sib = star_ephemerides(star, mission)
            if len(sib) < 3:
                continue
            flat = load_and_flatten(path, mission=mission, ephemerides=sib)
            print(f"\n  {mission} {star} ({len(sib)} KOIs, {path.name})")
            _print_table(verify_per_koi_folding(flat, sib))
        chosen = plot_sample_views(sample, EDA_DIR / "preprocessing_sample_views.png")
        print(f"\nsaved eda/preprocessing_sample_views.png ({', '.join(chosen)})")
    elif command == "kepler90":
        sib = star_ephemerides(11442793, "Kepler")
        flat = load_and_flatten(fetch_lightcurve_via_api(11442793, "Kepler"), mission="Kepler", ephemerides=sib)
        print(f"Kepler-90 (KIC 11442793): {len(sib)} KOIs; flatten meta {flat.meta}")
        _print_table(verify_per_koi_folding(flat, sib))
    elif command == "api-test":
        koi = catalog_for("Kepler")
        local = {s for (m, s) in local_lightcurve_index() if m == "Kepler"}
        targets = list(dict.fromkeys(koi[(koi.split == "train") & ~koi.kepid.isin(local)].kepid))[:5]
        for star in targets:
            started = _time.time()
            try:
                path = fetch_lightcurve_via_api(star, "Kepler")
                raw = pd.read_csv(path)
                sib = star_ephemerides(star, "Kepler")
                flat = load_and_flatten(path, mission="Kepler", ephemerides=sib)
                res = [process_koi(flat, e, sib) for e in sib]
                print(f"  KIC {star}: {len(raw)} cadences over {raw.time.max() - raw.time.min():.0f} d, columns "
                      f"{list(raw.columns)}; {flat.meta['n_final']} after flatten; "
                      + "; ".join(f"{e.object_id} ({e.label}) SNR {r['depth_snr']:.1f}" for e, r in zip(sib, res))
                      + f" ({_time.time() - started:.0f}s)")
            except Exception as exc:
                print(f"  KIC {star}: FAILED {type(exc).__name__}: {exc}")
    elif command == "build-test":
        koi = catalog_for("Kepler")
        out = VIEWS_DIR / "_test_kepler_train_first20"
        index = build_dataset(koi[koi.split == "train"].head(20), out, overwrite="--overwrite" in argv)
        _print_table(index)
        print(f"\n  ok {int((index.status == 'ok').sum())}/{len(index)}; output {out.relative_to(RESEARCH_DIR)}")
    elif command in ("build-train", "summarize-train", "build-split", "summarize-split"):
        # Full split builds (one output folder per split: detection_views/<mission>_<split>). Usage:
        #   python preprocessing_pipeline.py build-split {train|val|test} [--mission Kepler|TESS]
        #                                    [--workers N] [--discard-raw]
        #                                    [--min-free-ram GB] [--stall-timeout MIN] [--s3] [--cloud]
        #       --mission selects the catalog and output folder; Kepler if omitted (the original behaviour).
        #       --s3 fetches light curves straight from the public AWS mirror, for BOTH missions, and
        #       falls back to MAST per star if S3 fails. This is the route to use whenever MAST is
        #       degraded (it was at 1.7-6.4 KB/s on 2026-09-25 while the line itself did 9 MB/s).
        #       Kepler maps a kepid to one prefix; TESS has no per-star index, so its sector
        #       directories are listed once and scanned in parallel (~2 s, vs ~38 s sequentially).
        #       --cloud instead keeps MAST for the metadata search and pulls only the FITS from AWS.
        #       That still inherits MAST's latency, so prefer --s3 unless S3 lookup is failing.
        #       (resumable; rerun to continue. --discard-raw deletes each star's raw download once its
        #        views are saved; --min-free-ram lowers the 2 GB RAM guard on small-memory machines)
        #   python preprocessing_pipeline.py summarize-split {train|val|test} [--mission ...]
        #   build-train / summarize-train are the original names for the Kepler train split and still work.
        split = "train" if command.endswith("-train") else (argv[2] if len(argv) > 2 else "")
        if split not in ("train", "val", "test"):
            raise SystemExit(f"{command}: give a split, one of train / val / test")
        mission = argv[argv.index("--mission") + 1] if "--mission" in argv else "Kepler"
        if mission not in ("Kepler", "TESS"):
            raise SystemExit(f"--mission must be Kepler or TESS (K2 is out of scope: {K2_OUT_OF_SCOPE})")
        prefix = mission.lower()
        cat = catalog_for(mission)
        part = cat[cat.split == split]
        out = VIEWS_DIR / f"{prefix}_{split}"
        if command.startswith("build"):
            # Default dropped 4 -> 2 on 2026-09-18: with 4 workers each holding a full Kepler light
            # curve plus its flattened copy, a worker was killed (BrokenProcessPool) at 4.1/13.8 GB free.
            workers = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 2
            min_ram = float(argv[argv.index("--min-free-ram") + 1]) if "--min-free-ram" in argv else 2.0
            stall_min = float(argv[argv.index("--stall-timeout") + 1]) if "--stall-timeout" in argv else 20.0
            index = build_dataset(part, out, n_workers=workers, discard_raw="--discard-raw" in argv,
                                  min_free_ram_gb=min_ram, isolate_workers=True, stall_timeout_s=stall_min * 60,
                                  use_s3="--s3" in argv, use_cloud="--cloud" in argv)
            if (index.status == "not processed").any():
                print(f"build incomplete (stopped early); rerun build-split {split} to resume")
                return
        report = summarize_build(part, out, figure_path=EDA_DIR / f"{prefix}_{split}_build_sample.png")
        print(report)
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(f"\n\n{'#' * 88}\n# {mission.upper()} {split.upper()} BUILD REPORT (preprocessing_pipeline.py, "
                     f"{datetime.now():%Y-%m-%d %H:%M:%S})\n{'#' * 88}\n" + "\n".join("  " + l for l in report.splitlines()) + "\n")
    else:
        raise SystemExit(f"unknown command {command}")


if __name__ == "__main__":
    main(sys.argv)
