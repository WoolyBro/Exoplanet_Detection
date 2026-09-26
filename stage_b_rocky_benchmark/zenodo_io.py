"""Parse the Kreidberg & Stevenson (2025) JWST rocky-planet spectra (Zenodo 10.5281/zenodo.15084226)
and cross-check each file against the local NASA Exoplanet Archive .tbl spectra.

Rocky-planet benchmark only. Nothing here feeds the Stage B classifier, and nothing in stage_b/ or
labels/verified_gas_labels.csv is written. The .tbl files are read for verification only (Step 6),
through the unchanged stage_b_pipeline.parse_tbl / select_spectra.

Internal structure (the same as stage_b_pipeline.parse_tbl returns):
    DataFrame[wl (um), depth_ppm, err_ppm] + extra columns half_width_um, half_width_derived
"""
from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

RESEARCH = Path(__file__).resolve().parents[1]  # stage_b_rocky_benchmark/ -> project root
OUT_DIR = RESEARCH / "outputs" / "rocky_benchmark"
sys.path.insert(0, str(RESEARCH / "stage_b_atmospheres"))
import stage_b_pipeline as sb  # noqa: E402  (reused unchanged: parse_tbl, select_spectra, resample)

ZEN_DIR = RESEARCH / "datasets" / "zenodo_15084226_rocky_spectra"
PSCOMP = RESEARCH / "exoplanet_research_data" / "01_candidate_catalogs" / "pscomppars_confirmed_planets.csv"

# Zenodo file -> NASA Exoplanet Archive planet name (the file names carry no spaces or hyphens)
FILES = {
    "GJ1132b.txt": "GJ 1132 b", "GJ341b.txt": "GJ 341 b", "GJ486b.txt": "GJ 486 b",
    "L98-59c.txt": "L 98-59 c", "L98-59d.txt": "L 98-59 d",
    "LHS1140b-NIRISS.txt": "LHS 1140 b", "LHS1140b-NIRSpec.txt": "LHS 1140 b", "LHS1140c.txt": "LHS 1140 c",
    "LHS475b.txt": "LHS 475 b", "TOI836b.txt": "TOI-836 b",
    "TRAPPIST1b.txt": "TRAPPIST-1 b", "TRAPPIST1h.txt": "TRAPPIST-1 h",
}
PLANETS = sorted(set(FILES.values()))


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def planet_params() -> pd.DataFrame:
    """Composite parameters for the rocky planets (pscomppars, read-only)."""
    df = pd.read_csv(PSCOMP, comment="#", low_memory=False)
    df["_n"] = df.pl_name.map(_norm)
    cols = ["pl_name", "pl_rade", "pl_bmasse", "pl_bmasselim", "pl_bmassprov", "pl_eqt", "pl_trandep",
            "pl_ratror", "st_rad", "st_teff", "st_spectype"]
    out = df[df._n.isin([_norm(p) for p in PLANETS + ["TRAPPIST-1 c"]])][[c for c in cols if c in df.columns]]
    return out.set_index("pl_name")


def _read_numeric(path: Path) -> tuple[np.ndarray, list[str]]:
    comments, rows = [], []
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            comments.append(s)
            continue
        rows.append([float(x) for x in s.split()])
    widths = {len(r) for r in rows}
    if widths != {4}:
        raise ValueError(f"{path.name}: expected 4 numeric columns on every data row, found {sorted(widths)}")
    return np.array(rows), comments


def _half_widths_from_spacing(wl: np.ndarray) -> np.ndarray:
    """Half the mean distance to the neighbouring bin centres (NaN where a point has no distinct neighbour)."""
    order = np.argsort(wl)
    gaps = np.diff(wl[order])
    gaps = np.where(gaps > 0, gaps, np.nan)
    sp = np.nanmean(np.vstack([np.concatenate([[np.nan], gaps]), np.concatenate([gaps, [np.nan]])]), axis=0)
    out = np.empty_like(wl)
    out[order] = 0.5 * sp
    return out


def _declared_units(comments: list[str]) -> str:
    text = " ".join(comments).lower()
    if "ppm" in text:
        return "ppm"
    if "%" in text or "percent" in text:
        return "%"
    return "not stated"


def parse_zenodo(path: Path, params: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """One Zenodo file -> DataFrame[wl, depth_ppm, err_ppm, half_width_um, half_width_derived] + info."""
    a, comments = _read_numeric(path)
    info = {"file": path.name, "planet": FILES[path.name], "rows": len(a), "comment_lines": len(comments),
            "declared_units": _declared_units(comments)}

    c0, c1, dep, err = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    # Column 2 is documented as a half-width (<~0.2 um). If it instead holds values of order the wavelength
    # itself and every row has c1 > c0, the file gives bin START and END (LHS1140b-NIRSpec.txt).
    edges = bool(np.median(c1) > 0.5 and np.all(c1 > c0))
    if edges:
        wl, hw = (c0 + c1) / 2.0, (c1 - c0) / 2.0
        info["column_layout"] = "bin start, bin end, depth, err  (converted: centre=(start+end)/2, half-width=(end-start)/2)"
        info["before_after_first3"] = [
            {"start": float(c0[i]), "end": float(c1[i]), "centre": float(wl[i]), "half_width": float(hw[i])}
            for i in range(min(3, len(a)))]
    else:
        wl, hw = c0, c1.copy()
        info["column_layout"] = "centre, half-width, depth, err"

    # Zero half-widths (GJ1132b, L98-59c): derive from the spacing to neighbouring bin centres, flagged.
    # Binning never divides by a width (points go into grid bins by their centre, weights come from the
    # errors), so zero widths cannot cause a division by zero there; the derived width is only used where
    # a bin extent is needed (the Step 6 cross-check) and for the resolution sanity check.
    derived = ~(hw > 0)
    if derived.any():
        hw[derived] = _half_widths_from_spacing(wl)[derived]
    info["half_width_zero_rows"] = int(derived.sum())

    bad_err = ~(np.isfinite(err) & (err > 0))
    info["dropped_nonpositive_err"] = int(bad_err.sum())  # would give infinite inverse-variance weights
    keep = ~bad_err & np.isfinite(wl) & np.isfinite(dep)
    df = pd.DataFrame({"wl": wl[keep], "depth_ppm": dep[keep], "err_ppm": err[keep],
                       "half_width_um": hw[keep], "half_width_derived": derived[keep]}).sort_values("wl", kind="stable")
    df = df.reset_index(drop=True)

    # Units: verify the declared unit numerically against the catalogue transit depth (pl_trandep is in %).
    if params is not None and info["planet"] in params.index:
        cat_ppm = float(params.loc[info["planet"], "pl_trandep"]) * 1e4
        ratio = float(np.median(df.depth_ppm)) / cat_ppm if cat_ppm else np.nan
        info["catalogue_depth_ppm"] = round(cat_ppm, 1)
        info["median_depth_over_catalogue"] = round(ratio, 3)
        info["numeric_units"] = ("ppm" if 0.3 < ratio < 3 else "%" if 0.3e-4 < ratio < 3e-4
                                 else "fraction" if 0.3e-6 < ratio < 3e-6 else "UNRESOLVED")
    info["wl_min"], info["wl_max"] = float(df.wl.min()), float(df.wl.max())
    info["n_points"] = len(df)
    info["n_in_grid"] = int(((df.wl >= sb.GRID_MIN) & (df.wl < sb.GRID_MAX)).sum())
    info["grid_bins_covered"] = int(sb.resample(df)[2].sum())
    info["resolving_power_median"] = round(float(np.median(df.wl / (2 * df.half_width_um))), 0)
    info["restarts"] = int((np.diff(a[:, 0]) < 0).sum())  # >0 = several segments stacked in one file
    return df, info


# --------------------------------------------------------------------------- #
# Step 6: which local .tbl file(s) does each Zenodo file duplicate?
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def parse_local(file: str) -> pd.DataFrame:
    """A local archive .tbl spectrum via the unchanged stage_b parse_tbl, parsed once per process.
    Callers must not modify the returned frame (it is shared)."""
    return sb.parse_tbl(sb.SPEC_DIR / "spectra_tbl" / file)[0]


def compare(zen: pd.DataFrame, loc: pd.DataFrame) -> dict:
    """Bin the finer spectrum onto the coarser one's bins inside the overlap and compare depths.

    If the Zenodo file is a re-binning of the local data, the binned depths agree to a small fraction of the
    per-point uncertainty (norm_diff << 1)."""
    lo, hi = max(zen.wl.min(), loc.wl.min()), min(zen.wl.max(), loc.wl.max())
    if hi <= lo:
        return {"overlap_um": 0.0, "n_compared": 0}
    z = zen[(zen.wl >= lo) & (zen.wl <= hi)]
    l = loc[(loc.wl >= lo) & (loc.wl <= hi)].copy()
    l["half_width_um"] = _half_widths_from_spacing(l.wl.to_numpy())
    coarse, fine, coarse_is = (z, l, "zenodo") if len(z) <= len(l) else (l, z, "local")
    fw, fd, fe = fine.wl.to_numpy(), fine.depth_ppm.to_numpy(), fine.err_ppm.to_numpy()
    diffs, sig, pairs = [], [], []
    for c, hw, d, e in coarse[["wl", "half_width_um", "depth_ppm", "err_ppm"]].itertuples(index=False):
        sel = (fw >= c - hw - 1e-6) & (fw < c + hw - 1e-6) & np.isfinite(fe) & (fe > 0)
        if not sel.any():
            continue
        w = 1 / fe[sel] ** 2
        fb = np.sum(w * fd[sel]) / np.sum(w)
        diffs.append(d - fb)
        sig.append(e)
        pairs.append((d, fb))
    if not diffs:
        return {"overlap_um": round(hi - lo, 3), "n_compared": 0}
    diffs, sig, pairs = np.array(diffs), np.array(sig), np.array(pairs)
    r = np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1] if len(pairs) > 2 else np.nan
    return {"overlap_um": round(hi - lo, 3), "coarse_grid": coarse_is, "n_coarse": len(coarse),
            "n_compared": len(diffs), "coverage": round(len(diffs) / len(coarse), 2),
            "median_offset_ppm": round(float(np.median(diffs)), 1),
            "norm_diff": round(float(np.median(np.abs(diffs) / sig)), 3),
            "norm_diff_after_offset": round(float(np.median(np.abs(diffs - np.median(diffs)) / sig)), 3),
            "corr": round(float(r), 3)}


def local_candidates(planet: str) -> list[dict]:
    """Every local JWST transmission .tbl for the planet, singly and combined per paper+instrument."""
    sel = sb.select_spectra([planet])
    out = [{"label": r.file, "files": [r.file], "bibcode": r.bibcode, "authors": r.authors,
            "instrument": r.instrument, "df": parse_local(r.file)} for r in sel.itertuples()]
    for (bib, inst), g in sel.groupby(["bibcode", "instrument"]):
        if len(g) < 2:
            continue
        out.append({"label": f"COMBINED {len(g)} files ({g.authors.iloc[0]}, {inst})", "files": list(g.file),
                    "bibcode": bib, "authors": g.authors.iloc[0], "instrument": inst,
                    "df": pd.concat([parse_local(f) for f in g.file], ignore_index=True)})
    return out


def classify_match(m: dict) -> str:
    if m.get("n_compared", 0) == 0:
        return "no overlap"
    if m["coverage"] >= 0.8 and m["norm_diff"] <= 0.25:
        return "same data (re-binned)"
    if m["coverage"] >= 0.8 and m["norm_diff_after_offset"] <= 0.25:
        return "same data up to a constant depth offset"
    if m["coverage"] >= 0.5 and m["norm_diff"] <= 1.0:
        return "consistent within errors (different reduction or binning)"
    return "INCONSISTENT"
