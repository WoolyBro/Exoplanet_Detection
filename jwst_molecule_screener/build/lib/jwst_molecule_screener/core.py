"""Preprocessing, frozen models and the calibrated call rule of the JWST molecule screener.

A spectrum (wavelength in micron, transit depth and its uncertainty in ppm) is binned by inverse-variance
weighting onto 100 bins over 1.0-5.0 micron, centred on its own median, and scored for CH4 and CO2 by a
gradient-boosted model and a 1-D CNN, both trained only on simulated spectra that include unocculted spots,
faculae and hazes. Each score is mapped to a probability by a Platt map fitted on real labelled JWST planets
(41 for CH4, 34 for CO2), and a call is made only when both models agree.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import numpy as np

GRID_MIN, GRID_MAX, N_BINS = 1.0, 5.0, 100
EDGES = np.linspace(GRID_MIN, GRID_MAX, N_BINS + 1)
CENTERS = 0.5 * (EDGES[:-1] + EDGES[1:])
DEPTH_SCALE = 1000.0
GASES = {"CH4": {"index": 1, "bands": [(2.20, 2.40), (3.20, 3.45)]}, "CO2": {"index": 2, "bands": [(4.20, 4.45)]}}
_DATA = resources.files("jwst_molecule_screener") / "data"
_CACHE: dict = {}


def bin_spectrum(wl_um, depth_ppm, err_ppm) -> np.ndarray:
    """(3, 100): [depth minus its median, uncertainty, coverage mask] on the 1-5 micron grid."""
    wl, d, e = (np.asarray(a, float) for a in (wl_um, depth_ppm, err_ppm))
    ok = np.isfinite(wl) & np.isfinite(d)
    wl, d, e = wl[ok], d[ok], e[ok]
    depth, unc, mask = np.zeros(N_BINS), np.zeros(N_BINS), np.zeros(N_BINS)
    idx = np.digitize(wl, EDGES) - 1
    for b in range(N_BINS):
        sel = idx == b
        if not sel.any():
            continue
        db, eb = d[sel], e[sel]
        good = np.isfinite(eb) & (eb > 0)
        if good.all():
            w = 1.0 / eb ** 2
            depth[b], unc[b] = np.sum(w * db) / np.sum(w), 1.0 / np.sqrt(np.sum(w))
        else:
            depth[b] = np.mean(db)
            unc[b] = np.nanmean(eb[good]) / np.sqrt(good.sum()) if good.any() else (np.std(db) / np.sqrt(len(db)) if len(db) > 1 else np.nan)
        mask[b] = 1.0
    cov = mask > 0
    if not cov.any():
        raise ValueError("the spectrum has no data between 1.0 and 5.0 micron")
    depth = np.where(cov, depth - np.median(depth[cov]), 0.0)
    unc = np.where(cov, np.where(np.isfinite(unc), unc, np.nanmedian(unc[cov])), 0.0)
    return np.stack([depth, unc, mask])


def observable(x: np.ndarray, gas: str) -> bool:
    return any(x[2][(CENTERS >= lo) & (CENTERS <= hi)].sum() >= 3 for lo, hi in GASES[gas]["bands"])


def _features(X):
    d, u, m = X[:, 0] / DEPTH_SCALE, X[:, 1] / DEPTH_SCALE, X[:, 2]
    snr = np.where(m > 0, X[:, 0] / np.maximum(X[:, 1], 1e-6), 0.0)
    return np.concatenate([d, u, m, snr], axis=1)


def build_cnn(n_out: int = 5):
    from torch import nn
    return nn.Sequential(nn.Conv1d(3, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                         nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                         nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(4), nn.Flatten(),
                         nn.Dropout(0.3), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, n_out))


def _models():
    if not _CACHE:
        import joblib
        import torch
        _CACHE["meta"] = json.loads((_DATA / "calibration.json").read_text())
        _CACHE["gbm"] = {g: joblib.load(str(_DATA / f"gbm_{g.lower()}.joblib")) for g in GASES}
        net = build_cnn()
        net.load_state_dict(torch.load(str(_DATA / "cnn.pt"), map_location="cpu", weights_only=True))
        net.eval()
        _CACHE["cnn"] = net
    return _CACHE


def _platt(s, ab):
    s = np.clip(np.asarray(s, float), 1e-4, 1 - 1e-4)
    return 1 / (1 + np.exp(-(ab[0] * np.log(s / (1 - s)) + ab[1])))


def score_binned(X: np.ndarray) -> dict:
    """Raw scores for binned spectra X (n, 3, 100)."""
    import torch
    m = _models()
    with torch.no_grad():
        t = torch.tensor(np.stack([X[:, 0] / DEPTH_SCALE, X[:, 1] / DEPTH_SCALE, X[:, 2]], 1), dtype=torch.float32)
        cnn = torch.sigmoid(m["cnn"](t)).numpy()
    return {g: {"gbm": m["gbm"][g].predict_proba(_features(X))[:, 1], "cnn": cnn[:, GASES[g]["index"]]} for g in GASES}


def screen(spectra) -> dict:
    """Screen one planet. `spectra` is a list of (wavelength_um, depth_ppm, err_ppm) tuples (several visits or
    instruments of the same planet), or a single such tuple. Returns, per gas, the planet-level scores,
    calibrated probabilities, the spread across spectra and the call (present / absent / ambiguous /
    not observable)."""
    if isinstance(spectra, tuple):
        spectra = [spectra]
    binned = []
    for s in spectra:
        try:
            binned.append(bin_spectrum(*s))
        except ValueError:                       # e.g. a MIRI spectrum with no data in 1-5 micron
            pass
    out = {"n_spectra_given": len(spectra), "n_spectra_on_grid": len(binned)}
    if not binned:
        return {**out, **{g: {"call": "not observable", "n_spectra": 0} for g in GASES}}
    X = np.array(binned)
    meta = _models()["meta"]
    raw = score_binned(X)
    for g in GASES:
        ok = np.array([observable(x, g) for x in X])
        if not ok.any():
            out[g] = {"call": "not observable", "n_spectra": 0}
            continue
        sg, sc = raw[g]["gbm"][ok], raw[g]["cnn"][ok]
        pg, pc = float(_platt([sg.mean()], meta["platt"][g]["gbm"])[0]), float(_platt([sc.mean()], meta["platt"][g]["cnn"])[0])
        per = _platt(sg, meta["platt"][g]["gbm"])
        disagree = bool(len(per) > 1 and per.min() < 0.5 <= per.max())
        call = "present" if pg >= 0.8 and pc >= 0.8 else "absent" if pg <= 0.2 and pc <= 0.2 else "ambiguous"
        out[g] = {"call": "ambiguous" if disagree else call, "p_gbm": round(pg, 3), "p_cnn": round(pc, 3),
                  "score_gbm": round(float(sg.mean()), 3), "score_cnn": round(float(sc.mean()), 3),
                  "n_spectra": int(ok.sum()), "spectra_disagree": disagree,
                  "p_gbm_per_spectrum": [round(float(v), 3) for v in per]}
    return out


def read_table(path: str | Path, wl_col: int = 0, depth_col: int = 1, err_col: int = 2, depth_unit: str = "ppm"):
    """Read a whitespace or comma table (comments '#'); depth_unit: ppm | percent | fraction."""
    path = Path(path)
    a = np.loadtxt(path, comments="#", delimiter="," if path.suffix == ".csv" else None, ndmin=2)
    f = {"ppm": 1.0, "percent": 1e4, "fraction": 1e6}[depth_unit]
    return a[:, wl_col], a[:, depth_col] * f, a[:, err_col] * f
