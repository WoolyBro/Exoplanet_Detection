#!/usr/bin/env python3
"""
stage_b_pipeline.py - Stage B: molecule presence from real JWST transmission spectra
=====================================================================================

Parts
  1. Dataset: parse the ORIGINAL .tbl spectra (04_atmospheric_spectra/spectra_tbl/) for the 13 planets
     in outputs/labels/verified_gas_labels.csv; 1-5 um, 100 bins; channels = depth, uncertainty, coverage mask.
  2. Split: leave-one-planet-out (13 folds).
  3. Models: (a) gradient boosting per molecule, (b) small NumPy 1-D CNN with masked multi-label loss.
  4. Evaluation vs majority baseline; report + figure.

Targets: H2O, CH4, CO2, SO2, CO. "unknown" labels are masked out of loss and metrics, never treated as 0.
Never reads 05_final_ML_dataset_DO_NOT_USE/. Writes outputs/stage_b/, eda/STAGE_B_RESULTS.md and
eda/stage_b_spectra_sample.png.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import ascii as astro_ascii
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

warnings.filterwarnings("ignore")

RESEARCH = Path(__file__).resolve().parents[1]  # stage_b_atmospheres/ -> project root
SPEC_DIR = RESEARCH / "exoplanet_research_data" / "04_atmospheric_spectra"
LABELS = RESEARCH / "outputs" / "labels" / "verified_gas_labels.csv"
OUT = RESEARCH / "outputs" / "stage_b"
EDA = RESEARCH / "eda"

MOLECULES = ["H2O", "CH4", "CO2", "SO2", "CO"]
GRID_MIN, GRID_MAX, N_BINS = 1.0, 5.0, 100
EDGES = np.linspace(GRID_MIN, GRID_MAX, N_BINS + 1)
CENTERS = 0.5 * (EDGES[:-1] + EDGES[1:])

# Absorption bands inside 1-5 um used to decide whether a spectrum can show a molecule at all.
# A planet-level label is applied to a spectrum only if >= MIN_BAND_BINS covered bins fall in any band.
BANDS = {
    "H2O": [(1.30, 1.50), (1.80, 2.00), (2.60, 3.00)],
    "CH4": [(2.20, 2.40), (3.20, 3.45)],
    "CO2": [(1.95, 2.10), (4.20, 4.45)],
    "SO2": [(3.95, 4.15)],
    "CO": [(4.50, 4.90)],
}
MIN_BAND_BINS = 3
DEPTH_SCALE_PPM = 1000.0  # CNN inputs divided by this constant (numerics only)
SEED = 42


# =========================================================================== #
# Part 1: dataset
# =========================================================================== #
def select_spectra(planets: list[str]) -> pd.DataFrame:
    meta = pd.read_csv(SPEC_DIR / "metadata" / "spectra_metadata.csv", low_memory=False)
    sel = meta[(meta.spec_type == "Transmission") & meta.facility.astype(str).str.contains("James Webb")
               & meta.pl_name.isin(planets)].copy()
    files = {p.name for p in (SPEC_DIR / "spectra_tbl").glob("*.tbl")}
    def local(spec_path: str) -> str:
        base = spec_path.split("/")[-1]
        hits = [f for f in files if f == base or f.endswith("_" + base)]
        if len(hits) != 1:
            raise FileNotFoundError(f"{spec_path}: {len(hits)} local matches")
        return hits[0]
    sel["file"] = sel.spec_path.map(local)
    return sel.reset_index(drop=True)


def parse_tbl(path: Path) -> tuple[pd.DataFrame, dict]:
    """Wavelength (um), depth (ppm), uncertainty (ppm) from an IPAC .tbl; drops artifact rows."""
    t = astro_ascii.read(path, format="ipac")
    info = {"rows_raw": len(t)}
    wl_unit = str(t["CENTRALWAVELNG"].unit)
    if "micron" not in wl_unit:
        raise ValueError(f"{path.name}: unexpected wavelength unit {wl_unit!r}")

    def col(name):
        if name not in t.colnames:
            return np.full(len(t), np.nan)
        c = t[name]
        vals = np.ma.filled(np.ma.asarray(c, dtype=float), np.nan) if hasattr(c, "mask") else np.asarray(c, dtype=float)
        return vals

    wl = col("CENTRALWAVELNG")
    depth, e1, e2 = col("PL_TRANDEP"), col("PL_TRANDEPERR1"), col("PL_TRANDEPERR2")
    source = "PL_TRANDEP"
    dep_unit = str(t["PL_TRANDEP"].unit) if "PL_TRANDEP" in t.colnames else ""
    if np.all(~np.isfinite(depth)):
        r, r1, r2 = col("PL_RATROR"), col("PL_RATRORERR1"), col("PL_RATRORERR2")
        if np.all(~np.isfinite(r)):
            raise ValueError(f"{path.name}: neither PL_TRANDEP nor PL_RATROR present")
        depth = 100.0 * r ** 2
        e1, e2 = 100.0 * 2 * r * np.abs(r1), 100.0 * 2 * r * np.abs(r2)
        source, dep_unit = "PL_RATROR (converted: depth = 100*Rp/Rs^2 %)", "%"
    if dep_unit.strip() not in ("%", "percent"):
        raise ValueError(f"{path.name}: unexpected depth unit {dep_unit!r}")

    # artifact rows: all-null or all-zero in the key columns (the bug that corrupted the quarantined copies)
    key = np.vstack([wl, depth])
    all_null = np.all(~np.isfinite(key), axis=0)
    all_zero = np.all(np.nan_to_num(key, nan=1.0) == 0, axis=0) & np.all(np.nan_to_num(np.vstack([e1, e2]), nan=0) == 0, axis=0)
    partial = ~all_null & ~all_zero & (~np.isfinite(wl) | ~np.isfinite(depth))
    info.update({"artifact_all_null": int(all_null.sum()), "artifact_all_zero": int(all_zero.sum()),
                 "partial_null_dropped": int(partial.sum()), "depth_source": source})
    keep = ~(all_null | all_zero | partial)
    err = np.nanmean(np.vstack([np.abs(e1), np.abs(e2)]), axis=0) * 1e4  # % -> ppm
    df = pd.DataFrame({"wl": wl[keep], "depth_ppm": depth[keep] * 1e4, "err_ppm": err[keep]})
    info["rows_kept"] = len(df)
    info["wl_min"], info["wl_max"] = float(df.wl.min()), float(df.wl.max())
    info["n_in_grid"] = int(((df.wl >= GRID_MIN) & (df.wl < GRID_MAX)).sum())
    info["n_missing_err"] = int((~np.isfinite(df.err_ppm)).sum())
    return df, info


def resample(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse-variance weighted mean per bin; uncertainty 1/sqrt(sum w). Bins with no data get mask 0
    and are left at 0 in depth/uncertainty (after normalization 0 = the spectrum's median), so empty bins
    are identified by the mask channel, never by a depth value."""
    depth, unc, mask = np.zeros(N_BINS), np.zeros(N_BINS), np.zeros(N_BINS)
    idx = np.digitize(df.wl.to_numpy(), EDGES) - 1
    for b in range(N_BINS):
        sel = idx == b
        if not sel.any():
            continue
        d, e = df.depth_ppm.to_numpy()[sel], df.err_ppm.to_numpy()[sel]
        good = np.isfinite(e) & (e > 0)
        if good.all():
            w = 1.0 / e ** 2
            depth[b] = np.sum(w * d) / np.sum(w)
            unc[b] = 1.0 / np.sqrt(np.sum(w))
        else:  # no usable errors in this bin: plain mean, scatter-based uncertainty if possible
            depth[b] = np.mean(d)
            unc[b] = np.nanmean(e[good]) / np.sqrt(good.sum()) if good.any() else (np.std(d) / np.sqrt(len(d)) if len(d) > 1 else np.nan)
        mask[b] = 1.0
    return depth, unc, mask


def band_observable(mask: np.ndarray, molecule: str) -> bool:
    for lo, hi in BANDS[molecule]:
        if mask[(CENTERS >= lo) & (CENTERS <= hi)].sum() >= MIN_BAND_BINS:
            return True
    return False


def build_dataset() -> dict:
    labels = pd.read_csv(LABELS, dtype=str, keep_default_na=False)
    planets = labels.planet.tolist()
    sel = select_spectra(planets)
    X, infos, dropped = [], [], []
    for r in sel.itertuples():
        df, info = parse_tbl(SPEC_DIR / "spectra_tbl" / r.file)
        depth, unc, mask = resample(df)
        covered = mask > 0
        if not covered.any():  # e.g. MIRI 5-12 um spectra: nothing on the 1-5 um grid
            dropped.append({"planet": r.pl_name, "file": r.file, "authors": r.authors, "instrument": r.instrument,
                            "wl_min": info["wl_min"], "wl_max": info["wl_max"], **info,
                            "reason": f"no data inside {GRID_MIN}-{GRID_MAX} um"})
            continue
        med = np.median(depth[covered])
        depth = np.where(covered, depth - med, 0.0)  # normalization: subtract per-spectrum median (ppm)
        # bins whose uncertainty could not be computed: keep depth, flag with median uncertainty
        bad_unc = covered & ~np.isfinite(unc)
        if bad_unc.any():
            unc[bad_unc] = np.nanmedian(unc[covered & np.isfinite(unc)])
        unc = np.where(covered, unc, 0.0)
        X.append(np.stack([depth, unc, mask]))
        infos.append({"planet": r.pl_name, "file": r.file, "authors": r.authors, "instrument": r.instrument,
                      "median_depth_ppm": round(float(med), 1), "bins_covered": int(covered.sum()), **info})
    X = np.array(X)  # (n, 3, 100)
    meta = pd.DataFrame(infos)

    lab = labels.set_index("planet")
    Y = np.full((len(meta), len(MOLECULES)), np.nan)
    band_masked = np.zeros_like(Y, dtype=bool)
    for i, r in meta.iterrows():
        for j, m in enumerate(MOLECULES):
            v = lab.at[r.planet, m]
            if v == "unknown":
                continue
            if not band_observable(X[i, 2], m):
                band_masked[i, j] = True
                continue
            Y[i, j] = float(v)
    return {"X": X, "Y": Y, "meta": meta, "band_masked": band_masked, "labels": labels,
            "dropped": pd.DataFrame(dropped), "all_selected": sel}


# =========================================================================== #
# Part 3: models
# =========================================================================== #
def planet_weights(planets: np.ndarray) -> np.ndarray:
    """1 / (number of spectra of that planet): each planet contributes equally to training."""
    counts = pd.Series(planets).value_counts()
    return np.array([1.0 / counts[p] for p in planets])


def fit_gbm(Xtr, ytr, wtr):
    clf = GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate=0.1, subsample=0.8,
                                     random_state=SEED)
    clf.fit(Xtr.reshape(len(Xtr), -1), ytr, sample_weight=wtr)
    return clf


class TinyCNN:
    """conv(3->8, k7) ReLU -> maxpool(4) -> conv(8->8, k5) ReLU -> global average pool -> dense(8->5) -> sigmoid.
    ~550 parameters. Plain NumPy with manual backprop (verified by a numerical gradient check)."""

    def __init__(self, seed: int, c1=8, k1=7, c2=8, k2=5, pool=4, n_out=len(MOLECULES)):
        rng = np.random.default_rng(seed)
        self.k1, self.k2, self.pool = k1, k2, pool
        self.p = {
            "W1": rng.normal(0, np.sqrt(2 / (3 * k1)), (c1, 3, k1)), "b1": np.zeros(c1),
            "W2": rng.normal(0, np.sqrt(2 / (c1 * k2)), (c2, c1, k2)), "b2": np.zeros(c2),
            "W3": rng.normal(0, np.sqrt(1 / c2), (n_out, c2)), "b3": np.zeros(n_out),
        }

    @staticmethod
    def _conv(x, W, b):  # x (n, cin, L) -> (n, cout, L-k+1)
        k = W.shape[2]
        win = np.lib.stride_tricks.sliding_window_view(x, k, axis=2)  # (n, cin, Lo, k)
        return np.einsum("nclk,ock->nol", win, W) + b[None, :, None], win

    def forward(self, x):
        z1, win1 = self._conv(x, self.p["W1"], self.p["b1"])
        a1 = np.maximum(z1, 0)
        n, c, L = a1.shape
        Lp = L // self.pool
        a1c = a1[:, :, :Lp * self.pool].reshape(n, c, Lp, self.pool)
        arg = a1c.argmax(axis=3)
        p1 = a1c.max(axis=3)
        z2, win2 = self._conv(p1, self.p["W2"], self.p["b2"])
        a2 = np.maximum(z2, 0)
        g = a2.mean(axis=2)
        logits = g @ self.p["W3"].T + self.p["b3"]
        self.cache = (x, win1, z1, a1, arg, p1, win2, z2, a2, g)
        return logits

    def loss_grad(self, x, Y, w, l2):
        """Masked, weighted BCE. Y has NaN for masked labels; w per-sample weights."""
        logits = self.forward(x)
        prob = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        known = np.isfinite(Y)
        Yz = np.where(known, Y, 0.0)
        W = known * w[:, None]
        denom = W.sum()
        bce = -(Yz * np.log(prob + 1e-12) + (1 - Yz) * np.log(1 - prob + 1e-12))
        loss = (W * bce).sum() / denom + l2 * sum(np.sum(v ** 2) for k, v in self.p.items() if k.startswith("W"))
        dlogits = W * (prob - Yz) / denom
        x, win1, z1, a1, arg, p1, win2, z2, a2, g = self.cache
        grads = {"W3": dlogits.T @ g, "b3": dlogits.sum(0)}
        dg = dlogits @ self.p["W3"]
        da2 = np.repeat(dg[:, :, None], a2.shape[2], axis=2) / a2.shape[2]
        dz2 = da2 * (z2 > 0)
        grads["W2"] = np.einsum("nol,nclk->ock", dz2, win2)
        grads["b2"] = dz2.sum((0, 2))
        dp1 = self._conv_input_grad(dz2, self.p["W2"], p1.shape)
        da1 = np.zeros_like(a1)
        idx = arg + (np.arange(p1.shape[2]) * self.pool)[None, None, :]
        np.put_along_axis(da1, idx, dp1, axis=2)
        dz1 = da1 * (z1 > 0)
        grads["W1"] = np.einsum("nol,nclk->ock", dz1, win1)
        grads["b1"] = dz1.sum((0, 2))
        for k in grads:
            if k.startswith("W"):
                grads[k] = grads[k] + 2 * l2 * self.p[k]
        return loss, grads

    @staticmethod
    def _conv_input_grad(dz, W, in_shape):
        k = W.shape[2]
        dx = np.zeros(in_shape)
        for j in range(k):
            dx[:, :, j:j + dz.shape[2]] += np.einsum("nol,oc->ncl", dz, W[:, :, j])
        return dx

    def fit(self, x, Y, w, epochs=300, lr=3e-3, l2=1e-3):
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v = {k: np.zeros_like(v) for k, v in self.p.items()}
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, epochs + 1):
            _, grads = self.loss_grad(x, Y, w, l2)
            for k in self.p:
                m[k] = b1 * m[k] + (1 - b1) * grads[k]
                v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
                self.p[k] -= lr * (m[k] / (1 - b1 ** t)) / (np.sqrt(v[k] / (1 - b2 ** t)) + eps)
        return self

    def predict_proba(self, x):
        return 1 / (1 + np.exp(-np.clip(self.forward(x), -30, 30)))


def gradient_check() -> float:
    rng = np.random.default_rng(0)
    net = TinyCNN(seed=1)
    x = rng.normal(size=(4, 3, 100))
    Y = rng.integers(0, 2, size=(4, 5)).astype(float)
    Y[0, 1] = np.nan
    w = rng.uniform(0.5, 1.5, 4)
    _, grads = net.loss_grad(x, Y, w, l2=1e-3)
    worst = 0.0
    for k in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        flat = net.p[k].reshape(-1)
        for idx in rng.choice(flat.size, size=min(6, flat.size), replace=False):
            old = flat[idx]
            flat[idx] = old + 1e-5
            lp, _ = net.loss_grad(x, Y, w, 1e-3)
            flat[idx] = old - 1e-5
            lm, _ = net.loss_grad(x, Y, w, 1e-3)
            flat[idx] = old
            num = (lp - lm) / 2e-5
            ana = grads[k].reshape(-1)[idx]
            worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
    return worst


# =========================================================================== #
# Part 2 + 3: leave-one-planet-out
# =========================================================================== #
def lopo(data: dict, cnn_seeds=(0, 1, 2, 3, 4)) -> pd.DataFrame:
    X, Y, meta = data["X"], data["Y"], data["meta"]
    planets = meta.planet.to_numpy()
    w_all = planet_weights(planets)
    Xc = X.copy()
    Xc[:, :2, :] /= DEPTH_SCALE_PPM
    rows = []
    for held in sorted(set(planets)):
        tr, te = planets != held, planets == held
        cnn_prob = np.zeros((te.sum(), len(MOLECULES)))
        for s in cnn_seeds:
            net = TinyCNN(seed=SEED + s).fit(Xc[tr], Y[tr], w_all[tr])
            cnn_prob += net.predict_proba(Xc[te]) / len(cnn_seeds)
        for j, mol in enumerate(MOLECULES):
            known_te = np.isfinite(Y[te, j])
            ytr_known = np.isfinite(Y[tr, j])
            ytr = Y[tr, j][ytr_known]
            classes = np.unique(ytr)
            if len(classes) == 2:
                gbm = fit_gbm(X[tr][ytr_known], ytr, w_all[tr][ytr_known])
                gbm_prob = gbm.predict_proba(X[te].reshape(te.sum(), -1))[:, 1]
                degenerate = False
            else:  # training fold holds only one class: nothing to learn, predict that class
                gbm_prob = np.full(te.sum(), float(classes[0]) if len(classes) else 0.5)
                degenerate = True
            train_pos_rate = float(np.average(ytr, weights=w_all[tr][ytr_known])) if len(ytr) else np.nan
            for k, i in enumerate(np.flatnonzero(te)):
                rows.append({"fold": held, "planet": held, "spectrum": meta.file[i], "molecule": mol,
                             "label": Y[i, j], "label_known": bool(known_te[k]),
                             "gbm_prob": float(gbm_prob[k]), "cnn_prob": float(cnn_prob[k, j]),
                             "baseline_prob": 1.0 if train_pos_rate >= 0.5 else 0.0,
                             "train_single_class": degenerate, "train_pos_rate": train_pos_rate})
    return pd.DataFrame(rows)


def metrics(y, p) -> dict:
    y = np.asarray(y, int)
    pred = (np.asarray(p) >= 0.5).astype(int)
    out = {"n": len(y), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
           "accuracy": accuracy_score(y, pred),
           "precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred, zero_division=0),
           "f1": f1_score(y, pred, zero_division=0)}
    out["roc_auc"] = roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan
    return out


def evaluate(pred: pd.DataFrame) -> pd.DataFrame:
    """Baselines: 'baseline' = majority class of each fold's TRAINING planets (leave-one-out majority; on a
    near-balanced molecule this is systematically wrong on the held-out planet, a known LOPO artefact);
    'majority_global' = majority class over all evaluable planets for that molecule (a fixed reference
    constant, not a trained model)."""
    known = pred[pred.label_known].copy()
    rows = []
    for mol in MOLECULES:
        k = known[known.molecule == mol].copy()
        planet_level = k.groupby("planet").agg(label=("label", "first"), gbm_prob=("gbm_prob", "mean"),
                                               cnn_prob=("cnn_prob", "mean"), baseline_prob=("baseline_prob", "first"),
                                               single=("train_single_class", "first"))
        global_major = 1.0 if (planet_level.label == 1).sum() >= (planet_level.label == 0).sum() else 0.0
        planet_level["majority_global_prob"] = global_major
        k["majority_global_prob"] = global_major
        for level, table in [("planet", planet_level), ("spectrum", k)]:
            if len(table) == 0:
                rows.append({"molecule": mol, "level": level, "evaluable_folds": 0})
                continue
            for model in ["gbm", "cnn", "baseline", "majority_global"]:
                m = metrics(table.label, table[f"{model}_prob"])
                rows.append({"molecule": mol, "level": level, "model": model,
                             "evaluable_folds": int(k.planet.nunique()),
                             "folds_train_single_class": int(planet_level.single.sum()), **m})
    res = pd.DataFrame(rows)

    def reliability(r):
        if r.get("n_pos", 0) >= 3 and r.get("n_neg", 0) >= 3:
            return "meaningful (>=3 planets of each class)"
        if r.get("n_neg", 0) <= 1 or r.get("n_pos", 0) <= 1:
            return "NOT MEANINGFUL: only one example of a class - indicative at best"
        return "exploratory: only 2 planets of the minority class"
    planet_rel = {r["molecule"]: reliability(r) for r in res[(res.level == "planet") & (res.model == "gbm")].to_dict("records")}
    res["reliability"] = res.molecule.map(planet_rel)
    return res


# =========================================================================== #
# Part 4: figure + report
# =========================================================================== #
def plot_samples(data: dict, path: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X, meta, labels = data["X"], data["meta"], data["labels"].set_index("planet")
    wanted = ["WASP-39 b", "K2-18 b", "HAT-P-18 b", "WASP-80 b"]
    picks = []
    for pl in wanted:
        cand = meta[meta.planet == pl].sort_values("bins_covered", ascending=False)
        if len(cand):
            picks.append(cand.index[0])
    surface, ink, ink2, grid, series, band = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#b7d3f6"
    plt.rcParams.update({"figure.facecolor": surface, "axes.facecolor": surface, "savefig.facecolor": surface,
                         "axes.edgecolor": grid, "axes.labelcolor": ink2, "xtick.color": ink2, "ytick.color": ink2,
                         "text.color": ink, "axes.grid": True, "grid.color": grid, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 8})
    fig, axes = plt.subplots(len(picks), 1, figsize=(10, 2.5 * len(picks)), squeeze=False)
    for ax, i in zip(axes[:, 0], picks):
        d, u, m = X[i]
        cov = m > 0
        # shade uncovered bins so "no data" is visually distinct from "flat"
        for b in np.flatnonzero(~cov):
            ax.axvspan(EDGES[b], EDGES[b + 1], color=grid, alpha=0.8, linewidth=0, zorder=0)
        ax.fill_between(CENTERS, np.where(cov, d - u, np.nan), np.where(cov, d + u, np.nan), color=band,
                        linewidth=0, step="mid", label="±1σ binned uncertainty")
        ax.plot(CENTERS, np.where(cov, d, np.nan), color=series, linewidth=1.2, drawstyle="steps-mid",
                label="binned depth − median (ppm)")
        r = meta.loc[i]
        lab = labels.loc[r.planet]
        text = "  ".join(f"{mol}={lab[mol] if lab[mol] != 'unknown' else '?'}" for mol in MOLECULES)
        ax.set_title(f"{r.planet} | {r.authors} | {r.instrument[:45]} | {int(cov.sum())}/100 bins covered | labels: {text}",
                     loc="left", fontsize=7.5)
        ax.set_ylabel("Δ depth (ppm)")
        ax.set_xlim(GRID_MIN, GRID_MAX)
    axes[0, 0].legend(loc="upper right", fontsize=7, frameon=False)
    axes[-1, 0].set_xlabel("wavelength (µm); grey = no data in bin (coverage mask = 0)")
    fig.suptitle("Stage B input: resampled JWST transmission spectra (depth, uncertainty, coverage mask) with verified labels",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [f"{meta.planet[i]} ({meta.file[i]})" for i in picks]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    lines = []
    say = lambda s="": (print(s), lines.append(s))

    say("=== Part 1: dataset ===")
    data = build_dataset()
    meta, X, Y = data["meta"], data["X"], data["Y"]
    found = data["all_selected"].groupby("pl_name").size()
    say(f"JWST transmission spectra found in metadata: {len(data['all_selected'])}; per planet {found.to_dict()}")
    dropped = data["dropped"]
    if len(dropped):
        say(f"dropped (no data inside {GRID_MIN}-{GRID_MAX} um): {len(dropped)}")
        for r in dropped.itertuples():
            say(f"  {r.planet} | {r.authors} | {r.instrument} | {r.wl_min:.2f}-{r.wl_max:.2f} um | {r.file}")
        dropped.to_csv(OUT / "stage_b_dropped_spectra.csv", index=False)
    all_parsed = pd.concat([meta, dropped], ignore_index=True) if len(dropped) else meta
    counts = meta.groupby("planet").size()
    say(f"usable spectra (with data on the grid): {len(meta)} across {meta.planet.nunique()} planets")
    expected = {"WASP-39 b": 17, "WASP-107 b": 12, "TOI-270 d": 8, "WASP-17 b": 6, "K2-18 b": 4, "GJ 3470 b": 3,
                "HAT-P-18 b": 3, "TRAPPIST-1 c": 3, "HAT-P-26 b": 2, "HD 189733 b": 2, "HD 209458 b": 2,
                "WASP-52 b": 1, "WASP-80 b": 1}
    mism_found = {p: (int(found.get(p, 0)), n) for p, n in expected.items() if int(found.get(p, 0)) != n}
    mism_usable = {p: (int(counts.get(p, 0)), n) for p, n in expected.items() if int(counts.get(p, 0)) != n}
    say(f"usable per planet {counts.to_dict()}")
    say(f"found vs expected mismatches: {mism_found or 'none'}")
    say(f"usable vs expected (usable, expected): {mism_usable or 'none'}")
    say(f"artifact rows across all {len(all_parsed)} files: all-null {int(all_parsed.artifact_all_null.sum())}, all-zero "
        f"{int(all_parsed.artifact_all_zero.sum())}, partial-null dropped {int(all_parsed.partial_null_dropped.sum())} "
        f"(raw rows {int(all_parsed.rows_raw.sum())}, kept {int(all_parsed.rows_kept.sum())})")
    say(f"depth source: {all_parsed.depth_source.value_counts().to_dict()}; rows with missing errors: {int(all_parsed.n_missing_err.sum())}")
    say(f"bins covered per spectrum: min {meta.bins_covered.min()}, median {int(meta.bins_covered.median())}, max {meta.bins_covered.max()}")
    lab = data["labels"].set_index("planet")
    known_before = sum(1 for i in range(len(meta)) for m in MOLECULES if lab.at[meta.planet[i], m] != "unknown")
    say(f"spectrum-level labels: {known_before} known before band check; {int(data['band_masked'].sum())} masked because the "
        f"spectrum does not cover the molecule's band; {int(np.isfinite(Y).sum())} used")
    for j, m in enumerate(MOLECULES):
        say(f"  {m}: used {int(np.isfinite(Y[:, j]).sum())} (pos {int(np.nansum(Y[:, j] == 1))}, neg {int(np.nansum(Y[:, j] == 0))}), "
            f"band-masked {int(data['band_masked'][:, j].sum())}")
    meta.to_csv(OUT / "stage_b_spectra_index.csv", index=False)
    np.savez_compressed(OUT / "stage_b_dataset.npz", X=X, Y=Y, planets=meta.planet.to_numpy(), files=meta.file.to_numpy(),
                        molecules=np.array(MOLECULES), bin_centers=CENTERS)

    say("\n=== Part 2: leave-one-planet-out folds ===")
    fold_rows = []
    for pl in sorted(meta.planet.unique()):
        idx = np.flatnonzero(meta.planet == pl)
        known = [m for j, m in enumerate(MOLECULES) if np.isfinite(Y[idx, j]).any()]
        fold_rows.append({"held_out_planet": pl, "n_spectra": len(idx), "evaluable_molecules": ", ".join(known) or "none"})
    folds = pd.DataFrame(fold_rows)
    say(folds.to_string(index=False))
    per_mol = {m: int(sum(np.isfinite(Y[meta.planet == pl, j]).any() for pl in meta.planet.unique())) for j, m in enumerate(MOLECULES)}
    say(f"evaluable folds per molecule: {per_mol}")
    zero = [m for m, n in per_mol.items() if n == 0]
    if zero:
        say(f"STOP: molecules with zero evaluable folds: {zero}")
        (OUT / "stage_b_log.txt").write_text("\n".join(lines), encoding="utf-8")
        sys.exit(1)

    say("\n=== Part 3: models ===")
    ge = gradient_check()
    say(f"CNN numerical gradient check: worst relative error {ge:.2e}")
    if ge > 1e-4:
        say("STOP: gradient check failed")
        sys.exit(1)
    cached = OUT / "stage_b_lopo_predictions.csv"
    if "--reuse-predictions" in sys.argv and cached.is_file():
        pred = pd.read_csv(cached)
        say("(reusing saved LOPO predictions)")
    else:
        pred = lopo(data)
        pred.to_csv(cached, index=False)
    res = evaluate(pred)
    res.to_csv(OUT / "stage_b_metrics.csv", index=False)
    show = res[res.level == "planet"][["molecule", "model", "evaluable_folds", "n_pos", "n_neg", "accuracy", "precision",
                                       "recall", "f1", "roc_auc", "folds_train_single_class", "reliability"]]
    say(show.round(3).to_string(index=False))
    say("\nspectrum-level (secondary; spectra of one planet are not independent):")
    say(res[res.level == "spectrum"][["molecule", "model", "n", "n_pos", "n_neg", "accuracy", "f1", "roc_auc"]].round(3).to_string(index=False))

    say("\n=== Part 4: figure ===")
    picks = plot_samples(data, EDA / "stage_b_spectra_sample.png")
    say(f"saved eda/stage_b_spectra_sample.png: {picks}")
    say(f"\nruntime {time.time() - t0:.0f} s")
    (OUT / "stage_b_log.txt").write_text("\n".join(lines), encoding="utf-8")
    with open(OUT / "stage_b_summary.json", "w") as fh:
        json.dump({"folds": fold_rows, "evaluable_folds": per_mol, "grad_check": ge}, fh, indent=2)


if __name__ == "__main__":
    main()
