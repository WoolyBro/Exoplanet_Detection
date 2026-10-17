"""
methane_extension.py
====================

The pre-registered test of METHANE_EXTENSION_PROTOCOL.md. Does the simulated-spectra CH4 classifier
(Stage B Level 3) work on JWST planets it has never been near? And can its probabilities be calibrated?

Run (project .venv; needs the research pack's spectra):
    python stage_b_atmospheres/methane_extension.py

Steps
  1. Check the frozen label file against its recorded SHA-256, and refuse to run if it changed.
  2. Build the new planets' spectra exactly as Stage B does (stage_b_pipeline.parse_tbl / resample,
     median-subtracted depth, uncertainty, mask). A spectrum enters the CH4 test only if a CH4 band
     is observable on it (stage_b_pipeline.band_observable).
  3. Train the Level 3 GBM and CNN once, on all 3,000 synthetic spectra, with unchanged hyperparameters,
     and score the new planets. Planet score = mean over that planet's CH4-observable spectra.
  4. Metrics:
     - primary: GBM ROC-AUC on the new planets, with a permutation p-value and a planet-bootstrap CI;
     - secondary: CNN, all planets (old from strict LOPO plus new), and new planets excluding
       featureless spectra;
     - control: the gas-blind feature-strength baseline.
  5. Calibration: nested leave-one-planet-out Platt scaling on logit(score), fitted on all OTHER
     labelled planets. Brier and ECE are reported before and after, against the majority baseline.

Output: outputs/stage_b_methane_ext/{new_planet_scores.csv, all_planet_scores.csv, metrics.json}
        eda/methane_extension.png
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(HERE))
import stage_b_pipeline as sbp  # noqa: E402
import train_synthetic_classifier as l3  # noqa: E402

OUT = RESEARCH / "outputs" / "stage_b_methane_ext"
LABELS = RESEARCH / "outputs" / "labels" / "ch4_labels_extended.csv"
CH4 = sbp.MOLECULES.index("CH4")
N_BOOT, SEED = 2000, 20261017


def check_frozen_labels() -> pd.DataFrame:
    rec = json.loads((OUT / "protocol_hashes.json").read_text())
    h = hashlib.sha256(LABELS.read_bytes()).hexdigest()
    if h != rec["ch4_labels_extended.csv"]:
        raise SystemExit("label file changed after it was frozen - the pre-registered test is void")
    return pd.read_csv(LABELS, dtype=str, keep_default_na=False)


def build_new_spectra(planets: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    sel = sbp.select_spectra(planets)
    X, rows = [], []
    for r in sel.itertuples():
        try:
            df, info = sbp.parse_tbl(sbp.SPEC_DIR / "spectra_tbl" / r.file)
        except ValueError as e:                        # same rule as Stage B: unreadable spectra are skipped
            rows.append({"planet": r.pl_name, "file": r.file, "used": False, "reason": str(e)[:80]})
            continue
        depth, unc, mask = sbp.resample(df)
        cov = mask > 0
        if not cov.any():
            rows.append({"planet": r.pl_name, "file": r.file, "used": False, "reason": "no data in 1-5 um"})
            continue
        depth = np.where(cov, depth - np.median(depth[cov]), 0.0)
        bad = cov & ~np.isfinite(unc)
        if bad.any():
            unc[bad] = np.nanmedian(unc[cov & np.isfinite(unc)])
        unc = np.where(cov, unc, 0.0)
        x = np.stack([depth, unc, mask])
        ok = sbp.band_observable(mask, "CH4")
        rows.append({"planet": r.pl_name, "file": r.file, "used": bool(ok), "instrument": r.instrument,
                     "reason": "" if ok else "no CH4 band on the 1-5 um grid"})
        if ok:
            X.append(x)
    meta = pd.DataFrame(rows)
    return np.array(X), meta[meta.used].reset_index(drop=True), meta


def bootstrap_auc(y, p, rng) -> list[float]:
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            vals.append(roc_auc_score(y[i], p[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def evaluate(d: pd.DataFrame, rng, cols=("gbm", "cnn", "strength")) -> dict:
    y = d.label.to_numpy().astype(int)
    out = {"n": int(len(y)), "n_pos": int(y.sum())}
    for c in cols:
        if len(np.unique(y)) < 2:
            out[c] = None
            continue
        out[c] = {"roc_auc": round(float(roc_auc_score(y, d[c])), 3), "perm_p": round(l3.auc_permutation_p(y, d[c].to_numpy()), 4),
                  "ci95": [round(v, 3) for v in bootstrap_auc(y, d[c].to_numpy(), rng)]}
    return out


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def ece(y, p, bins=5) -> float:
    """Expected calibration error with equal-count bins."""
    order = np.argsort(p)
    err = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            err += len(chunk) / len(p) * abs(np.mean(p[chunk]) - np.mean(y[chunk]))
    return float(err)


def nested_platt(d: pd.DataFrame, col: str) -> np.ndarray:
    y, s = d.label.to_numpy().astype(int), logit(d[col])
    cal = np.zeros(len(d))
    for i in range(len(d)):
        tr = np.arange(len(d)) != i
        m = LogisticRegression(C=1.0).fit(s[tr, None], y[tr])
        cal[i] = m.predict_proba(s[[i], None])[0, 1]
    return cal


def main() -> int:
    rng = np.random.default_rng(SEED)
    lab = check_frozen_labels()
    known = lab[lab.CH4.isin(["0", "1"])]
    Xn, meta, all_meta = build_new_spectra(known.planet.tolist())
    all_meta.to_csv(OUT / "new_spectra_used.csv", index=False)

    # ---- frozen model: Level 3 hyperparameters, trained once on every synthetic spectrum ----
    syn = np.load(RESEARCH / "outputs" / "stage_b_synthetic" / "synthetic_grid.npz", allow_pickle=True)
    Xs, Ys = syn["X"].astype(np.float64), syn["Y"].astype(int)
    assert list(syn["molecules"].astype(str)) == sbp.MOLECULES
    gbm = l3.fit_gbm(Xs, Ys[:, CH4])
    cnn = l3.train_cnn(Xs, Ys)
    spec = meta.assign(gbm=l3.predict_gbm(gbm, Xn), cnn=cnn(Xn)[:, CH4], strength=l3.feature_strength(Xn))
    new = spec.groupby("planet").agg(gbm=("gbm", "mean"), cnn=("cnn", "mean"), strength=("strength", "mean"),
                                     n_spectra=("file", "size")).reset_index()
    new = new.merge(known.rename(columns={"CH4": "label"})[["planet", "label", "featureless", "source"]], on="planet")
    new["label"] = new.label.astype(int)
    new["featureless"] = new.featureless.astype(int)
    new["set"] = "new"
    new.to_csv(OUT / "new_planet_scores.csv", index=False)

    # ---- the original 13 planets: their strict leave-one-planet-out Level 3 scores ----
    old = pd.read_csv(RESEARCH / "outputs" / "stage_b_synthetic" / "level3_predictions.csv")
    old = old[(old.molecule == "CH4") & old.label.notna()].groupby("planet").agg(
        label=("label", "first"), gbm=("gbm_prob", "mean"), cnn=("cnn_prob", "mean"), strength=("strength", "mean"),
        n_spectra=("spectrum", "size")).reset_index()
    old["label"] = old.label.astype(int)
    old["featureless"] = 0
    old["set"] = "original"
    both = pd.concat([old, new], ignore_index=True)

    metrics = {
        "protocol": "stage_b_atmospheres/METHANE_EXTENSION_PROTOCOL.md",
        "labels": {"planets_labelled": int(len(lab)), "known": int(len(known)),
                   "new_scorable": int(len(new)), "new_unscorable_no_ch4_band": sorted(set(known.planet) - set(new.planet))},
        "primary_new_planets": evaluate(new, rng),
        "secondary_new_without_featureless": evaluate(new[new.featureless == 0], rng),
        "secondary_all_planets": evaluate(both, rng),
    }

    # ---- POST-HOC (not pre-registered): planets excluded only because their spectra are binned so
    # coarsely that fewer than 3 grid bins fall in a CH4 band. Scored with the same frozen model. ----
    excluded = sorted(set(known.planet) - set(new.planet))
    if excluded:
        sel = sbp.select_spectra(excluded)
        rows = []
        for r in sel.itertuples():
            df, _ = sbp.parse_tbl(sbp.SPEC_DIR / "spectra_tbl" / r.file)
            depth, unc, mask = sbp.resample(df)
            cov = mask > 0
            n_band = int(sum(mask[(sbp.CENTERS >= lo) & (sbp.CENTERS <= hi)].sum() for lo, hi in sbp.BANDS["CH4"]))
            if not cov.any() or n_band == 0:
                continue
            depth = np.where(cov, depth - np.median(depth[cov]), 0.0)
            unc = np.where(cov, np.where(np.isfinite(unc), unc, np.nanmedian(unc[cov])), 0.0)
            x = np.stack([depth, unc, mask])[None]
            rows.append({"planet": r.pl_name, "ch4_band_bins": n_band, "gbm": float(l3.predict_gbm(gbm, x)[0]),
                         "cnn": float(cnn(x)[0, CH4])})
        ph = pd.DataFrame(rows)
        if not ph.empty:
            ph = ph.groupby("planet").agg(ch4_band_bins=("ch4_band_bins", "max"), gbm=("gbm", "mean"), cnn=("cnn", "mean")).reset_index()
            ph = ph.merge(known.rename(columns={"CH4": "label"})[["planet", "label"]], on="planet")
            metrics["post_hoc_coarse_spectra"] = {"note": "NOT pre-registered; < 3 grid bins in any CH4 band",
                                                  "planets": ph.round(3).to_dict(orient="records")}

    # ---- calibration: nested leave-one-planet-out Platt over all labelled planets ----
    y = both.label.to_numpy().astype(int)
    cal = {}
    for col in ("gbm", "cnn"):
        both[f"{col}_cal"] = nested_platt(both, col)
        base = np.array([np.mean(np.delete(y, i)) for i in range(len(y))])     # majority-rate baseline, also nested
        cal[col] = {"brier_raw": round(float(np.mean((both[col] - y) ** 2)), 3),
                    "brier_calibrated": round(float(np.mean((both[f"{col}_cal"] - y) ** 2)), 3),
                    "brier_base_rate": round(float(np.mean((base - y) ** 2)), 3),
                    "ece_raw": round(ece(y, both[col].to_numpy()), 3),
                    "ece_calibrated": round(ece(y, both[f"{col}_cal"].to_numpy()), 3),
                    "accuracy_0.5_raw": round(float(((both[col] >= 0.5) == y).mean()), 3),
                    "accuracy_0.5_calibrated": round(float(((both[f"{col}_cal"] >= 0.5) == y).mean()), 3),
                    "roc_auc_calibrated": round(float(roc_auc_score(y, both[f"{col}_cal"])), 3)}
    metrics["calibration_all_planets_nested_lopo"] = cal
    both.to_csv(OUT / "all_planet_scores.csv", index=False)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    pd.set_option("display.width", 200)
    print(both.sort_values("gbm", ascending=False)[["planet", "set", "label", "featureless", "gbm", "gbm_cal", "cnn", "strength"]]
          .round(3).to_string(index=False))
    make_figure(both)
    return 0


def make_figure(both: pd.DataFrame):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    d = both.sort_values("gbm")
    from matplotlib.colors import to_rgba

    colours = [to_rgba("#d1495b" if l == 1 else "#2a6fdb", 1.0 if s == "new" else 0.45) for l, s in zip(d.label, d.set)]
    ax.barh(np.arange(len(d)), d.gbm, color=colours)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels([f"{p}{' *' if s == 'new' else ''}{' (flat)' if f else ''}" for p, s, f in zip(d.planet, d.set, d.featureless)],
                       fontsize=6.5)
    ax.set(xlabel="P(CH4), GBM trained on simulations only", title="Every labelled planet (* = new, never used before)")
    ax.text(0.98, 0.02, "red: CH4 detected\nblue: not detected", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    ax = axes[1]
    for name, sub, ls in [("new planets", both[both.set == "new"], "-"), ("all planets", both, "--")]:
        for col, c in [("gbm", "#d1495b"), ("strength", "0.5")]:
            if sub.label.nunique() == 2:
                fpr, tpr, _ = roc_curve(sub.label, sub[col])
                auc = roc_auc_score(sub.label, sub[col])
                ax.plot(fpr, tpr, ls, color=c, label=f"{'GBM' if col == 'gbm' else 'gas-blind control'}, {name} ({auc:.2f})")
    ax.plot([0, 1], [0, 1], ":", color="0.7")
    ax.set(xlabel="false-positive rate", ylabel="true-positive rate", title="CH4 identification, planet level")
    ax.legend(fontsize=7, loc="lower right")
    ax = axes[2]
    y = both.label.to_numpy()
    for col, c, lab in [("gbm", "0.6", "raw"), ("gbm_cal", "#d1495b", "calibrated (nested LOPO Platt)")]:
        p = both[col].to_numpy()
        order = np.argsort(p)
        xs, ys = [], []
        for chunk in np.array_split(order, 5):
            xs.append(p[chunk].mean())
            ys.append(y[chunk].mean())
        ax.plot(xs, ys, "o-", color=c, label=lab)
    ax.plot([0, 1], [0, 1], ":", color="0.7")
    ax.set(xlabel="predicted P(CH4)", ylabel="fraction with CH4 detected", title="Reliability (5 equal-count bins)",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESEARCH / "eda" / "methane_extension.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())
