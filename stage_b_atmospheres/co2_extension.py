"""
co2_extension.py
================

The pre-registered CO2 test (stage_b_atmospheres/CO2_EXTENSION_PROTOCOL.md): simulation-trained CO2 detectors
(GBM and 1-D CNN on the contamination + haze grid) on every labelled JWST planet whose transmission spectrum
covers the 4.3 um band, against the gas-blind feature-strength control, plus a band-occlusion test of
gas specificity. Labels: outputs/labels/co2_labels.csv, frozen by hash before this script existed.

    python stage_b_atmospheres/co2_extension.py
Output: outputs/stage_b_co2/{co2_results.json, co2_scores.csv, co2_spectra.csv}, eda/co2_extension.png
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(HERE))
import methane_extension as me  # noqa: E402
import methane_tls_robustness as tr  # noqa: E402
import stage_b_pipeline as sbp  # noqa: E402
import train_synthetic_classifier as l3  # noqa: E402

OUT = RESEARCH / "outputs" / "stage_b_co2"
LABELS = RESEARCH / "outputs" / "labels" / "co2_labels.csv"
REFRESH = sorted((RESEARCH / "external_data").glob("archive_spectra_*"))[-1]
CO2 = sbp.MOLECULES.index("CO2")
BAND = (4.20, 4.45)
IN_BAND = (sbp.CENTERS >= BAND[0]) & (sbp.CENTERS <= BAND[1])


def frozen_labels() -> pd.DataFrame:
    rec = json.loads((OUT / "protocol_hashes.json").read_text())
    if hashlib.sha256(LABELS.read_bytes()).hexdigest() != rec["co2_labels.csv"]:
        raise SystemExit("CO2 label file changed after it was frozen - the pre-registered test is void")
    d = pd.read_csv(LABELS, dtype=str, keep_default_na=False)
    return d[d.CO2.isin(["0", "1"])].assign(label=lambda x: x.CO2.astype(int)).reset_index(drop=True)


def _prepare(df: pd.DataFrame):
    depth, unc, mask = sbp.resample(df)
    cov = mask > 0
    if not cov.any():
        return None, "no data on the 1-5 um grid"
    if mask[IN_BAND].sum() < 3:
        return None, "fewer than 3 grid bins in 4.20-4.45 um"
    depth = np.where(cov, depth - np.median(depth[cov]), 0.0)
    unc = np.where(cov, np.where(np.isfinite(unc), unc, np.nanmedian(unc[cov])), 0.0)
    return np.stack([depth, unc, mask]), ""


def spectra(planets: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    """Every JWST transmission spectrum of these planets, pack and refresh, that covers the CO2 band."""
    items = [(r.pl_name, sbp.SPEC_DIR / "spectra_tbl" / r.file, r.instrument) for r in sbp.select_spectra(planets).itertuples()]
    added = pd.read_csv(REFRESH / "added_since_pack.csv")
    added = added[(added.spec_type == "Transmission") & added.facility.astype(str).str.contains("James Webb") & added.pl_name.isin(planets)]
    files = {p.name: p for p in (REFRESH / "spectra_tbl").glob("*.tbl")}
    for r in added.itertuples():
        f = next((p for n, p in files.items() if n.endswith("_" + Path(r.spec_path).name)), None)
        if f is not None:
            items.append((r.pl_name, f, r.instrument))
    X, rows = [], []
    for planet, f, inst in items:
        try:
            df, _ = sbp.parse_tbl(f)
        except ValueError as e:
            rows.append({"planet": planet, "file": f.name, "instrument": inst, "scored": False, "reason": str(e)[:60]})
            continue
        x, why = _prepare(df)
        rows.append({"planet": planet, "file": f.name, "instrument": inst, "scored": x is not None, "reason": why})
        if x is not None:
            X.append(x)
    return np.array(X), pd.DataFrame(rows)


def occlude(X: np.ndarray) -> np.ndarray:
    Z = X.copy()
    Z[:, :, IN_BAND] = 0.0
    return Z


def main() -> int:
    rng = np.random.default_rng(me.SEED)
    lab = frozen_labels()
    X, meta = spectra(lab.planet.tolist())
    sc = meta[meta.scored].reset_index(drop=True)
    Xg, Yg, _, anchors, noise_from, _ = tr.load_grid(tr.GRID["v3"])

    def models(keep):
        return l3.fit_gbm(Xg[keep], Yg[keep, CO2]), l3.train_cnn(Xg[keep], Yg[keep])

    full = models(np.ones(len(Xg), bool))
    gbm, cnn, gbm_occ, cnn_occ = (np.zeros(len(sc)) for _ in range(4))
    Xo = occlude(X)
    fold = {}
    for p in sorted(set(sc.planet)):
        idx = np.flatnonzero(sc.planet.to_numpy() == p)
        if p in set(anchors) or p in set(noise_from):                    # strict leave-one-planet-out
            fold[p] = int(((anchors != p) & (noise_from != p)).sum())
            g, c = models((anchors != p) & (noise_from != p))
        else:
            g, c = full
        gbm[idx], cnn[idx] = l3.predict_gbm(g, X[idx]), c(X[idx])[:, CO2]
        gbm_occ[idx], cnn_occ[idx] = l3.predict_gbm(g, Xo[idx]), c(Xo[idx])[:, CO2]
    sc = sc.assign(gbm=gbm, cnn=cnn, gbm_occluded=gbm_occ, cnn_occluded=cnn_occ, strength=l3.feature_strength(X))
    sc.to_csv(OUT / "co2_spectra.csv", index=False)
    meta.to_csv(OUT / "co2_spectra_all.csv", index=False)
    pl = sc.groupby("planet")[["gbm", "cnn", "gbm_occluded", "cnn_occluded", "strength"]].mean().reset_index()
    pl = pl.merge(lab[["planet", "label", "source"]], on="planet")
    pl["drop_gbm"], pl["drop_cnn"] = pl.gbm - pl.gbm_occluded, pl.cnn - pl.cnn_occluded
    pl.to_csv(OUT / "co2_scores.csv", index=False)

    res = {"protocol": "stage_b_atmospheres/CO2_EXTENSION_PROTOCOL.md", "labels_frozen": True,
           "n_labelled": int(len(lab)), "n_scored": int(len(pl)), "unscored_labelled": sorted(set(lab.planet) - set(pl.planet)),
           "n_spectra": int(len(sc)), "lopo_training_sizes": fold, "all": me.evaluate(pl, rng)}
    thr = float(pl.strength.median())
    strong = pl[pl.strength > thr]
    n_pos, n_neg = int(strong.label.sum()), int((strong.label == 0).sum())
    res["strong_feature_half"] = {"strength_threshold": round(thr, 3), "n_pos": n_pos, "n_neg": n_neg,
                                  "testable": bool(n_pos >= 3 and n_neg >= 3),
                                  **(me.evaluate(strong, rng) if n_pos >= 1 and n_neg >= 1 else {})}
    occ = {}
    for m in ("gbm", "cnn"):
        a, b = pl[pl.label == 1][f"drop_{m}"], pl[pl.label == 0][f"drop_{m}"]
        occ[m] = {"median_drop_co2": round(float(a.median()), 3), "median_drop_no_co2": round(float(b.median()), 3),
                  "mannwhitney_p": float(mannwhitneyu(a, b, alternative="greater").pvalue),
                  "roc_auc_after_occlusion": me.evaluate(pl.assign(**{m: pl[f"{m}_occluded"]}), rng, cols=(m,))[m]}
    res["band_occlusion"] = occ
    a, s = res["all"], res["strong_feature_half"]
    res["expectations"] = {
        "K1 GBM ROC-AUC >= 0.85 and p < 0.01": a["gbm"]["roc_auc"] >= 0.85 and a["gbm"]["perm_p"] < 0.01,
        "K2 strong half: GBM >= 0.80 and >= control + 0.10": (bool(s["testable"]) and s["gbm"]["roc_auc"] >= 0.80
                                                               and s["gbm"]["roc_auc"] >= s["strength"]["roc_auc"] + 0.10),
        "K3 occlusion drops CO2 planets more (p < 0.05)": occ["gbm"]["mannwhitney_p"] < 0.05,
        "K4 CNN ROC-AUC >= 0.80": a["cnn"]["roc_auc"] >= 0.80}
    res["failure"] = {"K1 CI includes 0.5": a["gbm"]["ci95"][0] <= 0.5,
                      "GBM not above control on the strong half": (not s["testable"]) or s["gbm"]["roc_auc"] <= s["strength"]["roc_auc"],
                      "K3 failed": not res["expectations"]["K3 occlusion drops CO2 planets more (p < 0.05)"]}
    (OUT / "co2_results.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in ["n_scored", "unscored_labelled", "all", "strong_feature_half", "band_occlusion", "expectations", "failure"]},
                     indent=2, default=float))
    pd.set_option("display.width", 200)
    print(pl.sort_values("gbm", ascending=False)[["planet", "label", "gbm", "cnn", "strength", "drop_gbm"]].round(3).to_string(index=False))
    figure(pl, res)
    return 0


def figure(pl: pd.DataFrame, res: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    for c, col, lab in [("gbm", "#2a78d6", "GBM"), ("cnn", "#e8743b", "1-D CNN"), ("strength", "#999999", "gas-blind control")]:
        f, t, _ = roc_curve(pl.label, pl[c])
        ax.plot(f, t, color=col, label=f"{lab}: {res['all'][c]['roc_auc']:.2f}")
    ax.plot([0, 1], [0, 1], ":", color="#bbbbbb")
    ax.set(xlabel="false-positive rate", ylabel="true-positive rate",
           title=f"CO2 on {res['n_scored']} labelled JWST planets ({int(pl.label.sum())} with CO2)")
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1]
    for y, col in [(1, "#2a78d6"), (0, "#999999")]:
        d = pl[pl.label == y]
        ax.scatter(d.strength, d.gbm, color=col, s=28, label="CO2 detected" if y else "not detected")
    ax.axvline(res["strong_feature_half"]["strength_threshold"], color="k", ls=":", lw=1)
    ax.set(xscale="log", xlabel="feature strength (gas-blind)", ylabel="GBM P(CO2)", title="Score vs feature strength")
    ax.legend(fontsize=8)
    ax = axes[2]
    for y, col in [(1, "#2a78d6"), (0, "#999999")]:
        d = pl[pl.label == y]
        ax.scatter(d.gbm, d.gbm_occluded, color=col, s=28)
    ax.plot([0, 1], [0, 1], ":", color="#bbbbbb")
    o = res["band_occlusion"]["gbm"]
    ax.set(xlabel="GBM P(CO2)", ylabel="with 4.20-4.45 um masked",
           title=f"Band occlusion: median drop {o['median_drop_co2']:.2f} (CO2) vs {o['median_drop_no_co2']:.2f}")
    fig.tight_layout()
    fig.savefig(RESEARCH / "eda" / "co2_extension.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())
