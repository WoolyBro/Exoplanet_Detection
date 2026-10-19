"""
molecule_register.py
====================

The prospective molecule register (stage_b_atmospheres/MOLECULE_REGISTER_PROTOCOL.md, commit d7f1f6b): the
frozen, real-spectrum-validated CH4 and CO2 detectors on every archive JWST transmission planet whose CH4 or
CO2 status is unresolved, calibrated on the labelled planets, with a two-model call rule.

All spectra (labelled and register planets alike) go through one builder, so the calibration and the register
use identical preprocessing. Grid-anchor planets are scored by strict leave-one-planet-out models.

    python stage_b_atmospheres/molecule_register.py
Output: outputs/stage_b_register/{molecule_register.csv, register_spectra.csv, register_summary.json}
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
import methane_round4 as m4  # noqa: E402
import methane_tls_robustness as tr  # noqa: E402
import stage_b_pipeline as sbp  # noqa: E402
import train_synthetic_classifier as l3  # noqa: E402

OUT = RESEARCH / "outputs" / "stage_b_register"
LAB = RESEARCH / "outputs" / "labels"
REFRESH = RESEARCH / "external_data" / "archive_spectra_2026-10-18"          # pinned: the snapshot every test used
CH4, CO2 = sbp.MOLECULES.index("CH4"), sbp.MOLECULES.index("CO2")
CO2_BAND = (sbp.CENTERS >= 4.20) & (sbp.CENTERS <= 4.45)


def check_hashes() -> None:
    h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    me = json.loads((RESEARCH / "outputs" / "stage_b_methane_ext" / "protocol_hashes.json").read_text())
    co = json.loads((RESEARCH / "outputs" / "stage_b_co2" / "protocol_hashes.json").read_text())
    for f in ["ch4_labels_extended.csv", "ch4_labels_round2.csv", "ch4_labels_round3.csv"]:
        assert h(LAB / f) == me[f], f"{f} changed after freezing"
    assert h(LAB / "co2_labels.csv") == co["co2_labels.csv"], "co2_labels.csv changed after freezing"


def labels() -> tuple[pd.DataFrame, pd.DataFrame]:
    v = pd.read_csv(LAB / "verified_gas_labels.csv", dtype=str)[["planet", "CH4"]]
    parts = [v] + [pd.read_csv(LAB / f, dtype=str)[["planet", "CH4"]] for f in
                   ["ch4_labels_extended.csv", "ch4_labels_round2.csv", "ch4_labels_round3.csv"]]
    ch4 = pd.concat(parts).drop_duplicates("planet", keep="last")
    ch4 = pd.concat([ch4, pd.DataFrame({"planet": ["TOI-1231 b"], "CH4": ["1"]})], ignore_index=True)   # round 4
    co2 = pd.read_csv(LAB / "co2_labels.csv", dtype=str)[["planet", "CO2"]]
    return ch4, co2


def prepare(df: pd.DataFrame):
    depth, unc, mask = sbp.resample(df)
    cov = mask > 0
    if not cov.any():
        return None
    depth = np.where(cov, depth - np.median(depth[cov]), 0.0)
    unc = np.where(cov, np.where(np.isfinite(unc), unc, np.nanmedian(unc[cov])), 0.0)
    return np.stack([depth, unc, mask])


def spectra(planets: set[str]) -> tuple[np.ndarray, pd.DataFrame]:
    items = [(r.pl_name, "pack:" + r.file, lambda f=r.file: sbp.parse_tbl(sbp.SPEC_DIR / "spectra_tbl" / f)[0])
             for r in sbp.select_spectra(sorted(planets)).itertuples()]
    added = pd.read_csv(REFRESH / "added_since_pack.csv")
    added = added[(added.spec_type == "Transmission") & added.facility.astype(str).str.contains("James Webb") & added.pl_name.isin(planets)]
    files = {p.name: p for p in (REFRESH / "spectra_tbl").glob("*.tbl")}
    for r in added.itertuples():
        f = next((p for n, p in files.items() if n.endswith("_" + Path(r.spec_path).name)), None)
        if f is not None:
            items.append((r.pl_name, "refresh:" + f.name, lambda f=f: sbp.parse_tbl(f)[0]))
    if "TOI-1231 b" in planets:
        for red in ["JexoRes", "JexoPipe"]:
            for mode, parts in [("SOSS", ["niriss_ord1", "niriss_ord2"]), ("G395H", ["nirspec_g395h_nrs1", "nirspec_g395h_nrs2"])]:
                items.append(("TOI-1231 b", f"osf:{mode}_{red}", lambda p=parts, r=red: m4.read([f"TOI-1231b_{q}_{r}.txt" for q in p])))
    X, rows = [], []
    for planet, src, load in items:
        try:
            x = prepare(load())
        except ValueError:
            x = None
        if x is None:
            continue
        X.append(x)
        rows.append({"planet": planet, "source": src, "ch4_ok": bool(sbp.band_observable(x[2], "CH4")),
                     "co2_ok": bool(x[2][CO2_BAND].sum() >= 3)})
    return np.array(X), pd.DataFrame(rows)


def platt(s_lab, y_lab):
    lg = lambda s: np.log(np.clip(s, 1e-4, 1 - 1e-4) / (1 - np.clip(s, 1e-4, 1 - 1e-4)))
    m = LogisticRegression(C=1.0).fit(lg(s_lab)[:, None], y_lab)
    return lambda s: m.predict_proba(lg(np.asarray(s, float))[:, None])[:, 1]


def loo_platt_logscore(s, y) -> dict:
    p, base = np.zeros(len(y)), np.zeros(len(y))
    for i in range(len(y)):
        k = np.arange(len(y)) != i
        p[i], base[i] = platt(s[k], y[k])(s[[i]])[0], y[k].mean()
    ls = lambda q: float(np.mean(y * np.log(np.clip(q, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - q, 1e-6, 1))))
    return {"loo_calibrated": round(ls(p), 4), "loo_base_rate": round(ls(base), 4)}


def main() -> int:
    check_hashes()
    OUT.mkdir(parents=True, exist_ok=True)
    ch4, co2 = labels()
    ch4_lab = ch4[ch4.CH4.isin(["0", "1"])]
    co2_lab = co2[co2.CO2.isin(["0", "1"])]
    ch4_reg = sorted(ch4[ch4.CH4 == "unknown"].planet)
    co2_reg = sorted(co2[co2.CO2 == "unknown"].planet)
    planets = set(ch4_lab.planet) | set(co2_lab.planet) | set(ch4_reg) | set(co2_reg)
    X, meta = spectra(planets)
    Xg, Yg, _, anchors, noise_from, _ = tr.load_grid(tr.GRID["v3"])
    anc = set(anchors) | set(noise_from)

    def models(keep):
        return (l3.fit_gbm(Xg[keep], Yg[keep, CH4]), l3.fit_gbm(Xg[keep], Yg[keep, CO2]), l3.train_cnn(Xg[keep], Yg[keep]))

    score = {k: np.full(len(meta), np.nan) for k in ["ch4_gbm", "ch4_cnn", "co2_gbm", "co2_cnn"]}
    full = models(np.ones(len(Xg), bool))
    folds = {}
    for p in sorted(set(meta.planet)):
        idx = np.flatnonzero(meta.planet.to_numpy() == p)
        if p in anc:
            keep = (anchors != p) & (noise_from != p)
            folds[p] = int(keep.sum())
            g4, g2, c = models(keep)
        else:
            g4, g2, c = full
        cp = c(X[idx])
        score["ch4_gbm"][idx], score["co2_gbm"][idx] = l3.predict_gbm(g4, X[idx]), l3.predict_gbm(g2, X[idx])
        score["ch4_cnn"][idx], score["co2_cnn"][idx] = cp[:, CH4], cp[:, CO2]
    meta = meta.assign(**score)
    meta.to_csv(OUT / "register_spectra.csv", index=False)

    def planet_table(gas, lab_or_reg):
        ok = meta[meta[f"{gas}_ok"]]
        t = ok.groupby("planet")[[f"{gas}_gbm", f"{gas}_cnn"]].mean()
        t["n_spectra"] = ok.groupby("planet").size()
        return t[t.index.isin(lab_or_reg)]

    summary = {"protocol": "stage_b_atmospheres/MOLECULE_REGISTER_PROTOCOL.md (commit d7f1f6b)", "lopo_folds": folds}
    regs = []
    for gas, lab, col, reg in [("ch4", ch4_lab, "CH4", ch4_reg), ("co2", co2_lab, "CO2", co2_reg)]:
        L = planet_table(gas, set(lab.planet)).join(lab.set_index("planet")[col].astype(int).rename("label"), how="inner")
        R = planet_table(gas, set(reg))
        cal = {m: platt(L[f"{gas}_{m}"].to_numpy(), L.label.to_numpy()) for m in ["gbm", "cnn"]}
        summary[gas] = {"n_labelled_scored": int(len(L)), "n_pos": int(L.label.sum()),
                        "labelled_roc_auc": {m: round(float(roc_auc_score(L.label, L[f"{gas}_{m}"])), 3) for m in ["gbm", "cnn"]},
                        "calibration_check_P1": {m: loo_platt_logscore(L[f"{gas}_{m}"].to_numpy(), L.label.to_numpy()) for m in ["gbm", "cnn"]},
                        "register_planets": len(reg), "register_scored": int(len(R)),
                        "register_unscored": sorted(set(reg) - set(R.index))}
        ok = meta[meta[f"{gas}_ok"] & meta.planet.isin(reg)]
        spec_p = cal["gbm"](ok[f"{gas}_gbm"].to_numpy())
        spread = pd.Series(spec_p, index=ok.planet.values).groupby(level=0).agg(["min", "max"])
        for p, r in R.iterrows():
            pg, pc = float(cal["gbm"]([r[f"{gas}_gbm"]])[0]), float(cal["cnn"]([r[f"{gas}_cnn"]])[0])
            disagree = bool(spread.loc[p, "min"] < 0.5 <= spread.loc[p, "max"]) if r.n_spectra > 1 else False
            call = ("present" if pg >= 0.8 and pc >= 0.8 else "absent" if pg <= 0.2 and pc <= 0.2 else "ambiguous")
            if disagree:
                call = "ambiguous"
            regs.append({"planet": p, "gas": col, "n_spectra": int(r.n_spectra), "score_gbm": round(float(r[f"{gas}_gbm"]), 3),
                         "score_cnn": round(float(r[f"{gas}_cnn"]), 3), "p_gbm": round(pg, 3), "p_cnn": round(pc, 3),
                         "p_gbm_spectra_min": round(float(spread.loc[p, "min"]), 3), "p_gbm_spectra_max": round(float(spread.loc[p, "max"]), 3),
                         "spectra_disagree": disagree, "call": call})
    reg = pd.DataFrame(regs).sort_values(["gas", "planet"]).reset_index(drop=True)
    reg.to_csv(OUT / "molecule_register.csv", index=False)
    h = hashlib.sha256((OUT / "molecule_register.csv").read_bytes()).hexdigest()
    n_planets = reg.planet.nunique()
    definite = reg[reg.call != "ambiguous"].planet.nunique()
    summary.update({"register_sha256": h, "n_register_planets_scored": int(n_planets), "n_with_a_definite_call": int(definite),
                    "calls": reg.groupby(["gas", "call"]).size().unstack(fill_value=0).to_dict(orient="index"),
                    "expectations": {
                        "P1 LOO calibration beats base rate, both gases (GBM)": all(
                            summary[g]["calibration_check_P1"]["gbm"]["loo_calibrated"] > summary[g]["calibration_check_P1"]["gbm"]["loo_base_rate"]
                            for g in ["ch4", "co2"]),
                        "P2 >= half the scored register planets get a definite call": bool(definite >= 0.5 * n_planets)}})
    (OUT / "register_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    pd.set_option("display.width", 200)
    print(reg.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
