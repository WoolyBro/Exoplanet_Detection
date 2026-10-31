"""
detector_injection.py
=====================

eda/POST_REVIEW_PROTOCOL.md, analysis I (post hoc): a known-truth test of the molecule detectors on real noise. For each planet labeled
without a gas, the difference between a model with and without that gas is added to its real JWST spectra, and the frozen detectors
score the result under the same leave-one-planet-out rule as the register.

    ~/micromamba/envs/prt/bin/python stage_b_atmospheres/detector_injection.py --templates    # model differences (petitRADTRANS)
    python stage_b_atmospheres/detector_injection.py --score                                  # frozen detectors (project venv)
Output: outputs/post_review/detector_injection_templates.npz, detector_injection.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
R = HERE.parent
OUT = R / "outputs" / "post_review"
LEVELS = [-5.0, -4.0, -3.0, -2.0]
LOGP_DECK = -1.0                                                  # opaque deck at 0.1 bar
BOUNDARY = {"CH4": 0.125, "CO2": 0.5}


def labeled() -> dict:
    L = R / "outputs" / "labels"
    parts = [pd.read_csv(L / f, dtype=str)[["planet", "CH4"]] for f in
             ["verified_gas_labels.csv", "ch4_labels_extended.csv", "ch4_labels_round2.csv", "ch4_labels_round3.csv"]]
    ch4 = pd.concat(parts).drop_duplicates("planet", keep="last")
    ch4 = ch4[ch4.CH4 == "0"].planet.tolist()
    co2 = pd.read_csv(L / "co2_labels.csv", dtype=str)
    return {"CH4": ch4, "CO2": co2[co2.CO2 == "0"].planet.tolist()}


def templates() -> int:
    import forward_model as fm
    import stage_b_pipeline as sbp
    ps = pd.read_csv(R / "exoplanet_research_data" / "01_candidate_catalogs" / "pscomppars_confirmed_planets.csv", low_memory=False).set_index("pl_name")
    out, skipped = {}, []
    for gas, planets in labeled().items():
        for p in planets:
            if p not in ps.index or not np.isfinite(ps.loc[p, ["pl_rade", "pl_bmasse", "st_rad", "pl_eqt"]].astype(float)).all():
                skipped.append((p, gas))
                continue
            r = ps.loc[p]
            bg = "H2He" if float(r.pl_rade) > 1.8 else "N2"
            args = dict(temperature=float(np.clip(r.pl_eqt, 150, 2500)), rp_earth=float(r.pl_rade), mp_earth=float(r.pl_bmasse), rs_sun=float(r.st_rad),
                        surface_pressure=10 ** LOGP_DECK, wl_range=(0.95, 5.1), species=("CH4", "CO2"), resolution=300)
            wl, base = fm.transmission_spectrum({gas: 1e-14}, bg, **args)
            b0 = fm.bin_spectrum(wl, base, sbp.EDGES)
            for lv in LEVELS:
                wl, d = fm.transmission_spectrum({gas: 10 ** lv}, bg, **args)
                out[f"{p}|{gas}|{lv:g}"] = np.nan_to_num(fm.bin_spectrum(wl, d, sbp.EDGES) - b0)
            print(p, gas, bg, round(float(np.max(out[f"{p}|{gas}|-3"])), 1), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "detector_injection_templates.npz", **out)
    print("templates:", len(out), "skipped:", skipped, flush=True)
    return 0


def score() -> int:
    from sklearn.metrics import roc_auc_score
    import molecule_register as mr

    npz = np.load(OUT / "detector_injection_templates.npz")
    keys = [k.split("|") for k in npz.files]
    planets = sorted({k[0] for k in keys})
    X, meta = mr.spectra(set(planets))
    Xg, Yg, _, anchors, noise_from, _ = mr.tr.load_grid(mr.tr.GRID["v3"])
    anc = set(anchors) | set(noise_from)
    idx_gas = {"CH4": mr.CH4, "CO2": mr.CO2}

    def models(k):
        return {"CH4": mr.l3.fit_gbm(Xg[k], Yg[k, mr.CH4]), "CO2": mr.l3.fit_gbm(Xg[k], Yg[k, mr.CO2])}, mr.l3.train_cnn(Xg[k], Yg[k])

    full = models(np.ones(len(Xg), bool))
    rows = []
    for p in planets:
        ok = np.flatnonzero(meta.planet.to_numpy() == p)
        if not len(ok):
            continue
        g, c = models((anchors != p) & (noise_from != p)) if p in anc else full
        for gas in ("CH4", "CO2"):
            flag = (meta.ch4_ok if gas == "CH4" else meta.co2_ok).astype(bool).to_numpy()[ok]
            sp = ok[flag]
            if not len(sp) or f"{p}|{gas}|-3" not in npz.files:
                continue
            base_g, base_c = float(np.mean(mr.l3.predict_gbm(g[gas], X[sp]))), float(np.mean(c(X[sp])[:, idx_gas[gas]]))
            for lv in LEVELS:
                Z = X[sp].copy()
                d = npz[f"{p}|{gas}|{lv:g}"]
                cov = Z[:, 2] > 0
                Z[:, 0] = np.where(cov, Z[:, 0] + d, 0.0)
                for i in range(len(Z)):
                    Z[i, 0] = np.where(cov[i], Z[i, 0] - np.median(Z[i, 0][cov[i]]), 0.0)
                rows.append({"planet": p, "gas": gas, "logx": lv, "base_gbm": base_g, "base_cnn": base_c, "inj_gbm": float(np.mean(mr.l3.predict_gbm(g[gas], Z))),
                             "inj_cnn": float(np.mean(c(Z)[:, idx_gas[gas]])), "peak_ppm": float(np.max(d)), "median_err_ppm": float(np.median(X[sp][:, 1][X[sp][:, 2] > 0]))})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "detector_injection_scores.csv", index=False)
    res = {"protocol": "eda/POST_REVIEW_PROTOCOL.md (I), post hoc", "boundary": BOUNDARY, "gases": {}}
    for gas, d in t.groupby("gas"):
        res["gases"][gas] = {"n_planets": int(d.planet.nunique())}
        for lv, e in d.groupby("logx"):
            r = {"median_peak_ppm": round(float(e.peak_ppm.median()), 1)}
            for m in ("gbm", "cnn"):
                y = np.r_[np.ones(len(e)), np.zeros(len(e))]
                r[m] = {"roc_auc_vs_uninjected": round(float(roc_auc_score(y, np.r_[e[f"inj_{m}"], e[f"base_{m}"]])), 3),
                        "recovered_above_boundary": round(float((e[f"inj_{m}"] > BOUNDARY[gas]).mean()), 3),
                        "false_rate_uninjected": round(float((e[f"base_{m}"] > BOUNDARY[gas]).mean()), 3)}
            b = e[e.peak_ppm > 3 * e.median_err_ppm]
            r["n_signal_gt_3x_error"] = int(len(b))
            for m in ("gbm", "cnn"):
                r[m]["recovered_when_signal_gt_3x_error"] = round(float((b[f"inj_{m}"] > BOUNDARY[gas]).mean()), 3) if len(b) else None
            res["gases"][gas][f"logx_{lv:g}"] = r
    (OUT / "detector_injection.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(templates() if "--templates" in sys.argv else score())
