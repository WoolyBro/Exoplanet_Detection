"""
ch4_band_mask.py
================

eda/POST_REVIEW_PROTOCOL.md, analysis D (post hoc): the CH4 counterpart of the CO2 band occlusion, on the 41 labeled planets, with
the same strict leave-one-planet-out rule as the register.

    python stage_b_atmospheres/ch4_band_mask.py
Output: outputs/post_review/ch4_band_mask.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import molecule_register as mr  # noqa: E402

R = HERE.parent
OUT = R / "outputs" / "post_review"
BANDS = [(2.15, 2.45), (3.05, 3.65)]


def main() -> int:
    L = R / "outputs" / "labels"
    parts = [pd.read_csv(L / f, dtype=str)[["planet", "CH4"]] for f in
             ["verified_gas_labels.csv", "ch4_labels_extended.csv", "ch4_labels_round2.csv", "ch4_labels_round3.csv"]]
    lab = pd.concat(parts).drop_duplicates("planet", keep="last")
    lab = pd.concat([lab, pd.DataFrame({"planet": ["TOI-1231 b"], "CH4": ["1"]})], ignore_index=True)
    lab = lab[lab.CH4.isin(["0", "1"])].set_index("planet").CH4.astype(int)
    X, meta = mr.spectra(set(lab.index))
    keep_sp = meta.ch4_ok.astype(bool).to_numpy()
    X, meta = X[keep_sp], meta[keep_sp].reset_index(drop=True)
    cen = mr.sbp.CENTERS
    inband = np.any([(cen >= a) & (cen <= b) for a, b in BANDS], axis=0)
    Z = X.copy()
    Z[:, :, inband] = 0.0
    Xg, Yg, _, anchors, noise_from, _ = mr.tr.load_grid(mr.tr.GRID["v3"])
    anc = set(anchors) | set(noise_from)

    def models(k):
        return mr.l3.fit_gbm(Xg[k], Yg[k, mr.CH4]), mr.l3.train_cnn(Xg[k], Yg[k])

    full = models(np.ones(len(Xg), bool))
    s = {k: np.zeros(len(meta)) for k in ("gbm", "cnn", "gbm_masked", "cnn_masked")}
    for p in sorted(set(meta.planet)):
        idx = np.flatnonzero(meta.planet.to_numpy() == p)
        g, c = models((anchors != p) & (noise_from != p)) if p in anc else full
        s["gbm"][idx], s["gbm_masked"][idx] = mr.l3.predict_gbm(g, X[idx]), mr.l3.predict_gbm(g, Z[idx])
        s["cnn"][idx], s["cnn_masked"][idx] = c(X[idx])[:, mr.CH4], c(Z[idx])[:, mr.CH4]
    t = meta.assign(**s).groupby("planet")[list(s)].mean().join(lab)
    assert len(t) == 41 and t.CH4.sum() == 11, (len(t), t.CH4.sum())
    res = {"protocol": "eda/POST_REVIEW_PROTOCOL.md (D), post hoc", "bands_um": BANDS, "n": int(len(t)), "n_ch4": int(t.CH4.sum())}
    for m in ("gbm", "cnn"):
        drop = t[m] - t[f"{m}_masked"]
        a, b = drop[t.CH4 == 1], drop[t.CH4 == 0]
        res[m] = {"roc_auc": round(float(roc_auc_score(t.CH4, t[m])), 4), "roc_auc_masked": round(float(roc_auc_score(t.CH4, t[f"{m}_masked"])), 4),
                  "median_drop_ch4": round(float(a.median()), 3), "median_drop_no_ch4": round(float(b.median()), 3),
                  "mannwhitney_p": float(mannwhitneyu(a, b, alternative="greater").pvalue)}
    res["specific"] = bool(res["gbm"]["mannwhitney_p"] < 0.05)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ch4_band_mask.json").write_text(json.dumps(res, indent=2))
    t.to_csv(OUT / "ch4_band_mask_scores.csv")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
