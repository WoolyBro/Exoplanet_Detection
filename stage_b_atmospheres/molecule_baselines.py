"""
molecule_baselines.py
=====================

eda/POST_REVIEW_PROTOCOL.md, analysis J (post hoc): do simple planet properties rank the labeled planets as well as the spectra do?

    python stage_b_atmospheres/molecule_baselines.py
Output: outputs/post_review/molecule_baselines.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

R = Path(__file__).resolve().parents[1]
OUT = R / "outputs" / "post_review"


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    k = np.isfinite(s)
    return round(float(roc_auc_score(y[k], s[k])), 3) if len(set(y[k])) == 2 else None


def run(t: pd.DataFrame, lab: str, cols: tuple) -> dict:
    ps = pd.read_csv(R / "exoplanet_research_data" / "01_candidate_catalogs" / "pscomppars_confirmed_planets.csv", low_memory=False).set_index("pl_name")
    t = t.join(ps[["pl_eqt", "pl_rade", "pl_bmasse"]])
    y = t[lab].astype(int)
    r = {"n": int(len(t)), "n_pos": int(y.sum()), **{c: auc(y, t[c]) for c in cols}, "cooler_alone": auc(y, -t.pl_eqt), "hotter_alone": auc(y, t.pl_eqt),
         "larger_alone": auc(y, t.pl_rade), "smaller_alone": auc(y, -t.pl_rade), "bands": {}}
    for lo, hi in [(0, 600), (600, 1000), (1000, 5000)]:
        e = t[(t.pl_eqt >= lo) & (t.pl_eqt < hi)]
        r["bands"][f"{lo}-{hi} K"] = {"n": int(len(e)), "n_pos": int(e[lab].astype(int).sum()), **{c: auc(e[lab].astype(int), e[c]) for c in cols}}
    return r


def main() -> int:
    ch4 = pd.read_csv(OUT / "ch4_band_mask_scores.csv", index_col=0)
    co2 = pd.read_csv(R / "outputs" / "stage_b_co2" / "co2_scores.csv").dropna(subset=["label", "gbm", "cnn"]).set_index("planet")
    res = {"protocol": "eda/POST_REVIEW_PROTOCOL.md (J), post hoc", "CH4": run(ch4, "CH4", ("gbm", "cnn")), "CO2": run(co2, "label", ("gbm", "cnn"))}
    (OUT / "molecule_baselines.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
