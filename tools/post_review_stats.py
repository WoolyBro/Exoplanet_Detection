"""
post_review_stats.py
====================

eda/POST_REVIEW_PROTOCOL.md, analysis H (post hoc): Holm over every significance p-value the paper prints, Newcombe score intervals
for the ROC-AUC values, detector minus gas-blind control on the strong-feature half, and CO2 labels against exoplanet.eu.

    python tools/post_review_stats.py
Output: outputs/post_review/stats.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.metrics import cohen_kappa_score, roc_auc_score

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "tools"))
from label_crosscheck import EU_DIR, key  # noqa: E402

OUT = R / "outputs" / "post_review"


def holm(p: dict) -> dict:
    names = sorted(p, key=p.get)
    m, adj, top = len(names), {}, 0.0
    for i, n in enumerate(names):
        top = max(top, min(1.0, (m - i) * p[n]))
        adj[n] = top
    return adj


def newcombe(a: float, m: int, n: int, z: float = 1.96) -> list:
    """Newcombe (2006), Method 5: a score interval for the Mann-Whitney ROC-AUC that stays usable at a perfect ranking."""
    ns = (m + n) / 2 - 1

    def var(t):
        return t * (1 - t) / (m * n) * (1 + ns * (1 - t) / (2 - t) + ns * t / (1 + t))

    f = lambda t: (a - t) ** 2 - z ** 2 * var(t)
    lo = brentq(f, 1e-6, a - 1e-9) if a > 1e-6 else 0.0
    hi = brentq(f, a + 1e-9, 1 - 1e-9) if a < 1 - 1e-6 else 1.0
    return [round(lo, 3), round(hi, 3)]


def paired_boot(y, a, b, rng, n=2000) -> dict:
    y, a, b = map(np.asarray, (y, a, b))
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return {"difference": round(float(roc_auc_score(y, a) - roc_auc_score(y, b)), 3), "ci95": [round(float(q), 3) for q in np.percentile(d, [2.5, 97.5])],
            "n": int(len(y)), "n_pos": int(np.sum(y)), "p_gt_0": round(float(np.mean(np.array(d) > 0)), 3)}


def main() -> int:
    rng = np.random.default_rng(20261031)
    res = {"protocol": "eda/POST_REVIEW_PROTOCOL.md (H), post hoc"}
    p = {"DR25: v2 keeps more injected planets than the Robovetter": 3e-8, "K2: deep confirmed planets underscored": 2e-6,
         "CH4 GBM, strong-feature half": 0.003, "CH4 CNN, strong-feature half": 0.006, "CO2 CNN, strong-feature half": 0.0008,
         "CO2 GBM, strong-feature half": 0.003, "CO2 band masking": 3e-6, "CH4 band masking": 6.5e-7, "v3 rejects more INV false alarms than v2": 2e-86,
         "v3 rejects more NTP false alarms than v2": 2e-64, "v3 rejects more INJ2 than v2": 1e-33, "v3 rejects more INJ3 than v2": 1e-9,
         "S/N balance raises deep K2 planets": 7e-7, "CO2 GBM on 34 planets": 1e-5, "CH4 GBM on 29 new planets": 3e-4}
    a = holm(p)
    res["holm"] = {"m": len(p), "largest_adjusted_p": round(max(a.values()), 4), "all_below_0.05": bool(max(a.values()) < 0.05), "adjusted": {k: float(f"{v:.2g}") for k, v in a.items()}}
    co2 = pd.read_csv(R / "outputs" / "stage_b_co2" / "co2_scores.csv").dropna(subset=["label", "gbm", "cnn", "strength"])
    m, n = int(co2.label.sum()), int((co2.label == 0).sum())
    res["newcombe"] = {"CH4 GBM 41 planets": {"auc": 1.0, "m": 11, "n": 30, "ci95": newcombe(1.0, 11, 30)},
                       "CH4 CNN 41 planets": {"auc": 0.967, "m": 11, "n": 30, "ci95": newcombe(0.967, 11, 30)}}
    for k in ("cnn", "gbm"):
        v = float(roc_auc_score(co2.label, co2[k]))
        res["newcombe"][f"CO2 {k.upper()} 34 planets"] = {"auc": round(v, 3), "m": m, "n": n, "ci95": newcombe(v, m, n)}
    thr = float(co2.strength.median())
    st = co2[co2.strength > thr]
    res["co2_strong_half_vs_control"] = {k: paired_boot(st.label, st[k], st.strength, rng) for k in ("cnn", "gbm")}
    sc = pd.read_csv(R / "outputs" / "stage_b_methane_ext" / "specnet_scores.csv")
    new = sc[~sc.set.astype(str).str.contains("original")].dropna(subset=["label", "gbm_v3", "cnn_v3", "strength"])
    st4 = new[new.strength > new.strength.median()]
    res["ch4_strong_half_vs_control"] = {"n_new": int(len(new)), **{k: paired_boot(st4.label.astype(int), st4[v], st4.strength, rng) for k, v in (("gbm", "gbm_v3"), ("cnn", "cnn_v3"))}}
    eu = pd.read_csv(sorted(EU_DIR.glob("exoplanet_eu_species_*.csv"))[-1])
    alt = eu.assign(target_name=eu.alt_target_name.fillna("").str.split("#")).explode("target_name")
    idx = pd.concat([eu, alt]).assign(k=lambda d: d.target_name.astype(str).map(key)).drop_duplicates("k").set_index("k").species
    lab = pd.read_csv(R / "outputs" / "labels" / "co2_labels.csv")
    lab = lab[lab.CO2.astype(str).isin(["0", "1"])].copy()
    lab["y"] = lab.CO2.astype(int)
    sp = lab.planet.map(key).map(idx)
    lab["eu"] = sp.fillna("").str.split("#").map(lambda s: "CO2" in s).astype(int)
    res["co2_crosscheck_exoplanet_eu"] = {"n": int(len(lab)), "n_with_species": int(sp.notna().sum()), "agreement": round(float((lab.y == lab.eu).mean()), 3),
                                          "kappa": round(float(cohen_kappa_score(lab.y, lab.eu)), 2), "ours1_eu0": int(((lab.y == 1) & (lab.eu == 0)).sum()), "ours0_eu1": int(((lab.y == 0) & (lab.eu == 1)).sum())}
    sub = lab[sp.notna()]
    res["co2_crosscheck_exoplanet_eu"]["kappa_listed_only"] = round(float(cohen_kappa_score(sub.y, sub.eu)), 2) if len(sub) else None
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stats.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
