"""
label_shift_baseline.py
=======================

eda/POST_REVIEW_PROTOCOL.md, analysis B (post hoc): is the 80-label depth correction better than ordinary recalibration on the
same labeled signals? Frozen ensembles and Platt maps only; 200 draws of labeled signals, as in addendum L5.

    python stage_a_transit_model/label_shift_baseline.py
Output: outputs/post_review/label_shift_baseline.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from stage_a_transit_model import label_shift as ls  # noqa: E402

OUT = R / "outputs" / "post_review"
M, DRAWS = 20, 200


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main() -> int:
    rng = np.random.default_rng(20261019)
    kval, tval = ls.scored("kepler", "kepler_val"), ls.scored("tess", "tess_val")
    k2 = pd.read_csv(R / "stage_a_transit_model" / "runs" / "k2_blind" / "k2_predictions.csv")
    k2 = k2[k2.duration_source.isin(["catalogue", "other set"])]
    k2 = pd.DataFrame({"y": (k2.true_label != "FALSE POSITIVE").astype(int), "p": ls.platt("kepler", k2.score_kepler), "depth": k2.depth_ppm})
    cases = [("Kepler -> K2", kval, k2), ("Kepler -> TESS", kval, ls.scored("kepler", "tess_test")), ("TESS -> Kepler", tval, ls.scored("tess", "kepler_test"))]
    out = {"protocol": "eda/POST_REVIEW_PROTOCOL.md (B), post hoc", "labels_per_band": M, "draws": DRAWS, "cases": {}}
    ok = True
    for name, src, tgt in cases:
        ss, ts = ls.stratum(src.depth.to_numpy()), ls.stratum(tgt.depth.to_numpy())
        y, p = tgt.y.to_numpy(), tgt.p.to_numpy()
        bands = [k for k in range(len(ls.STRATA)) if (ts == k).any()]
        pi_s = {k: float(src.y.to_numpy()[ss == k].mean()) if (ss == k).any() else float(src.y.mean()) for k in bands}
        pi_all = float(src.y.mean())
        rows = []
        for _ in range(DRAWS):
            lab, w = np.zeros(len(y), bool), np.zeros(len(y))
            for k in bands:
                idx = np.flatnonzero(ts == k)
                if len(idx) > M:
                    chosen = rng.choice(idx, M, replace=False)
                    lab[chosen] = True
                    w[chosen] = len(idx) / M                     # each band counts by its share of the survey
            q = p.copy()
            for k in bands:
                mk = ts == k
                q[mk] = ls.shift(p[mk], pi_s[k], float(y[mk & lab].mean()) if (mk & lab).any() else pi_s[k])
            h = ~lab
            est = float(np.sum(w * y) / np.sum(w))
            g = ls.shift(p, pi_all, est)
            if len(set(y[lab])) == 2:
                lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p[lab]).reshape(-1, 1), y[lab], sample_weight=w[lab])
                pl = lr.predict_proba(logit(p).reshape(-1, 1))[:, 1]
            else:
                pl = g
            rows.append([ls.ece(y[h], x[h]) for x in (p, q, pl, g)] + [roc_auc_score(y[h], x[h]) for x in (p, q, pl, g)])
        a = np.median(np.array(rows), axis=0)
        r = {"n": int(len(y)), "ece": dict(zip(["frozen", "depth_correction", "platt_refit", "global_shift"], np.round(a[:4], 4).tolist())),
             "roc_auc": dict(zip(["frozen", "depth_correction", "platt_refit", "global_shift"], np.round(a[4:], 4).tolist()))}
        r["depth_beats_platt_by_0.01"] = bool(r["ece"]["platt_refit"] - r["ece"]["depth_correction"] >= 0.01)
        ok &= r["depth_beats_platt_by_0.01"]
        out["cases"][name] = r
    out["may_be_called_label_shift_by_depth"] = bool(ok)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "label_shift_baseline.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
