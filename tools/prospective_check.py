"""
prospective_check.py
====================

Scores the registered PROSPECTIVE predictions (eda/PROSPECTIVE_PREDICTIONS.md): the planet-like scores
that the frozen v2 ensembles gave to test-split CANDIDATES on 2026-10-18 (commit ccd2eed), before
anyone knew what those candidates are. Each run downloads the current dispositions from the NASA
Exoplanet Archive (TAP) and evaluates only the candidates resolved since then.

    python tools/prospective_check.py

The prediction files are checked against their registered SHA-256, so they cannot be edited afterwards.
A result is reported only when at least MIN_PER_CLASS candidates have resolved each way; until then the
script reports the counts. Output: outputs/prospective/check_<date>.json (one file per run, kept).
"""

from __future__ import annotations

import hashlib
import io
import json
import ssl
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
PRED = R / "stage_a_transit_model" / "runs" / "v2_final"
OUT = R / "outputs" / "prospective"
TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
REGISTERED = {"kepler_test_predictions.csv": "3d897473c3f8a9545f53c5313c8f2789e66d652bf2ebe7d6724c0cd7bc0014b2",
              "tess_test_predictions.csv": "99206c087ab12381f56ad09a2b70df9c45eb085122a34d014b8dc266d259433e"}
MIN_PER_CLASS = 10
PLANET = {"tess": {"CP", "KP"}, "kepler": {"CONFIRMED"}}
FALSE = {"tess": {"FP", "FA"}, "kepler": {"FALSE POSITIVE"}}


def tap(query: str) -> pd.DataFrame:
    import certifi
    ctx = ssl.create_default_context(cafile=certifi.where())
    url = TAP + "?" + urllib.parse.urlencode({"query": query, "format": "csv"})
    req = urllib.request.Request(url, headers={"User-Agent": "research (prospective check)"})
    return pd.read_csv(io.BytesIO(urllib.request.urlopen(req, timeout=300, context=ctx).read()))


def current(mission: str) -> pd.Series:
    if mission == "tess":
        t = tap("select toi, tfopwg_disp from toi")
        return t.assign(object_id="TOI " + t.toi.map(lambda x: f"{x:.2f}")).set_index("object_id").tfopwg_disp
    k = tap("select kepoi_name, koi_disposition from cumulative")
    return k.set_index("kepoi_name").koi_disposition


def main() -> int:
    from sklearn.metrics import roc_auc_score

    out = {"checked_on": date.today().isoformat(), "registered_commit": "ccd2eed", "missions": {}}
    for f, h in REGISTERED.items():
        if hashlib.sha256((PRED / f).read_bytes()).hexdigest() != h:
            raise SystemExit(f"{f} differs from its registered hash; the prospective test is void")
    for mission in ["tess", "kepler"]:
        pred = pd.read_csv(PRED / f"{mission}_test_predictions.csv")
        cand = pred[pred.true_label == "CANDIDATE"].copy()
        cand["now"] = cand.object_id.map(current(mission))
        cand["y"] = np.where(cand.now.isin(PLANET[mission]), 1, np.where(cand.now.isin(FALSE[mission]), 0, -1))
        res = cand[cand.y >= 0]
        r = {"n_candidates": int(len(cand)), "now": cand.now.fillna("not in table").value_counts().to_dict(),
             "resolved_planet": int((res.y == 1).sum()), "resolved_false_positive": int((res.y == 0).sum())}
        if min(r["resolved_planet"], r["resolved_false_positive"]) >= MIN_PER_CLASS:
            r["roc_auc_planet_like_score"] = float(roc_auc_score(res.y, res.planet_like_score))
            r["roc_auc_calibrated"] = float(roc_auc_score(res.y, res.calibrated_score))
        else:
            r["status"] = f"not yet evaluable (needs >= {MIN_PER_CLASS} resolved each way)"
        r["resolved"] = res[["object_id", "now", "planet_like_score"]].to_dict(orient="records")
        out["missions"][mission] = r
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"check_{date.today().isoformat()}.json").write_text(json.dumps(out, indent=2, default=str))
    for m, r in out["missions"].items():
        print(f"{m}: {r['n_candidates']} registered candidates; resolved {r['resolved_planet']} planets, "
              f"{r['resolved_false_positive']} false positives; " + (f"ROC-AUC {r['roc_auc_planet_like_score']:.3f}"
                                                                    if "roc_auc_planet_like_score" in r else r["status"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
