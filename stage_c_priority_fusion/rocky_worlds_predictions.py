"""
rocky_worlds_predictions.py
===========================

eda/POST_REVIEW_PROTOCOL.md, analysis P (post hoc): the locked forecast for the open Rocky Worlds targets, from the posterior of the
observing design, with no refit.

    python stage_c_priority_fusion/rocky_worlds_predictions.py
Output: outputs/post_review/rocky_worlds_predictions.csv, rocky_worlds_predictions.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shoreline_design as sd  # noqa: E402

OUT = HERE.parent / "outputs" / "post_review"


def main() -> int:
    rm, ru = sd.rm, sd.ru
    ps = pd.read_csv(rm.PSCOMP, low_memory=False)
    old = ru.labelled(HERE / "rocky_atmosphere_observations.csv")
    new = ru.labelled(HERE / "rocky_atmosphere_observations_new.csv")
    allp = pd.concat([old, new], ignore_index=True)
    m_all, g_all, _ = ru.inputs(allp)
    theta, w, _ = ru.fit(allp.y.to_numpy(), m_all, g_all)
    th = theta[np.random.default_rng(rm.SEED).choice(len(theta), size=2000, p=w)]
    need = ["sy_kmag", "pl_trandur", "pl_eqt", "pl_rade", "st_rad", "st_teff", "pl_insol", "pl_orbsmax"]
    pool = ps[(ps.tran_flag == 1) & (ps.pl_rade <= 1.8)].dropna(subset=need)
    pool = pool[~pool.pl_name.isin(allp.planet)].drop_duplicates("pl_name").reset_index(drop=True)
    _, draws = rm.planets_with_draws(pool.pl_name)
    margin = draws["margin_primary"]
    magma = (np.nanmedian(draws["teq"], axis=1) >= rm.MAGMA_TEQ).astype(float)
    p = np.concatenate([rm.prob(th, margin[i:i + 100], magma[i:i + 100]) for i in range(0, len(pool), 100)], axis=1)
    sel = np.where(pool.pl_name.isin(sd.DDT))[0]
    q = p[:, sel]
    names = pool.pl_name.iloc[sel].to_numpy()
    t = pd.DataFrame({"pl_name": names, "p_retain_mean": q.mean(axis=0), "p_retain_p5": np.percentile(q, 5, axis=0), "p_retain_p95": np.percentile(q, 95, axis=0),
                      "margin_median_dex": np.median(margin[sel], axis=1)}).sort_values("p_retain_mean", ascending=False).round(3)
    t.to_csv(OUT / "rocky_worlds_predictions.csv", index=False)
    expected = q.sum(axis=1)
    none = np.prod(1 - q, axis=1)
    counts = np.zeros((len(q), q.shape[1] + 1))                                # exact distribution of the number of atmospheres, per draw
    counts[:, 0] = 1.0
    for j in range(q.shape[1]):
        counts[:, 1:] = counts[:, 1:] * (1 - q[:, [j]]) + counts[:, :-1] * q[:, [j]]
        counts[:, 0] *= 1 - q[:, j]
    res = {"protocol": "eda/POST_REVIEW_PROTOCOL.md (P), post hoc", "targets": list(t.pl_name), "expected_atmospheres": {"mean": round(float(expected.mean()), 2), "ci90": [round(float(x), 2) for x in np.percentile(expected, [5, 95])]},
           "p_none_of_six": round(float(none.mean()), 3), "p_five_or_six": round(float(counts[:, 5:].sum(axis=1).mean()), 3), "sha256": hashlib.sha256((OUT / "rocky_worlds_predictions.csv").read_bytes()).hexdigest()}
    (OUT / "rocky_worlds_predictions.json").write_text(json.dumps(res, indent=2))
    print(t.to_string(index=False))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
