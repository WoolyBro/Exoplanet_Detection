"""
methane_round4.py
=================

CH4 round 4 (METHANE_EXTENSION_PROTOCOL_ROUND4.md, commit 944f3dc): the frozen v3 CH4 GBM on TOI-1231 b, a
planet published after the models were frozen, from the authors' public spectra (OSF, osf.io/mryd6; two
independent reductions, JexoRes and JexoPipe). As in the archive, each instrument mode of each reduction is
one spectrum: NIRISS SOSS (orders 1 + 2) and NIRSpec G395H (NRS1 + NRS2). Preprocessing and model are those
of methane_round3.py.

    python stage_b_atmospheres/methane_round4.py
Output: outputs/stage_b_methane_ext/round4.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(HERE))
import methane_tls_robustness as tr  # noqa: E402
import stage_b_pipeline as sbp  # noqa: E402
import train_synthetic_classifier as l3  # noqa: E402

DATA = RESEARCH / "external_data" / "round4_toi1231b"
OUT = RESEARCH / "outputs" / "stage_b_methane_ext"
BOUNDARY = 0.125                          # midpoint of the rounds 1-3 class separation (0.08 / 0.17)


def read(files: list[str]) -> pd.DataFrame:
    parts = [np.loadtxt(DATA / f, comments="#") for f in files]
    a = np.vstack(parts)
    return pd.DataFrame({"wl": a[:, 0], "depth_ppm": a[:, 2] * 1e6, "err_ppm": a[:, 3] * 1e6}).sort_values("wl").reset_index(drop=True)


def main() -> int:
    spectra = {f"{mode} ({red})": [f"TOI-1231b_{p}_{red}.txt" for p in parts]
               for red in ["JexoRes", "JexoPipe"]
               for mode, parts in [("NIRISS SOSS", ["niriss_ord1", "niriss_ord2"]), ("NIRSpec G395H", ["nirspec_g395h_nrs1", "nirspec_g395h_nrs2"])]}
    X, rows = [], []
    for name, files in spectra.items():
        df = read(files)
        depth, unc, mask = sbp.resample(df)
        cov = mask > 0
        ok = bool(cov.any() and sbp.band_observable(mask, "CH4"))
        rows.append({"spectrum": name, "points": int(len(df)), "wl_range_um": [round(float(df.wl.min()), 2), round(float(df.wl.max()), 2)],
                     "grid_bins": int(cov.sum()), "scored": ok})
        if ok:
            depth = np.where(cov, depth - np.median(depth[cov]), 0.0)
            unc = np.where(cov, np.where(np.isfinite(unc), unc, np.nanmedian(unc[cov])), 0.0)
            X.append(np.stack([depth, unc, mask]))
    g = tr.load_grid(tr.GRID["v3"])
    model = l3.fit_gbm(g[0], g[1][:, tr.CH4])
    p = l3.predict_gbm(model, np.array(X))
    k = 0
    for r in rows:
        if r["scored"]:
            r["gbm_v3_p_ch4"] = round(float(p[k]), 3)
            k += 1
    planet = float(np.mean(p))
    res = {"protocol": "stage_b_atmospheres/METHANE_EXTENSION_PROTOCOL_ROUND4.md", "planet": "TOI-1231 b", "label_ch4": 1,
           "spectra": rows, "planet_score_gbm_v3": round(planet, 3), "boundary": BOUNDARY,
           "expectations": {"R4-1 TOI-1231 b on the CH4 side (>= 0.125)": bool(planet >= BOUNDARY)}}
    (OUT / "round4.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
