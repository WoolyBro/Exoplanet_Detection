"""
methane_round3.py
=================

Round 3 of the methane test (METHANE_EXTENSION_PROTOCOL_ROUND3.md). Covers the JWST planets added to
the NASA Exoplanet Archive since the research pack's snapshot, downloaded by
tools/refresh_archive_spectra.py into external_data/archive_spectra_<date>/. They are scored by the
frozen headline (v3) and reference (v1) GBMs, with exactly the Stage B preprocessing.

Run (project .venv): python stage_b_atmospheres/methane_round3.py
Output: outputs/stage_b_methane_ext/round3_scores.csv, round3.json
"""

from __future__ import annotations

import hashlib
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

OUT = RESEARCH / "outputs" / "stage_b_methane_ext"
REFRESH = sorted((RESEARCH / "external_data").glob("archive_spectra_*"))[-1]
LABELS = RESEARCH / "outputs" / "labels" / "ch4_labels_round3.csv"
EXTRA_PLANETS = ["HAT-P-12 b"]           # already labelled (round 1); new spectra scored as a consistency check


def frozen() -> pd.DataFrame:
    rec = json.loads((OUT / "protocol_hashes.json").read_text())
    if hashlib.sha256(LABELS.read_bytes()).hexdigest() != rec["ch4_labels_round3.csv"]:
        raise SystemExit("round-3 labels changed after freezing")
    return pd.read_csv(LABELS, dtype=str, keep_default_na=False)


def spectra(planets: list[str]):
    added = pd.read_csv(REFRESH / "added_since_pack.csv")
    added = added[(added.spec_type == "Transmission") & added.facility.astype(str).str.contains("James Webb") & added.pl_name.isin(planets)]
    files = {p.name: p for p in (REFRESH / "spectra_tbl").glob("*.tbl")}
    X, rows = [], []
    for r in added.itertuples():
        f = next((p for n, p in files.items() if n.endswith("_" + Path(r.spec_path).name)), None)
        if f is None:
            rows.append({"planet": r.pl_name, "file": Path(r.spec_path).name, "scored": False, "reason": "file missing"})
            continue
        try:
            df, _ = sbp.parse_tbl(f)
        except ValueError as e:
            rows.append({"planet": r.pl_name, "file": f.name, "scored": False, "reason": str(e)[:60]})
            continue
        depth, unc, mask = sbp.resample(df)
        cov = mask > 0
        if not cov.any() or not sbp.band_observable(mask, "CH4"):
            rows.append({"planet": r.pl_name, "file": f.name, "scored": False, "instrument": r.instrument,
                         "reason": "fewer than 3 grid bins in a CH4 band"})
            continue
        depth = np.where(cov, depth - np.median(depth[cov]), 0.0)
        unc = np.where(cov, np.where(np.isfinite(unc), unc, np.nanmedian(unc[cov])), 0.0)
        X.append(np.stack([depth, unc, mask]))
        rows.append({"planet": r.pl_name, "file": f.name, "scored": True, "instrument": r.instrument, "reason": ""})
    return np.array(X), pd.DataFrame(rows)


def main() -> int:
    lab = frozen()
    X, meta = spectra(lab.planet.tolist() + EXTRA_PLANETS)
    scored = meta[meta.scored].reset_index(drop=True)
    out = {"refresh_folder": REFRESH.name, "spectra": meta.to_dict(orient="records"), "planets": {}}
    if len(X):
        for k in ("v1", "v3"):
            g = tr.load_grid(tr.GRID[k])
            model = l3.fit_gbm(g[0], g[1][:, tr.CH4])
            scored[f"gbm_{k}"] = l3.predict_gbm(model, X)
        per = scored.groupby("planet")[["gbm_v1", "gbm_v3"]].mean()
        for p, r in per.iterrows():
            lbl = lab.set_index("planet").CH4.get(p, "0 (round 1)" if p in EXTRA_PLANETS else "?")
            out["planets"][p] = {"label": lbl, "gbm_v1": round(float(r.gbm_v1), 3), "gbm_v3": round(float(r.gbm_v3), 3),
                                 "n_spectra": int((scored.planet == p).sum())}
        scored.to_csv(OUT / "round3_scores.csv", index=False)
    out["unscored_labelled"] = sorted(set(lab[lab.CH4 != "unknown"].planet) - set(out["planets"]))
    (OUT / "round3.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "spectra"}, indent=2))
    print(meta.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
