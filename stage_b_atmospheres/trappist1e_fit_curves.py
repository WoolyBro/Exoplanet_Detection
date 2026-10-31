"""
trappist1e_fit_curves.py
========================

Saves the best-fit curves of the TRAPPIST-1 e retrieval for the paper's figure, without running a new retrieval.
Runs in the petitRADTRANS environment:

    OMP_NUM_THREADS=2 ~/micromamba/envs/prt/bin/python stage_b_atmospheres/trappist1e_fit_curves.py
Output: outputs/post_review/trappist1e_fit_curves.npz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import real_retrieval_report as rep  # noqa: E402

rr = rep.rr


def main() -> int:
    res = {r["model"]: r for r in json.loads((rep.OUT / "results.json").read_text())["runs"]}
    d = rr.load_visits()
    out = {"wl": d["wl"], "depth": d["depth"], "sigma": d["sigma"]}
    for m in ("bare_rock", "full", "archean"):
        bf = res[m]["best_fit"]
        gases = rr.MODELS[m]
        vmr = dict(rr.ARCHEAN) if gases == "fixed" else {g: 10 ** bf[f"log_{g}"] for g in gases}
        depth = rr.planet_depth(vmr if m != "bare_rock" else {}, bf.get("T", 0.0), bf.get("logP", 0.0), bf["Rp"])
        out[f"planet_{m}"] = depth
        out[f"visits_{m}"] = np.array([rep.conditional_fit(depth, d["depth"][v], d["sigma"][v])[1] for v in range(len(d["depth"]))])
    np.savez(rep.RESEARCH / "outputs" / "post_review" / "trappist1e_fit_curves.npz", **out)
    print({k: np.shape(v) for k, v in out.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
