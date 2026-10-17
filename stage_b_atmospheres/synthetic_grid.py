"""
synthetic_grid.py
=================

A training set of synthetic JWST transmission spectra for the Stage B molecule classifier, built so
a model trained on it can be TESTED on the 13 real planets without ever having seen them.

Runs in the petitRADTRANS environment (see forward_model.py):
    ~/micromamba/envs/prt/bin/python stage_b_atmospheres/synthetic_grid.py --n 3000 --workers 4

Why each design choice
  * Planets: each sample is anchored on one of the 13 real planets' (R_p, M_p, R_s, T_eq) and
    perturbed (radius x U(0.8,1.25), mass x U(0.7,1.4), T x U(0.7,1.1)), so the synthetic population
    spans the real one rather than Earth analogues (the reason INARA could not be used).
  * Composition: H2/He background - except that rocky anchors (R_p < 1.8 R_earth, i.e. TRAPPIST-1 c)
    get an N2 (secondary) background with probability 0.7, because a hydrogen envelope on a rocky
    planet would create features its real spectrum can never show. Each of H2O, CO2, CH4, CO, SO2 is present with probability 0.6 at
    log10 VMR ~ U(-7, -2.5); absent gases are left out entirely. Grey cloud deck log10 P ~ U(-5, 0) bar.
  * Coverage and noise: each sample borrows the coverage mask and per-bin uncertainty of a randomly
    chosen REAL spectrum (from outputs/stage_b/stage_b_dataset.npz), then Gaussian noise of that
    size is added. A model can therefore never see a cleaner or fuller spectrum than JWST delivers.
  * Labels mean what the literature labels mean - DETECTABLE, not merely present: for each gas the
    noise-free spectrum is recomputed WITHOUT that gas, and the label is 1 only if the difference,
    over the bins this sample actually covers, is >= 3 sigma (chi^2 >= 9) at this sample's noise.
    A gas that is present but undetectable at the borrowed noise is labelled 0, as a paper would
    report it.
  * Grid and preprocessing identical to Stage B: 1-5 um, 100 bins, depth minus its median over covered
    bins, channels (depth, uncertainty, mask).

Output: outputs/stage_b_synthetic/synthetic_grid.npz (X, Y, presence, anchors, params) + params.csv
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(HERE))

MOLECULES = ["H2O", "CO2", "CH4", "CO", "SO2"]       # Stage B order is H2O, CH4, CO2, SO2, CO; mapped on save
STAGE_B_ORDER = ["H2O", "CH4", "CO2", "SO2", "CO"]
EDGES = np.linspace(1.0, 5.0, 101)
OUT = RESEARCH / "outputs" / "stage_b_synthetic"
DETECT_CHI2 = 9.0

# The 13 real planets' parameters (Rp, Mp in Earth units, Rs solar, T_eq K) come from
# outputs/stage_b_level2/planet_scale_heights.csv, which Stage B Level 2 already wrote from pscomppars.
_REAL = None


def real_inputs():
    global _REAL
    if _REAL is None:
        import pandas as pd

        z = np.load(RESEARCH / "outputs" / "stage_b" / "stage_b_dataset.npz", allow_pickle=True)
        params = pd.read_csv(RESEARCH / "outputs" / "stage_b_level2" / "planet_scale_heights.csv")
        _REAL = {"X": z["X"], "planets": z["planets"].astype(str), "params": params}
    return _REAL


def make_sample(seed: int) -> dict:
    import forward_model as fm

    rng = np.random.default_rng(seed)
    R = real_inputs()
    anchor = R["params"].iloc[rng.integers(len(R["params"]))]
    rp = float(anchor.pl_rade) * rng.uniform(0.8, 1.25)
    mp = float(anchor.pl_bmasse) * rng.uniform(0.7, 1.4)
    t = float(anchor.pl_eqt) * rng.uniform(0.7, 1.1)
    rs = float(anchor.st_rad)
    vmr = {g: 10 ** rng.uniform(-7, -2.5) for g in MOLECULES if rng.random() < 0.6}
    if not vmr:
        vmr = {"H2O": 10 ** rng.uniform(-7, -2.5)}
    p_cloud = 10 ** rng.uniform(-5, 0)
    background = "N2" if (float(anchor.pl_rade) < 1.8 and rng.random() < 0.7) else "H2He"
    # borrow a real spectrum's coverage and noise
    k = rng.integers(len(R["X"]))
    _, unc, mask = R["X"][k]
    cov = mask > 0

    def binned(v):
        wl, dep = fm.transmission_spectrum(v, background, t, rp, mp, rs, surface_pressure=p_cloud,
                                         wl_range=(0.98, 5.02), species=tuple(MOLECULES))
        return fm.bin_spectrum(wl, dep, EDGES)

    full = binned(vmr)
    labels = {}
    for g in MOLECULES:
        if g not in vmr:
            labels[g] = 0
            continue
        without = {h: v for h, v in vmr.items() if h != g} or {"H2O": 1e-12}
        diff = full - binned(without)
        # median offsets cancel in the real preprocessing, so compare shapes, not absolute levels
        diff = diff - np.nanmedian(diff[cov])
        chi2 = np.nansum((diff[cov] / unc[cov]) ** 2)
        labels[g] = int(chi2 >= DETECT_CHI2)
    noisy = full + rng.normal(0, 1, len(full)) * unc
    depth = np.where(cov, noisy - np.nanmedian(noisy[cov]), 0.0)
    return {"X": np.stack([depth, np.where(cov, unc, 0.0), mask]).astype(np.float32),
            "Y": [labels[g] for g in STAGE_B_ORDER], "present": [int(g in vmr) for g in STAGE_B_ORDER],
            "anchor": str(anchor.pl_name), "noise_from": str(R["planets"][k]),
            "params": {"rp": rp, "mp": mp, "rs": rs, "T": t, "log_p_cloud": float(np.log10(p_cloud)), "background": background,
                       **{f"log_vmr_{g}": float(np.log10(vmr[g])) if g in vmr else None for g in MOLECULES}}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--out", default=str(OUT), help="output folder (a second, independent grid goes elsewhere)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    seeds = [args.seed + i for i in range(args.n)]
    # Each sample is checkpointed as it finishes, so an interrupted run resumes instead of restarting.
    parts = out / "parts"
    parts.mkdir(exist_ok=True)
    todo = [s for s in seeds if not (parts / f"{s}.pkl").is_file()]
    print(f"  {len(seeds) - len(todo)} samples already checkpointed, {len(todo)} to compute", flush=True)
    with Pool(args.workers) as pool:
        for i, (s, r) in enumerate(zip(todo, pool.imap(make_sample, todo, chunksize=4)), 1):
            tmp = parts / f"{s}.pkl.tmp"
            tmp.write_bytes(pickle.dumps(r))
            tmp.replace(parts / f"{s}.pkl")
            if i % 100 == 0:
                print(f"  {i}/{len(todo)} samples, {time.time() - t0:.0f} s", flush=True)
    rows = [pickle.loads((parts / f"{s}.pkl").read_bytes()) for s in seeds]
    import pandas as pd

    np.savez_compressed(out / "synthetic_grid.npz", X=np.stack([r["X"] for r in rows]),
                        Y=np.array([r["Y"] for r in rows], np.int8),
                        present=np.array([r["present"] for r in rows], np.int8),
                        anchors=np.array([r["anchor"] for r in rows]), noise_from=np.array([r["noise_from"] for r in rows]),
                        molecules=np.array(STAGE_B_ORDER))
    pd.DataFrame([{"anchor": r["anchor"], "noise_from": r["noise_from"], **r["params"]} for r in rows]).to_csv(
        out / "params.csv", index=False)
    Y = np.array([r["Y"] for r in rows]); P = np.array([r["present"] for r in rows])
    summary = {"n": args.n, "seed": args.seed, "seconds": round(time.time() - t0),
               "detectable_rate": dict(zip(STAGE_B_ORDER, Y.mean(0).round(3).tolist())),
               "present_rate": dict(zip(STAGE_B_ORDER, P.mean(0).round(3).tolist())),
               "detectable_given_present": dict(zip(STAGE_B_ORDER, (Y.sum(0) / np.maximum(P.sum(0), 1)).round(3).tolist()))}
    (out / "grid_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
