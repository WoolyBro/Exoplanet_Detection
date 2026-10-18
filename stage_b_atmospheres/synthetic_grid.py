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

  * Optional (2026-10-18), --tls and --haze:
      - unocculted spots (0.70-0.95 T_eff) or faculae (T_eff + 50-500 K) covering up to 25 % of the
        disc, with PHOENIX spectra at the anchor star's own T_eff, on 70 % of samples;
      - a power-law haze (kappa_350 0.01-30 cm^2/g, slope -6 to -2) on 50 % of samples.
    Labels are computed on the contaminated, hazy spectra, so "detectable" stays honest.

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


# ---- optional realism (2026-10-18): stellar contamination and hazes ------------------------------ #
# Off by default, so the original grid (seed 20261017) reproduces exactly; every extra random draw
# happens only when a flag is on.
TLS_PROB, SPOT_SHARE, F_MAX = 0.7, 0.6, 0.25
HAZE_PROB, LOG_KAPPA, GAMMA = 0.5, (-2.0, 1.5), (-6.0, -2.0)
_PHX = {}


def phoenix_on_grid():
    """PHOENIX F_nu on the 1-5 um grid, every 20 K from 2002 to 7500 K (petitRADTRANS table)."""
    if not _PHX:
        import os
        from petitRADTRANS.config import petitradtrans_config_parser as cfg
        cfg.set_input_data_path(os.path.expanduser("~/petitRADTRANS/input_data"))
        from petitRADTRANS.stellar_spectra.phoenix import PhoenixStarTable
        t = PhoenixStarTable()
        t.load()
        grid = np.arange(2002.0, 7501.0, 20.0)
        rows = []
        for T in grid:
            spec, _ = t.compute_spectrum(T)
            idx = np.digitize(spec[:, 0] * 1e4, EDGES) - 1
            rows.append([spec[idx == b, 1].mean() for b in range(len(EDGES) - 1)])
        _PHX.update(grid=grid, flux=np.array(rows))
    return _PHX


def phoenix_flux(T: float) -> np.ndarray:
    p = phoenix_on_grid()
    T = float(np.clip(T, p["grid"][0], p["grid"][-1]))
    i = min(np.searchsorted(p["grid"], T) - 1, len(p["grid"]) - 2)
    i = max(i, 0)
    w = (T - p["grid"][i]) / (p["grid"][i + 1] - p["grid"][i])
    return (1 - w) * p["flux"][i] + w * p["flux"][i + 1]


def make_sample(arg) -> dict:
    import forward_model as fm

    seed, use_tls, use_haze = arg if isinstance(arg, tuple) else (arg, False, False)
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
    extra = {}
    haze_k, haze_g, eps = None, -4.0, np.ones(len(EDGES) - 1)
    if use_haze and rng.random() < HAZE_PROB:
        haze_k, haze_g = 10 ** rng.uniform(*LOG_KAPPA), rng.uniform(*GAMMA)
    if use_tls and rng.random() < TLS_PROB:
        t_phot = float(anchor.st_teff)
        spot = rng.random() < SPOT_SHARE
        t_het = t_phot * rng.uniform(0.70, 0.95) if spot else t_phot + rng.uniform(50, 500)
        f = rng.uniform(0, F_MAX)
        eps = 1.0 / (1.0 - f * (1.0 - phoenix_flux(t_het) / phoenix_flux(t_phot)))
        extra.update(tls_f=f, tls_T_het=t_het, tls_type="spot" if spot else "facula")
    if haze_k is not None:
        extra.update(haze_kappa_350=haze_k, haze_gamma=haze_g)

    def binned(v):
        wl, dep = fm.transmission_spectrum(v, background, t, rp, mp, rs, surface_pressure=p_cloud,
                                         wl_range=(0.98, 5.02), species=tuple(MOLECULES),
                                         **({} if haze_k is None else {"haze_kappa_350": haze_k, "haze_gamma": haze_g}))
        return fm.bin_spectrum(wl, dep, EDGES) * eps        # contamination multiplies the observed depth

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
                       **{f"log_vmr_{g}": float(np.log10(vmr[g])) if g in vmr else None for g in MOLECULES}, **extra}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--out", default=str(OUT), help="output folder (a second, independent grid goes elsewhere)")
    ap.add_argument("--tls", action="store_true", help="add unocculted spots/faculae (PHOENIX) to 70 %% of samples")
    ap.add_argument("--haze", action="store_true", help="add a power-law haze to 50 %% of samples")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    seeds = [args.seed + i for i in range(args.n)]
    # Each sample is checkpointed as it finishes, so an interrupted run resumes instead of restarting.
    parts = out / "parts"
    parts.mkdir(exist_ok=True)
    todo = [s for s in seeds if not (parts / f"{s}.pkl").is_file()]
    jobs = [(s, args.tls, args.haze) for s in todo]
    print(f"  {len(seeds) - len(todo)} samples already checkpointed, {len(todo)} to compute", flush=True)
    with Pool(args.workers) as pool:
        for i, (s, r) in enumerate(zip(todo, pool.imap(make_sample, jobs, chunksize=4)), 1):
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
