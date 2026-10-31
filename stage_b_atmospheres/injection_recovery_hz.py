"""
injection_recovery_hz.py
========================

eda/POST_REVIEW_PROTOCOL.md, analysis G (post hoc): could the contamination-aware retrieval have found an atmosphere? A model
signal is added to each real spectrum, which keeps its real noise and contamination, and the same retrieval and credit rule are
rerun.

    TARGET="TRAPPIST-1 f" INJ=n2_co2 ~/micromamba/envs/prt/bin/python stage_b_atmospheres/injection_recovery_hz.py --workers 3
Output: outputs/post_review/injection/<target>_<case>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import retrieval_real_general as rr  # noqa: E402

CASES = {"none": None, "none_h2": None,
         "n2_earthlike": {"gases": {"CO2": 4e-4, "CH4": 1e-7, "H2O": 1e-4}, "bg": "N2", "model": "n2"},
         "n2_co2": {"gases": {"CO2": 1e-2, "CH4": 1e-3, "H2O": 1e-4}, "bg": "N2", "model": "n2"},
         "h2_envelope": {"gases": {"CO2": 1e-7, "CH4": 1e-3, "H2O": 1e-3}, "bg": "H2He", "model": "h2"}}
INJ = os.environ.get("INJ", "n2_co2")
OUT = rr.RESEARCH / "outputs" / "post_review" / "injection"
_READY: dict = {}


def injected_signal() -> list[np.ndarray]:
    pl = rr.planet()
    c = CASES[INJ]
    logp = 0.0                                                  # 1 bar
    T = float(np.clip(pl["teq"], *rr.T_RANGE))
    return [a - b for a, b in zip(rr.planet_depths(c["gases"], c["bg"], T, logp, pl["rp"]),
                                  rr.planet_depths({}, None, 0.0, 0.0, pl["rp"]))]


def ensure() -> None:
    if _READY:
        return
    spectra = rr.load()
    rr.templates()
    if CASES[INJ] is not None:
        sig = injected_signal()
        for s, d in zip(spectra, sig):
            s["depth"] = s["depth"] + d
        _READY["signal_ppm"] = [float(np.max(np.abs(d - np.median(d)))) for d in sig]
    _READY["ok"] = True


def loglike(x, model):
    ensure()
    return rr.log_likelihood(x, model)


def run(model, workers, nlive) -> dict:
    import dynesty
    from dynesty.pool import Pool

    t0 = time.time()
    with Pool(workers, loglike, rr.prior_transform, logl_args=(model,), ptform_args=(model,)) as pool:
        s = dynesty.NestedSampler(pool.loglike, pool.prior_transform, rr.ndim(model), nlive=nlive, pool=pool,
                                  queue_size=workers, sample="unif", rstate=np.random.default_rng(3))
        s.run_nested(dlogz=rr.DLOGZ, print_progress=False)
    r = s.results
    return {"model": model, "logz": float(r.logz[-1]), "logz_err": float(r.logzerr[-1]), "seconds": round(time.time() - t0),
            "best_fit": dict(zip(rr.names(model), map(float, r.samples[np.argmax(r.logl)])))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--nlive", type=int, default=rr.NLIVE)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{rr.SLUG}_{INJ}.json"
    ensure()
    c = CASES[INJ]
    atm = c["model"] if c else ("h2" if INJ == "none_h2" else "n2")
    res = {m: run(m, a.workers, a.nlive) for m in ("bare_rock", atm)}
    chis = {}
    for m, r in res.items():
        gases, bg, T, logp, rp = rr.unpack(np.array([r["best_fit"][n] for n in rr.names(m)]), m)
        dps = rr.planet_depths(gases, bg, T, logp, rp)
        chis[m] = [rr.conditional_chi2(dp, s, tp) for dp, s, tp in zip(dps, rr.load(), rr.templates())]
    gain = [round(b - x, 1) for x, b in zip(chis[atm], chis["bare_rock"])]
    lnb = res[atm]["logz"] - res["bare_rock"]["logz"]
    need = int(np.ceil(2 * len(gain) / 3))
    n_up = int(sum(g > 0 for g in gain))
    out = {"protocol": "eda/POST_REVIEW_PROTOCOL.md (G), post hoc", "target": rr.TARGET, "case": INJ, "injected": c,
           "max_signal_ppm_per_spectrum": _READY.get("signal_ppm"), "lnB_atm_vs_bare": round(lnb, 2),
           "lnB_err": round(float(np.hypot(res[atm]["logz_err"], res["bare_rock"]["logz_err"])), 2),
           "chi2_gain": gain, "n_spectra_improved": n_up, "n_needed": need, "credited": bool(lnb >= 3 and n_up >= need),
           "runs": res, "n_points": [int(len(s["wl"])) for s in rr.load()]}
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("target", "case", "lnB_atm_vs_bare", "n_spectra_improved", "credited", "max_signal_ppm_per_spectrum")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
