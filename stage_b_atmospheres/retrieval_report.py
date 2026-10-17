"""
retrieval_report.py
===================

Summarises the TRAPPIST-1 e retrievals (retrieval_trappist1e.py), with and without stellar
contamination, and compares them with the known-template (matched-filter) forecast.

Run (project .venv): python stage_b_atmospheres/retrieval_report.py

For each gas and set it reports:
  * the detection significance at each transit count. lnB = lnZ(full) - lnZ(without the gas), turned
    into sigma with the Benneke & Seager (2013) bound;
  * the transits needed for 3 sigma and 5 sigma, interpolated in log N - log sigma (flagged if extrapolated);
  * the "retrieval factor" = N_retrieval / N_matched-filter at 5 sigma, on the same data grid and noise.

Output: outputs/retrieval/summary.json, eda/retrieval_trappist1e.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(HERE))
from retrieval_trappist1e import bayes_sigma  # noqa: E402

SETS = {"clean": RESEARCH / "outputs" / "retrieval", "stellar_contamination": RESEARCH / "outputs" / "retrieval_tls"}
GASES = {"CH4": "no_CH4", "CO2": "no_CO2"}


def significance_table(runs: list[dict]) -> dict:
    """{(n, realisation): {gas: (lnB, err, sigma)}}"""
    groups = {}
    for d in runs:
        groups.setdefault((d["n_transits"], d.get("realisation")), {})[d["model"]] = d
    out = {}
    for key, g in groups.items():
        if "full" not in g:
            continue
        for gas, red in GASES.items():
            if red in g:
                ln_b = g["full"]["logz"] - g[red]["logz"]
                err = float(np.hypot(g["full"]["logz_err"], g[red]["logz_err"]))
                out.setdefault(key, {})[gas] = (round(ln_b, 2), round(err, 2), round(bayes_sigma(ln_b), 2))
    return out


def n_for_sigma(ns, sigmas, target):
    """Interpolate log N against log sigma. Returns (N, extrapolated?)."""
    ns, s = np.asarray(ns, float), np.maximum(np.asarray(sigmas, float), 1e-3)
    order = np.argsort(ns)
    ns, s = ns[order], s[order]
    if len(ns) < 2:
        return None, True
    ls, ln = np.log(s), np.log(ns)
    if s.max() >= target and s.min() <= target:
        i = np.flatnonzero(s >= target)[0]
        if i == 0:
            return float(ns[0]), False
        x = np.interp(np.log(target), ls[i - 1:i + 1], ln[i - 1:i + 1])
        return float(np.exp(x)), False
    # extrapolate with the slope of the last (or first) two points
    j = slice(-2, None) if s.max() < target else slice(0, 2)
    slope = (ls[j][1] - ls[j][0]) / (ln[j][1] - ln[j][0]) if ln[j][1] != ln[j][0] else 0.5
    slope = slope if slope > 0.05 else 0.5
    x = ln[j][1] + (np.log(target) - ls[j][1]) / slope
    return float(np.exp(x)), True


def main() -> int:
    summary, curves = {}, {}
    for name, d in SETS.items():
        f = d / "results.json"
        if not f.is_file():
            continue
        res = json.loads(f.read_text())
        mf = res["matched_filter_snr_1_transit"]
        table = significance_table(res["runs"])
        expected = {k[0]: v for k, v in table.items() if k[1] is None}
        noisy = {f"N={k[0]} #{k[1]}": v for k, v in table.items() if k[1] is not None}
        s = {"per_N_expected": {str(n): v for n, v in sorted(expected.items())}, "noise_realisations": noisy}
        for gas in GASES:
            ns = [n for n in sorted(expected) if gas in expected[n]]
            sig = [expected[n][gas][2] for n in ns]
            n3, e3 = n_for_sigma(ns, sig, 3.0)
            n5, e5 = n_for_sigma(ns, sig, 5.0)
            mf5 = (5.0 / mf[gas]) ** 2
            s[gas] = {"N_3sigma": None if n3 is None else round(n3, 1), "N_3sigma_extrapolated": e3,
                      "N_5sigma": None if n5 is None else round(n5, 1), "N_5sigma_extrapolated": e5,
                      "matched_filter_N_5sigma": round(mf5, 1),
                      "retrieval_factor_5sigma": None if n5 is None else round(n5 / mf5, 2)}
            curves[(name, gas)] = (ns, sig, mf[gas])
        summary[name] = s
    out = RESEARCH / "outputs" / "retrieval" / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    make_figure(curves)
    return 0


def make_figure(curves):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, gas in zip(axes, GASES):
        n_grid = np.geomspace(1, 64, 100)
        mf = next((v[2] for k, v in curves.items() if k[1] == gas), None)
        if mf is not None:
            ax.plot(n_grid, mf * np.sqrt(n_grid), color="0.55", ls="--", label="known-template (matched filter)")
        for (name, g), (ns, sig, _) in curves.items():
            if g != gas or not ns:
                continue
            c = "#2a6fdb" if name == "clean" else "#d1495b"
            lab = "Bayesian retrieval" + ("" if name == "clean" else " + stellar contamination fitted")
            ax.plot(ns, sig, "o-", color=c, label=lab)
        for lvl in (3, 5):
            ax.axhline(lvl, color="0.8", lw=0.8)
            ax.text(1.05, lvl + 0.1, f"{lvl} sigma", fontsize=7, color="0.5")
        ax.set(xscale="log", xlabel="JWST NIRSpec PRISM transits", title=f"TRAPPIST-1 e, Archean atmosphere: {gas}",
               xlim=(1, 64), ylim=(0, 12))
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("detection significance (sigma)")
    fig.tight_layout()
    fig.savefig(RESEARCH / "eda" / "retrieval_trappist1e.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())
