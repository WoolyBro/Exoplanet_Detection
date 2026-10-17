"""
validate_shoreline.py
=====================

Does the cosmic shoreline (cosmic_shoreline.py) sort the bodies whose atmospheres we actually know?

Two checks, run before the shoreline is allowed into the Stage C ranking:
  1. Solar System: the classic test. Venus, Earth, Mars and Titan have atmospheres; Mercury, the
     Moon, Ganymede and Callisto do not. Escape velocities and mean insolations are from NASA's
     planetary fact sheets. Solar luminosity = 1, so the cumulative XUV equals the insolation.
  2. Rocky exoplanets that JWST or Spitzer has observed (rocky_atmosphere_observations.csv, with
     sources): 13 with no thick atmosphere, and 5 with atmosphere evidence (1 moderate, 4 tentative).
     Each planet's margin uses the same Monte Carlo as Stage C (target_physics.monte_carlo).

Run: python stage_c_priority_fusion/validate_shoreline.py
Output: outputs/stage_c/shoreline_validation.csv, shoreline_validation.json, eda/cosmic_shoreline.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(HERE))
import cosmic_shoreline as cs  # noqa: E402
import target_physics as tp  # noqa: E402

PSCOMP = RESEARCH / "exoplanet_research_data" / "01_candidate_catalogs" / "pscomppars_confirmed_planets.csv"
OUT = RESEARCH / "outputs" / "stage_c"

# name: (v_esc km/s, mean insolation relative to Earth, has a substantial atmosphere)
SOLAR_SYSTEM = {"Mercury": (4.25, 6.67, False), "Venus": (10.36, 1.91, True), "Earth": (11.19, 1.0, True),
                "Moon": (2.38, 1.0, False), "Mars": (5.03, 0.431, True), "Ganymede": (2.74, 0.037, False),
                "Callisto": (2.44, 0.037, False), "Titan": (2.64, 0.011, True)}


def solar_system_check() -> pd.DataFrame:
    rows = []
    for name, (v, i, atm) in SOLAR_SYSTEM.items():
        m_lo = cs.margins(v, i, 1.0, cs.LOG_I0_RANGE[0], 0.0)
        m_hi = cs.margins(v, i, 1.0, cs.LOG_I0_RANGE[1], 0.0)
        rows.append({"body": name, "v_esc": v, "insolation": i, "has_atmosphere": atm,
                     "margin_primary_7e-4": round(float(m_lo["primary"]), 2), "margin_primary_1e-3": round(float(m_hi["primary"]), 2),
                     "margin_empirical": round(float(m_lo["empirical"]), 2), "margin_bolometric": round(float(m_lo["bolometric"]), 2)})
    d = pd.DataFrame(rows)
    for col in ["margin_primary_7e-4", "margin_primary_1e-3", "margin_empirical", "margin_bolometric"]:
        d[f"correct_{col.replace('margin_', '')}"] = (d[col] < 0) == d.has_atmosphere
    return d


def main() -> int:
    obs = pd.read_csv(HERE / "rocky_atmosphere_observations.csv")
    ps = pd.read_csv(PSCOMP, low_memory=False)
    rows = ps[ps.pl_name.isin(obs.planet)]
    phys = tp.monte_carlo(rows, n=4000, seed=20261018)
    keep = ["pl_name", "v_esc_kms", "cumulative_xuv_earth"] + [c for c in phys.columns if c.startswith(("p_retain", "shoreline_margin"))]
    v = obs.merge(phys[keep], left_on="planet", right_on="pl_name").drop(columns="pl_name")
    v["atmosphere"] = (v.status == "atmosphere_evidence").astype(int)
    v.to_csv(OUT / "shoreline_validation.csv", index=False)

    summary = {}
    for k in ("primary", "empirical", "bolometric"):
        bare = v[v.atmosphere == 0]
        atm = v[v.atmosphere == 1]
        summary[k] = {
            "no_thick_atmosphere_on_dry_side": f"{int((bare[f'p_retain_{k}'] < 0.5).sum())}/{len(bare)}",
            "atmosphere_evidence_on_wet_side": f"{int((atm[f'p_retain_{k}'] >= 0.5).sum())}/{len(atm)}",
            "roc_auc_minus_margin_vs_atmosphere": round(float(roc_auc_score(v.atmosphere, -v[f"shoreline_margin_{k}_dex"])), 3),
            "roc_auc_excluding_tentative": round(float(roc_auc_score(
                v[(v.atmosphere == 0) | (v.strength != "tentative")].atmosphere,
                -v[(v.atmosphere == 0) | (v.strength != "tentative")][f"shoreline_margin_{k}_dex"])), 3)
            if ((v.atmosphere == 1) & (v.strength != "tentative")).any() else None,
        }
    ss = solar_system_check()
    summary["solar_system"] = {c: f"{int(ss[c].sum())}/{len(ss)}" for c in ss.columns if c.startswith("correct_")}
    (OUT / "shoreline_validation.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 220)
    print(v[["planet", "status", "strength", "v_esc_kms", "cumulative_xuv_earth", "shoreline_margin_primary_dex", "p_retain_primary",
             "shoreline_margin_empirical_dex", "p_retain_empirical", "p_retain_bolometric"]].round(2).sort_values("shoreline_margin_primary_dex").to_string(index=False))
    print(ss.to_string(index=False))
    print(json.dumps(summary, indent=2))
    make_figure(v, ss)
    return 0


def make_figure(v: pd.DataFrame, ss: pd.DataFrame):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stage_c = pd.read_csv(OUT / "final_priority_table.csv") if (OUT / "final_priority_table.csv").is_file() else None
    fig, ax = plt.subplots(figsize=(10, 7))
    vg = np.geomspace(1, 40, 100)
    ax.fill_between(vg, 7e-4 * vg ** 4, 1e-3 * vg ** 4, color="#8f5bd6", alpha=0.25, label="shoreline I_XUV = I0 v^4 (I0 7e-4 to 1e-3)")
    ax.plot(vg, 10 ** (cs.ECS_INTERCEPT + cs.ECS_SLOPE * np.log10(vg)), color="#8f5bd6", ls="--", lw=1,
            label="empirical shoreline (slope 5.89, arXiv:2508.12865)")
    for _, r in ss.iterrows():
        ax.scatter(r.v_esc, r.insolation, marker="s", s=40, color="#2e933c" if r.has_atmosphere else "0.55", zorder=3)
        ax.annotate(r.body, (r.v_esc, r.insolation), fontsize=7, xytext=(3, 3), textcoords="offset points", color="0.3")
    # validation planets: numbered markers with a key (their names collide in the hot cluster)
    vv = v.sort_values("cumulative_xuv_earth", ascending=False).reset_index(drop=True)
    key = []
    for status, c, m in [("no_thick_atmosphere", "#d1495b", "o"), ("atmosphere_evidence", "#2a6fdb", "D")]:
        d = vv[vv.status == status]
        ax.scatter(d.v_esc_kms, d.cumulative_xuv_earth, color=c, marker=m, s=60, edgecolor="k", linewidth=0.4, zorder=4,
                   label={"no_thick_atmosphere": "JWST/Spitzer: no thick atmosphere",
                          "atmosphere_evidence": "atmosphere evidence (55 Cnc e moderate; others tentative)"}[status])
    for i, r in vv.iterrows():
        ax.annotate(str(i + 1), (r.v_esc_kms, r.cumulative_xuv_earth), fontsize=5.5, ha="center", va="center",
                    color="white", fontweight="bold", zorder=6)
        key.append(f"{i + 1:>2} {r.planet}" + ("" if r.status == "no_thick_atmosphere" else f" ({r.strength})"))
    ax.text(1.02, 0.99, "validation planets\n" + "\n".join(key), transform=ax.transAxes, va="top", ha="left",
            fontsize=7, family="monospace")
    if stage_c is not None:
        top = stage_c[stage_c.eligible & (stage_c.p_hz_optimistic >= 0.5)]
        ax.scatter(top.v_esc_kms, top.cumulative_xuv_earth, marker="*", s=110, color="#f2a541", edgecolor="k", linewidth=0.4,
                   zorder=5, label="Stage C habitable-zone targets")
        for j, (_, r) in enumerate(top.sort_values("cumulative_xuv_earth").iterrows()):
            dx, dy = [(-6, -2), (6, 2)][j % 2]
            ax.annotate(r.pl_name, (r.v_esc_kms, r.cumulative_xuv_earth), fontsize=6.5, xytext=(dx, dy), textcoords="offset points",
                        ha="right" if dx < 0 else "left", color="#a0522d")
    ax.set(xscale="log", yscale="log", xlabel="escape velocity (km/s)",
           ylabel="cumulative XUV fluence relative to Earth (Zahnle & Catling 2017 eq. 27)",
           title="Cosmic shoreline: bodies below the band can keep an atmosphere", xlim=(1, 40), ylim=(3e-3, 3e4))
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(RESEARCH / "eda" / "cosmic_shoreline.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())
