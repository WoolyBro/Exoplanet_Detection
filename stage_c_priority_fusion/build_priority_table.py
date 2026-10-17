"""
build_priority_table.py
=======================

Stage C: rank the 54 target planets for TRANSMISSION-SPECTROSCOPY follow-up of potentially temperate,
terrestrial-size worlds. This is a prioritisation with stated weights - not a habitability
probability, and not a measure of whether any planet is habitable.

Rebuilt 2026-10-17 after the audit (see run_log.txt). The previous version
  * scored "temperature" with a 400 K-wide Gaussian on catalogue T_eq values drawn from 45 papers
    with different albedo/redistribution conventions (a 600 K planet scored 0.76);
  * gave 20 % of the weight to Stage B molecule probabilities - a model Stage B shows has no skill;
  * scored transmission observability for 10 planets that DO NOT TRANSIT (radial-velocity and
    directly imaged planets), one of which ranked 11th;
  * read its planet list from the excluded 05_final_ML_dataset_DO_NOT_USE folder.

Now every component is a published, physical definition (target_physics.py), with measurement
uncertainty propagated by Monte Carlo:
  hz           0.40  P(conservative HZ) + 0.5 x P(optimistic-only HZ)      Kopparapu et al. 2014
  terrestrial  0.30  P(R_p < 1.5 R_earth)                                   Kempton et al. 2018 bins
  observable   0.30  TSM / Kempton follow-up cut, capped at 1; 0 if the planet does not transit
Changed 2026-10-18: "terrestrial" is now "terrestrial_atm" = P(R_p < 1.5 R_earth AND the planet lies on
the atmosphere-retaining side of the cosmic shoreline) (Zahnle & Catling 2017; cosmic_shoreline.py).
Weights unchanged. Before entering the score, the shoreline was checked against JWST/Spitzer rocky
planets: all 13 with no thick atmosphere lie on its airless side (validate_shoreline.py). It is
combined with size rather than added as a fourth term, because a giant planet trivially "retains"
an H/He envelope. A separate term put hot Jupiters in the top 15.
The weights are a stated preference; the ablation shows how much each one drives the order. The
score is computed for every Monte-Carlo draw, so each planet gets a rank interval, not just a rank.

Reported alongside, never inside the score:
  * the vetting-model scores of the TESS light curves (these are CONFIRMED planets, so the scores
    are a sanity check on the vetter, not evidence about the planet);
  * literature molecule detections and non-detections from outputs/labels/verified_gas_labels.csv.

Usage
    python stage_c_priority_fusion/build_priority_table.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RESEARCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import target_physics as tp  # noqa: E402

TARGETS = Path(__file__).resolve().parent / "target_list.csv"   # the 54 names (from the mentor's template)
_PACK = RESEARCH_DIR / "exoplanet_research_data" / "01_candidate_catalogs" / "pscomppars_confirmed_planets.csv"
LABELS = RESEARCH_DIR / "outputs" / "labels" / "verified_gas_labels.csv"
OUT_DIR = RESEARCH_DIR / "outputs" / "stage_c"
EDA_DIR = RESEARCH_DIR / "eda"
MOLECULES = ["H2O", "CO2", "CH4", "CO", "SO2"]

WEIGHTS = {"hz": 0.40, "terrestrial_atm": 0.30, "observable": 0.30}
WEIGHTS_WITHOUT_RETENTION = {"hz": 0.40, "terrestrial": 0.30, "observable": 0.30}     # the 2026-10-17 ranking
N_DRAWS = 4000


def score_draws(draws: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Weighted score for every planet x Monte-Carlo draw."""
    total = sum(weights.values())
    return sum(w * draws[k] for k, w in weights.items()) / total


def ranks(scores: np.ndarray) -> np.ndarray:
    """Rank 1 = highest score, per draw. Ties SHARE the better rank ("min" method): three planets tied
    at the top are all rank 1, never 1/2/3 by row order - that would fabricate a precise ordering."""
    from scipy.stats import rankdata

    return rankdata(-scores, axis=0, method="min")


def literature_molecules(names: pd.Series) -> pd.DataFrame:
    lab = pd.read_csv(LABELS, dtype=str, keep_default_na=False).set_index("planet")
    rows = []
    for n in names:
        if n not in lab.index:
            rows.append({"pl_name": n, "lit_detected": "", "lit_not_detected": ""})
            continue
        r = lab.loc[n]
        rows.append({"pl_name": n, "lit_detected": " ".join(m for m in MOLECULES if r[m] == "1"),
                     "lit_not_detected": " ".join(m for m in MOLECULES if r[m] == "0")})
    return pd.DataFrame(rows)


def vetting_scores() -> pd.DataFrame:
    """Planet-like scores of the TESS light curves from both vetting models (informational)."""
    out = None
    for tag, col in [("TESS", "transit_ML_probability_TESS"), ("KEPLER", "transit_ML_probability_TESS_crossmission")]:
        f = OUT_DIR / f"priority_target_scores_{tag}.csv"
        if f.is_file():
            d = pd.read_csv(f)[["pl_name", col]].rename(columns={col: f"vetting_score_{tag.lower()}_model"})
            out = d if out is None else out.merge(d, on="pl_name", how="outer")
    return out if out is not None else pd.DataFrame(columns=["pl_name"])


def make_figure(final: pd.DataFrame, abl: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4))

    # 1: the HZ diagram - incident flux vs stellar temperature, Kopparapu boundaries
    ax = axes[0]
    t = np.linspace(2600, 7200, 200)
    lim = tp.hz_flux_limits(t)
    ax.fill_betweenx(t, lim["early_mars"], lim["recent_venus"], color="#cfe5d6", label="optimistic HZ")
    ax.fill_betweenx(t, lim["maximum_greenhouse"], lim["runaway_greenhouse"], color="#7fbf95", label="conservative HZ")
    tr = final.transiting
    ax.scatter(final.insol_earth[tr], final.st_teff[tr], s=28, c="#2a78d6", edgecolor="black", lw=0.4, label="transiting", zorder=3)
    ax.scatter(final.insol_earth[~tr], final.st_teff[~tr], s=34, marker="^", facecolor="white",
               edgecolor="#555", lw=0.8, label="not transiting", zorder=3)
    for _, r in final[final.p_hz_optimistic > 0.5].iterrows():
        ax.annotate(r.pl_name, (r.insol_earth, r.st_teff), fontsize=6.5, xytext=(4, 2), textcoords="offset points")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set(xlabel="incident flux (Earth = 1)", ylabel="stellar effective temperature (K)",
           title="Habitable-zone membership (Kopparapu et al. 2014)")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)

    # 2: top 15 with rank intervals
    ax = axes[1]
    top = final[final.eligible].head(15).iloc[::-1]
    y = np.arange(len(top))
    left = np.zeros(len(top))
    for comp, col in [("hz", "#3b7a57"), ("terrestrial_atm", "#c8902a"), ("observable", "#2d6fa8")]:
        vals = top[f"c_{comp}"].to_numpy() * WEIGHTS[comp] / sum(WEIGHTS.values())
        ax.barh(y, vals, left=left, color=col, height=0.72, label=comp)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.pl_name}  (rank {int(r.rank_median)}, {int(r.rank_p5)}-{int(r.rank_p95)})"
                        for _, r in top.iterrows()], fontsize=7)
    ax.set_xlabel("weighted contribution to the priority score")
    ax.set_title("Top 15: what earns the score, with the 90 % rank interval", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    # 3: ablation
    ax = axes[2]
    a = abl.sort_values("spearman_vs_full")
    ax.barh(range(len(a)), a.spearman_vs_full, color="#2d6fa8", edgecolor="black", lw=0.5)
    ax.set_yticks(range(len(a)))
    ax.set_yticklabels([f"drop {r.dropped} (w={r.weight:.2f})" for _, r in a.iterrows()], fontsize=8)
    ax.set_xlim(max(0.0, float(a.spearman_vs_full.min()) - 0.1), 1.0)
    ax.set_xlabel("Spearman correlation with the full ranking")
    ax.set_title("Ablation: lower = the ranking depends on it more", fontsize=10)
    for i, (_, r) in enumerate(a.iterrows()):
        ax.text(r.spearman_vs_full - 0.005, i, f"top-10 overlap {int(r.top10_overlap)}/10",
                va="center", ha="right", fontsize=7, color="white")
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Stage C: transmission-follow-up priority for potentially temperate terrestrial planets - "
                 "a ranking with stated weights, not a habitability probability", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(EDA_DIR / "stage_c_priority_overview.png", dpi=150)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 88)
    print("Stage C: transmission-follow-up priority (HZ membership, terrestrial size, TSM), with uncertainty")
    print("=" * 88)

    names = pd.read_csv(TARGETS).pl_name
    ps = pd.read_csv(_PACK, low_memory=False)
    cat = (ps[ps.pl_name.isin(names)].assign(_n=lambda d: d.notna().sum(axis=1))
           .sort_values("_n", ascending=False).drop_duplicates("pl_name").drop(columns="_n"))
    missing = sorted(set(names) - set(cat.pl_name))
    if missing:
        raise SystemExit(f"no catalogue row for {missing}")
    cat = cat.set_index("pl_name").loc[names].reset_index()
    print(f"\n[1] {len(cat)} planets; transiting {int((cat.tran_flag == 1).sum())}, "
          f"not transiting {int((cat.tran_flag != 1).sum())} ({', '.join(cat.loc[cat.tran_flag != 1, 'pl_name'])})")

    phys, draws = tp.monte_carlo(cat, n=N_DRAWS, return_draws=True)
    # Eligibility: transmission spectroscopy needs a transit. Non-transiting planets are listed with
    # their HZ physics but are not ranked - otherwise HZ membership plus a radius ESTIMATED from mass
    # would put radial-velocity planets above observable ones (GJ 1061 d and GJ 1002 b were 8th and
    # 9th in the first draft of this rebuild).
    eligible = phys.transiting.to_numpy()
    s_all = score_draws(draws, WEIGHTS)
    s = s_all[eligible]
    r_el = ranks(s)
    r = np.full(s_all.shape, np.nan)
    r[eligible] = r_el
    for k in list(WEIGHTS) + ["terrestrial", "retention"]:
        phys[f"c_{k}"] = draws[k].mean(axis=1)
    r_old = np.full(s_all.shape, np.nan)
    r_old[eligible] = ranks(score_draws(draws, WEIGHTS_WITHOUT_RETENTION)[eligible])
    with np.errstate(all="ignore"):
        phys["rank_without_retention"] = np.nanmedian(r_old, axis=1)
    phys["priority_score"] = np.where(eligible, np.median(s_all, axis=1), np.nan)
    phys["score_p5"] = np.where(eligible, np.percentile(s_all, 5, axis=1), np.nan)
    phys["score_p95"] = np.where(eligible, np.percentile(s_all, 95, axis=1), np.nan)
    with np.errstate(all="ignore"):
        phys["rank_median"] = np.nanmedian(r, axis=1)
        phys["rank_p5"], phys["rank_p95"] = np.nanpercentile(r, [5, 95], axis=1)
    phys["p_top10"] = np.where(eligible, (r <= 10).mean(axis=1), np.nan)
    phys["eligible"] = eligible

    out = (phys.merge(cat[["pl_name", "st_teff", "st_rad", "pl_rade", "pl_bmasse", "pl_bmassprov", "sy_jmag",
                           "discoverymethod"]], on="pl_name")
           .merge(literature_molecules(names), on="pl_name", how="left")
           .merge(vetting_scores(), on="pl_name", how="left"))
    final = out.sort_values(["priority_score", "rank_median"], ascending=[False, True],
                            na_position="last").reset_index(drop=True)
    print(f"  ranked: {int(final.eligible.sum())} transiting planets; listed, not ranked: {int((~final.eligible).sum())}")

    print("\n[2] weights:", WEIGHTS)
    print(f"\n  {'planet':<14} {'score':>6} {'rank (90%)':>12} {'old':>4} {'P(HZcons)':>9} {'P(HZopt)':>8} "
          f"{'P(R<1.5)':>8} {'TSM':>7} {'cut':>4} {'P(keep)':>7} {'margin':>6}")
    for _, row in final[final.eligible].head(15).iterrows():
        print(f"  {row.pl_name:<14} {row.priority_score:>6.3f} {int(row.rank_median):>4} ({int(row.rank_p5):>2}-"
              f"{int(row.rank_p95):>2}) {int(row.rank_without_retention):>4} {row.p_hz_conservative:>9.2f} {row.p_hz_optimistic:>8.2f} "
              f"{row.p_radius_lt_1p5:>8.2f} {row.tsm_median:>7.1f} {row.tsm_followup_cut:>4.0f} {row.p_retain_primary:>7.2f} "
              f"{row.shoreline_margin_primary_dex:>6.2f}")

    # ---- ablation ------------------------------------------------------------------------------ #
    from scipy.stats import spearmanr

    full = np.median(s, axis=1)
    top10 = set(np.argsort(-full)[:10])
    abl = []
    for drop in WEIGHTS:
        w = {k: v for k, v in WEIGHTS.items() if k != drop}
        alt = np.median(score_draws({k: v[eligible] for k, v in draws.items()}, w), axis=1)
        abl.append({"dropped": drop, "weight": WEIGHTS[drop],
                    "spearman_vs_full": float(spearmanr(full, alt).statistic),
                    "top10_overlap": len(top10 & set(np.argsort(-alt)[:10]))})
    alt = np.median(score_draws({k: v[eligible] for k, v in draws.items()}, WEIGHTS_WITHOUT_RETENTION), axis=1)
    abl.append({"dropped": "retention (terrestrial_atm -> terrestrial)", "weight": 0.0,
                "spearman_vs_full": float(spearmanr(full, alt).statistic),
                "top10_overlap": len(top10 & set(np.argsort(-alt)[:10]))})
    abl = pd.DataFrame(abl).sort_values("spearman_vs_full")
    print("\n[3] ablation")
    print(abl.round(3).to_string(index=False))

    cols = ["pl_name", "eligible", "priority_score", "score_p5", "score_p95", "rank_median", "rank_p5", "rank_p95", "p_top10",
            "c_hz", "c_terrestrial_atm", "c_observable", "c_terrestrial", "c_retention", "rank_without_retention",
            "p_retain_primary", "shoreline_margin_primary_dex", "p_retain_empirical", "p_retain_bolometric",
            "v_esc_kms", "cumulative_xuv_earth",
            "p_hz_conservative", "p_hz_optimistic", "insol_earth", "hz_inner_runaway", "hz_outer_maxgreen",
            "p_radius_lt_1p5", "pl_rade", "pl_bmasse", "pl_bmassprov", "teq_kempton_K",
            "transiting", "discoverymethod", "tsm_median", "tsm_p16", "tsm_p84", "tsm_followup_cut", "p_tsm_above_cut",
            "st_teff", "st_rad", "sy_jmag", "lit_detected", "lit_not_detected",
            "vetting_score_tess_model", "vetting_score_kepler_model", "physics_notes"]
    final = final[[c for c in cols if c in final.columns]]
    final.to_csv(OUT_DIR / "final_priority_table.csv", index=False)
    abl.to_csv(OUT_DIR / "ablation.csv", index=False)
    make_figure(final.assign(st_teff=final.st_teff), abl)
    (OUT_DIR / "stage_c_summary.json").write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_planets": int(len(final)), "n_transiting": int(final.transiting.sum()),
        "n_ranked": int(final.eligible.sum()),
        "weights": WEIGHTS, "monte_carlo_draws": N_DRAWS,
        "definitions": {
            "hz": "Kopparapu et al. 2014 (ApJL 787 L29), 1 M_earth coefficients; P(conservative) + 0.5 P(optimistic only)",
            "terrestrial_atm": "P(R_p < 1.5 R_earth [Kempton et al. 2018 bin] AND atmosphere side of the cosmic shoreline)",
            "observable": "TSM (Kempton et al. 2018, PASP, doi:10.1088/1538-3873/aadf6f) / follow-up cut, capped at 1; "
                          "0 for non-transiting planets",
            "retention_definition": "P(atmosphere side of the cosmic shoreline): I_XUV = I (L/L_sun)^-0.6 (Zahnle & Catling 2017 eq. 27) "
                         "vs I0 v_esc^4, log10 I0 ~ U(log10 7e-4, -3); XUV scaling scatter 0.48 dex; see validate_shoreline.py"},
        "weights_without_retention": WEIGHTS_WITHOUT_RETENTION,
        "n_in_conservative_hz_p50": int((final.p_hz_conservative >= 0.5).sum()),
        "n_in_optimistic_hz_p50": int((final.p_hz_optimistic >= 0.5).sum()),
        "ablation": abl.to_dict("records"),
        "note": ("A ranking with stated weights for transmission follow-up; not a probability of habitability. "
                 "The molecule model and the vetting scores are not inputs to the score."),
    }, indent=2), encoding="utf-8")
    print(f"\nsaved {OUT_DIR.relative_to(RESEARCH_DIR)}/ (final_priority_table.csv, ablation.csv, "
          f"stage_c_summary.json) and eda/stage_c_priority_overview.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
