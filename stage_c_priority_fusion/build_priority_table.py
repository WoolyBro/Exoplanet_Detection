"""
build_priority_table.py
=======================

Stage C: fuse Stage A (transit vetting) and Stage B (atmospheres) into the 54-planet
priority table, replacing the template's placeholders with real model output.

The guide's instruction:
    "Current values are template placeholders (confirmed planets -> 0.99; gas scores from
     literature labels; priority from documented heuristic rules). Your job: replace each
     score column with a Stage A/B model output, calibrate probabilities, define `priority`
     from explicit science weights, and ablate each contributor. Keep the `notes` column
     honest about what is measured vs modelled."

The hard rule here: **a cell gets a number only if a model actually produced it.** Where no
model can speak, the cell is left empty and a provenance string says why. The template ships
transit_ML_probability = 0.99 for all 54 planets, which is not a prediction - it is "this
planet is confirmed". Copying that forward would be the single most misleading thing this
table could do.

Coverage, measured not assumed (see the report for the consequences):
  * transit model : 2 of 54 planets have a light curve in any built split, and only ONE
                    (Kepler-62 f, test split) is a genuine held-out prediction. Kepler-296 f
                    is in train, so its score is in-sample and is labelled as such.
                    The other 52 are WASP/HD/HAT-P planets that Kepler never observed.
  * gas model     : 13 of 54 have JWST spectra in the pack. Their scores come from Stage B's
                    leave-one-planet-out predictions, so they are out-of-fold, not fitted.
  * O3            : zero positive examples anywhere. Not modellable. Left empty everywhere.

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
sys.path.insert(0, str(RESEARCH_DIR))

def _load_level2():
    """Import stage_b_level2 by file path, since stage_b_atmospheres/ is not a package."""
    import importlib.util

    path = RESEARCH_DIR / "stage_b_atmospheres" / "stage_b_level2.py"
    spec = importlib.util.spec_from_file_location("stage_b_level2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_LEVEL2 = _load_level2()
planet_parameters = _LEVEL2.planet_parameters
MU_H2HE, MU_SECONDARY = _LEVEL2.MU_H2HE, _LEVEL2.MU_SECONDARY

# Which mean molecular weight applies to which planet. Below the radius valley (~1.8 Re,
# Fulton+2017) a planet has almost certainly lost any H2/He envelope, so its atmosphere - if it
# has one - is high-mu and MU_SECONDARY (N2, Earth-like) is the right assumption; that is the
# value stage_b_rocky_benchmark uses for exactly these planets. Above it an H2/He envelope is
# expected and MU_H2HE applies. Using 2.3 for everything overstated one scale height by 12.2x
# for the 21 rocky planets here, inflating `observable` for the very planets `small` already
# favours. The threshold is an assumption: amplitude_1H_ppm_mu2p3 and _mu28 are both carried in
# the output, so the effect of moving it is one column away.
RADIUS_VALLEY_RE = 1.8


def assign_mu(rade_earth):
    return np.where(np.asarray(rade_earth, float) < RADIUS_VALLEY_RE, MU_SECONDARY, MU_H2HE)

TEMPLATE = (RESEARCH_DIR / "exoplanet_research_data" / "05_final_ML_dataset_DO_NOT_USE"
            / "final_priority_table_TEMPLATE.csv")
STAGE_B = RESEARCH_DIR / "outputs" / "stage_b"
OUT_DIR = RESEARCH_DIR / "outputs" / "stage_c"
EDA_DIR = RESEARCH_DIR / "eda"
VIEWS = RESEARCH_DIR / "detection_views"
RUN = RESEARCH_DIR / "stage_a_transit_model" / "runs" / "bn_aug_sched"

MOLECULES = ["H2O", "CO2", "CH4", "O3"]

# Explicit science weights. Stated here, in one place, so they can be argued with - which is
# the point of the ablation below. "small + temperate + features + observable = High".
WEIGHTS = {
    "small": 0.25,        # smaller planets are the interesting ones for rocky/habitable work
    "temperate": 0.25,    # equilibrium temperature near ~300 K
    "observable": 0.30,   # can the atmosphere actually be measured (scale-height amplitude)
    "evidence": 0.20,     # molecules actually detected by the Stage B model
}


# =========================================================================== #
# Component scores (each 0-1, each from a measured quantity)
# =========================================================================== #
def score_small(rade: pd.Series) -> pd.Series:
    """1 at Earth size, falling smoothly to 0 by Jupiter size (log scale)."""
    x = np.log10(rade.clip(lower=0.3))
    return ((np.log10(11.2) - x) / (np.log10(11.2) - np.log10(1.0))).clip(0, 1)


def score_temperate(teq: pd.Series, centre: float = 300.0, width: float = 400.0) -> pd.Series:
    """Gaussian-ish peak at ~300 K. Not a habitability claim - a prioritisation preference."""
    return pd.Series(np.exp(-0.5 * ((teq - centre) / width) ** 2), index=teq.index)


def score_observable(amp_ppm: pd.Series) -> pd.Series:
    """Scale-height feature amplitude, log-scaled. 10 ppm -> 0, 1000 ppm -> 1."""
    x = np.log10(amp_ppm.clip(lower=1.0))
    return ((x - 1.0) / (3.0 - 1.0)).clip(0, 1)


# =========================================================================== #
# Stage A: transit probability from the trained CNN, where a light curve exists
# =========================================================================== #
def transit_scores() -> pd.DataFrame:
    """Planet-like probability from the Stage A model, only for planets with a built view."""
    import torch

    from stage_a_transit_model.cnn_lstm import PLANET_LIKE_IDX, consolidate_views, drop_unusable
    from stage_a_transit_model.failure_analysis import load_run_model, score_views

    koi = pd.read_csv(RESEARCH_DIR / "splits" / "koi_cumulative_split.csv", low_memory=False)
    named = koi[koi.kepler_name.notna()][["kepler_name", "kepoi_name", "split"]]

    model, _ = load_run_model(RUN)
    rows = []
    for split in ("train", "val", "test"):
        d = VIEWS / f"kepler_{split}"
        if not d.is_dir() or not any(d.glob("*.npz")):
            continue
        vs = drop_unusable(consolidate_views(d, verbose=False))
        proba = score_views(model, vs)
        planet_like = proba[:, PLANET_LIKE_IDX].sum(axis=1)
        sub = named[named.split == split]
        lookup = dict(zip(sub.kepoi_name.astype(str), sub.kepler_name.astype(str)))
        for oid, p in zip(vs.object_id, planet_like):
            if oid in lookup:
                rows.append({"pl_name": lookup[oid], "kepoi_name": oid, "split": split,
                             "transit_ML_probability": float(p)})
    return pd.DataFrame(rows)


# =========================================================================== #
# Stage B: out-of-fold molecule probabilities
# =========================================================================== #
def gas_scores() -> pd.DataFrame:
    """Per-planet molecule probability from Stage B's leave-one-planet-out predictions.

    Out-of-fold by construction: each planet's score comes from a model that never saw it.
    Averaged over that planet's spectra.
    """
    lp = pd.read_csv(STAGE_B / "stage_b_lopo_predictions.csv")
    # Only rows Stage B could actually evaluate. A spectrum whose coverage does not reach a
    # molecule's band carries label_known = False there; Stage B masks those out of its own loss
    # and metrics, so importing their predictions would put a number in a cell Stage B declined
    # to score. Without this filter 11 of 39 cells came from such rows, and for WASP-17 b and
    # WASP-80 b the whole evidence term did.
    n_all = len(lp)
    lp = lp[lp.label_known.astype(bool)]
    print(f"  using {len(lp)} of {n_all} LOPO rows ({n_all - len(lp)} dropped: label masked out "
          f"as unobservable in that spectrum)")
    agg = (lp.groupby(["planet", "molecule"]).gbm_prob.mean().unstack("molecule"))
    agg.columns = [f"{c}_score" for c in agg.columns]
    return agg.reset_index().rename(columns={"planet": "pl_name"})


# =========================================================================== #
# Priority
# =========================================================================== #
def priority_score(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Weighted sum over whichever components exist, renormalised, plus which were used.

    Renormalising matters: 41 of 54 planets have no spectral evidence term, and scoring them
    as if evidence were zero would push every un-observed planet to the bottom for the sole
    reason that nobody has pointed JWST at it yet. That is a statement about the observing
    log, not about the planet.
    """
    comp = pd.DataFrame({
        "small": df["c_small"], "temperate": df["c_temperate"],
        "observable": df["c_observable"], "evidence": df["c_evidence"],
    })
    w = pd.Series(WEIGHTS)
    avail = comp.notna()
    wsum = (avail * w).sum(axis=1)
    score = (comp.fillna(0) * w).sum(axis=1) / wsum.replace(0, np.nan)
    used = avail.apply(lambda r: "+".join(sorted(c for c in comp.columns if r[c])), axis=1)
    return score, used


def band(score: pd.Series) -> pd.Series:
    """Terciles of the computed score. Bands are relative to this sample of 54, nothing more."""
    q1, q2 = score.quantile([1 / 3, 2 / 3])
    return pd.cut(score, [-np.inf, q1, q2, np.inf], labels=["Low", "Medium", "High"])


def ablate(df: pd.DataFrame, full: pd.Series) -> pd.DataFrame:
    """Drop each contributor in turn; report how much the ranking moves."""
    from scipy.stats import spearmanr

    rows = []
    full_band = band(full)
    for drop in WEIGHTS:
        comp = pd.DataFrame({k: df[f"c_{k}"] for k in WEIGHTS if k != drop})
        w = pd.Series({k: v for k, v in WEIGHTS.items() if k != drop})
        avail = comp.notna()
        s = (comp.fillna(0) * w).sum(axis=1) / (avail * w).sum(axis=1).replace(0, np.nan)
        both = full.notna() & s.notna()
        rho = spearmanr(full[both], s[both]).statistic if both.sum() > 2 else np.nan
        moved = int((band(s)[both].astype(str) != full_band[both].astype(str)).sum())
        rows.append({"dropped": drop, "weight": WEIGHTS[drop], "spearman_vs_full": rho,
                     "planets_changing_band": moved, "n_scored": int(both.sum())})
    return pd.DataFrame(rows).sort_values("spearman_vs_full")


# =========================================================================== #
# Figure
# =========================================================================== #
def make_figure(final: pd.DataFrame, abl: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    colours = {"High": "#3b7a57", "Medium": "#c8902a", "Low": "#999999"}

    # 1: top 18 by score, stacked by contribution
    ax = axes[0, 0]
    top = final.head(18).iloc[::-1]
    y = np.arange(len(top))
    left = np.zeros(len(top))
    for comp, col in [("small", "#2d6fa8"), ("temperate", "#c23b22"),
                      ("observable", "#3b7a57"), ("evidence", "#8a5fa8")]:
        vals = (top[f"c_{comp}"].fillna(0) * WEIGHTS[comp]).to_numpy()
        ax.barh(y, vals, left=left, color=col, label=comp, height=0.75)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(top.pl_name, fontsize=7)
    ax.set_xlabel("weighted contribution to priority score")
    ax.set_title("Top 18 targets, by what earns them the score", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    # 2: coverage - which cells have a model behind them
    ax = axes[0, 1]
    cov = pd.DataFrame({
        "transit": final.transit_ML_probability.notna(),
        "H2O": final.H2O_score.notna(), "CO2": final.CO2_score.notna(),
        "CH4": final.CH4_score.notna(), "O3": final.O3_score.notna(),
    }).astype(int)
    ax.imshow(cov.T.to_numpy(), aspect="auto", cmap="Greens", vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_yticks(range(len(cov.columns)))
    ax.set_yticklabels(cov.columns, fontsize=8)
    ax.set_xlabel("planet (ordered by priority)")
    ax.set_title("Which cells have a MODEL behind them\n"
                 "(green = model output, white = left empty, not guessed)", fontsize=9)
    for i, c in enumerate(cov.columns):
        ax.text(len(cov) + 0.5, i, f"{cov[c].sum()}/{len(cov)}", va="center", fontsize=8)
    ax.set_xlim(-0.5, len(cov) + 4)

    # 3: ablation
    ax = axes[1, 0]
    a = abl.sort_values("spearman_vs_full")
    ax.barh(range(len(a)), a.spearman_vs_full, color="#2d6fa8", edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(a)))
    ax.set_yticklabels([f"{r.dropped} (w={r.weight:.2f})" for _, r in a.iterrows()], fontsize=8)
    # Data-driven, not a fixed window: after the mu fix the weakest ablation fell to 0.75, and a
    # hardcoded lower bound of 0.85 silently drew the two most important bars off the axis.
    lo = min(0.0 if a.spearman_vs_full.isna().all() else a.spearman_vs_full.min(), 0.95)
    ax.set_xlim(max(0.0, lo - 0.06), 1.0)
    ax.set_xlabel("Spearman vs the full ranking when this contributor is dropped")
    ax.set_title("Ablation: what the ranking actually depends on\n(lower = matters more)", fontsize=9)
    for i, (_, r) in enumerate(a.iterrows()):
        ax.text(r.spearman_vs_full - 0.002, i, f"{int(r.planets_changing_band)} change band",
                va="center", ha="right", fontsize=7, color="white")
    ax.grid(axis="x", alpha=0.3)

    # 4: the observability physics behind the ranking
    ax = axes[1, 1]
    for b, c in colours.items():
        m = final.priority.astype(str) == b
        ax.scatter(final.loc[m, "pl_eqt"], final.loc[m, "amplitude_1H_ppm"],
                   s=25 + 90 * final.loc[m, "c_small"].fillna(0), c=c, label=b,
                   edgecolor="black", linewidth=0.4, alpha=0.85)
    for _, r in final.head(6).iterrows():
        ax.annotate(r.pl_name, (r.pl_eqt, r.amplitude_1H_ppm), fontsize=6.5,
                    textcoords="offset points", xytext=(4, 3))
    ax.set_yscale("log")
    ax.set_xlabel("equilibrium temperature (K)")
    ax.set_ylabel("one scale height of absorption (ppm)")
    ax.set_title("Why those targets: observability vs temperature\n"
                 "(marker size = smallness score)", fontsize=9)
    ax.legend(fontsize=7, title="priority", title_fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle("Stage C: priority fusion over 54 planets - Stage A transit model + Stage B "
                 "atmosphere model + explicit science weights", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    out = EDA_DIR / "stage_c_priority_overview.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n[7] figure -> {out.relative_to(RESEARCH_DIR)}")


# =========================================================================== #
# Main
# =========================================================================== #
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 88)
    print("Stage C: priority fusion. A cell gets a number only if a model produced it.")
    print("=" * 88)

    tpl = pd.read_csv(TEMPLATE)
    out = tpl[["pl_name"]].copy()
    print(f"\n[1] template: {len(tpl)} planets")
    print(f"  template transit_ML_probABILITY: {tpl['transit_ML_probABILITY'].nunique()} unique value(s) "
          f"= {list(tpl['transit_ML_probABILITY'].unique())}  <- a placeholder, not a prediction")

    # ---- physical parameters -------------------------------------------- #
    params = planet_parameters(sorted(tpl.pl_name.unique()), mu=assign_mu)
    out = out.merge(params[["pl_name", "pl_eqt", "pl_rade", "st_rad", "st_teff",
                            "gravity_ms2", "mu_assumed", "scale_height_km", "amplitude_1H_ppm",
                            "amplitude_1H_ppm_mu2p3", "amplitude_1H_ppm_mu28"]],
                    on="pl_name", how="left")
    print(f"\n[2] physical parameters found for {out.pl_eqt.notna().sum()}/{len(out)} planets")
    n_rocky = int((out.mu_assumed == MU_SECONDARY).sum())
    print(f"  mu={MU_SECONDARY:g} (N2, secondary) for the {n_rocky} planets below "
          f"{RADIUS_VALLEY_RE} Re; mu={MU_H2HE:g} (H2/He) for the other {len(out) - n_rocky}")
    print(f"  one scale height of absorption spans {out.amplitude_1H_ppm.min():.2f}-"
          f"{out.amplitude_1H_ppm.max():.0f} ppm; at a single mu=2.3 it would read "
          f"{out.amplitude_1H_ppm_mu2p3.min():.2f}-{out.amplitude_1H_ppm_mu2p3.max():.0f} ppm")

    # ---- Stage A --------------------------------------------------------- #
    print("\n[3] Stage A transit probability (CNN+LSTM, run bn_aug_sched)")
    ts = transit_scores()
    ts = ts[ts.pl_name.isin(out.pl_name)]
    out = out.merge(ts[["pl_name", "transit_ML_probability", "split", "kepoi_name"]],
                    on="pl_name", how="left")
    out["transit_provenance"] = np.where(
        out.transit_ML_probability.isna(),
        "no light curve in any built split (Kepler never observed this target)",
        "model, " + out.split.astype(str) + " split"
        + np.where(out.split.eq("train"), " -> IN-SAMPLE, not a fair score", " -> held out"))
    n_model = int(out.transit_ML_probability.notna().sum())
    print(f"  scored by the model: {n_model}/{len(out)}")
    for _, r in out[out.transit_ML_probability.notna()].iterrows():
        print(f"    {r.pl_name:<15} {r.kepoi_name:<10} p={r.transit_ML_probability:.4f}  "
              f"({r.transit_provenance})")
    print(f"  the other {len(out) - n_model} keep an EMPTY cell, not the template's 0.99")

    # ---- TESS light-curve scores, from both models ----------------------- #
    # These are the real transit probabilities for the 54. Both models are carried because they
    # have OPPOSITE, measured biases (see the report): the Kepler model's score falls with
    # planet radius (Spearman -0.570), the TESS model's rises with it (+0.302). Reporting only
    # one would hide a known systematic.
    tess_path = OUT_DIR / "priority_target_scores_TESS.csv"
    kep_path = OUT_DIR / "priority_target_scores_KEPLER.csv"
    if tess_path.is_file():
        t = pd.read_csv(tess_path)[["pl_name", "transit_ML_probability_TESS"]]
        out = out.merge(t, on="pl_name", how="left")
    else:
        out["transit_ML_probability_TESS"] = np.nan
    if kep_path.is_file():
        k = (pd.read_csv(kep_path)[["pl_name", "transit_ML_probability_TESS_crossmission"]]
             .rename(columns={"transit_ML_probability_TESS_crossmission":
                              "transit_ML_probability_kepler_model"}))
        out = out.merge(k, on="pl_name", how="left")
    else:
        out["transit_ML_probability_kepler_model"] = np.nan

    n_t = int(out.transit_ML_probability_TESS.notna().sum())
    print(f"\n[3b] TESS light-curve scores for {n_t}/{len(out)} planets")
    print("  PRIMARY column is the TESS-trained model: these ARE TESS light curves, so it is the")
    print("  same-mission model, and the Kepler one is demonstrably wrong here - it rejected")
    print("  KELT-9 b (0.06), HD 189733 b (0.12) and the other canonical hot Jupiters, because")
    print("  93% of Kepler KOIs deeper than 10,000 ppm are false positives. In the TOI split that")
    print("  figure is 11.5%, below its own base rate.")
    print("  The Kepler-model column is kept alongside because the two biases are OPPOSITE:")
    print("  on small planets (<4 Re) the Kepler model scores higher (0.786 vs 0.633); on giants")
    print("  (>=8 Re) the TESS model does (0.741 vs 0.511). Neither is unbiased.")

    # ---- Stage B --------------------------------------------------------- #
    print("\n[4] Stage B molecule scores (leave-one-planet-out, out-of-fold)")
    gs = gas_scores()
    keep = ["pl_name"] + [f"{m}_score" for m in MOLECULES if f"{m}_score" in gs.columns]
    out = out.merge(gs[keep], on="pl_name", how="left")
    for m in MOLECULES:
        col = f"{m}_score"
        if col not in out.columns:
            out[col] = np.nan
        n = int(out[col].notna().sum())
        why = ("zero positive examples anywhere in the label set - not modellable"
               if m == "O3" else "no JWST spectrum in the pack")
        print(f"  {m:<4} scored for {n:>2}/{len(out)} planets"
              + (f"   (the rest: {why})" if n < len(out) else ""))
    out["gas_provenance"] = np.where(out[[f"{m}_score" for m in MOLECULES]].notna().any(axis=1),
                                     "Stage B LOPO (out-of-fold), mean over that planet's spectra; "
                                     "NO SKILL above a constant majority prediction at planet level "
                                     "(see eda/STAGE_B_RESULTS.md) - rank signal, not a detection",
                                     "no spectra in the pack")
    # Where the model's score disagrees with the verified literature label, say so in the table.
    # TRAPPIST-1 c is the case that matters: H2O = 0 in labels/verified_gas_labels.csv (a 2-sigma
    # non-detection) against a model score of 0.93, and it ranks near the top.
    labels = pd.read_csv(RESEARCH_DIR / "outputs" / "labels" / "verified_gas_labels.csv",
                         dtype=str, keep_default_na=False).set_index("planet")
    clashes = []
    for i, r in out.iterrows():
        for m in ("H2O", "CO2", "CH4"):
            score = r.get(f"{m}_score")
            if pd.isna(score) or r.pl_name not in labels.index:
                continue
            verified = labels.at[r.pl_name, m] if m in labels.columns else "unknown"
            if verified == "0" and score >= 0.5:
                clashes.append(f"{m} scored {score:.2f} vs verified non-detection")
        out.at[i, "score_vs_verified_label"] = "; ".join(clashes) if clashes else ""
        clashes = []
    n_clash = int((out.score_vs_verified_label != "").sum())
    print(f"  {n_clash} planet(s) carry a score >= 0.5 for a molecule their verified label records "
          f"as a non-detection; flagged in score_vs_verified_label")

    # ---- components + priority ------------------------------------------- #
    out["c_small"] = score_small(out.pl_rade)
    out["c_temperate"] = score_temperate(out.pl_eqt)
    out["c_observable"] = score_observable(out.amplitude_1H_ppm)
    detected = out[[f"{m}_score" for m in ("H2O", "CO2", "CH4")]]
    out["c_evidence"] = detected.max(axis=1)  # strongest molecular evidence available

    out["priority_score"], out["components_used"] = priority_score(out)
    out["priority"] = band(out.priority_score)
    print("\n[5] priority from explicit weights")
    for k, v in WEIGHTS.items():
        print(f"    {k:<11} {v:.2f}")
    print("  weights are renormalised over the components a planet actually has, so a planet")
    print("  is not penalised for never having been observed")
    print(f"\n  {'planet':<16} {'score':>6} {'band':>7} {'Rp':>6} {'Teq':>6} {'1H ppm':>8}  components")
    for _, r in out.sort_values("priority_score", ascending=False).head(12).iterrows():
        print(f"  {r.pl_name:<16} {r.priority_score:>6.3f} {str(r.priority):>7} {r.pl_rade:>6.2f} "
              f"{r.pl_eqt:>6.0f} {r.amplitude_1H_ppm:>8.1f}  {r.components_used}")

    # ---- ablation -------------------------------------------------------- #
    abl = ablate(out, out.priority_score)
    print("\n[6] ablation: drop each contributor, see how far the ranking moves")
    print(f"  {'dropped':<11} {'weight':>7} {'Spearman vs full':>17} {'planets changing band':>22}")
    for _, r in abl.iterrows():
        print(f"  {r.dropped:<11} {r.weight:>7.2f} {r.spearman_vs_full:>17.3f} "
              f"{r.planets_changing_band:>22}")
    print("  lowest Spearman = the contributor the ranking depends on most")

    # ---- notes + save ---------------------------------------------------- #
    out["notes"] = out.apply(
        lambda r: "; ".join([
            f"transit: {r.transit_provenance}",
            f"gas: {r.gas_provenance}",
            "O3: not modellable (no positive examples)",
            f"priority from {r.components_used}",
        ]), axis=1)

    cols = ["pl_name", "transit_ML_probability_TESS", "transit_ML_probability_kepler_model",
            "transit_ML_probability", "transit_provenance",
            "H2O_score", "CO2_score", "CH4_score", "O3_score", "gas_provenance",
            "score_vs_verified_label",
            "pl_rade", "pl_eqt", "st_teff", "st_rad",
            "mu_assumed", "amplitude_1H_ppm", "amplitude_1H_ppm_mu2p3", "amplitude_1H_ppm_mu28",
            "c_small", "c_temperate", "c_observable", "c_evidence",
            "priority_score", "priority", "components_used", "notes"]
    final = out[cols].sort_values("priority_score", ascending=False).reset_index(drop=True)
    final.to_csv(OUT_DIR / "final_priority_table.csv", index=False)
    abl.to_csv(OUT_DIR / "ablation.csv", index=False)
    make_figure(final, abl)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_planets": int(len(final)),
        "weights": WEIGHTS,
        "transit_scored_by_model": n_model,
        "transit_held_out": int((out.split == "test").sum()),
        "transit_in_sample": int((out.split == "train").sum()),
        "gas_scored_by_model": int(out[[f"{m}_score" for m in ("H2O", "CO2", "CH4")]]
                                   .notna().any(axis=1).sum()),
        "o3_scored": 0,
        "band_counts": {str(k): int(v) for k, v in final.priority.value_counts().items()},
        "ablation": abl.to_dict("records"),
        "integrity_note": ("The template shipped transit_ML_probability = 0.99 for all 54 planets, "
                           "which encodes 'confirmed', not a prediction. Cells with no model behind "
                           "them are left empty here rather than carried forward."),
    }
    (OUT_DIR / "stage_c_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  band counts: {dict(final.priority.value_counts())}")
    print(f"\nsaved {OUT_DIR.relative_to(RESEARCH_DIR)}/ "
          f"(final_priority_table.csv, ablation.csv, stage_c_summary.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
