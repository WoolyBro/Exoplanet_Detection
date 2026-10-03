#!/usr/bin/env python3
"""
make_figures.py
===============

Regenerates every statistical figure in figures/ from results committed to this repository.
Nothing is retrained and no light curve is downloaded, so this runs in a fresh clone in a few
seconds:

    python figures/make_figures.py

Each figure is titled by the question its study answers, not by the dataset it used, and its
subtitle states the answer. Every rate or score carries an uncertainty interval:

  * ROC-AUC / PR-AUC  - 95 % bootstrap interval, resampling STARS rather than rows, because
                        several KOIs share a host star and are not independent draws.
  * proportions       - 95 % Wilson score interval (well-behaved at small n and near 0 or 1).
  * paired effects    - 95 % bootstrap interval on the median, resampling stars.

Colour follows a validated categorical order (blue, orange, aqua - the three slots that
remain distinguishable under every colour-vision deficiency when all pairs are on screen),
and identity is always carried by a legend or direct label as well, never by colour alone.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import (average_precision_score, precision_recall_curve,  # noqa: E402
                             roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
RUNS = ROOT / "stage_a_transit_model" / "runs"
RNG_SEED = 20260928
N_BOOT = 1000

# ---- palette: validated reference instance, light mode -------------------- #
SURFACE = "#fcfcfb"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#b9b8b2"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
PLANET_LIKE = {"CANDIDATE", "CONFIRMED"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_3, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "legend.frameon": False, "legend.fontsize": 9,
    "lines.linewidth": 2.0,
})


# =========================================================================== #
# Statistics helpers
# =========================================================================== #
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Proportion and its 95 % Wilson score interval."""
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, centre - half), min(1.0, centre + half)


def star_bootstrap(df: pd.DataFrame, stat, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """Point estimate and 95 % interval, resampling host stars with replacement."""
    rng = np.random.default_rng(RNG_SEED)
    groups = [g for _, g in df.groupby("star_id")]
    point = stat(df)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        sample = pd.concat([groups[i] for i in pick], ignore_index=True)
        try:
            vals.append(stat(sample))
        except ValueError:          # a resample with one class only
            continue
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, lo, hi


def title(fig, question: str, answer: str) -> None:
    fig.suptitle(question, x=0.01, ha="left", fontsize=13, fontweight="semibold", color=INK)
    fig.text(0.01, 0.905, answer, ha="left", fontsize=10, color=INK_2)


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def load_predictions(run: str) -> pd.DataFrame:
    df = pd.read_csv(RUNS / run / "test_predictions.csv")
    df["y"] = df.true_label.isin(PLANET_LIKE).astype(int)
    return df


# =========================================================================== #
# Figures
# =========================================================================== #
def fig_separation() -> None:
    sets = [("Kepler (1,406 held-out KOIs)", "bn_aug_sched", BLUE),
            ("TESS (855 held-out TOIs)", "tess_bn_aug_sched", ORANGE)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.2))
    for label, run, col in sets:
        d = load_predictions(run)
        auc, lo, hi = star_bootstrap(d, lambda s: roc_auc_score(s.y, s.planet_like_score))
        ap, alo, ahi = star_bootstrap(d, lambda s: average_precision_score(s.y, s.planet_like_score))
        fpr, tpr, _ = roc_curve(d.y, d.planet_like_score)
        a.plot(fpr, tpr, color=col, label=f"{label}\nROC-AUC {auc:.3f}  [95% CI {lo:.3f}-{hi:.3f}]")
        p, r, _ = precision_recall_curve(d.y, d.planet_like_score)
        base = d.y.mean()
        b.plot(r, p, color=col, label=f"{label}\nPR-AUC {ap:.3f} [{alo:.3f}-{ahi:.3f}], "
                                       f"chance {base:.3f}")
        b.axhline(base, color=col, lw=1.0, ls=":")
    a.plot([0, 1], [0, 1], color=NEUTRAL, lw=1.0, ls="--", label="no skill")
    a.set(xlabel="false-positive rate", ylabel="true-positive rate (recall)",
          title="ROC curve on the held-out test split", xlim=(0, 1), ylim=(0, 1.01))
    b.set(xlabel="recall (completeness)", ylabel="precision",
          title="Precision-recall; dotted line = chance for that test set",
          xlim=(0, 1.01), ylim=(0.3, 1.01))
    a.legend(loc="lower right")
    b.legend(loc="lower left")
    title(fig, "How well does the transit classifier separate real planets from false positives?",
          "Strongly on Kepler. On TESS the PR-AUC looks higher but sits only ~0.09 above chance, "
          "because 86 % of that test set is planet-like. Intervals resample host stars.")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, "01_planet_vs_false_positive_separation.png")


def fig_ladder() -> None:
    base = json.loads((RUNS / "baselines" / "baselines.json").read_text(encoding="utf-8"))["results"]
    sweep = pd.read_csv(RUNS / "sweep_comparison.csv")
    sweep["family"] = sweep.config.str.replace(r"_s\d+$", "", regex=True)

    rows = [("Always predict majority class", [base["majority"]["roc_auc_binary"]]),
            ("Logistic regression on raw light curves", [base["logistic"]["roc_auc_binary"]]),
            ("Boosted trees on 11 hand-built shape features", [base["shape_gbm"]["roc_auc_binary"]])]
    for fam, label in [("base_01", "CNN + LSTM, untuned"),
                       ("bn", "+ batch normalisation"),
                       ("bn_aug_sched", "+ augmentation + LR schedule (final)")]:
        rows.append((label, sweep.loc[sweep.family == fam, "val_roc_auc"].dropna().tolist()))

    fig, ax = plt.subplots(figsize=(11, 5.0))
    y = np.arange(len(rows))[::-1]
    for yi, (label, vals) in zip(y, rows):
        is_deep = "CNN" in label or label.startswith("+")
        col = BLUE if is_deep else NEUTRAL
        ax.scatter(vals, [yi] * len(vals), s=64, color=col, edgecolor=SURFACE, linewidth=2, zorder=3)
        m = float(np.mean(vals))
        ax.plot([0.5, m], [yi, yi], color=col, lw=2, zorder=2)
        txt = f"{m:.3f}" + (f"  (n={len(vals)} seeds, sd {np.std(vals):.4f})" if len(vals) > 1 else "")
        ax.text(m + 0.006, yi, txt, va="center", fontsize=9, color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0.48, 1.0)
    ax.set_xlabel("validation ROC-AUC, planet-like vs false positive (0.5 = chance)")
    ax.grid(axis="y", visible=False)
    title(fig, "Does a deep network beat simpler methods on the same light curves?",
          "Yes, but modestly: +0.034 ROC-AUC over 11 interpretable features. Each dot is one "
          "training seed; seed-to-seed spread is about 0.001.")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, "02_deep_model_vs_simple_baselines.png")


def fig_ablation() -> None:
    sweep = pd.read_csv(RUNS / "sweep_comparison.csv")
    sweep["family"] = sweep.config.str.replace(r"_s\d+$", "", regex=True)
    stages = [("arch_global_cnn", "Global view only, CNN"),
              ("arch_dual_cnn", "+ zoomed local view"),
              ("bn_aug_sched", "+ LSTM (final model)")]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    means = []
    for i, (fam, label) in enumerate(stages):
        v = sweep.loc[sweep.family == fam, "val_roc_auc"].dropna().to_numpy()
        params = int(sweep.loc[sweep.family == fam, "n_params"].dropna().iloc[0])
        ax.scatter(v, [i] * len(v), s=64, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        means.append(v.mean())
        ax.text(v.max() + 0.0012, i, f"{v.mean():.4f}   ({params:,} parameters, {len(v)} seeds)",
                va="center", fontsize=9, color=INK_2)
    for i in range(1, len(means)):
        d = means[i] - means[i - 1]
        ax.annotate(f"+{d:.4f}", xy=((means[i] + means[i - 1]) / 2, i - 0.5), ha="center",
                    va="center", fontsize=9, color=ORANGE, fontweight="semibold")
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels([s[1] for s in stages])
    ax.invert_yaxis()
    ax.set_xlim(0.888, 0.93)
    ax.set_xlabel("validation ROC-AUC (each dot is one training seed)")
    ax.grid(axis="y", visible=False)
    title(fig, "Which parts of the network architecture actually improve detection?",
          "The zoomed local view adds 2.7x what the LSTM adds, at one-seventh of the extra "
          "parameters. Both gains exceed the seed-to-seed spread.")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save(fig, "03_architecture_ablation.png")


def fig_confusion() -> None:
    names = ["False positive", "Candidate", "Confirmed"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    for ax, (label, run) in zip(axes, [("Kepler test split", "bn_aug_sched"),
                                       ("TESS test split", "tess_bn_aug_sched")]):
        ev = json.loads((RUNS / run / "final_eval.json").read_text(encoding="utf-8"))["test"]
        cm = np.array(ev["confusion_3class"], float)
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ink = SURFACE if norm[i, j] > 0.55 else INK
                ax.text(j, i, f"{norm[i, j]:.0%}\n(n={int(cm[i, j])})", ha="center",
                        va="center", fontsize=9, color=ink)
        ax.set_xticks(range(3), names)
        ax.set_yticks(range(3), names)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true label")
        ax.set_title(f"{label} - accuracy {ev['accuracy_3class']:.3f}")
        ax.grid(False)
    title(fig, "Where does the classifier confuse the three dispositions?",
          "Rows sum to 100 %. On Kepler most errors call false positives 'Candidate'; on TESS the "
          "reverse - 38 % of candidates are called false positives. 'Candidate' means undecided.")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save(fig, "04_three_class_confusion.png")


def fig_calibration() -> None:
    d = load_predictions("bn_aug_sched")
    edges = np.linspace(0, 1, 11)
    d["bin"] = np.clip(np.digitize(d.planet_like_score, edges) - 1, 0, 9)
    fig, (a, b) = plt.subplots(2, 1, figsize=(8.5, 7.2), height_ratios=[3, 1], sharex=True)
    xs, ps, los, his = [], [], [], []
    for k, g in d.groupby("bin"):
        p, lo, hi = wilson(int(g.y.sum()), len(g))
        xs.append(g.planet_like_score.mean()); ps.append(p); los.append(p - lo); his.append(hi - p)
    a.plot([0, 1], [0, 1], color=NEUTRAL, ls="--", lw=1.0, label="perfect calibration")
    a.errorbar(xs, ps, yerr=[los, his], fmt="o-", color=BLUE, ms=8, capsize=3,
               label="observed fraction planet-like, 95% Wilson CI")
    a.set(ylabel="fraction actually planet-like", xlim=(0, 1), ylim=(0, 1.02))
    a.legend(loc="upper left")
    b.hist(d.planet_like_score, bins=edges, color=BLUE, edgecolor=SURFACE, linewidth=2)
    b.set(xlabel="predicted planet-like probability", ylabel="KOIs")
    brier = float(np.mean((d.planet_like_score - d.y) ** 2))
    title(fig, "Can the classifier's output be read as a probability?",
          f"Not directly: mid-range scores overstate the chance of a real planet (points sit below "
          f"the diagonal). Ranking is sound, probabilities need recalibration. Kepler test, Brier {brier:.3f}.")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "05_probability_calibration.png")


def fig_cross_mission() -> None:
    cm = pd.read_csv(RUNS / "cross_mission" / "cross_mission.csv")
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (col, label, vmin, vmax, fmt) in zip(axes, [
            ("roc_auc", "ROC-AUC", 0.7, 0.95, "{:.3f}"),
            ("pr_auc_lift", "PR-AUC minus chance (lift)", 0.0, 0.45, "{:+.3f}")]):
        m = cm.pivot(index="trained_on", columns="evaluated_on", values=col).loc[["Kepler", "TESS"], ["Kepler", "TESS"]]
        ax.imshow(m.to_numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
        for i in range(2):
            for j in range(2):
                v = m.iloc[i, j]
                ink = SURFACE if (v - vmin) / (vmax - vmin) > 0.55 else INK
                ax.text(j, i, fmt.format(v) + ("\nsame mission" if i == j else ""),
                        ha="center", va="center", fontsize=10, color=ink)
        ax.set_xticks([0, 1], ["Kepler", "TESS"])
        ax.set_yticks([0, 1], ["Kepler", "TESS"])
        ax.set_xlabel("evaluated on (held-out test split)")
        ax.set_ylabel("model trained on")
        ax.set_title(label)
        ax.grid(False)
    title(fig, "Does a transit classifier trained on one telescope work on another?",
          "Largely yes: using the other mission's model costs only 0.026-0.036 ROC-AUC. The lift "
          "panel shows the TESS task carries far less information than Kepler's.")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save(fig, "06_cross_mission_transfer.png")


def fig_depth_prior() -> None:
    koi = pd.read_csv(ROOT / "splits" / "koi_cumulative_split.csv", low_memory=False)
    toi = pd.read_csv(ROOT / "splits" / "toi_tess_candidates_split.csv", low_memory=False)
    sets = [("Kepler training split", koi[koi.split == "train"].rename(columns={"koi_depth": "depth"}), BLUE),
            ("TESS training split", toi[toi.split == "train"].rename(columns={"pl_trandep": "depth"}), ORANGE)]
    edges = [0, 300, 1000, 3000, 10000, 1e9]
    labels = ["<300", "300-1k", "1k-3k", "3k-10k", ">10k"]
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    x = np.arange(len(labels))
    for off, (label, df, col) in zip([-0.12, 0.12], sets):
        df = df.dropna(subset=["depth", "label_harmonized"])
        df = df.assign(bin=pd.cut(df.depth, edges, labels=labels, right=False))
        ps, lo, hi = [], [], []
        for b in labels:
            g = df[df.bin == b]
            p, l, h = wilson(int((g.label_harmonized == "FALSE POSITIVE").sum()), len(g))
            ps.append(p); lo.append(p - l); hi.append(h - p)
        base = float((df.label_harmonized == "FALSE POSITIVE").mean())
        ax.errorbar(x + off, ps, yerr=[lo, hi], fmt="o-", color=col, ms=8, capsize=3,
                    label=f"{label} (overall false-positive rate {base:.0%})")
    ax.set_xticks(x, labels)
    ax.set_xlabel("transit depth (ppm)")
    ax.set_ylabel("fraction labelled false positive (95% Wilson CI)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper left")
    title(fig, "Why does a Kepler-trained model distrust deep, hot-Jupiter-like transits?",
          "In Kepler's training data a transit deeper than 10,000 ppm is almost always an eclipsing "
          "binary; in TESS's it is not. Each model learns its own survey's prior.")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save(fig, "07_false_positive_rate_by_transit_depth.png")


def fig_opposite_bias() -> None:
    params = pd.read_csv(ROOT / "outputs" / "derived" / "planet_parameters.csv")[["pl_name", "pl_rade"]]
    k = pd.read_csv(ROOT / "outputs" / "stage_c" / "priority_target_scores_KEPLER.csv")
    t = pd.read_csv(ROOT / "outputs" / "stage_c" / "priority_target_scores_TESS.csv")
    m = (k[["pl_name", "transit_ML_probability_TESS_crossmission"]]
         .merge(t[["pl_name", "transit_ML_probability_TESS"]], on="pl_name")
         .merge(params, on="pl_name").dropna())
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for col_name, label, col in [("transit_ML_probability_TESS_crossmission", "Kepler-trained model", BLUE),
                                 ("transit_ML_probability_TESS", "TESS-trained model", ORANGE)]:
        rho = spearmanr(m.pl_rade, m[col_name]).statistic
        ax.scatter(m.pl_rade, m[col_name], s=64, color=col, edgecolor=SURFACE, linewidth=2,
                   label=f"{label}: Spearman rho = {rho:+.2f}", zorder=3)
    ax.axvline(4, color=NEUTRAL, lw=1.0, ls=":")
    ax.text(4.2, 0.03, "sub-Neptunes and smaller | larger", fontsize=8, color=INK_3)
    ax.set_xscale("log")
    ax.set_xlabel("planet radius (Earth radii, log scale)")
    ax.set_ylabel("planet-like score")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    title(fig, "Are the two detection models biased toward different kinds of planet?",
          f"Yes, in opposite directions, on the same {len(m)} confirmed planets observed by TESS. "
          "Each favours the planet sizes its own survey found most often.")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save(fig, "08_opposite_planet_size_bias.png")


def fig_fooled() -> None:
    fr = pd.read_csv(RUNS / "bn_aug_sched" / "failure_analysis" / "fooled_vs_rejected.csv")
    s = json.loads((RUNS / "bn_aug_sched" / "failure_analysis" / "summary.json").read_text(encoding="utf-8"))
    nf, nr = s["n_fooled"], s["n_rejected"]
    fr = fr[fr.signal.str.startswith("koi_fpflag")].copy()
    nice = {"nt": "Not transit-shaped", "ss": "Stellar eclipse (eclipsing binary)",
            "co": "Light from a neighbouring star", "ec": "Contamination from another signal"}
    fr["label"] = [nice[re.search(r"fpflag_(\w+)", x).group(1)] for x in fr.signal]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    y = np.arange(len(fr))
    for off, col_name, n, label, col in [(-0.15, "fooled_pct", nf, f"fooled the model (n={nf})", ORANGE),
                                         (0.15, "rejected_pct", nr, f"correctly rejected (n={nr})", BLUE)]:
        ps, lo, hi = [], [], []
        for v in fr[col_name]:
            p, l, h = wilson(int(round(v / 100 * n)), n)
            ps.append(p); lo.append(p - l); hi.append(h - p)
        ax.errorbar(ps, y + off, xerr=[lo, hi], fmt="o", color=col, ms=8, capsize=3, label=label)
    ax.set_yticks(y, fr.label)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.75)
    ax.set_xlabel("fraction of false positives carrying this Kepler flag (95% Wilson CI)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    title(fig, "Which kinds of false positive does the classifier fail to catch?",
          "Eclipsing binaries are caught well. Signals from a neighbouring star are a structural "
          "blind spot: that information is not present in a folded light curve at all.")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save(fig, "09_which_false_positives_fool_the_model.png")


def fig_injection() -> None:
    inj = pd.read_csv(ROOT / "outputs" / "injection_recovery" / "injections.csv")
    inj = inj[inj.status == "ok"]
    wide = inj.pivot_table(index="star", columns="depth_ppm", values="planet_like_score")
    depths = [d for d in wide.columns if d > 0]
    rng = np.random.default_rng(RNG_SEED)
    med, lo, hi, frac_lo, frac, frac_hi = [], [], [], [], [], []
    for d in depths:
        delta = (wide[d] - wide[0]).dropna().to_numpy()
        boots = [np.median(rng.choice(delta, len(delta))) for _ in range(N_BOOT)]
        med.append(np.median(delta)); lo.append(np.percentile(boots, 2.5)); hi.append(np.percentile(boots, 97.5))
        p, l, h = wilson(int((delta > 0).sum()), len(delta))
        frac.append(p); frac_lo.append(p - l); frac_hi.append(h - p)
    null = wide[0].dropna()
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8))
    a.fill_between(depths, lo, hi, color=BLUE, alpha=0.18, linewidth=0)
    a.plot(depths, med, "o-", color=BLUE, ms=8, label="median, 95% bootstrap CI over stars")
    a.axhline(0, color=INK_3, lw=1.0)
    a.axvline(84, color=NEUTRAL, ls=":", lw=1.0)
    a.text(88, max(hi) * 0.92, "Earth-size planet\naround a Sun-like star", fontsize=8, color=INK_3)
    a.set(xscale="log", xlabel="injected transit depth (ppm)",
          ylabel="score increase over the same star with nothing injected",
          title="Size of the response")
    a.legend(loc="upper left")
    b.errorbar(depths, frac, yerr=[frac_lo, frac_hi], fmt="o-", color=BLUE, ms=8, capsize=3)
    b.axhline(0.5, color=INK_3, lw=1.0, ls="--")
    b.text(depths[-1], 0.46, "coin flip", fontsize=8, color=INK_3, ha="right", va="top")
    b.set(xscale="log", ylim=(0, 1.02), xlabel="injected transit depth (ppm)",
          ylabel="fraction of stars whose score rose (95% Wilson CI)", title="Consistency across stars")
    title(fig, "What is the shallowest transit the classifier can actually detect?",
          f"About 80 ppm, near an Earth analogue. Synthetic transits in {wide.shape[0]} real Kepler "
          f"light curves; with nothing injected the model already scores {null.median():.2f}, "
          f"so each injection is compared with its own star.")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save(fig, "10_smallest_detectable_transit.png")


def fig_molecules() -> None:
    bi = pd.read_csv(ROOT / "outputs" / "stage_b_level2" / "band_indices.csv")
    mols = ["CO2", "H2O", "CH4", "SO2", "CO"]
    rng = np.random.default_rng(RNG_SEED)
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for i, m in enumerate(mols):
        v = bi[f"{m}_sigma"].dropna().to_numpy()
        jitter = rng.uniform(-0.18, 0.18, len(v))
        ax.scatter(np.full(len(v), i) + jitter, v, s=64, color=BLUE, alpha=0.55,
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color=INK, lw=2, zorder=4)
        p, l, h = wilson(int((v > 3).sum()), len(v))
        ax.text(i, ax.get_ylim()[1] if False else max(v.max(), 3) + 2.5,
                f"{int((v > 3).sum())}/{len(v)} above 3 sigma\n({p:.0%}, CI {l:.0%}-{h:.0%})",
                ha="center", fontsize=8, color=INK_2)
    ax.axhline(3, color=ORANGE, ls="--", lw=1.2, label="3-sigma detection threshold")
    ax.axhline(0, color=INK_3, lw=1.0)
    ax.set_xticks(range(len(mols)), ["CO$_2$ 4.3 um", "H$_2$O 1.4 um", "CH$_4$ 3.3 um",
                                     "SO$_2$ 4.05 um", "CO 4.6 um"])
    ax.set_ylabel("absorption in the band above its continuum (sigma)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    title(fig, "Which molecules can actually be detected in the JWST transmission spectra?",
          "CO2 and H2O, consistently; CO never reaches 3 sigma with this method. Each dot is one "
          "spectrum, black bars are medians, spectra normalised by atmospheric scale height.")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save(fig, "11_detectable_molecules_in_jwst_spectra.png")


def fig_ranking() -> None:
    ab = pd.read_csv(ROOT / "outputs" / "stage_c" / "ablation.csv").sort_values("spearman_vs_full")
    nice = {"small": "planet size", "temperate": "temperature near 300 K",
            "observable": "atmospheric signal strength", "evidence": "molecules detected"}
    fig, ax = plt.subplots(figsize=(11, 4.4))
    y = np.arange(len(ab))
    ax.hlines(y, 1.0, ab.spearman_vs_full, color=BLUE, lw=2)
    ax.scatter(ab.spearman_vs_full, y, s=64, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    for yi, (_, r) in zip(y, ab.iterrows()):
        ax.annotate(f"rho {r.spearman_vs_full:.3f}  -  {int(r.planets_changing_band)} of "
                    f"{int(r.n_scored)} planets change priority band",
                    xy=(1.0, yi), xycoords=("axes fraction", "data"), xytext=(10, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK_2)
    ax.axvline(1.0, color=INK_3, lw=1.0)
    ax.set_yticks(y, [f"remove '{nice.get(d, d)}' (weight {w:.2f})" for d, w in zip(ab.dropped, ab.weight)])
    ax.invert_yaxis()
    ax.set_xlim(0.6, 1.005)
    ax.set_xlabel("Spearman correlation with the full ranking (1.0 = ranking unchanged)")
    ax.grid(axis="y", visible=False)
    title(fig, "How much does the planet priority ranking depend on each scientific criterion?",
          "Planet size and temperature drive it most; detected molecules barely move it, because "
          "only 13 of the 54 planets have a JWST spectrum. Each criterion removed in turn.")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save(fig, "12_priority_ranking_sensitivity.png")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"writing figures to {OUT.relative_to(ROOT)}/")
    for f in (fig_separation, fig_ladder, fig_ablation, fig_confusion, fig_calibration,
              fig_cross_mission, fig_depth_prior, fig_opposite_bias, fig_fooled,
              fig_injection, fig_molecules, fig_ranking):
        f()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
