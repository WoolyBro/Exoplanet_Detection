#!/usr/bin/env python3
"""
make_figures.py
===============

Regenerates every statistical figure in figures/ (01-12) from results committed to this
repository. Nothing is retrained and no light curve is downloaded:

    python figures/make_figures.py

Since 2026-10-18 the Stage A figures (01-10) show the FINAL models on the leakage-free split v2:
the 5-seed Kepler CNN+BiLSTM ensemble with the flux-error channel and the 5-seed TESS dual-CNN
ensemble, selected on validation and read once on test (stage_a_transit_model/evaluate_v2.py,
eda/STAGE_A_V2_FINAL_RESULTS.md). Every number in a subtitle is computed from those saved results,
so the text cannot drift from the data. Reading the saved test predictions again here is not a
second test read: no model is changed or selected.

Each figure is titled by the question its study answers, not by the dataset it used, and its
subtitle states the answer. Every rate or score carries an uncertainty interval:

  * ROC-AUC / PR-AUC  - 95 % bootstrap interval, resampling STARS rather than rows, because
                        several signals share a host star and are not independent draws. The
                        test-set intervals are the ones evaluate_v2.py reported, read from
                        final_eval_v2.json, so figure and paper quote identical numbers.
  * proportions       - 95 % Wilson score interval (well-behaved at small n and near 0 or 1).
  * paired effects    - 95 % bootstrap interval on the median, resampling stars.

Colour follows a validated categorical order (blue, orange, aqua - the three slots that
remain distinguishable under every colour-vision deficiency when all pairs are on screen),
and identity is always carried by a legend or direct label as well, never by colour alone.
Kepler is always blue and TESS always orange.
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
from scipy.stats import fisher_exact, spearmanr  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
RUNS = ROOT / "stage_a_transit_model" / "runs"
V2 = RUNS / "v2_final"
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
MISSIONS = [("kepler", "Kepler", BLUE), ("tess", "TESS", ORANGE)]

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
# Helpers
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


def finish(fig, question: str, answer: str) -> None:
    """Question as the title, the answer (one or more lines) under it, axes laid out below both.
    Offsets are in inches so figures of any height get the same spacing."""
    h = fig.get_figheight()
    lines = answer.count("\n") + 1
    # plain fig.text, not suptitle: tight_layout would reserve room for a suptitle a second time
    fig.text(0.01, 1 - 0.10 / h, question, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 1 - 0.42 / h, answer, ha="left", va="top", fontsize=10, color=INK_2, linespacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.50 + 0.2 * lines) / h))


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def final_eval() -> dict:
    return json.loads((V2 / "final_eval_v2.json").read_text(encoding="utf-8"))


def load_final(mission: str) -> pd.DataFrame:
    df = pd.read_csv(V2 / f"{mission}_test_predictions.csv")
    df["y"] = df.true_label.isin(PLANET_LIKE).astype(int)
    return df


def family_of(config: str) -> str:
    return re.sub(r"_s\d+$", "", config)


# =========================================================================== #
# Stage A: detecting transiting planets
# =========================================================================== #
def fig_separation() -> None:
    ev = final_eval()
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 5.4))
    for m, name, col in MISSIONS:
        d, e = load_final(m), ev[m]["binary"]
        label = f"{name} ({e['n']:,} test signals, {ev[m]['n_test_stars']:,} stars)"
        (rlo, rhi), (plo, phi) = e["roc_auc_ci"], e["pr_auc_ci"]
        fpr, tpr, _ = roc_curve(d.y, d.planet_like_score)
        a.plot(fpr, tpr, color=col, label=f"{label}\nROC-AUC {e['roc_auc']:.3f}  [95% CI {rlo:.3f}-{rhi:.3f}]")
        p, r, _ = precision_recall_curve(d.y, d.planet_like_score)
        b.plot(r, p, color=col, label=f"{label}\nPR-AUC {e['pr_auc']:.3f} [{plo:.3f}-{phi:.3f}], "
                                       f"chance {e['positive_rate']:.3f}")
        b.axhline(e["positive_rate"], color=col, lw=1.0, ls=":")
    a.plot([0, 1], [0, 1], color=NEUTRAL, lw=1.0, ls="--", label="no skill")
    a.set(xlabel="false-positive rate", ylabel="true-positive rate (recall)",
          title="ROC curve on the held-out test split", xlim=(0, 1), ylim=(0, 1.01))
    b.set(xlabel="recall (completeness)", ylabel="precision",
          title="Precision-recall; dotted line = chance for that test set",
          xlim=(0, 1.01), ylim=(0.3, 1.01))
    a.legend(loc="lower right")
    b.legend(loc="lower left")
    k, t = ev["kepler"]["binary"], ev["tess"]["binary"]
    finish(fig, "How well does the transit classifier separate real planets from false positives?",
           f"Strongly on Kepler (ROC-AUC {k['roc_auc']:.3f}); less so on TESS ({t['roc_auc']:.3f}), whose PR-AUC looks "
           f"higher but sits only {t['pr_auc_lift']:.2f} above chance because {t['positive_rate']:.0%} of its test set\n"
           "is planet-like. Final 5-seed ensembles on the leakage-free split v2 (no shared stars, no signal twins), "
           "test split read once; intervals resample host stars.")
    save(fig, "01_planet_vs_false_positive_separation.png")


def fig_ladder() -> None:
    ev = final_eval()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.9), sharey=True)
    for ax, (m, name, col) in zip(axes, MISSIONS):
        e, base = ev[m], ev[m]["baselines"]
        auc, (clo, chi) = e["binary"]["roc_auc"], e["binary"]["roc_auc_ci"]
        rows = [("Always predict the majority class", [0.5], NEUTRAL),
                ("Logistic regression on the folded views", [base["logistic_views"]["roc_auc"]], NEUTRAL),
                ("Boosted trees on 12 transit-shape features", [base["shape_gbm"]["roc_auc"]], NEUTRAL),
                ("Final network, each seed alone", e["per_seed_test_roc_auc"]["values"], col),
                ("Final network, 5-seed ensemble", [auc], col)]
        y = np.arange(len(rows))[::-1]
        for yi, (label, vals, c) in zip(y, rows):
            ax.scatter(vals, [yi] * len(vals), s=64, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
            mean = float(np.mean(vals))
            ax.plot([0.5, mean], [yi, yi], color=c, lw=2, zorder=2)
            txt = f"{mean:.3f}" + (f"  (mean of {len(vals)} seeds, sd {np.std(vals, ddof=1):.3f})" if len(vals) > 1 else "")
            x_txt = max(max(vals), chi if label.endswith("ensemble") else 0) + 0.008
            ax.text(x_txt, yi, txt, va="center", fontsize=9, color=INK_2)
        ax.errorbar(auc, y[-1], xerr=[[auc - clo], [chi - auc]], fmt="none", ecolor=INK, capsize=3, lw=1.2, zorder=4)
        dlt = base["ensemble_minus_shape_gbm"]
        ax.set_title(f"{name} test split: ensemble minus boosted trees "
                     f"{dlt['delta_roc_auc']:+.3f} [{dlt['ci'][0]:+.3f}, {dlt['ci'][1]:+.3f}]")
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_xlim(0.48, 1.0)
        ax.set_xlabel("test ROC-AUC, planet-like vs false positive (0.5 = chance)")
        ax.grid(axis="y", visible=False)
    k, t = ev["kepler"]["baselines"]["ensemble_minus_shape_gbm"], ev["tess"]["baselines"]["ensemble_minus_shape_gbm"]
    finish(fig, "Does a deep network beat simpler methods on the same light curves?",
           f"Yes, on both missions: by {k['delta_roc_auc']:+.3f} ROC-AUC on Kepler and {t['delta_roc_auc']:+.3f} on TESS over "
           "boosted trees on interpretable shape features (95 % star-bootstrap intervals exclude zero).\n"
           "Every method trained on the same v2 training split and was scored on the same test split. "
           "Black bar: 95 % interval of the ensemble.")
    save(fig, "02_deep_model_vs_simple_baselines.png")


ABLATION = {
    "kepler": [("global_cnn", "Global view only, CNN"),
               ("dual_cnn", "+ zoomed local view"),
               ("dual_cnn_err", "+ local view + flux-error channel"),
               ("dual_cnn_lstm", "+ local view + BiLSTM"),
               ("dual_cnn_lstm_err", "+ local view + BiLSTM + error channel")],
    "tess": [("global_cnn", "Global view only, CNN"),
             ("dual_cnn", "+ zoomed local view"),
             ("dual_cnn_err", "+ local view + flux-error channel"),
             ("dual_cnn_lstm", "+ local view + BiLSTM"),
             ("dual_cnn_lstm_err", "+ local view + BiLSTM + error channel"),
             ("dual_cnn_lstm_cwsqrt", "+ local view + BiLSTM, sqrt class weights"),
             ("dual_cnn_lstm_cwnone", "+ local view + BiLSTM, no class weights")],
}


def fig_ablation() -> None:
    ev = final_eval()
    sweep = pd.read_csv(RUNS / "v2_comparison.csv")
    sweep["family"] = sweep.config.map(family_of)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.8), height_ratios=[len(ABLATION["kepler"]), len(ABLATION["tess"])])
    stats = {}
    for ax, (m, name, col) in zip(axes, MISSIONS):
        chosen = "v2_" + ev[m]["decision"]["family"]
        rows = ABLATION[m]
        vals = {fam: sweep[sweep.family == f"v2_{m}_{fam}"] for fam, _ in rows}
        x_txt = max(g.val_roc_auc.max() for g in vals.values()) + 0.004
        for i, (fam, label) in enumerate(rows):
            g = vals[fam]
            v = g.val_roc_auc.dropna().to_numpy()
            stats[(m, fam)] = v.mean()
            sel = f"v2_{m}_{fam}" == chosen
            ax.scatter(v, i + np.linspace(-0.12, 0.12, len(v)), s=52, color=col if sel else NEUTRAL,
                       edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color=INK, lw=1.6, zorder=4)
            ax.text(x_txt, i, f"{v.mean():.4f} +/- {v.std(ddof=1):.4f}   ({int(g.n_params.iloc[0]):,} params)"
                    + ("   <- selected" if sel else ""), va="center", fontsize=9,
                    color=INK if sel else INK_2, fontweight="bold" if sel else "normal")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[1] for r in rows])
        ax.invert_yaxis()
        lo = min(g.val_roc_auc.min() for g in vals.values())
        ax.set_xlim(lo - 0.006, x_txt + 0.042)
        ax.set_xlabel("validation ROC-AUC (each dot one training seed; black bar = mean)")
        ax.set_title(f"{name}: 5 seeds per configuration", loc="left")
        ax.grid(axis="y", visible=False)
    k = {f: stats[("kepler", f)] for f, _ in ABLATION["kepler"]}
    t = {f: stats[("tess", f)] for f, _ in ABLATION["tess"]}
    finish(fig, "Which parts of the network architecture actually improve detection?",
           f"On Kepler each part helps: the local view adds {k['dual_cnn'] - k['global_cnn']:+.3f}, the BiLSTM "
           f"{k['dual_cnn_lstm'] - k['dual_cnn']:+.3f} and the flux-error channel {k['dual_cnn_lstm_err'] - k['dual_cnn_lstm']:+.3f}.\n"
           f"On TESS only the local view does ({t['dual_cnn'] - t['global_cnn']:+.3f}); nothing else clears the seed-to-seed "
           "spread, so the rule fixed before any v2 run kept the dual CNN\n(simplest architecture within one seed-sd of the "
           "best; the error channel only if it adds more than one sd).")
    save(fig, "03_architecture_ablation.png")


def fig_confusion() -> None:
    ev = final_eval()
    names = ["False positive", "Candidate", "Confirmed"]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    norms = {}
    for ax, (m, name, _) in zip(axes, MISSIONS):
        tc = ev[m]["three_class"]
        cm = np.array(tc["confusion"], float)
        norm = cm / cm.sum(axis=1, keepdims=True)
        norms[m] = norm
        ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ink = SURFACE if norm[i, j] > 0.55 else INK
                ax.text(j, i, f"{norm[i, j]:.0%}\n(n={int(cm[i, j])})", ha="center", va="center", fontsize=9, color=ink)
        ax.set_xticks(range(3), names)
        ax.set_yticks(range(3), names)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true label")
        ax.set_title(f"{name} test split - 3-class accuracy {tc['accuracy']:.3f}")
        ax.grid(False)
    k, t = norms["kepler"], norms["tess"]
    finish(fig, "Where does the classifier confuse the three dispositions?",
           f"Rows sum to 100 %. On Kepler the main error calls false positives 'Candidate' ({k[0, 1]:.0%}); on TESS "
           f"{t[1, 0]:.0%} of candidates are called false positives.\nErrors concentrate on 'Candidate', which means "
           "undecided, not a third kind of object.")
    save(fig, "04_three_class_confusion.png")


def fig_calibration() -> None:
    ev = final_eval()
    edges = np.linspace(0, 1, 11)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6), height_ratios=[3, 1], sharex=True)
    for j, (m, name, _) in enumerate(MISSIONS):
        d, cal = load_final(m), ev[m]["calibration"]
        a, b = axes[0, j], axes[1, j]
        a.plot([0, 1], [0, 1], color=NEUTRAL, ls="--", lw=1.0, label="perfect calibration")
        for col_name, label, col, off, key in [
                ("planet_like_score", "raw ensemble score", ORANGE, -0.006, "raw"),
                ("calibrated_score", "Platt-scaled (fitted on validation)", BLUE, 0.006, "platt_fitted_on_val")]:
            bins = np.clip(np.digitize(d[col_name], edges) - 1, 0, 9)
            xs, ps, los, his = [], [], [], []
            for kbin in sorted(set(bins)):
                g = d[bins == kbin]
                if len(g) < 5:
                    continue
                p, lo, hi = wilson(int(g.y.sum()), len(g))
                xs.append(g[col_name].mean() + off); ps.append(p); los.append(p - lo); his.append(hi - p)
            a.errorbar(xs, ps, yerr=[los, his], fmt="o-", color=col, ms=6, capsize=3,
                       label=f"{label}: Brier {cal[key]['brier']:.3f}, ECE {cal[key]['ece']:.3f}")
            b.hist(d[col_name], bins=edges, color=col, histtype="step", linewidth=2.0)
        a.set(ylabel="fraction actually planet-like (95% Wilson CI)" if j == 0 else "", xlim=(0, 1), ylim=(0, 1.02),
              title=f"{name} test split ({len(d):,} signals; bins with < 5 omitted)")
        a.legend(loc="upper left", fontsize=8.5)
        b.set(xlabel="predicted planet-like probability", ylabel="signals" if j == 0 else "")
    k, t = ev["kepler"]["calibration"], ev["tess"]["calibration"]
    finish(fig, "Can the classifier's output be read as a probability?",
           f"Raw, no: the TESS ensemble is under-confident and the Kepler one over-confident in mid-range. After a Platt map "
           f"fitted on validation only, yes:\ntest ECE {k['raw']['ece']:.3f} -> {k['platt_fitted_on_val']['ece']:.3f} (Kepler) "
           f"and {t['raw']['ece']:.3f} -> {t['platt_fitted_on_val']['ece']:.3f} (TESS). The map is monotonic, so the ranking "
           "(ROC-AUC) is unchanged.")
    save(fig, "05_probability_calibration.png")


def fig_cross_mission() -> None:
    ev = final_eval()
    cm = pd.DataFrame(ev["cross_mission"])
    cm["lift"] = cm.pr_auc - cm.base_rate
    for c in ("trained_on", "evaluated_on"):
        cm[c] = cm[c].str.capitalize().str.replace("Tess", "TESS")
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (col, label, vmin, vmax, fmt) in zip(axes, [
            ("roc_auc", "ROC-AUC", 0.7, 0.95, "{:.3f}"),
            ("lift", "PR-AUC minus chance (lift)", 0.0, 0.45, "{:+.3f}")]):
        mat = cm.pivot(index="trained_on", columns="evaluated_on", values=col).loc[["Kepler", "TESS"], ["Kepler", "TESS"]]
        ax.imshow(mat.to_numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
        for i in range(2):
            for j in range(2):
                v = mat.iloc[i, j]
                ink = SURFACE if (v - vmin) / (vmax - vmin) > 0.55 else INK
                ax.text(j, i, fmt.format(v) + ("\nsame mission" if i == j else ""), ha="center", va="center",
                        fontsize=10, color=ink)
        ax.set_xticks([0, 1], ["Kepler", "TESS"])
        ax.set_yticks([0, 1], ["Kepler", "TESS"])
        ax.set_xlabel("evaluated on (v2 test split)")
        ax.set_ylabel("5-seed ensemble trained on")
        ax.set_title(label)
        ax.grid(False)
    r = cm.set_index(["trained_on", "evaluated_on"]).roc_auc
    finish(fig, "Does a transit classifier trained on one telescope work on another?",
           f"Yes. The Kepler-trained model is as good on TESS as TESS's own ({r['Kepler', 'TESS']:.3f} vs "
           f"{r['TESS', 'TESS']:.3f}), and the TESS-trained model does better on Kepler ({r['TESS', 'Kepler']:.3f})\n"
           "than on TESS. Both suggest that TESS's labels and class balance, more than its light curves, limit what can be learned.")
    save(fig, "06_cross_mission_transfer.png")


# =========================================================================== #
# Why the models behave as they do
# =========================================================================== #
def depth_prior_table() -> dict:
    koi = pd.read_csv(ROOT / "splits" / "koi_cumulative_split_v2.csv", low_memory=False)
    toi = pd.read_csv(ROOT / "splits" / "toi_tess_candidates_split_v2.csv", low_memory=False)
    return {"Kepler": koi[koi.split == "train"].rename(columns={"koi_depth": "depth"}),
            "TESS": toi[toi.split == "train"].rename(columns={"pl_trandep": "depth"})}


def fig_depth_prior() -> None:
    edges = [0, 300, 1000, 3000, 10000, 1e9]
    labels = ["<300", "300-1k", "1k-3k", "3k-10k", ">10k"]
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    x = np.arange(len(labels))
    deep, base = {}, {}
    for off, (name, df), col in zip([-0.12, 0.12], depth_prior_table().items(), [BLUE, ORANGE]):
        df = df.dropna(subset=["depth", "label_harmonized"])
        df = df.assign(bin=pd.cut(df.depth, edges, labels=labels, right=False))
        ps, lo, hi = [], [], []
        for b in labels:
            g = df[df.bin == b]
            p, l, h = wilson(int((g.label_harmonized == "FALSE POSITIVE").sum()), len(g))
            ps.append(p); lo.append(p - l); hi.append(h - p)
        deep[name] = ps[-1]
        base[name] = float((df.label_harmonized == "FALSE POSITIVE").mean())
        ax.errorbar(x + off, ps, yerr=[lo, hi], fmt="o-", color=col, ms=8, capsize=3,
                    label=f"{name} v2 training split, n = {len(df):,} (overall false-positive rate {base[name]:.0%})")
    ax.set_xticks(x, labels)
    ax.set_xlabel("transit depth (ppm)")
    ax.set_ylabel("fraction labelled false positive (95% Wilson CI)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper left")
    finish(fig, "Why does a Kepler-trained model distrust deep, hot-Jupiter-like transits?",
           f"In Kepler's training data {deep['Kepler']:.0%} of transits deeper than 10,000 ppm are false positives "
           f"(base rate {base['Kepler']:.0%}); in TESS's only {deep['TESS']:.0%} (base rate {base['TESS']:.0%}).\n"
           "Each model learns its own survey's prior, which explains figure 8.")
    save(fig, "07_false_positive_rate_by_transit_depth.png")


def fig_opposite_bias() -> None:
    params = pd.read_csv(ROOT / "outputs" / "derived" / "planet_parameters.csv")[["pl_name", "pl_rade"]]
    k = pd.read_csv(ROOT / "outputs" / "stage_c" / "priority_target_scores_KEPLER_v2.csv")
    t = pd.read_csv(ROOT / "outputs" / "stage_c" / "priority_target_scores_TESS_v2.csv")
    m = (k[["pl_name", "transit_ML_probability_TESS_crossmission"]]
         .merge(t[["pl_name", "transit_ML_probability_TESS"]], on="pl_name")
         .merge(params, on="pl_name").dropna())
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.vlines(m.pl_rade, m.transit_ML_probability_TESS_crossmission, m.transit_ML_probability_TESS,
              color=NEUTRAL, lw=0.8, zorder=1)          # one planet, two models
    frac = {}
    for col_name, label, col in [("transit_ML_probability_TESS_crossmission", "Kepler-trained ensemble", BLUE),
                                 ("transit_ML_probability_TESS", "TESS-trained ensemble", ORANGE)]:
        rho = spearmanr(m.pl_rade, m[col_name]).statistic
        frac[label] = float((m[col_name] >= 0.5).mean())
        ax.scatter(m.pl_rade, m[col_name], s=64, color=col, edgecolor=SURFACE, linewidth=2,
                   label=f"{label}: Spearman rho = {rho:+.2f}; {frac[label]:.0%} scored >= 0.5", zorder=3)
    for name in ["HD 189733 b", "KELT-9 b", "TRAPPIST-1 e", "GJ 1132 b"]:
        r = m[m.pl_name == name]
        if len(r):
            r = r.iloc[0]
            ax.annotate(name, (r.pl_rade, r.transit_ML_probability_TESS_crossmission), xytext=(6, -10),
                        textcoords="offset points", fontsize=8, color=INK_3)
    ax.axvline(4, color=NEUTRAL, lw=1.0, ls=":")
    ax.text(4.2, 0.03, "sub-Neptunes and smaller | larger", fontsize=8, color=INK_3)
    ax.set_xscale("log")
    ax.set_xlabel("planet radius (Earth radii, log scale)")
    ax.set_ylabel("planet-like score")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    finish(fig, "Are the two detection models biased toward different kinds of planet?",
           f"Yes, in opposite directions, on the same {len(m)} confirmed planets observed by TESS (the final v2 5-seed "
           "ensembles). The Kepler-trained model\ndistrusts large planets, as figure 7 predicts; the TESS-trained one "
           "favours them. Every planet here is real, so this measures bias, not skill.")
    save(fig, "08_opposite_planet_size_bias.png")


FLAG_NAMES = {"ss": "Stellar eclipse (eclipsing binary)", "ec": "Contamination from another signal",
              "nt": "Not transit-shaped", "co": "Light from a neighbouring star (centroid offset)"}


def fooled_table() -> tuple[pd.DataFrame, float, float, int]:
    """Kepler v2 test false positives, grouped by WHICH of Kepler's own false-positive flags
    (koi_fpflag_*) they carry, and the fraction the model rejects at the validation-fixed recall-90
    threshold. Groups are exclusive ("only this flag"), because the flags co-occur: most centroid-
    offset false positives also carry an eclipse or contamination flag the light curve does show,
    which hides the centroid blind spot in a per-flag comparison."""
    ev = final_eval()
    thr = ev["kepler"]["operating_points_fixed_on_val"]["recall_90"]["threshold"]
    flags = ["koi_fpflag_ss", "koi_fpflag_ec", "koi_fpflag_nt", "koi_fpflag_co"]
    koi = pd.read_csv(ROOT / "splits" / "koi_cumulative_split_v2.csv", low_memory=False, usecols=["kepoi_name", *flags])
    d = load_final("kepler").merge(koi, left_on="object_id", right_on="kepoi_name", how="left")
    fp = d[d.true_label == "FALSE POSITIVE"].copy()
    fp["rejected"] = fp.planet_like_score < thr
    n_flags = fp[flags].sum(axis=1)
    groups = [(f"{FLAG_NAMES[f.split('_')[-1]]}, only", (n_flags == 1) & (fp[f] == 1)) for f in flags]
    groups += [("Two or more flags", n_flags >= 2), ("No flag set", n_flags == 0)]
    rest_rate = lambda mask: float(fp[~mask].rejected.mean())
    rows = []
    for label, mask in groups:
        g = fp[mask]
        k, n = int(g.rejected.sum()), len(g)
        _, p = fisher_exact([[k, n - k], [int(fp[~mask].rejected.sum()), int((~fp[~mask].rejected).sum())]])
        rows.append({"group": label, "n": n, "rejected": k, "median_score": float(g.planet_like_score.median()),
                     "rest_rejection": rest_rate(mask), "fisher_p_vs_rest": p})
    return pd.DataFrame(rows), thr, float(fp.rejected.mean()), len(fp)


def fig_fooled() -> None:
    tab, thr, overall, n_fp = fooled_table()
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    y = np.arange(len(tab))
    for yi, r in zip(y, tab.itertuples()):
        p, l, h = wilson(r.rejected, r.n)
        weak = h < overall
        col = ORANGE if weak else BLUE
        ax.errorbar(p, yi, xerr=[[p - l], [h - p]], fmt="o", color=col, ms=9, capsize=3)
        ptxt = "p < 0.001" if r.fisher_p_vs_rest < 0.001 else f"p = {r.fisher_p_vs_rest:.2g}"
        ax.text(1.0, yi, f"{r.rejected}/{r.n} rejected ({p:.0%}); median score {r.median_score:.2f}; vs rest {ptxt}",
                transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8.5, color=INK_2)
    ax.axvline(overall, color=INK_3, lw=1.0, ls="--")
    ax.text(overall, len(tab) - 0.45, f" all {n_fp} test false positives: {overall:.0%}", fontsize=8.5, color=INK_3, va="bottom")
    ax.axvline(0.5, color=NEUTRAL, lw=1.0, ls=":")
    ax.set_yticks(y, tab.group)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_ylim(len(tab) - 0.3, -0.45)
    ax.set_xlabel(f"fraction of these false positives the model rejects (score < {thr:.2f}, the validation-fixed "
                  "recall-90 threshold; 95% Wilson CI)", fontsize=9)
    ax.grid(axis="y", visible=False)
    by = tab.set_index("group")
    co = by.loc[f"{FLAG_NAMES['co']}, only"]
    ss = by.loc[f"{FLAG_NAMES['ss']}, only"]
    nt = by.loc[f"{FLAG_NAMES['nt']}, only"]
    finish(fig, "Which kinds of false positive does the classifier fail to catch?",
           f"False positives whose only warning sign is light from a neighbouring star are rejected {co.rejected / co.n:.0%} "
           f"of the time, a coin flip, against {ss.rejected / ss.n:.0%} for eclipsing binaries.\nA folded light curve "
           "carries no centroid information, so that blind spot is structural; it is where vetters with centroid "
           f"diagnostics gain (figure 14).\nSignals flagged only as not transit-shaped are also weak ({nt.rejected / nt.n:.0%}). "
           "Kepler v2 test split; Fisher exact tests against all other false positives; groups are exclusive because "
           "the flags co-occur.")
    save(fig, "09_which_false_positives_fool_the_model.png")


def fig_injection() -> None:
    src = ROOT / "outputs" / "injection_recovery_v2"
    inj = pd.read_csv(src / "injections.csv")
    inj = inj[inj.status == "ok"]
    thr = float(json.loads((src / "summary.json").read_text(encoding="utf-8"))["recovery_threshold"])
    wide = inj.pivot_table(index="star", columns="depth_ppm", values="planet_like_score")
    depths = [d for d in wide.columns if d > 0]
    rng = np.random.default_rng(RNG_SEED)
    med, lo, hi, frac, frac_lo, frac_hi, rec, rec_lo, rec_hi = ([] for _ in range(9))
    for d in depths:
        delta = (wide[d] - wide[0]).dropna().to_numpy()
        boots = [np.median(rng.choice(delta, len(delta))) for _ in range(N_BOOT)]
        med.append(np.median(delta)); lo.append(np.percentile(boots, 2.5)); hi.append(np.percentile(boots, 97.5))
        p, l, h = wilson(int((delta > 0).sum()), len(delta))
        frac.append(p); frac_lo.append(p - l); frac_hi.append(h - p)
        s = wide[d].dropna()
        p, l, h = wilson(int((s >= thr).sum()), len(s))
        rec.append(p); rec_lo.append(p - l); rec_hi.append(h - p)
    null = wide[0].dropna()
    null_rec = wilson(int((null >= thr).sum()), len(null))
    # detection floor: the shallowest depth whose median paired response is above zero (95 % CI) AND
    # at which significantly more than half the stars respond (Wilson lower bound > 0.5)
    floor = next((d for d, l_, f, fl in zip(depths, lo, frac, frac_lo) if l_ > 0 and f - fl > 0.5), None)

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15.5, 4.9))
    a.fill_between(depths, lo, hi, color=BLUE, alpha=0.18, linewidth=0)
    a.plot(depths, med, "o-", color=BLUE, ms=7, label="median, 95% bootstrap CI over stars")
    a.axhline(0, color=INK_3, lw=1.0)
    a.axvline(84, color=NEUTRAL, ls=":", lw=1.0)
    a.text(78, a.get_ylim()[1] * 0.9, "Earth-size planet\naround a Sun-like star", fontsize=8, color=INK_3, va="top", ha="right")
    a.set(xscale="log", xlabel="injected transit depth (ppm)",
          ylabel="score increase over the same star, nothing injected", title="Size of the response (paired)")
    a.legend(loc="upper left", fontsize=8.5)
    b.errorbar(depths, frac, yerr=[frac_lo, frac_hi], fmt="o-", color=BLUE, ms=7, capsize=3)
    b.axhline(0.5, color=INK_3, lw=1.0, ls="--")
    b.text(depths[-1], 0.46, "coin flip", fontsize=8, color=INK_3, ha="right", va="top")
    b.set(xscale="log", ylim=(0, 1.02), xlabel="injected transit depth (ppm)",
          ylabel="fraction of stars whose score rose (95% Wilson CI)", title="Consistency across stars")
    c.errorbar(depths, rec, yerr=[rec_lo, rec_hi], fmt="o-", color=BLUE, ms=7, capsize=3,
               label=f"injected transit scored >= {thr:.2f}")
    c.axhspan(null_rec[1], null_rec[2], color=NEUTRAL, alpha=0.35, lw=0,
              label=f"same stars, nothing injected: {null_rec[0]:.0%}")
    c.set(xscale="log", ylim=(0, 1.02), xlabel="injected transit depth (ppm)",
          ylabel="fraction above the recall-90 threshold (Wilson CI)", title="Absolute recovery at the operating point")
    c.legend(loc="lower right", fontsize=8.5)
    if floor:
        for ax in (a, b, c):
            ax.axvline(floor, color=ORANGE, lw=1.2, ls="--")
    ftxt = (f"The model responds reliably from about {floor:.0f} ppm (orange line: median response above zero and "
            "most stars responding, both at 95 %)") if floor else "No depth in this grid gives a reliable response"
    finish(fig, "What is the shallowest transit the classifier can actually detect?",
           f"{ftxt}.\nSynthetic transits injected into {wide.shape[0]} real Kepler light curves of v2 test stars, scored by the "
           f"final 5-seed ensemble. With nothing injected it already scores {null.median():.2f}\n(a candidate vetter, not a "
           "blind search), so the left two panels compare every injection with its own star.")
    save(fig, "10_smallest_detectable_transit.png")


# =========================================================================== #
# Atmospheres and target selection
# =========================================================================== #
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
        ax.text(i, max(v.max(), 3) + 2.5,
                f"{int((v > 3).sum())}/{len(v)} above 3 sigma\n({p:.0%}, CI {l:.0%}-{h:.0%})",
                ha="center", fontsize=8, color=INK_2)
    ax.axhline(3, color=ORANGE, ls="--", lw=1.2, label="3-sigma detection threshold")
    ax.axhline(0, color=INK_3, lw=1.0)
    ax.set_xticks(range(len(mols)), ["CO$_2$ 4.3 um", "H$_2$O 1.4 um", "CH$_4$ 3.3 um",
                                     "SO$_2$ 4.05 um", "CO 4.6 um"])
    ax.set_ylabel("absorption in the band above its continuum (sigma)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    finish(fig, "Which molecules can actually be detected in the JWST transmission spectra?",
           "CO2 and H2O, consistently; CO never reaches 3 sigma with this method. Each dot is one "
           "spectrum, black bars are medians, spectra normalised by atmospheric scale height.")
    save(fig, "11_detectable_molecules_in_jwst_spectra.png")


def fig_ranking() -> None:
    ab = pd.read_csv(ROOT / "outputs" / "stage_c" / "ablation.csv").sort_values("spearman_vs_full")
    nice = {"hz": "Habitable-zone membership\n(Kopparapu et al. 2014)",
            "terrestrial_atm": "Rocky AND able to keep\nan atmosphere",
            "observable": "Transmission signal, TSM\n(Kempton et al. 2018)",
            "retention (terrestrial_atm -> terrestrial)": "Atmosphere retention only\n(cosmic shoreline; size kept)"}
    n_ranked = int(json.loads((ROOT / "outputs" / "stage_c" / "stage_c_summary.json").read_text())["n_ranked"])
    fig, ax = plt.subplots(figsize=(11, 5.0))
    y = np.arange(len(ab))
    ax.hlines(y, 1.0, ab.spearman_vs_full, color=BLUE, lw=2)
    ax.scatter(ab.spearman_vs_full, y, s=64, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    for yi, (_, r) in zip(y, ab.iterrows()):
        ax.text(0.995, yi - 0.2, f"rho {r.spearman_vs_full:.3f}, top-10 overlap {int(r.top10_overlap)}/10",
                ha="right", va="bottom", fontsize=9, color=INK_2)
    ax.axvline(1.0, color=INK_3, lw=1.0)
    ax.set_yticks(y, [nice.get(d, d) + (f", weight {w:.2f}" if w > 0 else ", no weight of its own")
                      for d, w in zip(ab.dropped, ab.weight)], fontsize=9)
    ax.set_ylabel("criterion removed")
    ax.invert_yaxis()
    ax.set_ylim(len(ab) - 0.5, -0.7)
    ax.set_xlim(0, 1.01)
    ax.set_xlabel(f"Spearman correlation with the full ranking of {n_ranked} transiting planets (1.0 = unchanged)")
    ax.grid(axis="y", visible=False)
    by = ab.set_index("dropped")
    hz, ret = by.loc["hz"], by.loc["retention (terrestrial_atm -> terrestrial)"]
    finish(fig, "How much does the follow-up priority ranking depend on each criterion?",
           f"Habitable-zone membership drives it: without it the order is nearly unrelated (rho {hz.spearman_vs_full:.2f}, "
           f"top-10 overlap {int(hz.top10_overlap)}/10).\nAtmosphere retention (the cosmic shoreline) reorders the list "
           f"(rho {ret.spearman_vs_full:.2f}) but keeps {int(ret.top10_overlap)} of the top 10. A ranking for transmission "
           "follow-up\nwith stated weights, not a habitability probability.")
    save(fig, "12_priority_ranking_sensitivity.png")


FIGURES = (fig_separation, fig_ladder, fig_ablation, fig_confusion, fig_calibration, fig_cross_mission,
           fig_depth_prior, fig_opposite_bias, fig_fooled, fig_injection, fig_molecules, fig_ranking)


def main(argv: list[str] | None = None) -> int:
    import sys
    only = set(argv if argv is not None else sys.argv[1:])     # e.g. `make_figures.py fig_injection`
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"writing figures to {OUT.relative_to(ROOT)}/")
    for f in FIGURES:
        if not only or f.__name__ in only:
            f()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
