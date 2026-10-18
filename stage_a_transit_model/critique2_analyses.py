"""
critique2_analyses.py
=====================

What the vetter does, beyond its architecture (critique 2). Three analyses of the frozen v2 Kepler ensemble,
on the Kepler v2 VALIDATION split (and, descriptively, on sets already read once: K2 and the DR25 benchmark).
No test split is read.

  1. Uncertainty. Does disagreement between the 5 seeds predict the ensemble's errors? Selective prediction:
     the error rate when the most uncertain signals are set aside for a human.
  2. Interpretability. Occlusion: which phases of the global and local views move the score, for planets and
     for eclipsing binaries.
  3. The mechanism of the survey prior. Views are depth-normalised (deepest bin -1), so the network can only
     sense depth through signal-to-noise: how smooth the baseline is relative to the dip, and how small the
     error channel is. If the Kepler prior "deep = eclipsing binary" acts through S/N, then ADDING noise to a
     high-S/N confirmed planet must RAISE its planet score. A causal test.

    python stage_a_transit_model/critique2_analyses.py
Output: outputs/stage_a_mechanisms/{mechanisms.json, ...}, figures/18_vetter_mechanisms.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from stage_a_transit_model.cnn_lstm import PLANET_LIKE_IDX, ViewSet, consolidate_views, drop_unusable  # noqa: E402
from stage_a_transit_model.failure_analysis import load_run_model, score_views  # noqa: E402

OUT = R / "outputs" / "stage_a_mechanisms"
RUNS = R / "stage_a_transit_model" / "runs"
SEEDS = [f"v2_kepler_dual_cnn_lstm_err{s}" for s in ["", "_s1", "_s2", "_s7", "_s11"]]
THR = 0.5926
NOISE = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]


def per_seed(models, vs: ViewSet) -> np.ndarray:
    return np.array([score_views(m, vs)[:, PLANET_LIKE_IDX].sum(1) for m in models])     # (seeds, n)


def uncertainty(models, vs: ViewSet, y: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score
    s = per_seed(models, vs)
    mean, sd = s.mean(0), s.std(0)
    wrong = ((mean >= THR).astype(int) != y)
    order = np.argsort(sd)                                       # most certain first
    cov, err, auc = [], [], []
    for c in [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]:
        keep = order[: int(round(c * len(y)))]
        cov.append(c)
        err.append(float(wrong[keep].mean()))
        auc.append(float(roc_auc_score(y[keep], mean[keep])) if len(np.unique(y[keep])) == 2 else None)
    return {"n": int(len(y)), "error_rate_at_threshold": float(wrong.mean()),
            "auroc_seed_sd_detects_errors": float(roc_auc_score(wrong.astype(int), sd)),
            "coverage": cov, "error_rate": err, "roc_auc": auc, "_mean": mean, "_sd": sd}


def occlusion(models, vs: ViewSet, which: str, width: int) -> np.ndarray:
    """Mean score change when a window of the view is replaced by 0 (the out-of-transit level), per window."""
    base = per_seed(models, vs).mean(0)
    L = vs.global_view.shape[1] if which == "global" else vs.local_view.shape[1]
    starts = np.arange(0, L - width + 1, width)
    out = np.zeros((len(starts), len(vs)))
    for k, a in enumerate(starts):
        g, l = vs.global_view.copy(), vs.local_view.copy()
        ge, le = vs.global_err.copy(), vs.local_err.copy()
        if which == "global":
            g[:, a:a + width] = 0.0
        else:
            l[:, a:a + width] = 0.0
        out[k] = per_seed(models, ViewSet(g, l, vs.label_index, vs.object_id, vs.star_id, ge, le)).mean(0) - base
    return out                                                    # (windows, n)


def degrade(vs: ViewSet, sigma: float, rng) -> ViewSet:
    """Lower the signal-to-noise: add noise of sigma (in depth units) to the views, and to their errors."""
    g = vs.global_view + rng.normal(0, sigma, vs.global_view.shape).astype(np.float32)
    l = vs.local_view + rng.normal(0, sigma, vs.local_view.shape).astype(np.float32)
    ge = np.sqrt(vs.global_err.astype(np.float64) ** 2 + sigma ** 2).astype(np.float32)
    le = np.sqrt(vs.local_err.astype(np.float64) ** 2 + sigma ** 2).astype(np.float32)
    return ViewSet(g, l, vs.label_index, vs.object_id, vs.star_id, ge, le)


def snr_of(views_dir: Path, ids) -> np.ndarray:
    from preprocessing_pipeline import _npz_name
    out = []
    for oid in ids:
        with np.load(views_dir / _npz_name(oid), allow_pickle=False) as z:
            out.append(json.loads(str(z["metadata"])).get("depth_snr", np.nan))
    return np.array(out, float)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261019)
    models = [load_run_model(RUNS / r)[0] for r in SEEDS]
    vdir = R / "detection_views_v2" / "kepler_val"
    val = drop_unusable(consolidate_views(vdir, verbose=False), verbose=False)
    y = np.isin(val.label_index, PLANET_LIKE_IDX).astype(int)
    res = {"split": "Kepler v2 validation", "uncertainty": {}}

    # 1. uncertainty: validation, plus the two sets already read once
    u = uncertainty(models, val, y)
    res["uncertainty"]["kepler_val"] = {k: v for k, v in u.items() if not k.startswith("_")}
    for name, d, ylab in [("k2_blind_everest", R / "detection_views_err" / "k2_everest", None),
                          ("dr25_benchmark_inj1_inv", None, None)]:
        if d is not None:
            vs = drop_unusable(consolidate_views(d, verbose=False), verbose=False)
            yy = np.isin(vs.label_index, PLANET_LIKE_IDX).astype(int)
        else:
            a = drop_unusable(consolidate_views(R / "detection_views_err" / "dr25_inj1", verbose=False), verbose=False)
            b = drop_unusable(consolidate_views(R / "detection_views_err" / "dr25_inv", verbose=False), verbose=False)
            vs = ViewSet(np.r_[a.global_view, b.global_view], np.r_[a.local_view, b.local_view], np.r_[a.label_index, b.label_index],
                         np.r_[a.object_id, b.object_id], np.r_[a.star_id, b.star_id], np.r_[a.global_err, b.global_err],
                         np.r_[a.local_err, b.local_err])
            yy = np.isin(vs.label_index, PLANET_LIKE_IDX).astype(int)
        uu = uncertainty(models, vs, yy)
        res["uncertainty"][name] = {k: v for k, v in uu.items() if not k.startswith("_")}

    # 2. occlusion on validation: confirmed planets vs eclipsing-binary false positives
    koi = pd.read_csv(R / "splits" / "koi_cumulative_split_v2.csv", low_memory=False,
                      usecols=["kepoi_name", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_nt", "koi_fpflag_ec"]).set_index("kepoi_name")
    f = koi.reindex(val.object_id)
    conf = np.flatnonzero(val.label_index == 2)
    eb = np.flatnonzero((val.label_index == 0) & (f.koi_fpflag_ss.to_numpy() == 1))
    pick_c = rng.choice(conf, size=min(250, len(conf)), replace=False)
    pick_e = rng.choice(eb, size=min(250, len(eb)), replace=False)
    occ = {}
    for which, width in [("global", 41), ("local", 9)]:
        occ[which] = {"planets": occlusion(models, val.take(pick_c), which, width).mean(1).tolist(),
                      "eclipsing_binaries": occlusion(models, val.take(pick_e), which, width).mean(1).tolist(),
                      "window_bins": width}
    res["occlusion"] = occ

    # 3. the survey-prior mechanism: add noise to high-S/N confirmed planets and eclipsing binaries
    snr = snr_of(vdir, val.object_id)
    hi_c = np.flatnonzero((val.label_index == 2) & (snr > 100))
    hi_e = np.flatnonzero((val.label_index == 0) & (f.koi_fpflag_ss.to_numpy() == 1) & (snr > 100))
    lo_c = np.flatnonzero((val.label_index == 2) & (snr > 10) & (snr < 30))
    mech = {}
    for name, idx in [("confirmed_snr_gt_100", hi_c), ("eclipsing_binaries_snr_gt_100", hi_e), ("confirmed_snr_10_30", lo_c)]:
        sub = val.take(idx)
        med = []
        for s in NOISE:
            sc = per_seed(models, degrade(sub, s, np.random.default_rng(1))).mean(0)
            med.append(float(np.median(sc)))
        mech[name] = {"n": int(len(idx)), "noise_sigma_depth_units": NOISE, "median_score": med}
    # where the prior actually bites: deep confirmed planets OUTSIDE Kepler's training distribution
    # (K2 confirmed > 10,000 ppm, median 0.72 in the K2 read; TESS hot Jupiters of Stage C, 0.09-0.45)
    k2 = drop_unusable(consolidate_views(R / "detection_views_err" / "k2_everest", verbose=False), verbose=False)
    from preprocessing_pipeline import _npz_name
    dep = []
    for oid in k2.object_id:
        with np.load(R / "detection_views_err" / "k2_everest" / _npz_name(oid), allow_pickle=False) as z:
            dep.append(json.loads(str(z["metadata"])).get("depth_ppm", np.nan))
    dep = np.array(dep, float)
    sc = drop_unusable(consolidate_views(R / "detection_views_err" / "stage_c_targets", verbose=False), verbose=False)
    for name, sub in [("k2_confirmed_deeper_than_10000ppm", k2.take(np.flatnonzero((k2.label_index == 2) & (dep > 10000)))),
                      ("k2_confirmed_shallower_than_3000ppm", k2.take(np.flatnonzero((k2.label_index == 2) & (dep < 3000))[:200])),
                      ("tess_stage_c_targets", sc)]:
        med = [float(np.median(per_seed(models, degrade(sub, s, np.random.default_rng(1))).mean(0))) for s in NOISE]
        mech[name] = {"n": int(len(sub)), "noise_sigma_depth_units": NOISE, "median_score": med}
    res["noise_intervention"] = mech
    d = mech["k2_confirmed_deeper_than_10000ppm"]["median_score"]
    res["prior_acts_through_snr"] = bool(max(d[1:4]) > d[0] + 0.05)      # small noise (<= 5 % of depth) raises deep planets
    (OUT / "mechanisms.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({"uncertainty": {k: {kk: vv for kk, vv in v.items() if kk in ("n", "error_rate_at_threshold", "auroc_seed_sd_detects_errors", "error_rate")}
                                      for k, v in res["uncertainty"].items()},
                      "noise_intervention": mech, "prior_acts_through_snr": res["prior_acts_through_snr"]}, indent=2, default=float))
    figure(res, u)
    return 0


def figure(res, u):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.9))
    ax = axes[0]
    for k, c, lab in [("kepler_val", "#2a78d6", "Kepler validation"), ("k2_blind_everest", "#eb6834", "K2 (never seen)"),
                      ("dr25_benchmark_inj1_inv", "#1baf7a", "DR25 injections + inverted")]:
        r = res["uncertainty"][k]
        ax.plot(r["coverage"], r["error_rate"], "o-", color=c,
                label=f"{lab}: seed spread flags errors, AUROC {r['auroc_seed_sd_detects_errors']:.2f}")
    ax.set(xlabel="coverage (fraction kept; the most uncertain set aside)", ylabel="error rate at the operating threshold",
           title="Abstaining on seed disagreement removes errors")
    ax.invert_xaxis()
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax = axes[1]
    o = res["occlusion"]["local"]
    x = np.linspace(-4, 4, len(o["planets"]))
    ax.plot(x, o["planets"], "o-", color="#2a78d6", label="confirmed planets")
    ax.plot(x, o["eclipsing_binaries"], "s-", color="#eb6834", label="eclipsing binaries")
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xlabel="phase (transit durations, local view)", ylabel="score change when this window is blanked",
           title="Where the decision lives (local view)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[2]
    for k, c, lab in [("k2_confirmed_deeper_than_10000ppm", "#2a78d6", "K2 confirmed, deeper than 10,000 ppm"),
                      ("tess_stage_c_targets", "#1baf7a", "TESS light curves of Stage C planets"),
                      ("eclipsing_binaries_snr_gt_100", "#eb6834", "Kepler EBs, S/N > 100 (validation)"),
                      ("confirmed_snr_gt_100", "#8a8984", "Kepler confirmed, S/N > 100 (validation)")]:
        m = res["noise_intervention"][k]
        ax.plot(np.array(m["noise_sigma_depth_units"]) + 1e-3, m["median_score"], "o-", color=c, label=f"{lab} (n={m['n']})")
    ax.set(xscale="log", xlabel="noise added (fraction of the transit depth)", ylabel="median planet-like score",
           title="Adding noise: what changes the score")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(R / "figures" / "18_vetter_mechanisms.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())
