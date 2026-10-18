"""
evaluate_k2.py
==============

The single read of the K2 blind test, exactly as pre-registered in eda/K2_BLIND_TEST_PROTOCOL.md
(commit fbc60d8). Frozen v2 ensembles, frozen Platt maps and thresholds, metric code shared with
evaluate_v2.py; every expectation E1-E7 gets an automatic verdict.

    python stage_a_transit_model/evaluate_k2.py --dry-run    # Kepler v2 VALIDATION views stand in for K2:
                                                            # checks the code end to end, reads no K2 score
    python stage_a_transit_model/evaluate_k2.py              # the read; refuses to run a second time

Output: stage_a_transit_model/runs/k2_blind/{k2_eval.json, k2_predictions.csv, TEST_READ.json},
        figures/15_k2_blind_test.png  (dry run: runs/k2_blind/dry_run/, no figure in figures/)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from stage_a_transit_model import evaluate_v2 as ev  # noqa: E402
from stage_a_transit_model.calibrate import apply as cal_apply, ece  # noqa: E402
from stage_a_transit_model.cnn_lstm import CLASSES, PLANET_LIKE_IDX, consolidate_views, drop_unusable, evaluate  # noqa: E402

OUT = ev.RUNS / "k2_blind"
GUARD = OUT / "TEST_READ.json"
VIEWS = {"EVEREST": R / "detection_views_err" / "k2_everest", "K2SFF": R / "detection_views_err" / "k2_k2sff"}
EPH = R / "external_data" / "k2" / "k2_ephemerides.csv"
SEEDS = ["", "_s1", "_s2", "_s7", "_s11"]
MODELS = {"kepler": [f"v2_kepler_dual_cnn_lstm_err{s}" for s in SEEDS], "tess": [f"v2_tess_dual_cnn{s}" for s in SEEDS]}
# Frozen in the protocol (section 2); checked against final_eval_v2.json below so they cannot drift.
FROZEN = {"kepler": {"platt": {"method": "platt", "a": 1.0945, "b": -0.9221}, "thr90": 0.5926},
          "tess": {"platt": {"method": "platt", "a": 1.0121, "b": 1.0341}, "thr90": 0.4793}}
DEEP_PPM = 10000.0


def check_frozen() -> None:
    fe = json.loads((ev.RUNS / "v2_final" / "final_eval_v2.json").read_text())
    for m in FROZEN:
        assert fe[m]["decision"]["runs"] == MODELS[m], f"{m}: ensemble differs from the frozen one"
        pl = fe[m]["calibration"]["platt"]
        assert abs(pl["a"] - FROZEN[m]["platt"]["a"]) < 1e-4 and abs(pl["b"] - FROZEN[m]["platt"]["b"]) < 1e-4
        assert abs(fe[m]["operating_points_fixed_on_val"]["recall_90"]["threshold"] - FROZEN[m]["thr90"]) < 1e-4


def load(views_dir: Path, dry: bool) -> tuple:
    vs = drop_unusable(consolidate_views(views_dir, verbose=False), verbose=False)
    from preprocessing_pipeline import _npz_name
    meta = []
    for oid in vs.object_id:
        with np.load(views_dir / _npz_name(oid), allow_pickle=False) as z:
            m = json.loads(str(z["metadata"]))
        meta.append({"object_id": oid, "depth_snr": m.get("depth_snr", np.nan), "depth_ppm": m.get("depth_ppm", np.nan),
                     "scatter_over_error": (m.get("flatten") or {}).get("p2p_scatter_over_error")})
    meta = pd.DataFrame(meta)
    if dry:
        meta["duration_source"], meta["tess_overlap"] = "catalogue", False
    else:
        e = pd.read_csv(EPH)
        tess = pd.read_csv(R / "splits" / "toi_tess_candidates_split_v2.csv", low_memory=False, usecols=["tid", "split"])
        seen = set(tess[tess.split.isin(["train", "val"])].tid)
        e["tess_overlap"] = pd.to_numeric(e.tic_id.astype(str).str.replace("TIC", "").str.strip(), errors="coerce").isin(seen)
        meta = meta.merge(e[["pl_name", "duration_source", "tess_overlap"]].rename(columns={"pl_name": "object_id"}),
                          on="object_id", how="left")
    return vs, meta


def block(y3: np.ndarray, s: np.ndarray, stars: np.ndarray) -> dict:
    yb = np.isin(y3, PLANET_LIKE_IDX).astype(int)
    cf = np.isin(y3, [0, 2])
    return {"planet_like_vs_fp": ev.binary_block(yb, s, stars),
            "confirmed_vs_fp": ev.binary_block((y3[cf] == 2).astype(int), s[cf], stars[cf])}


def analyse(vs, meta: pd.DataFrame, scores: dict, per_seed: dict, probs: dict) -> dict:
    from scipy.stats import mannwhitneyu
    from sklearn.metrics import roc_auc_score

    y3 = vs.label_index.astype(int)
    yb = np.isin(y3, PLANET_LIKE_IDX).astype(int)
    stars = vs.star_id
    primary = meta.duration_source.isin(["catalogue", "other set"]).to_numpy()
    tess_ok = primary & ~meta.tess_overlap.fillna(False).astype(bool).to_numpy()
    res = {"n_views": int(len(y3)), "labels_all_views": {CLASSES[i]: int((y3 == i).sum()) for i in range(3)}}
    for name, mask, model in [("primary_kepler", primary, "kepler"), ("primary_tess", tess_ok, "tess"),
                              ("secondaryA_kepler", np.ones(len(y3), bool), "kepler")]:
        s, m = scores[model], mask
        r = {"n": int(m.sum()), "n_stars": int(len(set(stars[m].tolist()))),
             "labels": {CLASSES[i]: int((y3[m] == i).sum()) for i in range(3)}, **block(y3[m], s[m], stars[m])}
        if name.startswith("primary"):
            cal = cal_apply(FROZEN[model]["platt"], s[m])
            r["calibration"] = {"raw": {"brier": float(np.mean((s[m] - yb[m]) ** 2)), "ece": ece(yb[m], s[m])},
                                "frozen_platt": {"brier": float(np.mean((cal - yb[m]) ** 2)), "ece": ece(yb[m], cal)}}
            r["at_frozen_recall90_threshold"] = ev.at_threshold(yb[m], s[m], FROZEN[model]["thr90"])
            full = evaluate(y3[m], probs[model][m])
            r["three_class"] = {"accuracy": full["accuracy_3class"], "confusion": full["confusion_3class"]}
            r["per_seed_roc_auc"] = [float(roc_auc_score(yb[m], p[m])) for p in per_seed[model]]
            snr = meta.depth_snr.to_numpy(float)
            r["by_depth_snr"] = {f"{lo:g}-{hi:g}": ev.binary_block(yb[m & b], s[m & b], stars[m & b])
                                 for lo, hi in ev.SNR_BINS for b in [(snr >= lo) & (snr < hi)]}
            conf = m & (y3 == 2)
            deep = meta.depth_ppm.to_numpy(float) > DEEP_PPM
            a, b = s[conf & deep], s[conf & ~deep]
            r["depth_prior_confirmed"] = {"n_deep": int(len(a)), "n_shallow": int(len(b)),
                                          "median_deep": float(np.median(a)) if len(a) else None,
                                          "median_shallow": float(np.median(b)) if len(b) else None,
                                          "mannwhitney_p_deep_lower": float(mannwhitneyu(a, b, alternative="less").pvalue)
                                          if len(a) > 2 and len(b) > 2 else None}
        res[name] = r
    both = tess_ok
    res["kepler_minus_tess_paired"] = {"n": int(both.sum()), **ev.paired_auc_diff(yb[both], scores["kepler"][both],
                                                                                  scores["tess"][both], stars[both])}
    sr = meta.scatter_over_error.astype(float).dropna()
    if len(sr):
        res["scatter_over_error"] = {"median": float(sr.median()), "p16_p84": [float(sr.quantile(0.16)), float(sr.quantile(0.84))]}
    return res


def robustness(vs, meta: pd.DataFrame, s1: np.ndarray, vs2) -> dict:
    """Paired EVEREST - K2SFF ROC-AUC for the Kepler ensemble, on primary rows present in both builds."""
    s2 = ev.ensemble_proba(MODELS["kepler"], vs2)[0][:, PLANET_LIKE_IDX].sum(1)
    common = sorted(set(vs.object_id) & set(vs2.object_id))
    i1 = pd.Series(np.arange(len(vs.object_id)), index=vs.object_id)[common].to_numpy()
    i2 = pd.Series(np.arange(len(vs2.object_id)), index=vs2.object_id)[common].to_numpy()
    prim = meta.duration_source.isin(["catalogue", "other set"]).to_numpy()[i1]
    y = np.isin(vs.label_index.astype(int)[i1], PLANET_LIKE_IDX).astype(int)[prim]
    return {"n_common_primary": int(prim.sum()),
            **ev.paired_auc_diff(y, s1[i1][prim], s2[i2][prim], vs.star_id[i1][prim])}


def yields(views_dir: Path) -> dict:
    """Signals surviving each stage, by class (protocol section 5: a class-dependent yield biases every metric)."""
    e = pd.read_csv(EPH)
    idx = pd.read_csv(views_dir / "index.csv")
    lab = e.set_index("pl_name").label_harmonized
    idx["label"] = idx.object_id.map(lab)
    out = {"in_split": e.label_harmonized.value_counts().to_dict(),
           "complete_ephemeris": e[e.ephemeris_complete].label_harmonized.value_counts().to_dict(),
           "view_built": idx[idx.status == "ok"].label.value_counts().to_dict(),
           "excluded_unreliable_ephemeris": idx[idx.status == "excluded"].label.value_counts().to_dict(),
           "failed": {f"{c} | {l}": int(n) for (c, l), n in idx[idx.status == "failed"].groupby(["category", "label"]).size().items()}}
    tot = out["complete_ephemeris"]
    out["fraction_of_complete_built"] = {k: round(out["view_built"].get(k, 0) / v, 3) for k, v in tot.items()}
    return out


def verdicts(res: dict, robust: dict | None) -> dict:
    pk, pt = res["primary_kepler"], res["primary_tess"]
    p1, p2 = pk["planet_like_vs_fp"]["roc_auc"], pk["confirmed_vs_fp"]["roc_auc"]
    snr = [pk["by_depth_snr"][k].get("roc_auc") for k in pk["by_depth_snr"]]
    snr = [x for x in snr if x is not None]
    dk, dt = pk["depth_prior_confirmed"], pt["depth_prior_confirmed"]
    v = {"E1 P1 in [0.80, 0.92]": 0.80 <= p1 <= 0.92,
         "E2 P2 in [0.88, 0.97]": 0.88 <= p2 <= 0.97,
         "E3 Kepler > TESS on K2 (paired)": res["kepler_minus_tess_paired"]["delta_roc_auc"] > 0,
         "E4 Kepler ensemble under-scores deep confirmed planets, TESS does not":
             (dk["median_deep"] is not None and dk["median_deep"] < dk["median_shallow"])
             and (dt["median_deep"] is None or dt["median_deep"] >= dt["median_shallow"]),
         "E5 ROC-AUC rises with S/N (first bin < last bin)": len(snr) >= 2 and snr[0] < snr[-1],
         "E6 EVEREST >= K2SFF (Kepler ensemble)": None if robust is None else robust["delta_roc_auc"] >= 0,
         "E7 frozen Platt map does not transfer (ECE > 0.05)": pk["calibration"]["frozen_platt"]["ece"] > 0.05,
         "FAILURE criterion P1 < 0.75": p1 < 0.75}
    return {k: (None if x is None else bool(x)) for k, x in v.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--figure-only", action="store_true", help="redraw figures/15 from the saved read; scores nothing")
    a = ap.parse_args(argv)
    if a.figure_only:
        make_figure(json.loads((OUT / "k2_eval.json").read_text()), OUT)
        return 0
    check_frozen()
    out = OUT / "dry_run" if a.dry_run else OUT
    if not a.dry_run and GUARD.is_file():
        raise SystemExit(f"K2 was already read ({GUARD.read_text().strip()}); a second read is not allowed (protocol §7)")
    out.mkdir(parents=True, exist_ok=True)
    src = {"EVEREST": R / "detection_views_v2" / "kepler_val"} if a.dry_run else VIEWS
    vs, meta = load(src["EVEREST"], a.dry_run)
    # The guard is written together with the results, before anything is printed: a crash before that
    # point means nobody has seen a K2 number, so the read has not happened yet.
    probs, per_seed, scores = {}, {}, {}
    for m, runs in MODELS.items():
        p, per = ev.ensemble_proba(runs, vs)
        probs[m], scores[m] = p, p[:, PLANET_LIKE_IDX].sum(1)
        per_seed[m] = [q[:, PLANET_LIKE_IDX].sum(1) for q in per]
    res = {"generated_at": datetime.now().isoformat(timespec="seconds"), "dry_run": a.dry_run,
           "light_curves": "EVEREST", **analyse(vs, meta, scores, per_seed, probs)}
    robust = None
    second = (R / "detection_views_v2" / "kepler_val") if a.dry_run else VIEWS["K2SFF"]
    if second.is_dir() and any(second.glob("*.npz")):
        vs2, _ = load(second, a.dry_run)
        robust = robustness(vs, meta, scores["kepler"], vs2)
        res["robustness_everest_minus_k2sff"] = robust
    if not a.dry_run:
        res["yields"] = yields(VIEWS["EVEREST"])
    res["expectations"] = verdicts(res, robust)
    pd.DataFrame({"object_id": vs.object_id, "star_id": vs.star_id, "true_label": [CLASSES[i] for i in vs.label_index.astype(int)],
                  "score_kepler": scores["kepler"], "score_tess": scores["tess"],
                  "calibrated_kepler": cal_apply(FROZEN["kepler"]["platt"], scores["kepler"]),
                  **meta.drop(columns=["object_id"]).to_dict("list")}).to_csv(out / "k2_predictions.csv", index=False)
    (out / "k2_eval.json").write_text(json.dumps(res, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    if not a.dry_run:
        GUARD.write_text(json.dumps({"reads": 1, "at": datetime.now().isoformat(timespec="seconds"),
                                     "protocol_commit": "fbc60d8"}))
    print(json.dumps({k: res[k] for k in ["n_views", "labels_all_views", "expectations"]}, indent=2))
    for k in ["primary_kepler", "primary_tess"]:
        r = res[k]
        ci = lambda c: f"[{float(c[0]):.3f}, {float(c[1]):.3f}]"
        print(f"{k}: n={r['n']} planet-like vs FP {r['planet_like_vs_fp']['roc_auc']:.3f} {ci(r['planet_like_vs_fp']['roc_auc_ci'])} | "
              f"confirmed vs FP {r['confirmed_vs_fp']['roc_auc']:.3f} {ci(r['confirmed_vs_fp']['roc_auc_ci'])}")
    if not a.dry_run:
        make_figure(res, out)
    return 0


def make_figure(res: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    pr = pd.read_csv(out / "k2_predictions.csv")
    prim = pr.duration_source.isin(["catalogue", "other set"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    for model, col, sel in [("kepler", "#2a78d6", prim), ("tess", "#eb6834", prim & ~pr.tess_overlap.fillna(False).astype(bool))]:
        d = pr[sel]
        y = (d.true_label != "FALSE POSITIVE").astype(int)
        f, t, _ = roc_curve(y, d[f"score_{model}"])
        r = res[f"primary_{model}"]["planet_like_vs_fp"]
        ax.plot(f, t, color=col, lw=2, label=f"{model.capitalize() if model == 'kepler' else 'TESS'}-trained: {r['roc_auc']:.3f} "
                                              f"[{r['roc_auc_ci'][0]:.3f}, {r['roc_auc_ci'][1]:.3f}] (n={r['n']})")
    ax.plot([0, 1], [0, 1], ls="--", color="0.7", lw=1)
    ax.set(xlabel="false-positive rate", ylabel="true-positive rate", title="K2 (never seen): planet-like vs false positive")
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1]
    bins = list(res["primary_kepler"]["by_depth_snr"])
    x = np.arange(len(bins))
    for off, model, col in [(-0.12, "kepler", "#2a78d6"), (0.12, "tess", "#eb6834")]:
        vals = [res[f"primary_{model}"]["by_depth_snr"][b] for b in bins]
        auc = [v.get("roc_auc", np.nan) for v in vals]
        lo = [v.get("roc_auc_ci", [np.nan, np.nan])[0] for v in vals]
        hi = [v.get("roc_auc_ci", [np.nan, np.nan])[1] for v in vals]
        auc, lo, hi = np.array(auc, float), np.array(lo, float), np.array(hi, float)
        ax.errorbar(x + off, auc, yerr=[auc - lo, hi - auc], fmt="o-", color=col, capsize=3, label=f"{model}-trained")
    ax.set(xticks=x, xticklabels=[f"S/N {b}" for b in bins], ylim=(0.4, 1.02), ylabel="ROC-AUC (95 % star bootstrap)",
           title="By transit signal-to-noise")
    ax.axhline(0.5, color="0.7", lw=1, ls="--")
    ax.legend(fontsize=8)
    ax = axes[2]
    c = pr[prim & (pr.true_label == "CONFIRMED")]
    for model, col in [("kepler", "#2a78d6"), ("tess", "#eb6834")]:
        ax.scatter(c.depth_ppm.clip(lower=10), c[f"score_{model}"], s=12, alpha=0.55, color=col, label=f"{model}-trained")
    ax.axvline(DEEP_PPM, color="k", lw=1, ls=":")
    dk = res["primary_kepler"]["depth_prior_confirmed"]
    ax.set(xscale="log", xlabel="measured transit depth (ppm; <= 10 plotted at 10)", ylabel="planet-like score",
           title=f"Confirmed K2 planets: Kepler model median {dk['median_deep']:.2f} deeper than 10,000 ppm\n"
                 f"vs {dk['median_shallow']:.2f} shallower (p = {dk['mannwhitney_p_deep_lower']:.0e}): its survey prior again")
    ax.legend(fontsize=8)
    pk = res["primary_kepler"]
    fig.suptitle(f"Stage A on K2, a mission neither model saw (single pre-registered read): Kepler-trained ROC-AUC "
                 f"{pk['planet_like_vs_fp']['roc_auc']:.3f}, confirmed vs FP {pk['confirmed_vs_fp']['roc_auc']:.3f}",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(R / "figures" / "15_k2_blind_test.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())
