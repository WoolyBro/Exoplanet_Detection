"""
evaluate_v3.py
==============

Tests T1-T3 of eda/VETTER_V3_PROTOCOL.md (commit ae91f4e), read once, for v2 and the v3 configs A, B, C:

  T1  fresh DR25 simulated signals on held-out stars: INJ1 planets kept; INV false alarms, INJ2 off-target
      transits and INJ3 eclipsing-binary mimics rejected. Compared with the Robovetter on the same TCEs.
  T2  the 900 real NTP test TCEs (held-out signal families): rejected.
  T3  the Kepler v2 test KOIs: the SECOND, pre-declared read of that split (guarded), ROC-AUC, confirmed vs
      FP, ExoMiner on the common KOIs, and rejection of false positives by exclusive Kepler flag.

Thresholds: each v3 ensemble's recall-90 threshold on its validation set (KOI + NTP val); v2 keeps its frozen
0.5926. Holm correction over the hypothesis tests behind H1-H7.

    python stage_a_transit_model/evaluate_v3.py --dry-run     # validation stands in for every test; no test read
    python stage_a_transit_model/evaluate_v3.py               # the read (refuses a second time)
Output: stage_a_transit_model/runs/v3_final/{v3_eval.json, predictions_*.csv, TEST_READ.json}, figures/17_vetter_v3.png
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
from stage_a_transit_model import vetter_v3 as v3  # noqa: E402
from stage_a_transit_model.cnn_lstm import CLASSES, PLANET_LIKE_IDX, consolidate_views, drop_unusable, precision_at_recall  # noqa: E402
from stage_a_transit_model.evaluate_dr25_simulated import mcnemar, wilson  # noqa: E402

OUT = v3.RUNS / "v3_final"
GUARD = OUT / "TEST_READ.json"
SIM_SETS = {"INJ1": "v3_eval_inj1", "INV": "v3_eval_inv", "INJ2": "v3_eval_inj2", "INJ3": "v3_eval_inj3"}
V2_RUNS = [f"v2_kepler_dual_cnn_lstm_err{s}" for s in ["", "_s1", "_s2", "_s7", "_s11"]]
V2_THR = 0.5926
MODELS = ["v2", "A", "B", "C"]


def runs_of(cfg: str) -> list[Path]:
    return [v3.RUNS / (f"v3_{cfg}" + ("" if s == 42 else f"_s{s}")) for s in v3.SEEDS]


def proba(model: str, vs) -> np.ndarray:
    if model == "v2":
        return ev.ensemble_proba(V2_RUNS, vs)[0]
    return np.mean([v3.score(r, vs) for r in runs_of(model)], axis=0)


def planet(p: np.ndarray) -> np.ndarray:
    return p[:, PLANET_LIKE_IDX].sum(1)


def load_dir(d: Path, keep=None):
    vs = drop_unusable(consolidate_views(d, verbose=False), verbose=False)
    if keep is not None:
        vs = vs.take(np.flatnonzero(np.isin(vs.object_id, list(keep))))
    return vs


def robovetter_disp(set_name: str, ids) -> np.ndarray:
    from astropy.io import ascii
    t = ascii.read(v3.R / "external_data" / "kepler_dr25_simulated" / f"kplr_dr25_{set_name.lower()}_tces.txt", format="ipac").to_pandas()
    m = t.set_index(t.TCE_ID + "_" + set_name).Disp
    return pd.Series(ids).map(m).to_numpy()


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def main(argv=None) -> int:
    from sklearn.metrics import roc_auc_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    out = OUT / "dry_run" if a.dry_run else OUT
    if not a.dry_run and GUARD.is_file():
        raise SystemExit(f"v3 tests already read ({GUARD.read_text().strip()})")
    out.mkdir(parents=True, exist_ok=True)

    # thresholds on validation (KOI + NTP val), the same for every v3 config
    val = v3.load_split("val", ntp=True)
    yv = np.isin(val.label_index, PLANET_LIKE_IDX).astype(int)
    thr, val_auc = {"v2": V2_THR}, {}
    for m in MODELS:
        s = planet(proba(m, val))
        val_auc[m] = float(roc_auc_score(yv, s))
        if m != "v2":
            thr[m] = next(r["threshold"] for r in precision_at_recall(yv, s) if r["target_recall"] == 0.9)
    res = {"generated_at": datetime.now().isoformat(timespec="seconds"), "dry_run": a.dry_run, "thresholds": thr,
           "val_roc_auc_koi_plus_ntp": val_auc, "T1": {}, "T2": {}, "T3": {}}

    # ---- T1 simulated (dry run: validation KOIs/NTP stand in) ----
    tests = {}
    if a.dry_run:
        tests["INJ1"] = val.take(np.flatnonzero(val.label_index == 2))
        tests["INV"] = val.take(np.flatnonzero(np.char.endswith(val.object_id.astype(str), "_OBS")))
    else:
        tests = {s: load_dir(R / "detection_views_err" / d) for s, d in SIM_SETS.items()}
    scores = {}
    for s, vs in tests.items():
        sc = {m: planet(proba(m, vs)) for m in MODELS}
        keep = {m: (sc[m] >= thr[m]) for m in MODELS}             # passed as planet-like
        good = (lambda m: keep[m]) if s == "INJ1" else (lambda m: ~keep[m])
        r = {"n": int(len(vs)), "metric": "completeness" if s == "INJ1" else "rejection",
             **{m: wilson(int(good(m).sum()), len(vs)) for m in MODELS}}
        if not a.dry_run:
            rv = robovetter_disp(s, vs.object_id)
            rgood = (rv == "PC") if s == "INJ1" else (rv == "FP")
            r["robovetter"] = wilson(int(rgood.sum()), len(vs))
            r["C_vs_robovetter"] = mcnemar(good("C"), rgood)
        r["C_vs_v2"] = mcnemar(good("C"), good("v2"))
        res["T1"][s] = r
        scores[s] = (vs, sc)
        pd.DataFrame({"object_id": vs.object_id, **{f"score_{m}": sc[m] for m in MODELS}}).to_csv(out / f"predictions_{s}.csv", index=False)
    if "INJ1" in scores:
        for s in [k for k in scores if k != "INJ1"]:
            y = np.r_[np.ones(len(scores["INJ1"][1]["C"])), np.zeros(len(scores[s][1]["C"]))]
            res["T1"][s]["roc_auc_vs_INJ1"] = {m: round(float(roc_auc_score(y, np.r_[scores["INJ1"][1][m], scores[s][1][m]])), 4) for m in MODELS}

    # ---- T2 real NTP test ----
    if not a.dry_run:
        ntp = v3.ntp_split_map()
        nt = load_dir(v3.NTP_VIEWS, keep=set(ntp[ntp == "test"].index))
        sc = {m: planet(proba(m, nt)) for m in MODELS}
        rej = {m: sc[m] < thr[m] for m in MODELS}
        res["T2"] = {"n": int(len(nt)), **{m: wilson(int(rej[m].sum()), len(nt)) for m in MODELS},
                     "C_vs_v2": mcnemar(rej["C"], rej["v2"])}
        pd.DataFrame({"object_id": nt.object_id, **{f"score_{m}": sc[m] for m in MODELS}}).to_csv(out / "predictions_NTP_test.csv", index=False)

    # ---- T3 Kepler v2 test KOIs (second, pre-declared read) ----
    if not a.dry_run:
        te = load_dir(v3.KOI_VIEWS / "kepler_test")
    else:
        te = v3.load_split("val", ntp=False)
    y3 = te.label_index.astype(int)
    yb = np.isin(y3, PLANET_LIKE_IDX).astype(int)
    cf = np.isin(y3, [0, 2])
    sc = {m: planet(proba(m, te)) for m in MODELS}
    res["T3"] = {"n": int(len(te)), "split": "val (dry run)" if a.dry_run else "test",
                 **{m: {"planet_like_vs_fp": ev.binary_block(yb, sc[m], te.star_id),
                        "confirmed_vs_fp": ev.binary_block((y3[cf] == 2).astype(int), sc[m][cf], te.star_id[cf])} for m in MODELS},
                 "C_minus_v2_confirmed_vs_fp": ev.paired_auc_diff((y3[cf] == 2).astype(int), sc["C"][cf], sc["v2"][cf], te.star_id[cf])}
    koi = pd.read_csv(R / "splits" / "koi_cumulative_split_v2.csv", low_memory=False,
                      usecols=["kepoi_name", "kepid", "koi_tce_plnt_num", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"])
    d = pd.DataFrame({"object_id": te.object_id, "true_label": [CLASSES[i] for i in y3], **{f"score_{m}": sc[m] for m in MODELS}})
    d = d.merge(koi, left_on="object_id", right_on="kepoi_name", how="left")
    d.to_csv(out / "predictions_kepler_test.csv", index=False)
    flags = ["koi_fpflag_ss", "koi_fpflag_ec", "koi_fpflag_nt", "koi_fpflag_co"]
    fp = d[d.true_label == "FALSE POSITIVE"]
    nfl = fp[flags].sum(axis=1)
    groups = {f.split("_")[-1] + "_only": (nfl == 1) & (fp[f] == 1) for f in flags}
    res["T3"]["fp_rejection_by_exclusive_flag"] = {
        g: {"n": int(msk.sum()), **{m: wilson(int((fp[msk][f"score_{m}"] < thr[m]).sum()), int(msk.sum())) for m in MODELS}}
        for g, msk in groups.items()}
    if not a.dry_run:
        from stage_a_transit_model.compare_published_vetters import read_exominer
        em = read_exominer()
        c = d.merge(em, left_on=["kepid", "koi_tce_plnt_num"], right_on=["target_id", "tce_plnt_num"]).dropna(subset=["exominer"])
        c = c[c.true_label != "CANDIDATE"]
        yc = (c.true_label == "CONFIRMED").astype(int).to_numpy()
        res["T3"]["exominer_common"] = {"n": int(len(c)), "exominer": round(float(roc_auc_score(yc, c.exominer)), 4),
                                        **{m: round(float(roc_auc_score(yc, c[f"score_{m}"])), 4) for m in MODELS},
                                        "C_minus_exominer": ev.paired_auc_diff(yc, c.score_C.to_numpy(), c.exominer.to_numpy(), c.kepid.to_numpy())}

    # ---- verdicts ----
    t1 = res["T1"]
    co = res["T3"]["fp_rejection_by_exclusive_flag"]["co_only"]
    v = {}
    if not a.dry_run:
        v = {"H1 C rejects >= 0.95 of INV": t1["INV"]["C"][0] >= 0.95,
             "H2 C keeps >= 0.90 of INJ1": t1["INJ1"]["C"][0] >= 0.90,
             "H3 C rejects INJ2 >= v2 + 0.15 and >= Robovetter": t1["INJ2"]["C"][0] >= t1["INJ2"]["v2"][0] + 0.15 and t1["INJ2"]["C"][0] >= t1["INJ2"]["robovetter"][0],
             "H4 C rejects INJ3 >= v2": t1["INJ3"]["C"][0] >= t1["INJ3"]["v2"][0],
             "H5 C rejects >= 0.95 of NTP test": res["T2"]["C"][0] >= 0.95,
             "H6 C confirmed-vs-FP >= ExoMiner - 0.01 (common KOIs)": res["T3"]["exominer_common"]["C"] >= res["T3"]["exominer_common"]["exominer"] - 0.01,
             "H7 C rejects >= 0.80 of centroid-only FPs": co["C"][0] is not None and co["C"][0] >= 0.80,
             "H8a NTP drives the INV gain (A, C > B)": min(t1["INV"]["A"][0], t1["INV"]["C"][0]) > t1["INV"]["B"][0],
             "H8b diagnostics drive the INJ2 gain (B, C > A)": min(t1["INJ2"]["B"][0], t1["INJ2"]["C"][0]) > t1["INJ2"]["A"][0]}
        p = {"INV": t1["INV"]["C_vs_v2"]["p_exact"], "INJ2": t1["INJ2"]["C_vs_v2"]["p_exact"], "INJ3": t1["INJ3"]["C_vs_v2"]["p_exact"],
             "NTP": res["T2"]["C_vs_v2"]["p_exact"], "INJ1_vs_v2": t1["INJ1"]["C_vs_v2"]["p_exact"]}
        res["holm_adjusted_p_C_vs_v2"] = holm(p)
    res["expectations"] = {k: bool(x) for k, x in v.items()}
    # the guard is written with the results, before anything is printed: a crash earlier means no number was seen
    if not a.dry_run:
        GUARD.write_text(json.dumps({"reads": 1, "at": datetime.now().isoformat(timespec="seconds"), "protocol_commit": "ae91f4e",
                                     "note": "second read of the Kepler v2 test split, declared before v3 training"}))
    (out / "v3_eval.json").write_text(json.dumps(res, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    print(json.dumps({k: res[k] for k in ["thresholds", "val_roc_auc_koi_plus_ntp", "expectations"] if k in res}, indent=2, default=str))
    for s, r in t1.items():
        print(s, r["metric"], {m: r[m][0] for m in MODELS + (["robovetter"] if "robovetter" in r else [])})
    if res["T2"]:
        print("NTP test rejection", {m: res["T2"][m][0] for m in MODELS})
    print("T3", {m: round(res["T3"][m]["planet_like_vs_fp"]["roc_auc"], 4) for m in MODELS},
          "CONF-vs-FP", {m: round(res["T3"][m]["confirmed_vs_fp"]["roc_auc"], 4) for m in MODELS})
    print("exclusive-flag rejection", {g: {m: r[m][0] for m in MODELS} for g, r in res["T3"]["fp_rejection_by_exclusive_flag"].items()})
    if not a.dry_run:
        make_figure(res)
    return 0


def make_figure(res: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = {"v2": "#8a8984", "A": "#eb6834", "B": "#1baf7a", "C": "#2a78d6", "robovetter": "#0b0b0b"}
    labels = {"v2": "v2 (views, KOIs only)", "A": "v3-A (+ real false alarms)", "B": "v3-B (+ diagnostics)",
              "C": "v3-C (+ both)", "robovetter": "Robovetter (tuned on DR25 sims)"}
    rows = [("INJ1", "planets kept\n(INJ1)"), ("INV", "false alarms rejected\n(INV)"), ("INJ2", "off-target rejected\n(INJ2)"),
            ("INJ3", "EB mimics rejected\n(INJ3)")]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.4), width_ratios=[1.6, 1])
    ax = axes[0]
    x = np.arange(len(rows) + 1)
    for j, m in enumerate(["v2", "A", "B", "C", "robovetter"]):
        vals = [res["T1"][s].get(m) for s, _ in rows] + [res["T2"].get(m) if m != "robovetter" else None]
        for xi, val in zip(x, vals):
            if val is None:
                continue
            ax.errorbar(xi + (j - 2) * 0.15, val[0], yerr=[[val[0] - val[1]], [val[2] - val[0]]], fmt="o", color=cols[m],
                        capsize=3, label=labels[m] if xi == 0 or (m == "robovetter" and xi == 0) else None)
    ax.set(xticks=x, xticklabels=[r[1] for r in rows] + ["real false alarms\nrejected (NTP test)"], ylim=(0, 1.02),
           ylabel="fraction (95 % Wilson CI)", title="Fresh DR25 simulations on held-out stars, and real held-out false alarms")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    ax = axes[1]
    g = res["T3"]["fp_rejection_by_exclusive_flag"]
    keys = [k for k in ["ss_only", "ec_only", "nt_only", "co_only"] if g[k]["n"]]
    x = np.arange(len(keys))
    for j, m in enumerate(["v2", "A", "B", "C"]):
        v = np.array([g[k][m] for k in keys], dtype=float)
        ax.errorbar(x + (j - 1.5) * 0.15, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o", color=cols[m], capsize=3, label=labels[m])
    ax.set(xticks=x, xticklabels=[f"{k.replace('_only', '')} only\n(n={g[k]['n']})" for k in keys], ylim=(0, 1.02),
           title="Kepler test false positives rejected, by their only flag\n(co = light from a neighbouring star)")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)
    t = res["T1"]
    fig.suptitle(f"Vetter v3 (pre-registered): false alarms rejected {t['INV']['v2'][0]:.2f} -> {t['INV']['C'][0]:.2f}, "
                 f"off-target {t['INJ2']['v2'][0]:.2f} -> {t['INJ2']['C'][0]:.2f}, planets kept {t['INJ1']['v2'][0]:.2f} -> {t['INJ1']['C'][0]:.2f}",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(R / "figures" / "17_vetter_v3.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())
