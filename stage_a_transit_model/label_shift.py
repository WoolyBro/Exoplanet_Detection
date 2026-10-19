"""
label_shift.py
==============

eda/LABEL_SHIFT_PROTOCOL.md: the survey prior as label shift. A label-free per-depth-stratum correction
(Saerens, Latinne & Decaestecker 2002 EM) of each ensemble's frozen Platt-calibrated probabilities on a survey
it was not trained on, in three cases (Kepler->K2, Kepler->TESS, TESS->Kepler). Target labels are used only to
evaluate.

    python stage_a_transit_model/label_shift.py               # L1-L4 (registered)
    python stage_a_transit_model/label_shift.py --few-label   # addendum L5: m labelled signals per stratum
Output: outputs/stage_a_mechanisms/label_shift.json, label_shift_few_label.json, figures/21_label_shift.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from stage_a_transit_model import evaluate_v2 as ev  # noqa: E402
from stage_a_transit_model.cnn_lstm import PLANET_LIKE_IDX, consolidate_views, drop_unusable  # noqa: E402

OUT = R / "outputs" / "stage_a_mechanisms"
EDGES = [0, 1000, 3000, 10000, np.inf]
STRATA = ["<1,000", "1,000-3,000", "3,000-10,000", ">10,000"]
RUNS = {"kepler": [f"v2_kepler_dual_cnn_lstm_err{s}" for s in ["", "_s1", "_s2", "_s7", "_s11"]],
        "tess": [f"v2_tess_dual_cnn{s}" for s in ["", "_s1", "_s2", "_s7", "_s11"]]}
PLATT = {"kepler": (1.0945, -0.9221), "tess": (1.0121, 1.0341)}


def platt(model, s):
    s = np.clip(np.asarray(s, float), 1e-6, 1 - 1e-6)
    a, b = PLATT[model]
    return 1 / (1 + np.exp(-(a * np.log(s / (1 - s)) + b)))


def depth_of(views_dir: Path, ids) -> np.ndarray:
    from preprocessing_pipeline import _npz_name
    out = []
    for oid in ids:
        with np.load(views_dir / _npz_name(oid), allow_pickle=False) as z:
            out.append(json.loads(str(z["metadata"])).get("depth_ppm", np.nan))
    return np.array(out, float)


def scored(model: str, folder: str):
    d = R / "detection_views_v2" / folder
    vs = drop_unusable(consolidate_views(d, verbose=False), verbose=False)
    s = ev.ensemble_proba(RUNS[model], vs)[0][:, PLANET_LIKE_IDX].sum(1)
    return pd.DataFrame({"object_id": vs.object_id, "label": vs.label_index.astype(int), "y": np.isin(vs.label_index, PLANET_LIKE_IDX).astype(int),
                         "p": platt(model, s), "depth": depth_of(d, vs.object_id)})


def stratum(depth):
    return np.clip(np.digitize(np.nan_to_num(depth, nan=0.0), EDGES) - 1, 0, len(STRATA) - 1)


def em(p, pi_s, tol=1e-6, it=1000):
    pi_s = float(np.clip(pi_s, 0.01, 0.99))
    pi = pi_s
    for _ in range(it):
        r1, r0 = pi / pi_s * p, (1 - pi) / (1 - pi_s) * (1 - p)
        post = r1 / (r1 + r0)
        new = float(post.mean())
        if abs(new - pi) < tol:
            break
        pi = new
    return pi, post


def shift(p, pi_s, pi_t):
    pi_s, pi_t = float(np.clip(pi_s, 0.01, 0.99)), float(np.clip(pi_t, 0.01, 0.99))
    r1, r0 = pi_t / pi_s * p, (1 - pi_t) / (1 - pi_s) * (1 - p)
    return r1 / (r1 + r0)


def ece(y, p, bins=10):
    o = np.argsort(p)
    return float(sum(len(c) / len(p) * abs(p[c].mean() - y[c].mean()) for c in np.array_split(o, bins) if len(c)))


def run_case(name, model, src, tgt) -> tuple[dict, pd.DataFrame]:
    ss, ts = stratum(src.depth.to_numpy()), stratum(tgt.depth.to_numpy())
    corrected, oracle, cells = tgt.p.to_numpy().copy(), tgt.p.to_numpy().copy(), []
    for k, lab in enumerate(STRATA):
        m = ts == k
        if not m.any():
            continue
        pi_s = float(src.y.to_numpy()[ss == k].mean()) if (ss == k).any() else float(src.y.mean())
        pi_em, post = em(tgt.p.to_numpy()[m], pi_s)
        corrected[m] = post
        pi_true = float(tgt.y.to_numpy()[m].mean())
        oracle[m] = shift(tgt.p.to_numpy()[m], pi_s, pi_true)
        cells.append({"stratum": lab, "n": int(m.sum()), "source_prior": round(pi_s, 3), "em_target_prior": round(pi_em, 3),
                      "true_target_fraction": round(pi_true, 3), "abs_error": round(abs(pi_em - pi_true), 3)})
    y = tgt.y.to_numpy()
    res = {"case": name, "n": int(len(y)), "strata": cells}
    for k, p in [("frozen_platt", tgt.p.to_numpy()), ("label_free", corrected), ("oracle", oracle)]:
        res[k] = {"roc_auc": round(float(roc_auc_score(y, p)), 4), "ece": round(ece(y, p), 4)}
    return res, tgt.assign(p_corrected=corrected, p_oracle=oracle)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    kval, tval = scored("kepler", "kepler_val"), scored("tess", "tess_val")
    k2 = pd.read_csv(R / "stage_a_transit_model" / "runs" / "k2_blind" / "k2_predictions.csv")
    k2 = k2[k2.duration_source.isin(["catalogue", "other set"])]
    k2 = pd.DataFrame({"object_id": k2.object_id, "label": k2.true_label.map({"FALSE POSITIVE": 0, "CANDIDATE": 1, "CONFIRMED": 2}),
                       "y": (k2.true_label != "FALSE POSITIVE").astype(int), "p": platt("kepler", k2.score_kepler), "depth": k2.depth_ppm})
    cases = [("Kepler -> K2", "kepler", kval, k2), ("Kepler -> TESS", "kepler", kval, scored("kepler", "tess_test")),
             ("TESS -> Kepler", "tess", tval, scored("tess", "kepler_test"))]
    out, frames = {"protocol": "eda/LABEL_SHIFT_PROTOCOL.md", "cases": []}, {}
    for name, model, src, tgt in cases:
        r, f = run_case(name, model, src, tgt)
        out["cases"].append(r)
        frames[name] = f
    f = frames["Kepler -> K2"]
    conf = f[f.label == 2]
    deep = conf.depth > 10000
    gap = lambda col: float(conf[~deep][col].median() - conf[deep][col].median())
    out["k2_depth_prior"] = {"n_deep": int(deep.sum()), "n_shallow": int((~deep).sum()),
                             "deep_median": {c: round(float(conf[deep][c].median()), 3) for c in ["p", "p_corrected", "p_oracle"]},
                             "shallow_median": {c: round(float(conf[~deep][c].median()), 3) for c in ["p", "p_corrected", "p_oracle"]},
                             "gap": {c: round(gap(c), 3) for c in ["p", "p_corrected", "p_oracle"]}}
    cs = out["cases"]
    cells = [c for r in cs for c in r["strata"] if c["n"] >= 20]
    kd = out["k2_depth_prior"]
    out["expectations"] = {
        "L1 label-free correction halves ECE in every case": all(r["label_free"]["ece"] <= 0.5 * r["frozen_platt"]["ece"] for r in cs),
        "L2 K2 deep median >= 0.85, gap halves, AUC drop <= 0.005": bool(
            kd["deep_median"]["p_corrected"] >= 0.85 and kd["gap"]["p_corrected"] <= 0.5 * kd["gap"]["p"]
            and cs[0]["label_free"]["roc_auc"] >= cs[0]["frozen_platt"]["roc_auc"] - 0.005),
        "L3 EM prior within 0.15 of truth in >= 75 % of cells (n >= 20)": bool(np.mean([c["abs_error"] <= 0.15 for c in cells]) >= 0.75),
        "L4 ROC-AUC improves in >= 2 of 3 cases": bool(sum(r["label_free"]["roc_auc"] > r["frozen_platt"]["roc_auc"] for r in cs) >= 2)}
    (OUT / "label_shift.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))
    figure(out, frames)
    return 0


def figure(out, frames):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    cols = ["#2a78d6", "#e8743b", "#1a9e77"]
    for i, r in enumerate(out["cases"]):
        st = r["strata"]
        x = np.arange(len(st)) + (i - 1) * 0.25
        ax.plot(x, [c["source_prior"] for c in st], "s", color=cols[i], mfc="white", ms=7)
        ax.plot(x, [c["true_target_fraction"] for c in st], "o", color=cols[i], ms=7, label=f"{r['case']}: target truth")
        ax.plot(x, [c["em_target_prior"] for c in st], "x", color="k", ms=8)
    ax.set(xticks=range(len(STRATA)), xticklabels=[s + "\nppm" for s in STRATA], ylabel="planet-like fraction", ylim=(-0.03, 1.03),
           title="Base rate by depth: source (open), target truth (filled),\nlabel-free EM estimate (x)")
    ax.legend(fontsize=7, loc="lower left")
    ax = axes[1]
    x = np.arange(len(out["cases"]))
    for j, (k, lab, c) in enumerate([("frozen_platt", "frozen Platt map", "#9a9a9a"), ("label_free", "label-free correction", "#2a78d6"),
                                     ("oracle", "oracle (true base rates)", "#1a9e77")]):
        ax.bar(x + (j - 1) * 0.27, [r[k]["ece"] for r in out["cases"]], 0.27, color=c, label=lab)
    ax.set(xticks=x, xticklabels=[r["case"] for r in out["cases"]], ylabel="expected calibration error", title="Calibration on a survey never trained on")
    ax.legend(fontsize=8)
    ax = axes[2]
    f = frames["Kepler -> K2"]
    c = f[f.label == 2]
    ax.scatter(c.depth.clip(lower=10), c.p, s=9, color="#9a9a9a", alpha=0.6, label="frozen Platt map")
    ax.scatter(c.depth.clip(lower=10), c.p_corrected, s=9, color="#2a78d6", alpha=0.6, label="label-free correction")
    ax.axvline(10000, color="k", ls=":", lw=1)
    kd = out["k2_depth_prior"]
    ax.set(xscale="log", xlabel="measured depth (ppm)", ylabel="P(planet-like)",
           title=f"K2 confirmed planets: deep median {kd['deep_median']['p']:.2f} -> {kd['deep_median']['p_corrected']:.2f}")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(R / "figures" / "21_label_shift.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__" and "--few-label" not in sys.argv:
    raise SystemExit(main())


def few_label() -> int:
    """Addendum L5: per-stratum base rates from m labelled target signals per stratum (200 draws each)."""
    rng = np.random.default_rng(20261019)
    kval, tval = scored("kepler", "kepler_val"), scored("tess", "tess_val")
    k2 = pd.read_csv(R / "stage_a_transit_model" / "runs" / "k2_blind" / "k2_predictions.csv")
    k2 = k2[k2.duration_source.isin(["catalogue", "other set"])]
    k2 = pd.DataFrame({"y": (k2.true_label != "FALSE POSITIVE").astype(int), "p": platt("kepler", k2.score_kepler), "depth": k2.depth_ppm})
    cases = [("Kepler -> K2", kval, k2), ("Kepler -> TESS", kval, scored("kepler", "tess_test")), ("TESS -> Kepler", tval, scored("tess", "kepler_test"))]
    out = {"addendum": "eda/LABEL_SHIFT_PROTOCOL.md (addendum, L5)", "cases": {}}
    ok_all = True
    for name, src, tgt in cases:
        ss, ts = stratum(src.depth.to_numpy()), stratum(tgt.depth.to_numpy())
        y, p = tgt.y.to_numpy(), tgt.p.to_numpy()
        pi_s = {k: float(src.y.to_numpy()[ss == k].mean()) if (ss == k).any() else float(src.y.mean()) for k in range(len(STRATA))}
        pi_true = {k: float(y[ts == k].mean()) for k in range(len(STRATA)) if (ts == k).any()}
        res = {}
        for m in [5, 10, 20, 50]:
            rows = []
            for _ in range(200):
                lab = np.zeros(len(y), bool)
                for k in pi_true:
                    idx = np.flatnonzero(ts == k)
                    if len(idx) > m:
                        lab[rng.choice(idx, m, replace=False)] = True
                q, o = p.copy(), p.copy()
                for k in pi_true:
                    mk = ts == k
                    est = float(y[mk & lab].mean()) if (mk & lab).any() else pi_s[k]
                    q[mk], o[mk] = shift(p[mk], pi_s[k], est), shift(p[mk], pi_s[k], pi_true[k])
                h = ~lab
                rows.append((ece(y[h], q[h]), ece(y[h], o[h]), ece(y[h], p[h]),
                             roc_auc_score(y[h], q[h]), roc_auc_score(y[h], p[h])))
            a = np.array(rows)
            res[m] = {"labels_used": int(m * len(pi_true)), "ece_few": round(float(np.median(a[:, 0])), 4),
                      "ece_oracle": round(float(np.median(a[:, 1])), 4), "ece_frozen": round(float(np.median(a[:, 2])), 4),
                      "auc_few": round(float(np.median(a[:, 3])), 4), "auc_frozen": round(float(np.median(a[:, 4])), 4)}
        r20 = res[20]
        ok_all &= (r20["ece_few"] - r20["ece_oracle"] <= 0.03) and (r20["auc_few"] >= r20["auc_frozen"] - 0.01)
        out["cases"][name] = res
    out["expectations"] = {"L5 at m = 20: ECE within 0.03 of the oracle and AUC within 0.01 of the frozen map, all cases": bool(ok_all)}
    (OUT / "label_shift_few_label.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))
    return 0


if __name__ == "__main__" and "--few-label" in sys.argv:
    raise SystemExit(few_label())
