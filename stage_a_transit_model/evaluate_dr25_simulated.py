"""
evaluate_dr25_simulated.py
==========================

The Kepler DR25 simulated-data benchmark, as pre-registered in eda/DR25_SIMULATED_PROTOCOL.md (commit 6c92294):
the frozen Kepler v2 ensemble and its validation recall-90 threshold on 1,000 injected transits (INJ1) and
1,000 inverted-light-curve false alarms (INV), head to head with the Robovetter on the same TCEs.

    python stage_a_transit_model/evaluate_dr25_simulated.py
Output: outputs/dr25_simulated/{dr25_eval.json, dr25_predictions.csv}, figures/16_dr25_simulated.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from stage_a_transit_model import evaluate_v2 as ev  # noqa: E402
from stage_a_transit_model.cnn_lstm import PLANET_LIKE_IDX, consolidate_views, drop_unusable  # noqa: E402

OUT = R / "outputs" / "dr25_simulated"
VIEWS = {"INJ1": R / "detection_views_err" / "dr25_inj1", "INV": R / "detection_views_err" / "dr25_inv"}
RUNS = [f"v2_kepler_dual_cnn_lstm_err{s}" for s in ["", "_s1", "_s2", "_s7", "_s11"]]
THR = 0.5926
MES_BINS = [(7.1, 10), (10, 15), (15, 25), (25, np.inf)]
PERIOD_BINS = [(0, 10), (10, 100), (100, 300), (300, np.inf)]


def wilson(k: int, n: int, z: float = 1.96) -> list:
    if n == 0:
        return [None, None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(p, 4), round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact McNemar test for two paired binary outcomes (ours vs Robovetter on the same TCEs)."""
    from scipy.stats import binomtest
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    p = binomtest(n10, n10 + n01).pvalue if n10 + n01 else 1.0
    return {"ours_only": n10, "robovetter_only": n01, "p_exact": float(p)}


def load() -> pd.DataFrame:
    from preprocessing_pipeline import _npz_name
    frames = []
    for s, d in VIEWS.items():
        vs = drop_unusable(consolidate_views(d, verbose=False), verbose=False)
        meta = []
        for oid in vs.object_id:
            with np.load(d / _npz_name(oid), allow_pickle=False) as z:
                meta.append(json.loads(str(z["metadata"])))
        m = pd.DataFrame(meta)
        p, per = ev.ensemble_proba(RUNS, vs)
        m["score"] = p[:, PLANET_LIKE_IDX].sum(1)
        m["kic"] = vs.star_id
        frames.append(m)
    return pd.concat(frames, ignore_index=True)


def rates(d: pd.DataFrame, positive: bool) -> dict:
    ours = (d.score >= THR) if positive else (d.score < THR)
    robo = (d.robovetter_disp == "PC") if positive else (d.robovetter_disp == "FP")
    return {"n": int(len(d)), "ours": wilson(int(ours.sum()), len(d)), "robovetter": wilson(int(robo.sum()), len(d)),
            "paired": mcnemar(ours.to_numpy(), robo.to_numpy())}


def main() -> int:
    from sklearn.metrics import roc_auc_score

    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    d["y"] = (d["set"] == "INJ1").astype(int)
    inj, inv = d[d.y == 1], d[d.y == 0]
    sample = {s: pd.read_csv(R / "external_data" / "kepler_dr25_simulated" / f"sample_{s.lower()}.csv") for s in VIEWS}
    res = {"threshold": THR, "views_built": {s: int((d["set"] == s).sum()) for s in VIEWS},
           "sampled": {s: int(len(t)) for s, t in sample.items()},
           "completeness_INJ1": rates(inj, True), "effectiveness_INV": rates(inv, False),
           "completeness_by_MES": {f"{lo:g}-{hi:g}": rates(inj[(inj.MES >= lo) & (inj.MES < hi)], True) for lo, hi in MES_BINS},
           "effectiveness_by_MES": {f"{lo:g}-{hi:g}": rates(inv[(inv.MES >= lo) & (inv.MES < hi)], False) for lo, hi in MES_BINS},
           "completeness_by_period": {f"{lo:g}-{hi:g}": rates(inj[(inj.period_days >= lo) & (inj.period_days < hi)], True)
                                      for lo, hi in PERIOD_BINS}}
    stars = d.kic.to_numpy()
    y = d.y.to_numpy()
    res["roc_auc_inj_vs_inv"] = {
        "ours": round(float(roc_auc_score(y, d.score)), 4),
        "ours_ci": ev.star_boot(stars, lambda i: roc_auc_score(y[i], d.score.to_numpy()[i])),
        "robovetter_score": round(float(roc_auc_score(y, d.robovetter_score)), 4),
        "ours_minus_robovetter": ev.paired_auc_diff(y, d.score.to_numpy(), d.robovetter_score.to_numpy(), stars)}
    c, e, auc = res["completeness_INJ1"], res["effectiveness_INV"], res["roc_auc_inj_vs_inv"]
    mes = [res["completeness_by_MES"][k]["ours"][0] for k in res["completeness_by_MES"]]
    mes = [x for x in mes if x is not None]
    res["expectations"] = {
        "D1 completeness in [0.85, 0.97] and >= Robovetter": bool(0.85 <= c["ours"][0] <= 0.97 and c["ours"][0] >= c["robovetter"][0]),
        "D1a completeness in [0.85, 0.97]": bool(0.85 <= c["ours"][0] <= 0.97),
        "D1b completeness >= Robovetter on the same TCEs": bool(c["ours"][0] >= c["robovetter"][0]),
        "D2 effectiveness in [0.40, 0.85], far below Robovetter": bool(0.40 <= e["ours"][0] <= 0.85 and e["ours"][0] < e["robovetter"][0]),
        "D3 ROC-AUC in [0.80, 0.95] and below Robovetter": bool(0.80 <= auc["ours"] <= 0.95 and auc["ours"] < auc["robovetter_score"]),
        "D4 completeness rises with MES (first bin < last bin)": bool(len(mes) >= 2 and mes[0] < mes[-1])}
    keep = ["object_id", "set", "kic", "tce_id", "period_days", "MES", "expected_mes", "depth_ppm", "depth_snr",
            "robovetter_disp", "robovetter_score", "score"]
    d[keep].to_csv(OUT / "dr25_predictions.csv", index=False)
    (OUT / "dr25_eval.json").write_text(json.dumps(res, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    print(json.dumps({k: res[k] for k in ["views_built", "completeness_INJ1", "effectiveness_INV", "roc_auc_inj_vs_inv", "expectations"]},
                     indent=2, default=str))
    make_figure(res, d)
    return 0


def make_figure(res: dict, d: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    BLUE, GREY = "#2a78d6", "#8a8984"
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, key, title in [(axes[0], "completeness_by_MES", "Completeness: injected planets kept (INJ1)"),
                           (axes[1], "effectiveness_by_MES", "Effectiveness: inverted-light-curve false alarms rejected (INV)")]:
        bins = list(res[key])
        x = np.arange(len(bins))
        for off, who, col, lab in [(-0.1, "ours", BLUE, "CNN+BiLSTM ensemble (frozen)"), (0.1, "robovetter", GREY, "Robovetter (tuned on these sets)")]:
            v = np.array([res[key][b][who] for b in bins], dtype=float)
            ax.errorbar(x + off, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o-", color=col, capsize=3, label=lab)
        ax.set(xticks=x, xticklabels=[f"MES {b}\n(n={res[key][b]['n']})" for b in bins], ylim=(0, 1.02), title=title,
               ylabel="fraction (95 % Wilson CI)")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
    ax = axes[2]
    y = (d["set"] == "INJ1").astype(int)
    for col, lab, c in [("score", f"ours: {res['roc_auc_inj_vs_inv']['ours']:.3f}", BLUE),
                        ("robovetter_score", f"Robovetter score: {res['roc_auc_inj_vs_inv']['robovetter_score']:.3f}", GREY)]:
        f, t, _ = roc_curve(y, d[col])
        ax.plot(f, t, color=c, lw=2, label=lab)
    ax.plot([0, 1], [0, 1], ls="--", color="0.75", lw=1)
    ax.set(xlabel="false-alarm rate (INV)", ylabel="true-positive rate (INJ1)", title="Injected planets vs inverted false alarms")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    c, e = res["completeness_INJ1"], res["effectiveness_INV"]
    fig.suptitle(f"Kepler DR25 simulated data, stars never seen in training: the vetter keeps more injected planets than the Robovetter "
                 f"({c['ours'][0]:.3f} vs {c['robovetter'][0]:.3f})\nbut passes {1 - e['ours'][0]:.0%} of inverted-light-curve "
                 f"false alarms (Robovetter {1 - e['robovetter'][0]:.1%}), mostly long-period, few-transit ones",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(R / "figures" / "16_dr25_simulated.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())
