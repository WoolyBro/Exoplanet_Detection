"""
evaluate_architectures.py
=========================

eda/ARCHITECTURE_PROTOCOL.md: v2's CNN + BiLSTM against a 1-D ResNet (R) and a Transformer (T), each a 5-seed
ensemble trained on the same Kepler v2 data with the same recipe, on the Kepler v2 validation split (every test
set is spent). The data effect for scale: config B (v2 + pipeline diagnostics) on the same split.

    python stage_a_transit_model/evaluate_architectures.py
Output: outputs/stage_a_mechanisms/architectures.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from stage_a_transit_model import evaluate_v2 as ev  # noqa: E402
from stage_a_transit_model import vetter_v3 as v3  # noqa: E402
from stage_a_transit_model.cnn_lstm import PLANET_LIKE_IDX  # noqa: E402

OUT = R / "outputs" / "stage_a_mechanisms"
V2_RUNS = [f"v2_kepler_dual_cnn_lstm_err{s}" for s in ["", "_s1", "_s2", "_s7", "_s11"]]


def runs(cfg):
    return [v3.RUNS / (f"v3_{cfg}" + ("" if s == 42 else f"_s{s}")) for s in v3.SEEDS]


def main() -> int:
    val = v3.load_split("val", ntp=False)
    y = np.isin(val.label_index, PLANET_LIKE_IDX).astype(int)
    scores = {"v2": ev.ensemble_proba(V2_RUNS, val)[0][:, PLANET_LIKE_IDX].sum(1)}
    for c in ["R", "T", "B"]:
        missing = [r.name for r in runs(c) if not (r / "report.json").is_file()]
        if missing:
            raise SystemExit(f"config {c} not finished: {missing}")
        scores[c] = np.mean([v3.score(r, val)[:, PLANET_LIKE_IDX].sum(1) for r in runs(c)], axis=0)
    res = {"protocol": "eda/ARCHITECTURE_PROTOCOL.md", "split": "Kepler v2 validation (KOIs)", "n": int(len(y)),
           "params": {"v2": 40451, **{c: json.loads((runs(c)[0] / "report.json").read_text())["n_params"] for c in ["R", "T", "B"]}},
           "per_seed_val_roc_auc": {c: [json.loads((r / "report.json").read_text())["val_roc_auc"] for r in runs(c)] for c in ["R", "T", "B"]},
           "ensemble": {m: ev.binary_block(y, s, val.star_id) for m, s in scores.items()},
           "minus_v2": {m: ev.paired_auc_diff(y, scores[m], scores["v2"], val.star_id) for m in ["R", "T", "B"]}}
    auc = {m: res["ensemble"][m]["roc_auc"] for m in scores}
    arch_gain = max(auc["R"], auc["T"]) - auc["v2"]
    data_gain = auc["B"] - auc["v2"]
    res["gains"] = {"best_architecture_minus_v2": round(arch_gain, 4), "data_B_minus_v2": round(data_gain, 4)}
    res["expectations"] = {"X1 neither R nor T beats v2 by more than 0.01": bool(arch_gain <= 0.01),
                           "X2 best architecture gain < a quarter of the data gain": bool(arch_gain < 0.25 * data_gain)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "architectures.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in ["params", "gains", "expectations"]}, indent=2, default=float))
    print({m: round(v, 4) for m, v in auc.items()}, {m: res["minus_v2"][m] for m in ["R", "T", "B"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
