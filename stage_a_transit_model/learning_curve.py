"""
learning_curve.py
=================

eda/POST_REVIEW_PROTOCOL.md, analysis O (post hoc): does more of the same data help? Retrains the CNN-BiLSTM of analysis A on nested
fractions of each training set and scores the same held-out folds. No test set is read.

    python stage_a_transit_model/learning_curve.py --mission kepler --jobs 4
    python stage_a_transit_model/learning_curve.py --mission kepler --report
Output: outputs/post_review/learning_curve/, outputs/post_review/learning_curve_<mission>.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from stage_a_transit_model import nested_architectures as na  # noqa: E402

FRACTIONS = (0.125, 0.25, 0.5)
ARCH = "cnn_lstm"
A_OUT = na.OUT
BASE_FOLDS = na.folds
OUT = R / "outputs" / "post_review" / "learning_curve"


def subset(mission: str, fold: int, frac: float) -> tuple:
    tr, va, te = BASE_FOLDS(mission)[fold]
    g = na.data(mission)["groups"]
    order = np.random.default_rng(na.SEED + 100 + fold).permutation(np.unique(g[tr]))     # one order per fold, so the fractions nest
    return tr[np.isin(g[tr], order[: int(round(frac * len(order)))])], va, te


def one(mission: str, fold: int, frac: float, lr: float, dropout: float, seed: int) -> dict:
    na.OUT = OUT / f"frac{frac:g}"
    (na.OUT / mission).mkdir(parents=True, exist_ok=True)

    def patched(m):
        f = BASE_FOLDS(m)
        f[fold] = subset(m, fold, frac)
        return f

    na.folds = patched
    return na.train(mission, fold, ARCH, lr, dropout, seed)


def sweep(mission: str, jobs: int) -> None:
    cfg = na.chosen(mission)
    tasks = [(mission, f, frac, *cfg[(f, ARCH)], s) for f in range(na.FOLDS) for frac in FRACTIONS for s in [42] + na.EXTRA_SEEDS]
    with ProcessPoolExecutor(jobs, mp_context=mp.get_context("spawn")) as ex:
        for f in as_completed([ex.submit(one, *t) for t in tasks]):
            f.result()
    print("sweep complete", mission, flush=True)


def report(mission: str) -> int:
    from sklearn.metrics import roc_auc_score
    d = na.data(mission)
    y = np.isin(d["y"], na.PLANET_LIKE_IDX).astype(int)
    grp = d["groups"]
    cfg = na.chosen(mission)
    fs = BASE_FOLDS(mission)
    fracs = (*FRACTIONS, 1.0)
    oof = {f: np.full(len(y), np.nan) for f in fracs}
    n_train = {f: [] for f in fracs}
    for fold, (tr, _, te) in enumerate(fs):
        lr, dr = cfg[(fold, ARCH)]
        for f in fracs:
            na.OUT = A_OUT if f == 1.0 else OUT / f"frac{f:g}"
            oof[f][te] = np.mean([np.load(na.run_path(mission, fold, ARCH, lr, dr, s).with_suffix(".npy")) for s in [42] + na.EXTRA_SEEDS], axis=0)
            n_train[f].append(int(len(tr) if f == 1.0 else len(subset(mission, fold, f)[0])))
    assert not any(np.isnan(x).any() for x in oof.values())
    aucs = {f: float(roc_auc_score(y, oof[f])) for f in fracs}
    rng = np.random.default_rng(na.SEED)
    ug = np.unique(grp)
    members = pd.Series(np.arange(len(y))).groupby(grp).apply(np.array)
    boots = {f: [] for f in fracs}
    for _ in range(2000):
        idx = np.concatenate(members.loc[rng.choice(ug, len(ug))].to_numpy())
        if len(set(y[idx])) < 2:
            continue
        for f in fracs:
            boots[f].append(roc_auc_score(y[idx], oof[f][idx]))
    b = {f: np.array(v) for f, v in boots.items()}
    gain = aucs[1.0] - aucs[0.5]
    ci = [round(float(q), 4) for q in np.percentile(b[1.0] - b[0.5], [2.5, 97.5])]
    verdict = "examples no longer the limit" if ci[1] < 0.01 else "examples still the limit" if ci[0] >= 0.01 else "unresolved"
    res = {"protocol": "eda/POST_REVIEW_PROTOCOL.md (O), post hoc", "mission": mission, "n": int(len(y)), "seeds": [42] + na.EXTRA_SEEDS,
           "mean_train_signals": {str(f): round(float(np.mean(n_train[f]))) for f in fracs},
           "oof_roc_auc": {str(f): round(aucs[f], 4) for f in fracs},
           "oof_roc_auc_ci": {str(f): [round(float(q), 4) for q in np.percentile(b[f], [2.5, 97.5])] for f in fracs},
           "gain_50_to_100": {"mean": round(gain, 4), "ci95": ci}, "verdict": verdict}
    (OUT.parent / f"learning_curve_{mission}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", choices=["kepler", "tess"], required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if not a.report:
        sweep(a.mission, a.jobs)
    return report(a.mission)


if __name__ == "__main__":
    raise SystemExit(main())
