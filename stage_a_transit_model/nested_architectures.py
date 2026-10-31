"""
nested_architectures.py
=======================

eda/POST_REVIEW_PROTOCOL.md, analysis A (post hoc): the CNN-BiLSTM, ResNet and Transformer with equal tuning, by nested
cross-validation on train + validation. Test folders are never opened.

    python stage_a_transit_model/nested_architectures.py --mission kepler --jobs 4
    python stage_a_transit_model/nested_architectures.py --mission kepler --report
Output: outputs/post_review/nested_arch/<mission>/ (one .json and .npy per run), nested_arch_<mission>.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from stage_a_transit_model import vetter_v3 as v  # noqa: E402
from stage_a_transit_model.cnn_lstm import (PLANET_LIKE_IDX, class_weights, consolidate_views, drop_unusable,  # noqa: E402
                                            model_inputs)

OUT = R / "outputs" / "post_review" / "nested_arch"
ARCHES = ["cnn_lstm", "resnet", "transformer"]
GRID = [(3e-4, 0.3), (1e-3, 0.3), (3e-3, 0.3), (1e-3, 0.1)]
EXTRA_SEEDS, FOLDS, SEED = [1], 5, 20261031
_D: dict = {}


def groups(mission: str, ids: np.ndarray, stars: np.ndarray) -> np.ndarray:
    """Connected components of star and signal family, so that no star or family crosses a fold."""
    f = R / "splits" / ("koi_cumulative_split_v2.csv" if mission == "kepler" else "toi_tess_candidates_split_v2.csv")
    t = pd.read_csv(f, low_memory=False)
    t["oid"] = t.kepoi_name.astype(str) if mission == "kepler" else "TOI " + t.toi.astype(str)
    t = t.drop_duplicates("oid")
    assert pd.Series(ids.astype(str)).isin(set(t.oid)).mean() > 0.99
    fam = t.set_index("oid").signal_family_id.astype(str)
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for oid, star in zip(ids.astype(str), stars.astype(str)):
        a, b = find("s" + star), find("f" + str(fam.get(oid, oid)))
        parent[a] = b
    roots = [find("s" + s) for s in stars.astype(str)]
    return pd.factorize(np.array(roots))[0]


def data(mission: str) -> dict:
    if not _D:
        import torch
        sets = [drop_unusable(consolidate_views(v.KOI_VIEWS / f"{mission}_{s}", verbose=False), verbose=False) for s in ("train", "val")]
        vs = v.concat(*sets)
        g, l = model_inputs(vs, True)
        _D.update(g=torch.from_numpy(g), l=torch.from_numpy(l), y=vs.label_index.astype(np.int64),
                  ids=vs.object_id.astype(str), groups=groups(mission, vs.object_id, vs.star_id))
    return _D


def folds(mission: str):
    from sklearn.model_selection import StratifiedGroupKFold
    d = data(mission)
    out = []
    n = len(d["y"])
    for tr, te in StratifiedGroupKFold(FOLDS, shuffle=True, random_state=SEED).split(np.zeros(n), d["y"], d["groups"]):
        i, h = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED + 1).split(tr, d["y"][tr], d["groups"][tr]))
        out.append((tr[i], tr[h], te))
    return out


def predict(model, g, l, device, batch=512) -> np.ndarray:
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(g), batch):
            out.append(torch.softmax(model(g[i:i + batch].to(device), l[i:i + batch].to(device), None), dim=1).cpu())
    return torch.cat(out).numpy()


def run_path(mission, fold, arch, lr, dropout, seed) -> Path:
    return OUT / mission / f"f{fold}_{arch}_lr{lr:g}_d{dropout:g}_s{seed}"


def train(mission: str, fold: int, arch: str, lr: float, dropout: float, seed: int) -> dict:
    import torch
    from sklearn.metrics import roc_auc_score
    from torch import nn

    p = run_path(mission, fold, arch, lr, dropout, seed)
    if p.with_suffix(".json").is_file():
        return json.loads(p.with_suffix(".json").read_text())
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    d = data(mission)
    tr, va, te = folds(mission)[fold]
    G, L, Y = d["g"], d["l"], d["y"]
    v.HP["dropout"] = dropout
    model = (v.build_v3(False) if arch == "cnn_lstm" else v.build_arch(arch)).to(device)
    w = torch.tensor(class_weights(Y[tr], "inverse"), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=v.HP["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    yv = np.isin(Y[va], PLANET_LIKE_IDX).astype(int)
    ytr = torch.from_numpy(Y[tr])
    gtr, ltr = G[tr], L[tr]
    best, t0 = {"epoch": -1, "auc": -np.inf, "state": None}, time.time()
    for epoch in range(1, v.HP["epochs"] + 1):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        order = rng.permutation(len(tr))
        for i in range(0, len(order), v.HP["batch_size"]):
            idx = torch.from_numpy(order[i:i + v.HP["batch_size"]])
            gb, lb = gtr[idx].clone(), ltr[idx].clone()
            flip = torch.from_numpy(rng.random(len(idx)) < 0.5)
            gb[flip] = torch.flip(gb[flip], dims=[-1])
            lb[flip] = torch.flip(lb[flip], dims=[-1])
            opt.zero_grad()
            loss = crit(model(gb.to(device), lb.to(device), None), ytr[idx].to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"{p.name}: non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), v.HP["clip"])
            opt.step()
        auc = float(roc_auc_score(yv, predict(model, G[va], L[va], device)[:, PLANET_LIKE_IDX].sum(1)))
        sched.step(auc)
        if auc > best["auc"] + v.HP["min_delta"]:
            best = {"epoch": epoch, "auc": auc, "state": {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}}
        if epoch - best["epoch"] >= v.HP["patience"]:
            break
    model.load_state_dict(best["state"])
    np.save(p.with_suffix(".npy"), predict(model, G[te], L[te], device)[:, PLANET_LIKE_IDX].sum(1).astype(np.float32))
    rep = {"fold": fold, "arch": arch, "lr": lr, "dropout": dropout, "seed": seed, "inner_val_auc": best["auc"], "best_epoch": best["epoch"],
           "minutes": round((time.time() - t0) / 60, 2), "n_params": sum(x.numel() for x in model.parameters())}
    p.with_suffix(".json").write_text(json.dumps(rep))
    print(f"{p.name}: inner val {best['auc']:.4f}, epoch {best['epoch']}, {rep['minutes']} min", flush=True)
    return rep


def chosen(mission: str) -> dict:
    res = {}
    for fold, arch in itertools.product(range(FOLDS), ARCHES):
        runs = [json.loads(run_path(mission, fold, arch, lr, d, 42).with_suffix(".json").read_text()) for lr, d in GRID]
        b = max(runs, key=lambda r: r["inner_val_auc"])
        res[(fold, arch)] = (b["lr"], b["dropout"])
    return res


def sweep(mission: str, jobs: int) -> None:
    (OUT / mission).mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    tune = [(mission, f, a, lr, d, 42) for f, a, (lr, d) in itertools.product(range(FOLDS), ARCHES, GRID)]
    with ProcessPoolExecutor(jobs, mp_context=ctx) as ex:
        for f in as_completed([ex.submit(train, *t) for t in tune]):
            f.result()
    extra = [(mission, f, a, *cfg, s) for (f, a), cfg in chosen(mission).items() for s in EXTRA_SEEDS]
    with ProcessPoolExecutor(jobs, mp_context=ctx) as ex:
        for f in as_completed([ex.submit(train, *t) for t in extra]):
            f.result()
    print("sweep complete", flush=True)


def report(mission: str) -> int:
    from sklearn.metrics import roc_auc_score
    d = data(mission)
    y3, grp = d["y"], d["groups"]
    y = np.isin(y3, PLANET_LIKE_IDX).astype(int)
    cfg = chosen(mission)
    fs = folds(mission)
    oof = {a: np.full(len(y), np.nan) for a in ARCHES}
    default = {a: np.full(len(y), np.nan) for a in ARCHES}
    per_fold: dict = {a: [] for a in ARCHES}
    for fold, (_, _, te) in enumerate(fs):
        for a in ARCHES:
            lr, dr = cfg[(fold, a)]
            s = np.mean([np.load(run_path(mission, fold, a, lr, dr, sd).with_suffix(".npy")) for sd in [42] + EXTRA_SEEDS], axis=0)
            oof[a][te] = s
            default[a][te] = np.load(run_path(mission, fold, a, 1e-3, 0.3, 42).with_suffix(".npy"))
            per_fold[a].append(round(float(roc_auc_score(y[te], s)), 4))
    assert not any(np.isnan(x).any() for x in oof.values())
    rng = np.random.default_rng(SEED)
    ug = np.unique(grp)
    members = pd.Series(np.arange(len(y))).groupby(grp).apply(np.array)
    aucs = {a: float(roc_auc_score(y, oof[a])) for a in ARCHES}
    boots = {a: [] for a in ARCHES}
    for _ in range(2000):
        idx = np.concatenate(members.loc[rng.choice(ug, len(ug))].to_numpy())
        if len(set(y[idx])) < 2:
            continue
        for a in ARCHES:
            boots[a].append(roc_auc_score(y[idx], oof[a][idx]))
    b = {a: np.array(x) for a, x in boots.items()}
    mask_cf = y3 != 1
    res = {"protocol": "eda/POST_REVIEW_PROTOCOL.md (A), post hoc", "mission": mission, "n": int(len(y)), "positive_rate": round(float(y.mean()), 3),
           "folds": FOLDS, "grid": GRID, "seeds": [42] + EXTRA_SEEDS,
           "oof_roc_auc": {a: round(aucs[a], 4) for a in ARCHES},
           "oof_roc_auc_ci": {a: [round(float(q), 4) for q in np.percentile(b[a], [2.5, 97.5])] for a in ARCHES},
           "confirmed_vs_fp_oof": {a: round(float(roc_auc_score(y[mask_cf], oof[a][mask_cf])), 4) for a in ARCHES},
           "default_config_single_seed_oof": {a: round(float(roc_auc_score(y, default[a])), 4) for a in ARCHES},
           "per_fold": per_fold,
           "difference_vs_cnn_lstm": {a: {"mean": round(aucs[a] - aucs["cnn_lstm"], 4),
                                          "ci95": [round(float(q), 4) for q in np.percentile(b[a] - b["cnn_lstm"], [2.5, 97.5])]}
                                      for a in ("resnet", "transformer")},
           "configs_chosen": {a: [list(cfg[(f, a)]) for f in range(FOLDS)] for a in ARCHES},
           "n_params": {a: json.loads(run_path(mission, 0, a, 1e-3, 0.3, 42).with_suffix(".json").read_text())["n_params"] for a in ARCHES}}
    (OUT.parent / f"nested_arch_{mission}.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", choices=["kepler", "tess"], required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="one fold, one configuration")
    a = ap.parse_args()
    if a.report:
        return report(a.mission)
    if a.smoke:
        (OUT / a.mission).mkdir(parents=True, exist_ok=True)
        d = data(a.mission)
        print(len(d["y"]), "signals", len(set(d["groups"])), "groups", [(len(x), len(y), len(z)) for x, y, z in folds(a.mission)], flush=True)
        for x in folds(a.mission):
            assert not set(d["groups"][x[2]]) & set(d["groups"][x[0]]) and not set(d["groups"][x[1]]) & set(d["groups"][x[0]])
        return 0
    sweep(a.mission, a.jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
