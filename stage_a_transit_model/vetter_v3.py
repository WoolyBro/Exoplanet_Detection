"""
vetter_v3.py
============

Vetter v3 (eda/VETTER_V3_PROTOCOL.md): the v2 Kepler network (dual CNN + BiLSTM over the global and local
views with the flux-error channel), trained the same way, with two additions that the v2 failure analysis
asked for:

  * real false alarms (DR25 non-KOI TCEs, "NTP") in training, labelled FALSE POSITIVE;
  * a diagnostic branch: an MLP over 27 raw Robovetter measurements (no Robovetter decision flags),
    concatenated with the view features before the classification head.

Configs (fixed by the protocol; nothing else is trained):
    C  NTP + diagnostics (primary)      A  NTP, no diagnostics      B  diagnostics, no NTP
    N  the v2 recipe with S/N-balanced loss weights, no NTP, no diagnostics (eda/SNR_BALANCE_PROTOCOL.md)

    python stage_a_transit_model/vetter_v3.py --config C --seed 42
    python stage_a_transit_model/vetter_v3.py --sweep --jobs 3          # all 15 runs
Output: stage_a_transit_model/runs/v3_<config>[_s<seed>]/{best.pt, report.json, history.csv}

The v2 code (cnn_lstm.py) is not modified: every reported v2 number stays reproducible.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from stage_a_transit_model.cnn_lstm import (CLASSES, PLANET_LIKE_IDX, ViewSet, build_model, class_weights,  # noqa: E402
                                            consolidate_views, drop_unusable, evaluate, model_inputs,
                                            precision_at_recall)

RUNS = R / "stage_a_transit_model" / "runs"
KOI_VIEWS = R / "detection_views_v2"
NTP_VIEWS = R / "detection_views_err" / "v3_ntp"
V3 = R / "external_data" / "kepler_dr25_simulated" / "v3"
CONFIGS = {"C": {"ntp": True, "diag": True}, "A": {"ntp": True, "diag": False}, "B": {"ntp": False, "diag": True},
           "N": {"ntp": False, "diag": False, "snr_balance": True}}
SEEDS = [42, 1, 2, 7, 11]
FEATURES = ["cent_score", "halo_ghost", "oesig_dv", "mod_oe_dv", "mod_sig_pri_dv", "mod_sig_sec_dv", "mod_sig_ter_dv",
            "mod_sig_pos_dv", "mod_fred_dv", "mod_fa1_dv", "mod_fa2_dv", "mod_depth_sec_dv", "mod_ph_sec_dv", "alb_dv",
            "sdepthsig_dv", "lpp_dv", "all_tran_chases", "sweet_snr", "shape_metric", "ses_to_mes", "mes", "new_mes",
            "nrealt", "ngoodt", "period", "duration", "depth"]
# the v2 Kepler recipe (sweep.py, v2_kepler_dual_cnn_lstm_err)
HP = {"epochs": 60, "batch_size": 64, "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.3, "clip": 5.0, "patience": 8,
      "min_delta": 1e-4, "width": 1.0}


# ----------------------------------------------------------------------------------------------- data
def concat(a: ViewSet, b: ViewSet) -> ViewSet:
    return ViewSet(np.concatenate([a.global_view, b.global_view]), np.concatenate([a.local_view, b.local_view]),
                   np.concatenate([a.label_index, b.label_index]), np.concatenate([a.object_id, b.object_id]),
                   np.concatenate([a.star_id, b.star_id]), np.concatenate([a.global_err, b.global_err]),
                   np.concatenate([a.local_err, b.local_err]))


def ntp_split_map() -> pd.Series:
    s = pd.read_csv(V3 / "ntp_sample.csv", usecols=["id", "split"])
    return s.set_index("id").split


def load_split(split: str, ntp: bool) -> ViewSet:
    vs = drop_unusable(consolidate_views(KOI_VIEWS / f"kepler_{split}", verbose=False), verbose=False)
    if ntp:
        n = drop_unusable(consolidate_views(NTP_VIEWS, verbose=False), verbose=False)
        keep = np.flatnonzero(pd.Series(n.object_id).map(ntp_split_map()).to_numpy() == split)
        vs = concat(vs, n.take(keep))
    return vs


def snr_balance_weights(vs: ViewSet, bins: int = 10, clip: tuple = (0.2, 10.0)) -> np.ndarray:
    """SNR_BALANCE_PROTOCOL.md: w = P(c) / P(c | S/N bin) for the binary class c, clipped, mean 1.
    In the weighted training set the label carries (almost) no information about depth S/N."""
    from preprocessing_pipeline import _npz_name
    snr = []
    for oid in vs.object_id:
        with np.load(KOI_VIEWS / "kepler_train" / _npz_name(oid), allow_pickle=False) as z:
            snr.append(json.loads(str(z["metadata"])).get("depth_snr", np.nan))
    x = np.log10(np.clip(np.nan_to_num(np.asarray(snr, float), nan=1e-3), 1e-3, None))
    b = pd.qcut(x, bins, labels=False, duplicates="drop")
    y = np.isin(vs.label_index, PLANET_LIKE_IDX)
    p, pb = y.mean(), pd.Series(y).groupby(b).mean().to_numpy()
    w = np.where(y, p / pb[b], (1 - p) / (1 - pb[b]))
    w = np.clip(w, *clip)
    return (w / w.mean()).astype(np.float32)


class Diagnostics:
    """Signed-log, robust-standardised (training median / IQR), clipped diagnostics + a missing flag."""

    def __init__(self):
        d = pd.read_csv(V3 / "diagnostics.csv").drop_duplicates("object_id").set_index("object_id")
        self.table = d[FEATURES].apply(pd.to_numeric, errors="coerce")
        self.center = self.scale = None

    @staticmethod
    def _slog(x):
        return np.sign(x) * np.log10(1 + np.abs(x))

    def fit(self, ids) -> "Diagnostics":
        x = self._slog(self.table.reindex(ids).to_numpy(float))
        self.center = np.nanmedian(x, axis=0)
        iqr = np.nanpercentile(x, 75, axis=0) - np.nanpercentile(x, 25, axis=0)
        self.scale = np.where(np.isfinite(iqr) & (iqr > 0), iqr, 1.0)
        return self

    def transform(self, ids) -> np.ndarray:
        x = self._slog(self.table.reindex(ids).to_numpy(float))
        missing = np.isnan(x).all(axis=1, keepdims=True).astype(np.float32)
        z = np.clip((x - self.center) / self.scale, -5, 5)
        return np.concatenate([np.nan_to_num(z, nan=0.0).astype(np.float32), missing], axis=1)

    def state(self) -> dict:
        return {"center": self.center.tolist(), "scale": self.scale.tolist(), "features": FEATURES}

    @classmethod
    def from_state(cls, st: dict) -> "Diagnostics":
        d = cls()
        d.center, d.scale = np.array(st["center"]), np.array(st["scale"])
        return d


# ---------------------------------------------------------------------------------------------- model
def build_v3(use_diag: bool, n_diag: int = len(FEATURES) + 1):
    import torch
    from torch import nn

    base = build_model(HP["dropout"], batchnorm=True, width=HP["width"], arch="dual_cnn_lstm", use_errors=True)

    class V3(nn.Module):
        def __init__(self):
            super().__init__()
            self.global_branch, self.local_branch = base.global_branch, base.local_branch
            n = self.global_branch.out_features + self.local_branch.out_features
            self.diag = (nn.Sequential(nn.Linear(n_diag, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU()) if use_diag else None)
            n += 32 if use_diag else 0
            self.head = nn.Sequential(nn.Dropout(HP["dropout"]), nn.Linear(n, 64), nn.ReLU(),
                                      nn.Dropout(HP["dropout"]), nn.Linear(64, len(CLASSES)))

        def forward(self, g, l, d=None):
            f = [self.global_branch(g), self.local_branch(l)]
            if self.diag is not None:
                f.append(self.diag(d))
            return self.head(torch.cat(f, dim=1))

    m = V3()
    m.use_errors, m.use_diag = True, use_diag
    return m


def predict(model, g, l, d, batch: int = 256) -> np.ndarray:
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(g), batch):
            out.append(torch.softmax(model(g[i:i + batch], l[i:i + batch], None if d is None else d[i:i + batch]), dim=1))
    return torch.cat(out).numpy()


def tensors(vs: ViewSet, diag: Diagnostics | None):
    import torch
    g, l = model_inputs(vs, True)
    d = None if diag is None else torch.from_numpy(diag.transform(vs.object_id))
    return torch.from_numpy(g), torch.from_numpy(l), d


# ------------------------------------------------------------------------------------------- training
def train_one(config: str, seed: int) -> dict:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(2)
    cfg = CONFIGS[config]
    name = f"v3_{config}" + ("" if seed == 42 else f"_s{seed}")
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    tr, va = load_split("train", cfg["ntp"]), load_split("val", cfg["ntp"])
    diag = Diagnostics().fit(tr.object_id) if cfg["diag"] else None
    gtr, ltr, dtr = tensors(tr, diag)
    gva, lva, dva = tensors(va, diag)
    ytr = torch.from_numpy(tr.label_index.astype(np.int64))
    model = build_v3(cfg["diag"])
    w = class_weights(tr.label_index, "inverse")
    sw = None
    if cfg.get("snr_balance"):
        if cfg["ntp"]:
            raise SystemExit("S/N balancing is defined on the KOI training views only")
        sw = torch.from_numpy(snr_balance_weights(tr))
        crit_each = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32), reduction="none")
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name}: train {len(tr)} ({np.bincount(tr.label_index, minlength=3).tolist()}), val {len(va)}, "
          f"{n_params:,} params", flush=True)
    best, history, t0 = {"epoch": -1, "monitor": -np.inf, "state": None}, [], time.time()
    for epoch in range(1, HP["epochs"] + 1):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        order = rng.permutation(len(tr))
        tot, nb = 0.0, 0
        for i in range(0, len(order), HP["batch_size"]):
            idx = torch.from_numpy(order[i:i + HP["batch_size"]])
            gb, lb = gtr[idx], ltr[idx]
            flip = torch.from_numpy(rng.random(len(idx)) < 0.5)
            if flip.any():                                     # mirror about mid-transit (views only)
                gb, lb = gb.clone(), lb.clone()
                gb[flip] = torch.flip(gb[flip], dims=[-1])
                lb[flip] = torch.flip(lb[flip], dims=[-1])
            opt.zero_grad()
            logits = model(gb, lb, None if dtr is None else dtr[idx])
            loss = crit(logits, ytr[idx]) if sw is None else (crit_each(logits, ytr[idx]) * sw[idx]).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"epoch {epoch}: non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), HP["clip"])
            opt.step()
            tot, nb = tot + float(loss.item()), nb + 1
        vm = evaluate(va.label_index.astype(int), predict(model, gva, lva, dva))
        monitor = vm["roc_auc_binary"]
        sched.step(monitor)
        history.append({"epoch": epoch, "train_loss": tot / nb, "val_roc_auc_binary": monitor,
                        "val_accuracy_3class": vm["accuracy_3class"]})
        if monitor > best["monitor"] + HP["min_delta"]:
            best = {"epoch": epoch, "monitor": monitor, "state": {k: v.clone() for k, v in model.state_dict().items()}}
        if epoch - best["epoch"] >= HP["patience"]:
            break
    model.load_state_dict(best["state"])
    vp = predict(model, gva, lva, dva)
    yb = np.isin(va.label_index, PLANET_LIKE_IDX).astype(int)
    thr = {r["target_recall"]: r["threshold"] for r in precision_at_recall(yb, vp[:, PLANET_LIKE_IDX].sum(1)) if r.get("achievable")}
    rep = {"name": name, "config": config, **cfg, "seed": seed, "best_epoch": best["epoch"], "val_roc_auc": best["monitor"],
           "val_n": int(len(va)), "train_n": int(len(tr)), "n_params": n_params, "thresholds_val": thr,
           "minutes": round((time.time() - t0) / 60, 1), "diag_state": None if diag is None else diag.state()}
    torch.save({"state_dict": model.state_dict(), "config": config, "use_diag": cfg["diag"],
                "diag_state": rep["diag_state"]}, out / "best.pt")
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=float))
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    print(f"{name}: best epoch {best['epoch']}, val ROC-AUC {best['monitor']:.4f}, {rep['minutes']} min", flush=True)
    return rep


def load_v3(run_dir: Path):
    import torch
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    m = build_v3(ck["use_diag"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    diag = Diagnostics.from_state(ck["diag_state"]) if ck["use_diag"] else None
    return m, diag


def score(run_dir: Path, vs: ViewSet) -> np.ndarray:
    m, diag = load_v3(run_dir)
    g, l, d = tensors(vs, diag)
    return predict(m, g, l, d)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sweep", action="store_true", help="all runs of --configs (5 seeds each)")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--configs", default="CAB", help="subset for --sweep, e.g. B")
    a = ap.parse_args(argv)
    if not a.sweep:
        train_one(a.config, a.seed)
        return 0
    todo = [(c, s) for c in a.configs for s in SEEDS
            if not (RUNS / (f"v3_{c}" + ("" if s == 42 else f"_s{s}")) / "report.json").is_file()]
    running = []
    log = RUNS / "v3_sweep.log"
    while todo or running:
        while todo and len(running) < a.jobs:
            c, s = todo.pop(0)
            f = open(RUNS / f"v3_{c}_{s}.out", "w")
            running.append((subprocess.Popen([sys.executable, "-u", __file__, "--config", c, "--seed", str(s)],
                                             stdout=f, stderr=subprocess.STDOUT), c, s, f))
        time.sleep(10)
        for item in list(running):
            if item[0].poll() is not None:
                running.remove(item)
                item[3].close()
                with open(log, "a") as fh:
                    fh.write(f"{time.strftime('%H:%M:%S')} v3_{item[1]} seed {item[2]} exit {item[0].returncode}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
