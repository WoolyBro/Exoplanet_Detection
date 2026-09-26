"""
cnn_lstm.py
===========

Track A Level 2: dual-view CNN+LSTM transit classifier (Astronet-style inputs).

Inputs are the views already produced by preprocessing_pipeline.build_dataset:
one .npz per KOI holding a GLOBAL view (2001 bins, whole folded orbit) and a
LOCAL view (201 bins, zoom on the transit), both already normalised (median 0,
deepest bin -1), plus label / label_index / metadata.

Architecture
    global (1, 2001) -> 3x [conv k5 -> conv k5 -> maxpool 5] -> (32, ~16) -> BiLSTM(32) -> 64
    local  (1,  201) -> 2x [conv k5 -> conv k5 -> maxpool 4] -> (16,  ~7) -> BiLSTM(16) -> 32
    concat 96 -> dropout -> dense 64 -> dropout -> dense 3
The CNN reads transit morphology; the LSTM reads the ordered sequence of CNN
features across phase. ~40k parameters, sized for a few thousand KOIs.

Outputs
    3-class logits (FALSE POSITIVE / CANDIDATE / CONFIRMED, indices 0/1/2 as in
    preprocessing_pipeline.LABEL_INDEX) and a derived binary planet-like score
    P(CANDIDATE) + P(CONFIRMED).

Protocol rules this file obeys (see DATA_BOUNDARY.md and the project hard rules)
  * It NEVER re-splits the official data. Train/val/test membership is read from
    the `split` column of splits/koi_cumulative_split.csv, loaded through
    data_manifest.load_stage1_catalogs().
  * No 0.5 threshold is hardcoded anywhere. The operating point is swept after
    training and reported as precision at fixed recall (90 / 95 / 99 %).
  * Class weights come from the actual balance of the rows trained on, and both
    that balance and the official split balance are printed so any skew from a
    partial build is visible.
  * Rows with label_index -1 (no harmonized label) are dropped, never coerced.

Usage
    # smoke test on whatever the train build has written so far (a few epochs)
    python stage_a_transit_model/cnn_lstm.py --smoke

    # full run later, no code changes: same command without --smoke, once
    # detection_views/kepler_{val,test} exist
    python stage_a_transit_model/cnn_lstm.py

Requires torch, numpy, pandas, scikit-learn. See --help for all options.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RESEARCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_DIR))

import data_manifest as dm  # noqa: E402  (manifest gate: data only via the loaders)

MODEL_DIR = RESEARCH_DIR / "stage_a_transit_model"
VIEWS_ROOT = RESEARCH_DIR / "detection_views"
KOI_SPLIT_FILE = "koi_cumulative_split.csv"   # kept: other modules import this name

# Per-mission catalogue wiring. `object_id` is how the built view names itself, and it must be
# reconstructed from the catalogue exactly as preprocessing_pipeline.ephemeris_from_row wrote
# it, or the split check below rejects every row.
MISSIONS = {
    "Kepler": {"split_file": "koi_cumulative_split.csv",
               "object_id": lambda df: df["kepoi_name"].astype(str),
               "views_prefix": "kepler"},
    "TESS":   {"split_file": "toi_tess_candidates_split.csv",
               "object_id": lambda df: "TOI " + df["toi"].astype(str),
               "views_prefix": "tess"},
}

# Same order/indices preprocessing_pipeline.LABEL_INDEX writes into label_index.
CLASSES = ["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"]
PLANET_LIKE_IDX = [1, 2]  # CANDIDATE + CONFIRMED
GLOBAL_BINS, LOCAL_BINS = 2001, 201
RECALL_TARGETS = (0.90, 0.95, 0.99)


# =========================================================================== #
# Loading views
# =========================================================================== #
@dataclass
class ViewSet:
    """Consolidated views plus the identity of every row, in one aligned block."""

    global_view: np.ndarray  # (n, 2001) float32
    local_view: np.ndarray   # (n, 201)  float32
    label_index: np.ndarray  # (n,) int8
    object_id: np.ndarray    # (n,) str
    star_id: np.ndarray      # (n,) int64

    def __len__(self) -> int:
        return len(self.object_id)

    def take(self, positions: np.ndarray) -> "ViewSet":
        return ViewSet(self.global_view[positions], self.local_view[positions],
                       self.label_index[positions], self.object_id[positions], self.star_id[positions])


def consolidate_views(views_dir: Path, cache: Path | None = None, verbose: bool = True) -> ViewSet:
    """Read every .npz in `views_dir` into one aligned block, caching the result.

    Identity comes from each file's own metadata (object_id, star_id), not from the
    file name, so this cannot drift from preprocessing_pipeline's naming rule. The
    cache is invalidated when the number of .npz files changes, so a resumed build
    is picked up on the next run.

    The cache is written under stage_a_transit_model/cache/, never into `views_dir`: that
    directory is the build's own output, and preprocessing_pipeline.summarize_build
    and tools/build_status.py both count *.npz there to report build progress.
    A cache file sitting among them would inflate those counts and would also be
    read back as if it were a view.
    """
    files = sorted(p for p in views_dir.glob("*.npz") if not p.name.startswith("_"))
    if not files:
        raise FileNotFoundError(f"no view files in {views_dir}; run preprocessing_pipeline.py build-train first")

    cache = cache or MODEL_DIR / "cache" / f"{views_dir.name}_consolidated.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as z:
            if int(z["n_source_files"]) == len(files):
                if verbose:
                    print(f"  loaded cache {cache.name} ({len(z['object_id'])} rows from {len(files)} view files)")
                return ViewSet(z["global_view"], z["local_view"], z["label_index"],
                               z["object_id"].astype(str), z["star_id"])
            if verbose:
                print(f"  cache stale ({int(z['n_source_files'])} -> {len(files)} view files); rebuilding")

    gv = np.zeros((len(files), GLOBAL_BINS), np.float32)
    lv = np.zeros((len(files), LOCAL_BINS), np.float32)
    li = np.zeros(len(files), np.int8)
    oid, sid = [], []
    started = time.time()
    for i, path in enumerate(files):
        with np.load(path, allow_pickle=False) as z:
            g, l = z["global_view"], z["local_view"]
            if g.shape != (GLOBAL_BINS,) or l.shape != (LOCAL_BINS,):
                raise ValueError(f"{path.name}: unexpected view shapes {g.shape}, {l.shape}")
            gv[i], lv[i], li[i] = g, l, int(z["label_index"])
            meta = json.loads(str(z["metadata"]))
        oid.append(meta["object_id"])
        sid.append(int(meta["star_id"]))
        if verbose and (i + 1) % 1000 == 0:
            print(f"    read {i + 1}/{len(files)} view files ({time.time() - started:.0f}s)", flush=True)

    out = ViewSet(gv, lv, li, np.array(oid, dtype=object).astype(str), np.array(sid, np.int64))
    np.savez_compressed(cache, global_view=gv, local_view=lv, label_index=li,
                        object_id=out.object_id, star_id=out.star_id, n_source_files=len(files))
    if verbose:
        print(f"  consolidated {len(files)} view files in {time.time() - started:.0f}s -> {cache.name}")
    return out


def drop_unusable(vs: ViewSet, verbose: bool = True) -> ViewSet:
    """Drop rows with no harmonized label or a non-finite view (never coerce them)."""
    finite = np.isfinite(vs.global_view).all(axis=1) & np.isfinite(vs.local_view).all(axis=1)
    labelled = vs.label_index >= 0
    keep = finite & labelled
    if verbose and not keep.all():
        print(f"  dropped {int((~labelled).sum())} unlabelled and {int((~finite).sum())} "
              f"non-finite row(s); {int(keep.sum())} usable")
    return vs.take(np.flatnonzero(keep))


# =========================================================================== #
# Split assignment (read, never computed)
# =========================================================================== #
def official_split_map(mission: str = "Kepler") -> tuple[dict[str, str], pd.Series]:
    """object_id -> official split, plus the official train-split label balance.

    Read from the `split` column of that mission's split catalog via the manifest
    loader. Nothing here recomputes a split.
    """
    import contextlib
    import io

    cfg = MISSIONS[mission]
    with contextlib.redirect_stdout(io.StringIO()):
        cat = dm.load_stage1_catalogs()[cfg["split_file"]]
    mapping = dict(zip(cfg["object_id"](cat), cat["split"].astype(str)))
    balance = cat.loc[cat["split"] == "train", "label_harmonized"].value_counts(normalize=True)
    return mapping, balance.reindex(CLASSES).fillna(0.0) * 100


def star_grouped_holdout(vs: ViewSet, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """SMOKE-TEST ONLY: carve a star-grouped, label-stratified holdout out of train rows.

    This exists because the official val/test views have not been built yet (see
    Â§7 of the handoff). It uses split_catalogs' own verified grouping helper, so
    whole stars move together, and asserts star-disjointness afterwards. Numbers
    measured against this holdout are NOT results and must not be reported as
    val/test performance: the real val/test sets come from running
    preprocessing_pipeline over the val and test splits.
    """
    import split_catalogs as sc  # project's verified grouped-split machinery

    strata = np.array([CLASSES[i] for i in vs.label_index])
    kept, held = sc._hold_out_groups(strata, vs.star_id, fraction, seed)
    overlap = set(vs.star_id[kept]) & set(vs.star_id[held])
    if overlap:
        raise RuntimeError(f"smoke holdout leaked {len(overlap)} star(s) across the boundary: {sorted(overlap)[:5]}")
    return kept, held


# =========================================================================== #
# Model
# =========================================================================== #
def build_model(dropout: float, batchnorm: bool = False, width: float = 1.0,
                arch: str = "dual_cnn_lstm"):
    """Dual-branch CNN + BiLSTM. Imported lazily so --help works without torch.

    `width` scales every conv/LSTM size, so capacity can be swept without editing the file.

    `arch` selects how much of the architecture is actually built, so the guide's sequence
    (plain CNN on the global view first, then the local branch, then the LSTM) can be run as a
    real ablation instead of asserted:
        global_cnn     - one CNN over the global view only, mean-pooled. No local view, no LSTM.
        dual_cnn       - both views, both CNN-only, concatenated.
        dual_cnn_lstm  - both views, each CNN followed by a BiLSTM (the default, unchanged).
    """
    if arch not in ("global_cnn", "dual_cnn", "dual_cnn_lstm"):
        raise ValueError(f"unknown arch {arch!r}")
    use_lstm = arch == "dual_cnn_lstm"
    use_local = arch != "global_cnn"
    import torch
    from torch import nn

    def scaled(n: int) -> int:
        return max(4, int(round(n * width)))

    class ConvStack(nn.Module):
        """Astronet-style repeated [conv, conv, maxpool] blocks over one view."""

        def __init__(self, channels: list[int], kernel: int, pool: int):
            super().__init__()
            layers = []
            in_c = 1
            for out_c in channels:
                layers += [nn.Conv1d(in_c, out_c, kernel, padding=kernel // 2)]
                layers += [nn.BatchNorm1d(out_c)] if batchnorm else []
                layers += [nn.ReLU(), nn.Conv1d(out_c, out_c, kernel, padding=kernel // 2)]
                layers += [nn.BatchNorm1d(out_c)] if batchnorm else []
                layers += [nn.ReLU(), nn.MaxPool1d(pool)]
                in_c = out_c
            self.net = nn.Sequential(*layers)
            self.out_channels = in_c

        def forward(self, x):
            return self.net(x)

    class Branch(nn.Module):
        """CNN over one view, optionally followed by a BiLSTM over the feature sequence."""

        def __init__(self, channels: list[int], kernel: int, pool: int, hidden: int):
            super().__init__()
            self.conv = ConvStack(channels, kernel, pool)
            self.lstm = (nn.LSTM(self.conv.out_channels, hidden, batch_first=True, bidirectional=True)
                         if use_lstm else None)
            # Without the LSTM the branch mean-pools the CNN features, so the two variants differ
            # only by the recurrent layer - which is the point of the ablation.
            self.out_features = 2 * hidden if use_lstm else self.conv.out_channels

        def forward(self, x):
            h = self.conv(x.unsqueeze(1))          # (B, C, L')
            seq = h.transpose(1, 2)                # (B, L', C) -> phase-ordered sequence
            if self.lstm is None:
                return seq.mean(dim=1)
            out, _ = self.lstm(seq)
            # mean over the sequence: less sensitive to where the transit sits than the last state
            return out.mean(dim=1)

    class TransitCNNLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.global_branch = Branch([scaled(8), scaled(16), scaled(32)], kernel=5, pool=5, hidden=scaled(32))
            self.local_branch = (Branch([scaled(8), scaled(16)], kernel=5, pool=4, hidden=scaled(16))
                                 if use_local else None)
            n = self.global_branch.out_features
            if self.local_branch is not None:
                n += self.local_branch.out_features
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(n, scaled(64)), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(scaled(64), len(CLASSES)))

        def forward(self, g, l):
            # The local view is still passed in every call so the training loop is identical
            # across architectures; global_cnn simply ignores it.
            feats = self.global_branch(g)
            if self.local_branch is not None:
                feats = torch.cat([feats, self.local_branch(l)], dim=1)
            return self.head(feats)

    return TransitCNNLSTM()


# =========================================================================== #
# Metrics
# =========================================================================== #
def planet_like_score(proba: np.ndarray) -> np.ndarray:
    return proba[:, PLANET_LIKE_IDX].sum(axis=1)


def precision_at_recall(y_true: np.ndarray, score: np.ndarray, targets=RECALL_TARGETS) -> list[dict]:
    """Precision (and the threshold that achieves it) at each target recall.

    The mentor asked for this explicitly: planet hunting cares about completeness,
    so the operating point is chosen by recall, never by a fixed 0.5 cut.
    """
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, score)
    rows = []
    for target in targets:
        ok = np.flatnonzero(recall >= target)
        if len(ok) == 0:
            rows.append({"target_recall": target, "achievable": False})
            continue
        i = ok[np.argmax(precision[ok])]  # best precision among points meeting the recall floor
        rows.append({"target_recall": target, "achievable": True,
                     "precision": float(precision[i]), "recall": float(recall[i]),
                     # thresholds is one shorter than precision/recall by construction
                     "threshold": float(thresholds[min(i, len(thresholds) - 1)])})
    return rows


def evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict:
    from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                                 confusion_matrix, roc_auc_score)

    pred = proba.argmax(axis=1)
    y_bin = np.isin(y_true, PLANET_LIKE_IDX).astype(int)
    score = planet_like_score(proba)
    present = sorted(set(y_true.tolist()))
    out = {
        "n": int(len(y_true)),
        "class_counts": {CLASSES[i]: int((y_true == i).sum()) for i in range(len(CLASSES))},
        "accuracy_3class": float(accuracy_score(y_true, pred)),
        "confusion_3class": confusion_matrix(y_true, pred, labels=range(len(CLASSES))).tolist(),
        "report_3class": classification_report(y_true, pred, labels=range(len(CLASSES)),
                                               target_names=CLASSES, digits=3, zero_division=0),
        "precision_at_recall": precision_at_recall(y_bin, score),
    }
    # AUCs need both classes present; on a tiny smoke holdout that is not guaranteed.
    out["roc_auc_binary"] = float(roc_auc_score(y_bin, score)) if 0 < y_bin.sum() < len(y_bin) else None
    out["pr_auc_binary"] = float(average_precision_score(y_bin, score)) if 0 < y_bin.sum() < len(y_bin) else None
    out["roc_auc_3class_macro_ovr"] = (
        float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro")) if len(present) == len(CLASSES) else None)
    return out


def threshold_sweep(y_true: np.ndarray, proba: np.ndarray, n_points: int = 201) -> pd.DataFrame:
    """Binary planet-like metrics across the whole threshold range (no 0.5 anywhere)."""
    y_bin = np.isin(y_true, PLANET_LIKE_IDX).astype(int)
    score = planet_like_score(proba)
    rows = []
    for t in np.linspace(0.0, 1.0, n_points):
        pred = (score >= t).astype(int)
        tp = int(((pred == 1) & (y_bin == 1)).sum())
        fp = int(((pred == 1) & (y_bin == 0)).sum())
        fn = int(((pred == 0) & (y_bin == 1)).sum())
        tn = int(((pred == 0) & (y_bin == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec and rec and np.isfinite(prec) and np.isfinite(rec) else float("nan")
        rows.append({"threshold": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "precision": prec, "recall": rec, "f1": f1,
                     "accuracy": (tp + tn) / len(y_bin)})
    return pd.DataFrame(rows)


# =========================================================================== #
# Training
# =========================================================================== #
def class_weights(label_index: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights from the rows actually being trained on."""
    counts = np.array([(label_index == i).sum() for i in range(len(CLASSES))], float)
    counts = np.where(counts > 0, counts, np.nan)
    w = np.nanmean(counts) / counts
    return np.where(np.isfinite(w), w, 0.0)


def train(model, train_vs: ViewSet, val_vs: ViewSet, args) -> dict:
    import torch
    from torch import nn

    device = torch.device(args.device)
    model = model.to(device)

    def tensors(vs: ViewSet):
        return (torch.from_numpy(vs.global_view).to(device),
                torch.from_numpy(vs.local_view).to(device),
                torch.from_numpy(vs.label_index.astype(np.int64)).to(device))

    gtr, ltr, ytr = tensors(train_vs)
    gva, lva, yva = tensors(val_vs)

    w = class_weights(train_vs.label_index)
    print(f"  class weights (inverse frequency): " + ", ".join(f"{c} {v:.3f}" for c, v in zip(CLASSES, w)))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model: {n_params:,} trainable parameters | device {device} | "
          f"train {len(train_vs)} rows, val {len(val_vs)} rows")

    def predict(g, l) -> np.ndarray:
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(g), args.batch_size):
                outs.append(torch.softmax(model(g[i:i + args.batch_size], l[i:i + args.batch_size]), dim=1).cpu())
        return torch.cat(outs).numpy()

    monitor_key = {"roc": "roc_auc_binary", "pr": "pr_auc_binary"}[getattr(args, "monitor", "roc")]
    monitor_name = {"roc": "val binary ROC-AUC", "pr": "val binary PR-AUC"}[getattr(args, "monitor", "roc")]
    scheduler = None
    if getattr(args, "scheduler", False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    history, best = [], {"epoch": -1, "monitor": -np.inf, "state": None}
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = np.random.default_rng(args.seed + epoch)
        order = rng.permutation(len(train_vs))
        total, n_batches = 0.0, 0
        for i in range(0, len(order), args.batch_size):
            idx = torch.from_numpy(order[i:i + args.batch_size]).to(device)
            gb, lb = gtr[idx], ltr[idx]
            if getattr(args, "augment", False):
                # Mirror about mid-transit. A phase-folded transit is symmetric in ingress/egress,
                # so the flipped view is a physically real view of the same event, not a distortion.
                flip = torch.from_numpy(rng.random(len(idx)) < 0.5).to(device)
                if flip.any():
                    gb, lb = gb.clone(), lb.clone()
                    gb[flip] = torch.flip(gb[flip], dims=[1])
                    lb[flip] = torch.flip(lb[flip], dims=[1])
            optimizer.zero_grad()
            loss = criterion(model(gb, lb), ytr[idx])
            if not torch.isfinite(loss):
                raise RuntimeError(f"epoch {epoch}: loss became {loss.item()} - stopping (NaN guard)")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            total += float(loss.item())
            n_batches += 1

        val_proba = predict(gva, lva)
        val_metrics = evaluate(val_vs.label_index.astype(int), val_proba)
        monitor = val_metrics[monitor_key]
        # Early stopping monitors the chosen val AUC; if it is undefined (one class only,
        # possible on a tiny smoke holdout) fall back to 3-class accuracy so the loop
        # still makes progress instead of silently never improving.
        if monitor is None:
            monitor, monitor_name = val_metrics["accuracy_3class"], "val 3-class accuracy (AUC undefined)"
        if scheduler is not None:
            scheduler.step(monitor)
        history.append({"epoch": epoch, "train_loss": total / max(n_batches, 1),
                        "val_accuracy_3class": val_metrics["accuracy_3class"],
                        "val_roc_auc_binary": val_metrics["roc_auc_binary"],
                        "val_pr_auc_binary": val_metrics["pr_auc_binary"]})
        flag = ""
        if monitor > best["monitor"] + args.min_delta:
            best = {"epoch": epoch, "monitor": monitor,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            flag = "  <- best"
        print(f"  epoch {epoch:>3}/{args.epochs}  train loss {total / max(n_batches, 1):.4f}  "
              f"val acc {val_metrics['accuracy_3class']:.3f}  "
              f"{monitor_name} {monitor:.3f}{flag}", flush=True)
        if epoch - best["epoch"] >= args.patience:
            print(f"  early stopping: no improvement in {monitor_name} for {args.patience} epoch(s)")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return {"history": history, "best_epoch": best["epoch"], "best_monitor": best["monitor"],
            "monitor": monitor_name, "n_params": n_params, "class_weights": w.tolist(),
            "predict": predict, "val_tensors": (gva, lva)}


# =========================================================================== #
# CLI
# =========================================================================== #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mission", choices=sorted(MISSIONS), default="Kepler",
                   help="which mission's catalogue and views to train on; also sets the default "
                        "view folders to <mission>_{train,val,test}")
    p.add_argument("--train-views", type=Path, default=None)
    p.add_argument("--val-views", type=Path, default=None,
                   help="official val views; if absent, --smoke-holdout is used instead")
    p.add_argument("--test-views", type=Path, default=None,
                   help="evaluated only if it exists (built by preprocessing_pipeline over the test split)")
    p.add_argument("--eval-test", action="store_true",
                   help="evaluate the official TEST views. OFF by default on purpose: tuning is done "
                        "against val, and every extra look at test erodes it as a held-out set. Pass this "
                        "only for the single final run of the chosen configuration.")
    p.add_argument("--smoke", action="store_true",
                   help="short throwaway run: few epochs, row cap, results explicitly not reportable")
    p.add_argument("--smoke-holdout", type=float, default=0.15,
                   help="star-grouped holdout carved from TRAIN when official val views do not exist")
    p.add_argument("--limit", type=int, default=0, help="cap usable rows (0 = all); --smoke sets 600")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--width", type=float, default=1.0, help="scale every conv/LSTM size (capacity sweep)")
    p.add_argument("--arch", choices=["global_cnn", "dual_cnn", "dual_cnn_lstm"], default="dual_cnn_lstm",
                   help="how much of the architecture to build, for the guide's staged ablation: "
                        "global_cnn (plain CNN, global view only), dual_cnn (both views, no LSTM), "
                        "dual_cnn_lstm (full, default)")
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--augment", action="store_true",
                   help="random left-right flip of both views during training. Valid for a phase-folded "
                        "transit: ingress and egress are symmetric about mid-transit, so a mirrored view is "
                        "still a physically real view of the same event.")
    p.add_argument("--batchnorm", action="store_true", help="BatchNorm after each conv in the CNN stacks")
    p.add_argument("--scheduler", action="store_true",
                   help="ReduceLROnPlateau on the monitored validation metric (factor 0.5, patience 3)")
    p.add_argument("--monitor", choices=["roc", "pr"], default="roc",
                   help="validation metric for early stopping and best-epoch selection: binary ROC-AUC "
                        "(default) or PR-AUC. PR-AUC is the stricter choice on an imbalanced positive class.")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2, help="torch CPU threads; kept low so a running build is not starved")
    p.add_argument("--out", type=Path, default=None, help="run directory (default stage_a_transit_model/runs/<tag>)")
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args(argv)
    # View folders follow the mission unless the caller named them explicitly.
    prefix = MISSIONS[args.mission]["views_prefix"]
    for split in ("train", "val", "test"):
        if getattr(args, f"{split}_views") is None:
            setattr(args, f"{split}_views", VIEWS_ROOT / f"{prefix}_{split}")
    if args.smoke:
        args.epochs = min(args.epochs, 4)
        args.limit = args.limit or 600
        args.patience = max(args.patience, args.epochs)  # never early-stop a 4-epoch smoke run
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))

    tag = ("smoke_" if args.smoke else "") + datetime.now().strftime("%Y%m%d_%H%M%S")
    # Resolved, so a relative --out (as the sweep driver passes) still prints and saves correctly.
    run_dir = Path(args.out).resolve() if args.out else MODEL_DIR / "runs" / tag

    print("=" * 88)
    print(f"CNN+LSTM transit classifier | run {tag}")
    if args.smoke:
        print("*** SMOKE TEST: few epochs on capped, partial data. The numbers below are NOT results. ***")
    print("=" * 88)

    print("\n[1] views")
    vs = drop_unusable(consolidate_views(args.train_views))
    split_map, official_balance = official_split_map(args.mission)

    # Every row must be a known KOI/TOI of the official TRAIN split. A row whose split
    # is anything else means the build wrote outside its remit: stop rather than train on it.
    assigned = np.array([split_map.get(o, "<unknown>") for o in vs.object_id])
    wrong = sorted(set(assigned) - {"train"})
    if wrong:
        offenders = [o for o, a in zip(vs.object_id, assigned) if a != "train"][:10]
        raise SystemExit(f"{args.train_views.name} contains rows whose official split is {wrong}: {offenders}")

    loaded_balance = pd.Series({CLASSES[i]: int((vs.label_index == i).sum()) for i in range(len(CLASSES))})
    loaded_pct = 100 * loaded_balance / loaded_balance.sum()
    print(f"  usable rows: {len(vs)} on {len(set(vs.star_id.tolist()))} stars")
    print("  class balance, official train split vs views built so far:")
    for c in CLASSES:
        shift = loaded_pct[c] - official_balance[c]
        print(f"    {c:<15} official {official_balance[c]:5.1f}%  built {loaded_pct[c]:5.1f}% "
              f"({loaded_balance[c]:>5}) | shift {shift:+.1f} pp")
    skew = max(abs(loaded_pct[c] - official_balance[c]) for c in CLASSES)
    if skew > 2.0:
        print(f"  WARNING: the built subset is skewed by up to {skew:.1f} pp vs the official train split. "
              f"Class weights below reflect the SUBSET, and any metric from this run is not comparable "
              f"to a full-build run.")

    if args.limit and args.limit < len(vs):
        pick = np.random.default_rng(args.seed).choice(len(vs), size=args.limit, replace=False)
        vs = vs.take(np.sort(pick))
        print(f"  --limit: capped to {len(vs)} rows")

    print("\n[2] split assignment")
    if args.val_views.is_dir() and any(args.val_views.glob("*.npz")):
        train_vs = vs
        val_vs = drop_unusable(consolidate_views(args.val_views))
        print(f"  official val views: {len(val_vs)} rows from {args.val_views.name}")
    else:
        kept, held = star_grouped_holdout(vs, args.smoke_holdout, args.seed)
        train_vs, val_vs = vs.take(kept), vs.take(held)
        print(f"  NO official val views at {args.val_views} - using a star-grouped holdout carved from TRAIN")
        print(f"  ({args.smoke_holdout:.0%}: {len(train_vs)} train / {len(val_vs)} holdout rows, "
              f"{len(set(val_vs.star_id.tolist()))} holdout stars, star-disjoint verified)")
        print("  this holdout is a smoke-test device; real val/test numbers need the val/test views built")

    print("\n[3] training")
    model = build_model(args.dropout, batchnorm=args.batchnorm, width=args.width, arch=args.arch)
    result = train(model, train_vs, val_vs, args)

    print("\n[4] evaluation on the validation rows (best epoch)")
    predict, (gva, lva) = result.pop("predict"), result.pop("val_tensors")
    val_proba = predict(gva, lva)
    val_metrics = evaluate(val_vs.label_index.astype(int), val_proba)
    print(f"  3-class accuracy {val_metrics['accuracy_3class']:.3f} | "
          f"binary ROC-AUC {val_metrics['roc_auc_binary']} | PR-AUC {val_metrics['pr_auc_binary']}")
    print(val_metrics["report_3class"])
    print("  confusion (rows true, cols pred; FP, CAND, CONF):", val_metrics["confusion_3class"])

    print("\n[5] threshold sweep on the binary planet-like score (no 0.5 assumed)")
    sweep = threshold_sweep(val_vs.label_index.astype(int), val_proba)
    usable = sweep[np.isfinite(sweep.f1)]
    if len(usable):
        bf1 = usable.loc[usable.f1.idxmax()]
        print(f"  best-F1 operating point: threshold {bf1.threshold:.3f} -> "
              f"precision {bf1.precision:.3f}, recall {bf1.recall:.3f}, F1 {bf1.f1:.3f}")
    print("  precision at fixed recall (the completeness-first view the mentor asked for):")
    for row in val_metrics["precision_at_recall"]:
        if row["achievable"]:
            # 6 dp: on a degenerate or tiny-holdout run the scores cluster so tightly that
            # 3 dp makes genuinely different operating points look like the same threshold.
            print(f"    recall >= {row['target_recall']:.0%}: precision {row['precision']:.3f} "
                  f"at threshold {row['threshold']:.6f} (actual recall {row['recall']:.3f})")
        else:
            print(f"    recall >= {row['target_recall']:.0%}: not achievable on these rows")

    test_metrics = None
    if not args.eval_test:
        print("\n[6] TEST NOT EVALUATED (--eval-test not passed). Tuning runs stay off the held-out set; "
              "pass --eval-test for the single final run of the chosen configuration.")
    elif args.test_views.is_dir() and any(args.test_views.glob("*.npz")):
        print("\n[6] official test views found - evaluating once")
        test_vs = drop_unusable(consolidate_views(args.test_views))
        gte = torch.from_numpy(test_vs.global_view)
        lte = torch.from_numpy(test_vs.local_view)
        test_metrics = evaluate(test_vs.label_index.astype(int), predict(gte, lte))
        print(f"  test 3-class accuracy {test_metrics['accuracy_3class']:.3f} | "
              f"binary ROC-AUC {test_metrics['roc_auc_binary']}")
    else:
        print(f"\n[6] no official test views at {args.test_views} - skipped (build them with preprocessing_pipeline)")

    if not args.no_save:
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "classes": CLASSES,
                    "args": {k: str(v) for k, v in vars(args).items()}}, run_dir / "best.pt")
        sweep.to_csv(run_dir / "threshold_sweep.csv", index=False)
        pd.DataFrame(result["history"]).to_csv(run_dir / "history.csv", index=False)
        report = {"run": tag, "smoke": args.smoke, "generated_at": datetime.now().isoformat(timespec="seconds"),
                  "args": {k: str(v) for k, v in vars(args).items()},
                  "n_train_rows": len(train_vs), "n_val_rows": len(val_vs),
                  "official_train_balance_pct": official_balance.round(2).to_dict(),
                  "built_subset_balance_pct": loaded_pct.round(2).to_dict(),
                  "training": {k: v for k, v in result.items()},
                  "val_metrics": val_metrics, "test_metrics": test_metrics}
        (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
        print(f"\nsaved {run_dir.relative_to(RESEARCH_DIR)}/ (best.pt, report.json, history.csv, threshold_sweep.csv)")

    if args.smoke:
        print("\nSMOKE TEST COMPLETE: shapes, loss, early stopping, evaluation and threshold sweep all ran. "
              "Discard the numbers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
