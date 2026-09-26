"""
embeddings.py
=============

Checklist item 7.4: "T-SNE on your model's embeddings, coloured by true label - confirm it's
actually learned separable structure."

A t-SNE picture on its own is weak evidence: the algorithm will produce apparent clusters from
pure noise if the perplexity is low enough, and it does not preserve global structure. So this
plots the map AND measures separability directly, in the untransformed embedding space:

  * kNN accuracy (leave-one-out, star-grouped) - can a neighbour vote recover the label?
  * silhouette score - are same-label points closer to each other than to other labels?
  * a permutation control - the same numbers with the labels shuffled, which is what "no
    structure" actually looks like on this sample size.

The embedding is the 64-unit hidden layer of the classifier head: the last representation
before the logits, which is what "the model's embedding" normally means.

Usage
    python stage_a_transit_model/embeddings.py --run stage_a_transit_model/runs/bn_aug_sched
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RESEARCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_DIR))

from stage_a_transit_model.cnn_lstm import CLASSES, VIEWS_ROOT, consolidate_views, drop_unusable  # noqa: E402
from stage_a_transit_model.failure_analysis import load_run_model  # noqa: E402

EDA_DIR = RESEARCH_DIR / "eda"


def extract(model, vs, batch: int = 256) -> np.ndarray:
    """The 64-unit hidden representation of the head, for every row."""
    import torch

    feats = []

    def hook(_module, _inp, out):
        feats.append(out.detach().cpu().numpy())

    # head = [Dropout, Linear, ReLU, Dropout, Linear]; index 2 is the ReLU after the first Linear.
    handle = model.head[2].register_forward_hook(hook)
    g = torch.from_numpy(vs.global_view)
    l = torch.from_numpy(vs.local_view)
    with torch.no_grad():
        for i in range(0, len(g), batch):
            model(g[i:i + batch], l[i:i + batch])
    handle.remove()
    return np.vstack(feats)


def separability(emb: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0) -> dict:
    """kNN accuracy and silhouette in the real embedding space, with a shuffled control."""
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GroupKFold

    def knn(labels: np.ndarray) -> float:
        # Star-grouped folds: two KOIs of the same star must never be each other's neighbour
        # across the split, or the score measures memorising the star, not the class.
        cv = GroupKFold(n_splits=5)
        scores = cross_val_score(KNeighborsClassifier(n_neighbors=10), emb, labels,
                                 groups=groups, cv=cv, scoring="accuracy")
        return float(scores.mean())

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(y)
    return {
        "knn_accuracy": knn(y),
        "knn_accuracy_shuffled_labels": knn(shuffled),
        "silhouette": float(silhouette_score(emb, y)),
        "silhouette_shuffled_labels": float(silhouette_score(emb, shuffled)),
        "majority_class_rate": float(pd.Series(y).value_counts(normalize=True).max()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="stage_a_transit_model/runs/bn_aug_sched")
    ap.add_argument("--views", type=Path, default=VIEWS_ROOT / "kepler_val")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    run_dir = Path(args.run).resolve()
    if not (run_dir / "best.pt").is_file():
        run_dir = (RESEARCH_DIR / args.run).resolve()

    print("=" * 88)
    print(f"Embedding structure | run {run_dir.name} | {args.views.name}")
    print("=" * 88)

    model, arch = load_run_model(run_dir)
    vs = drop_unusable(consolidate_views(args.views, verbose=False))
    emb = extract(model, vs)
    y = vs.label_index.astype(int)
    print(f"\n[1] embeddings: {emb.shape[0]} rows x {emb.shape[1]} dims  (architecture {arch})")
    dead = int((emb.std(axis=0) == 0).sum())
    print(f"  constant (dead) units: {dead}/{emb.shape[1]}")

    print("\n[2] is the structure real? measured in the embedding space, not in the t-SNE map")
    sep = separability(emb, y, vs.star_id, seed=args.seed)
    print(f"  kNN accuracy (star-grouped 5-fold) : {sep['knn_accuracy']:.4f}")
    print(f"    same, with labels shuffled       : {sep['knn_accuracy_shuffled_labels']:.4f}   <- the null")
    print(f"    majority-class rate              : {sep['majority_class_rate']:.4f}")
    print(f"  silhouette                         : {sep['silhouette']:.4f}")
    print(f"    same, with labels shuffled       : {sep['silhouette_shuffled_labels']:.4f}")
    verdict = ("real structure" if sep["knn_accuracy"] > sep["majority_class_rate"] + 0.05
               else "NOT clearly better than predicting the majority class")
    print(f"  verdict: {verdict}")

    print("\n[3] t-SNE map (for looking at, not for measuring)")
    from sklearn.manifold import TSNE

    ts = TSNE(n_components=2, perplexity=args.perplexity, init="pca",
              random_state=args.seed, max_iter=1000).fit_transform(emb)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    colours = {0: "#c23b22", 1: "#c8902a", 2: "#3b7a57"}
    for i, cls in enumerate(CLASSES):
        m = y == i
        axes[0].scatter(ts[m, 0], ts[m, 1], s=9, c=colours[i], label=f"{cls} (n={m.sum()})",
                        alpha=0.65, edgecolor="none")
    axes[0].set_title(f"t-SNE of the 64-d embedding, coloured by true label\n"
                      f"perplexity {args.perplexity:g}, seed {args.seed}", fontsize=9)
    axes[0].legend(fontsize=7, markerscale=1.6)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    bars = ["kNN acc", "kNN acc\n(shuffled)", "majority\nrate"]
    vals = [sep["knn_accuracy"], sep["knn_accuracy_shuffled_labels"], sep["majority_class_rate"]]
    axes[1].bar(bars, vals, color=["#2d6fa8", "#999999", "#c23b22"], edgecolor="black", linewidth=0.6)
    for i, v in enumerate(vals):
        axes[1].text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Separability measured in the embedding space\n"
                      "(the t-SNE picture is not the evidence)", fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(f"Learned representation - {run_dir.name} on {args.views.name} "
                 f"({len(vs)} rows, star-grouped)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    out = EDA_DIR / f"{args.views.name}_embedding_tsne.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    payload = {"run": run_dir.name, "views": args.views.name,
               "generated_at": datetime.now().isoformat(timespec="seconds"),
               "n": int(len(vs)), "dims": int(emb.shape[1]), "dead_units": dead,
               "perplexity": args.perplexity, "seed": args.seed, **sep, "verdict": verdict}
    (run_dir / f"embedding_{args.views.name}.json").write_text(json.dumps(payload, indent=2),
                                                               encoding="utf-8")
    print(f"  saved {out.relative_to(RESEARCH_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
