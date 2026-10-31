"""
embeddings_v2.py
================

The t-SNE of embeddings.py on the final Vetter v2 network, on Kepler v2 validation data only (no test set is read).
The map is for looking at; the separation is measured in the 64-unit embedding itself.

    python stage_a_transit_model/embeddings_v2.py
Output: outputs/post_review/embedding_v2.npz, embedding_v2.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from stage_a_transit_model.cnn_lstm import consolidate_views, drop_unusable  # noqa: E402
from stage_a_transit_model.embeddings import extract, separability  # noqa: E402
from stage_a_transit_model.failure_analysis import load_run_model  # noqa: E402

OUT = R / "outputs" / "post_review"


def main() -> int:
    from sklearn.manifold import TSNE

    model, arch = load_run_model(R / "stage_a_transit_model" / "runs" / "v2_kepler_dual_cnn_lstm_err")
    vs = drop_unusable(consolidate_views(R / "detection_views_v2" / "kepler_val", verbose=False), verbose=False)
    emb = extract(model, vs)
    y = vs.label_index.astype(int)
    sep = separability(emb, y, vs.star_id, seed=0)
    ts = TSNE(n_components=2, perplexity=30.0, init="pca", random_state=0, max_iter=1000).fit_transform(emb)
    np.savez(OUT / "embedding_v2.npz", tsne=ts, label=y)
    res = {"model": "v2_kepler_dual_cnn_lstm_err", "architecture": str(arch), "split": "Kepler v2 validation", "n": int(len(y)), "dims": int(emb.shape[1]), **sep}
    (OUT / "embedding_v2.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
