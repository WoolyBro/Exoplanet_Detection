"""
build_screener.py
=================

Build the frozen artefacts of the installable JWST molecule screener (jwst_molecule_screener/): the CH4 and
CO2 gradient-boosted models and the multi-label 1-D CNN, trained on the full contamination + haze grid
exactly as in the validated tests, and the Platt calibration maps fitted on the labelled real JWST planets'
scores from the molecule register run (outputs/stage_b_register/register_spectra.csv; anchors by leave-one-
planet-out). Then verify that the package reproduces the register's scores on non-anchor planets.

    python tools/build_screener.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "stage_b_atmospheres"))
sys.path.insert(0, str(R / "jwst_molecule_screener"))
import methane_tls_robustness as tr  # noqa: E402
import molecule_register as mr  # noqa: E402
import stage_b_pipeline as sbp  # noqa: E402
import train_synthetic_classifier as l3  # noqa: E402

PKG = R / "jwst_molecule_screener" / "jwst_molecule_screener"
DATA = PKG / "data"


def train_cnn_net(Xtr, Ytr, epochs=40):
    """train_synthetic_classifier.train_cnn, line for line, returning the network itself."""
    import torch
    import torch.nn as nn
    from jwst_molecule_screener.core import build_cnn

    torch.manual_seed(l3.SEED)
    torch.set_num_threads(2)
    x = torch.tensor(np.stack([Xtr[:, 0] / l3.DEPTH_SCALE, Xtr[:, 1] / l3.DEPTH_SCALE, Xtr[:, 2]], 1), dtype=torch.float32)
    y = torch.tensor(Ytr, dtype=torch.float32)
    net = build_cnn(Ytr.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(l3.SEED)
    for _ in range(epochs):
        net.train()
        for idx in torch.randperm(len(x), generator=g).split(64):
            xb = x[idx]
            xb = torch.cat([xb[:, :1] + torch.randn_like(xb[:, :1]) * xb[:, 1:2] * 0.3, xb[:, 1:]], 1)
            opt.zero_grad()
            loss_fn(net(xb), y[idx]).backward()
            opt.step()
    net.eval()
    return net


def main() -> int:
    import joblib
    import sklearn
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    DATA.mkdir(parents=True, exist_ok=True)
    Xg, Yg, *_ = tr.load_grid(tr.GRID["v3"])
    CH4, CO2 = sbp.MOLECULES.index("CH4"), sbp.MOLECULES.index("CO2")
    joblib.dump(l3.fit_gbm(Xg, Yg[:, CH4]), DATA / "gbm_ch4.joblib", compress=3)
    joblib.dump(l3.fit_gbm(Xg, Yg[:, CO2]), DATA / "gbm_co2.joblib", compress=3)
    torch.save(train_cnn_net(Xg, Yg).state_dict(), DATA / "cnn.pt")

    # Platt maps on the labelled planets, from the register run's planet means (identical to the register)
    spec = pd.read_csv(R / "outputs" / "stage_b_register" / "register_spectra.csv")
    ch4, co2 = mr.labels()
    lg = lambda s: np.log(np.clip(s, 1e-4, 1 - 1e-4) / (1 - np.clip(s, 1e-4, 1 - 1e-4)))
    meta = {"version": "1.0.0", "training_grid": "outputs/stage_b_synthetic_realistic (3,000 spectra; spots, faculae, hazes)",
            "calibration": "Platt maps on real labelled JWST planets (leave-one-planet-out scores for grid anchors)",
            "register_protocol": "stage_b_atmospheres/MOLECULE_REGISTER_PROTOCOL.md (commit d7f1f6b)",
            "versions": {"scikit-learn": sklearn.__version__, "torch": torch.__version__, "numpy": np.__version__},
            "platt": {}, "validation": {}}
    for gas, lab, col in [("CH4", ch4, "CH4"), ("CO2", co2, "CO2")]:
        k = gas.lower()
        ok = spec[spec[f"{k}_ok"]]
        L = ok.groupby("planet")[[f"{k}_gbm", f"{k}_cnn"]].mean().join(
            lab[lab[col].isin(["0", "1"])].set_index("planet")[col].astype(int).rename("y"), how="inner")
        meta["platt"][gas], meta["validation"][gas] = {}, {"n_planets": int(len(L)), "n_positive": int(L.y.sum())}
        for m in ["gbm", "cnn"]:
            f = LogisticRegression(C=1.0).fit(lg(L[f"{k}_{m}"].to_numpy())[:, None], L.y.to_numpy())
            meta["platt"][gas][m] = [float(f.coef_[0, 0]), float(f.intercept_[0])]
            meta["validation"][gas][f"roc_auc_{m}"] = round(float(roc_auc_score(L.y, L[f"{k}_{m}"])), 3)
    (DATA / "calibration.json").write_text(json.dumps(meta, indent=2))

    # verify: package scores == register scores for non-anchor planets' spectra
    import importlib
    import jwst_molecule_screener.core as core
    importlib.reload(core)
    Xs, meta_s = mr.spectra({"WASP-39 b", "WASP-127 b", "LHS 1140 b", "TOI-421 b", "HATS-6 b", "GJ 1214 b"})
    sc = core.score_binned(Xs)
    ref = spec.set_index("source")
    worst = 0.0
    for i, src in enumerate(meta_s.source):
        if src in ref.index and meta_s.planet[i] not in {"WASP-39 b"}:          # WASP-39 b is a grid anchor (LOPO in the register)
            for gas in ["CH4", "CO2"]:
                for m in ["gbm", "cnn"]:
                    worst = max(worst, abs(float(sc[gas][m][i]) - float(ref.loc[src, f"{gas.lower()}_{m}"])))
    print(json.dumps({"platt": meta["platt"], "validation": meta["validation"], "max_abs_diff_vs_register": worst}, indent=2))
    sizes = {p.name: round(p.stat().st_size / 1e6, 2) for p in DATA.iterdir()}
    print("artefacts (MB):", sizes)
    return 0 if worst < 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
