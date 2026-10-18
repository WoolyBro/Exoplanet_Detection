"""The 2026-10-18 critique fixes: the strength-blind SpecNet input, the v3 diagnostic features, the fitted
retention model, the candidate factor's parameter completion, and the statistics of the v3 and S/N-balance
evaluations (no network, no training)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in [ROOT, ROOT / "stage_b_atmospheres", ROOT / "stage_c_priority_fusion"]:
    sys.path.insert(0, str(p))


def test_specnet_input_cannot_see_feature_strength():
    """d -> a + k d and sigma -> c sigma (any a, k > 0, c > 0) leave the input unchanged: the network sees
    the shape of the residual, never its overall strength or the absolute noise level."""
    ms = pytest.importorskip("methane_specnet")
    rng = np.random.default_rng(0)
    n, bins = 4, 60
    d = rng.normal(5000, 80, (n, bins))
    u = rng.uniform(20, 60, (n, bins))
    m = (rng.random((n, bins)) > 0.1).astype(float)
    X = np.stack([d, u, m], axis=1)
    Y = np.stack([300 + 7.5 * d, 3.0 * u, m], axis=1)
    a, b = ms.strength_blind(X), ms.strength_blind(Y)
    assert np.allclose(a, b, atol=1e-4)
    for i in range(n):
        c = m[i] > 0
        assert np.sqrt(np.mean(a[i, 0, c] ** 2)) == pytest.approx(1.0, rel=1e-4)


def test_v3_diagnostics_are_bounded_with_a_missing_flag():
    from stage_a_transit_model.vetter_v3 import FEATURES, Diagnostics
    d = Diagnostics()
    ids = d.table.index[:500].tolist()
    d.fit(ids)
    z = d.transform(ids + ["not-a-tce"])
    assert z.shape == (501, len(FEATURES) + 1)
    assert np.all(np.abs(z[:, :-1]) <= 5) and np.isfinite(z).all()
    assert z[-1, -1] == 1 and np.all(z[-1, :-1] == 0)
    assert not set(FEATURES) & {"disposition", "koi_pdisposition", "fpflag", "robovetter_decision"}


def test_retention_probability_follows_the_parameters():
    import retention_model as rm
    margin = np.array([[-1.0] * 100, [0.0] * 100, [1.0] * 100])
    magma = np.array([0.0, 0.0, 0.0])
    p = rm.prob(np.array([[0.0, -1.3, 2.6]]), margin, magma, thin=10)[0]
    assert p[0] > p[1] > p[2]
    assert p[1] == pytest.approx(0.5)
    hot = rm.prob(np.array([[0.0, -1.3, 2.6]]), margin, np.array([1.0, 1.0, 1.0]), thin=10)[0]
    assert np.all(hot > p)


def test_retention_results_are_internally_consistent():
    f = ROOT / "outputs" / "stage_c" / "retention_model.json"
    if not f.is_file():
        pytest.skip("retention model not run")
    r = json.loads(f.read_text())
    th = np.load(ROOT / "outputs" / "stage_c" / "retention_posterior.npz")["theta"]
    assert np.mean(th[:, 1] < 0) == pytest.approx(r["p_beta_negative"], abs=0.01)
    assert r["expectations"]["A1 P(beta<0) >= 0.90"] == (r["p_beta_negative"] >= 0.90)
    assert set(r["magma_planets"]) == {"GJ 367 b", "55 Cnc e", "TOI-561 b"}


def test_candidate_completion_flags_period_aliases(monkeypatch):
    import candidate_factor as cf
    monkeypatch.setattr(cf, "tic_table", lambda tics: pd.DataFrame(
        {"Teff": [3300.0], "rad": [0.3], "mass": [0.3], "Jmag": [9.0]}, index=pd.Index(["1"], name="ID")))
    d = pd.DataFrame({"tic": [1, 1], "pl_orbper": [10.0, 700.0], "pl_insol": [np.nan, 0.6],
                      "st_teff": [np.nan, 3300.0], "st_rad": [np.nan, 0.3], "st_mass": [np.nan, 0.3], "sy_jmag": [np.nan, 9.0]})
    out = cf.complete(d, "tic")
    lum = 0.3 ** 2 * (3300 / 5772) ** 4
    a10 = (cf.G * 0.3 * cf.MSUN * (10 * cf.DAY) ** 2 / (4 * np.pi ** 2)) ** (1 / 3) / cf.AU
    assert out.st_teff.iloc[0] == 3300 and "from TIC" in out.param_notes.iloc[0]
    assert out.pl_insol.iloc[0] == pytest.approx(lum / a10 ** 2, rel=1e-6)            # derived, 30 % error
    assert not out.period_insolation_conflict.iloc[0]
    assert out.period_insolation_conflict.iloc[1]                                       # 700 d vs S = 0.6: alias
    assert out.pl_orbsmax.iloc[1] == pytest.approx(np.sqrt(lum / 0.6))                  # orbit from insolation


def test_candidate_platt_is_the_tess_map():
    import candidate_factor as cf
    s = np.array([0.1, 0.5, 0.9])
    a, b = cf.TESS_PLATT
    expect = 1 / (1 + np.exp(-(a * np.log(s / (1 - s)) + b)))
    assert np.allclose(cf.platt(s), expect) and np.all(np.diff(cf.platt(s)) > 0)


def test_holm_adjustment():
    from stage_a_transit_model.evaluate_v3 import holm
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)                    # monotone: never below the previous step


def test_snr_balance_gap_ci_is_zero_without_change():
    from stage_a_transit_model.evaluate_snr_balance import bootstrap_diff
    rng = np.random.default_rng(3)
    a, b = rng.random(30), rng.random(200)
    assert bootstrap_diff(a, a, b, b, n=200) == [0.0, 0.0]
    assert bootstrap_diff(a + 0.2, a, b, b, n=200) == pytest.approx([-0.2, -0.2])   # deep up 0.2 -> gap down 0.2


def test_mass_relation_option_leaves_the_default_unchanged():
    import target_physics as tp
    row = pd.DataFrame({"pl_name": ["X b"], "pl_rade": [1.1], "pl_radeerr1": [0.05], "pl_radeerr2": [-0.05],
                        "pl_insol": [0.8], "pl_insolerr1": [0.1], "pl_insolerr2": [-0.1], "st_teff": [3300.0],
                        "st_tefferr1": [50.0], "st_tefferr2": [-50.0], "st_rad": [0.3], "st_raderr1": [0.01], "st_raderr2": [-0.01],
                        "st_lum": [np.nan], "pl_orbsmax": [0.05], "pl_orbsmaxerr1": [0.002], "pl_orbsmaxerr2": [-0.002],
                        "sy_jmag": [9.0], "tran_flag": [1], "discoverymethod": ["Transit"], "pl_bmassprov": ["Estimated"],
                        "pl_bmasse": [np.nan], "pl_bmasseerr1": [np.nan], "pl_bmasseerr2": [np.nan], "pl_bmasselim": [0]})
    _, a = tp.monte_carlo(row, n=500, return_draws=True)
    _, b = tp.monte_carlo(row, n=500, return_draws=True, mass_relation="chen_kipping")
    _, c = tp.monte_carlo(row, n=500, return_draws=True, mass_relation="otegi_rocky")
    assert np.array_equal(a["margin_primary"], b["margin_primary"])
    assert not np.array_equal(a["margin_primary"], c["margin_primary"])
    with pytest.raises(ValueError):
        tp.monte_carlo(row, n=10, mass_relation="nonsense")
