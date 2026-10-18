"""The external-data benchmarks added on 2026-10-18: K2 ephemerides and time system, the duration estimator,
the exoplanet.eu name matching and the paired statistics of the DR25 evaluation (no network, no models)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import k2_ephemerides as k2e  # noqa: E402
import label_crosscheck as lc  # noqa: E402


def test_duration_estimator_gives_the_earth_sun_transit():
    # Earth across the Sun, central: T14 = (P/pi) asin((1+k) R/a) = 13.0 h
    row = pd.DataFrame({"pl_orbper": [365.25], "st_rad": [1.0], "st_mass": [1.0], "st_logg": [np.nan],
                        "pl_ratdor_set": [np.nan], "pl_ratror_set": [0.00917], "pl_rade": [np.nan], "pl_imppar_set": [0.0]})
    assert k2e.estimate_duration_h(row).iloc[0] == pytest.approx(13.0, rel=0.02)


def test_unknown_impact_parameter_averages_the_chord_over_b():
    # <sqrt(1 - b^2)> over b ~ U(0, 1) is pi/4 for a vanishing planet. The estimator takes a plain mean over
    # 201 grid points, 0.2 % low; it is kept because the K2 views were built with it, and 0.2 % is far
    # inside the estimator's own spread (68 % of estimates within 0.90-1.25 of the catalogue duration).
    assert k2e.mean_chord(np.array([0.0]), np.array([np.nan]))[0] == pytest.approx(np.pi / 4, rel=3e-3)
    assert k2e.mean_chord(np.array([0.1]), np.array([0.6]))[0] == pytest.approx(np.sqrt(1.1 ** 2 - 0.36))


def test_k2_ephemeris_uses_bkjd_and_days():
    import preprocessing_pipeline as p
    e = p.ephemeris_from_row({"pl_name": "K2-415 b", "epic": 211414619, "pl_orbper": 4.0179, "pl_tranmid": 2457812.75,
                              "pl_trandur": 1.12, "label_harmonized": "CONFIRMED"}, "K2")
    assert e.epoch == pytest.approx(2457812.75 - 2454833.0)
    assert e.duration == pytest.approx(1.12 / 24)
    assert e.star_id == 211414619 and e.mission == "K2"


def test_planet_names_match_across_catalogue_conventions():
    assert lc.key("HAT-P-18 Ab") == lc.key("HAT-P-18 b")
    assert lc.key("TRAPPIST-1 e") == lc.key("TRAPPIST-1e")
    assert lc.key("WASP-39 b") != lc.key("WASP-39 c")


def test_wilson_interval_and_exact_mcnemar():
    from stage_a_transit_model.evaluate_dr25_simulated import mcnemar, wilson
    p, lo, hi = wilson(90, 100)
    assert p == 0.9 and lo < 0.9 < hi and hi - lo < 0.13
    a = np.array([True] * 30 + [False] * 10)
    b = np.array([True] * 30 + [True] * 10)              # the second method wins all 10 discordant pairs
    r = mcnemar(a, b)
    assert r["ours_only"] == 0 and r["robovetter_only"] == 10 and r["p_exact"] == pytest.approx(2 * 0.5 ** 10)
