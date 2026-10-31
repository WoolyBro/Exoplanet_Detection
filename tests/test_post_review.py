"""Checks of the post-review statistics (tools/post_review_stats.py) and their saved results."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "tools"))
from post_review_stats import holm, newcombe  # noqa: E402

OUT = R / "outputs" / "post_review"


def test_holm_matches_a_hand_calculation():
    a = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert a == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_holm_is_monotone_and_capped():
    p = dict(zip("abcde", [1e-6, 0.2, 0.2, 0.9, 0.95]))
    a = holm(p)
    v = [a[k] for k in sorted(p, key=p.get)]
    assert v == sorted(v) and max(v) <= 1.0


def test_newcombe_stays_usable_at_a_perfect_ranking():
    lo, hi = newcombe(1.0, 11, 30)
    assert hi == 1.0 and 0.8 < lo < 0.95


def test_newcombe_narrows_with_more_data():
    a, b = newcombe(0.9, 10, 10), newcombe(0.9, 100, 100)
    assert (b[1] - b[0]) < (a[1] - a[0]) and a[0] < 0.9 < a[1]


@pytest.mark.skipif(not (OUT / "retention_prior_sensitivity.json").is_file(), reason="post-review outputs not present")
def test_registered_prior_row_reproduces_the_stored_fit():
    rows = json.loads((OUT / "retention_prior_sensitivity.json").read_text())["rows"]
    stored = json.loads((R / "outputs" / "stage_c" / "retention_model.json").read_text())
    assert rows[0]["p_beta_negative"] == stored["p_beta_negative"]
    assert rows[0]["beta_median"] == pytest.approx(stored["posterior"]["beta"]["median"], abs=0.01)
    assert all(r["beta_ci95"][1] > 0 for r in rows if r["beta_prior"][0] == 0)


@pytest.mark.skipif(not (OUT / "detector_injection.json").is_file(), reason="post-review outputs not present")
def test_injection_false_rate_and_recovery():
    g = json.loads((OUT / "detector_injection.json").read_text())["gases"]
    for gas in ("CH4", "CO2"):
        for lv in ("-5", "-4", "-3", "-2"):
            r = g[gas][f"logx_{lv}"]["gbm"]
            assert r["false_rate_uninjected"] == 0 and r["recovered_when_signal_gt_3x_error"] == 1.0
