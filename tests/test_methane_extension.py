"""Methane extension: frozen labels, label rules, and the calibration metrics."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "stage_b_atmospheres"))
LABELS = R / "outputs" / "labels" / "ch4_labels_extended.csv"


def test_labels_are_frozen_and_follow_the_rules():
    rec = json.loads((R / "outputs" / "stage_b_methane_ext" / "protocol_hashes.json").read_text())
    assert hashlib.sha256(LABELS.read_bytes()).hexdigest() == rec["ch4_labels_extended.csv"]
    lab = pd.read_csv(LABELS, dtype=str, keep_default_na=False)
    assert lab.planet.is_unique
    assert set(lab.CH4) <= {"0", "1", "unknown"}
    assert (lab.source.str.len() > 0).all()                      # every label cites its source
    assert not ((lab.featureless == "1") & (lab.CH4 == "1")).any()  # a featureless spectrum cannot show CH4
    old = pd.read_csv(R / "outputs" / "labels" / "verified_gas_labels.csv", dtype=str)
    assert not set(lab.planet) & set(old.planet)                 # the extension shares no planet with Level 1/3


def test_ece_is_zero_for_perfect_calibration_and_large_for_overconfidence():
    import methane_extension as me

    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0])
    assert me.ece(y, np.full(10, 0.5), bins=1) == pytest.approx(0.0)
    assert me.ece(np.zeros(10), np.full(10, 0.9), bins=5) == pytest.approx(0.9)


def test_nested_platt_never_sees_the_planet_it_scores():
    import methane_extension as me

    d = pd.DataFrame({"label": [0, 0, 0, 1, 1, 1], "gbm": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]})
    cal = me.nested_platt(d, "gbm")
    flipped = d.copy()
    flipped.loc[0, "label"] = 1                                   # changing planet 0's own label...
    assert me.nested_platt(flipped, "gbm")[0] == pytest.approx(cal[0])   # ...cannot change its own score


def test_round2_labels_are_frozen_and_only_fill_round1_unknowns():
    rec = json.loads((R / "outputs" / "stage_b_methane_ext" / "protocol_hashes.json").read_text())
    p2 = R / "outputs" / "labels" / "ch4_labels_round2.csv"
    assert hashlib.sha256(p2.read_bytes()).hexdigest() == rec["ch4_labels_round2.csv"]
    r1 = pd.read_csv(LABELS, dtype=str, keep_default_na=False).set_index("planet")
    r2 = pd.read_csv(p2, dtype=str, keep_default_na=False)
    assert set(r2.CH4) <= {"0", "1", "unknown"} and (r2.source.str.len() > 0).all()
    assert (r1.loc[r2.planet, "CH4"] == "unknown").all()       # round 2 never overrides a round-1 label
    tentative = {"GJ 1214 b", "TOI-1130 b", "Kepler-51 d", "LTT 9779 b"}
    assert not tentative & set(r2.planet)                        # tentative evidence was not reopened
