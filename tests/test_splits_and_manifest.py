"""The leakage guarantees every Stage A result rests on, checked on the committed split files."""

from pathlib import Path

import pandas as pd
import pytest

import data_manifest as dm
import split_signal_families as ssf

SPLITS = Path(__file__).resolve().parents[1] / "splits"


def test_manifest_validates_and_rejects_excluded():
    dm._validate_manifest()
    with pytest.raises(ValueError):
        dm.role_of("exoplanet_research_data/05_final_ML_dataset_DO_NOT_USE/anything.csv")
    assert dm.role_of("splits/koi_cumulative_split_v2.csv") == "STAGE_1_TRAINING"


@pytest.mark.parametrize("mission", ["kepler", "tess"])
def test_v1_is_star_disjoint(mission):
    cfg = ssf.MISSIONS[mission]
    df = pd.read_csv(SPLITS / cfg["v1"], low_memory=False)
    assert (df.groupby(cfg["star"])["split"].nunique() == 1).all()


@pytest.mark.parametrize("mission", ["kepler", "tess"])
def test_v2_is_star_and_signal_family_disjoint(mission):
    cfg = ssf.MISSIONS[mission]
    df = pd.read_csv(SPLITS / cfg["v2"], low_memory=False)
    ssf.check(df, cfg, mission)            # raises LeakageError on any shared star or signal twin


@pytest.mark.parametrize("mission", ["kepler", "tess"])
def test_v2_keeps_rows_labels_and_balance(mission):
    cfg = ssf.MISSIONS[mission]
    v1 = pd.read_csv(SPLITS / cfg["v1"], low_memory=False)
    v2 = pd.read_csv(SPLITS / cfg["v2"], low_memory=False)
    assert len(v1) == len(v2)
    assert (v1.label_harmonized.fillna("") .values == v2.label_harmonized.fillna("").values).all()
    whole = v2.label_harmonized.value_counts(normalize=True)
    for _, part in v2.groupby("split"):
        share = part.label_harmonized.value_counts(normalize=True).reindex(whole.index).fillna(0)
        assert (share - whole).abs().max() < 0.02, "a v2 split drifted > 2 pp from the catalogue balance"


def test_twin_detector_finds_a_planted_twin():
    df = pd.DataFrame({"kepid": [1, 2, 3], "koi_period": [5.0, 5.0 * (1 + 5e-5), 7.3],
                       "koi_time0bk": [100.0, 100.02, 100.0], "koi_duration": [3.0, 3.0, 3.0]})
    assert ssf.signal_twins(df, ssf.MISSIONS["kepler"]) == [(0, 1)]
