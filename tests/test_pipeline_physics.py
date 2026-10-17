"""Folding, view construction and the error channel, on synthetic light curves with a known answer."""

import numpy as np
import pytest

import preprocessing_pipeline as pp
from stage_a_transit_model.cnn_lstm import error_channel


def synthetic(period=3.0, epoch=1.0, duration=0.12, depth=1e-3, noise=1e-4, days=200.0, cadence_min=30, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, days, cadence_min / 1440.0)
    phase = (t - epoch + 0.5 * period) % period - 0.5 * period
    flux = 1.0 - depth * (np.abs(phase) < duration / 2) + rng.normal(0, noise, len(t))
    return pp.FlattenedLightCurve(time=t, flux=flux, flux_err=np.full(len(t), noise), mission="Kepler")


def test_fold_puts_transit_at_phase_zero():
    lc = synthetic()
    folded = pp.phase_fold(lc, 3.0, 1.0)
    assert np.all(np.diff(folded.phase) >= 0), "fold must be sorted by phase"
    assert folded.phase.min() >= -1.5 and folded.phase.max() < 1.5
    inside = np.abs(folded.phase) < 0.05
    assert folded.flux[inside].mean() < folded.flux[~inside].mean() - 5e-4


def test_views_are_normalised_and_centred():
    folded = pp.phase_fold(synthetic(), 3.0, 1.0)
    g = pp.build_global_view(folded, duration=0.12)
    loc = pp.build_local_view(folded, 0.12)
    for v, n in ((g, pp.GLOBAL_BINS), (loc, pp.LOCAL_BINS)):
        assert v.values.shape == (n,)
        assert np.isclose(v.values.min(), -1.0), "deepest bin is -1 by construction"
        assert abs(np.median(v.values)) < 1e-9, "median is 0 by construction"
    # A flat-bottomed transit's deepest bin can be anywhere on its floor: +/- half a duration, which in
    # the local view (+/- 4 durations over 201 bins) is +/- 12.5 bins - the pipeline's own criterion.
    half_width_bins = (pp.LOCAL_BINS - 1) * 0.5 / (2 * pp.LOCAL_NUM_DURATIONS)
    assert abs(int(np.argmin(loc.values)) - pp.LOCAL_BINS // 2) <= half_width_bins, "dip must sit within the transit"


def test_error_channel_does_not_change_the_depth_view():
    lc = synthetic()
    with_err = pp.build_local_view(pp.phase_fold(lc, 3.0, 1.0), 0.12)
    lc.flux_err = None
    without = pp.build_local_view(pp.phase_fold(lc, 3.0, 1.0), 0.12)
    assert np.array_equal(with_err.values, without.values)
    assert with_err.errors is not None and np.isfinite(with_err.errors).all()


def test_error_channel_scales_with_noise():
    quiet = pp.build_local_view(pp.phase_fold(synthetic(noise=1e-4), 3.0, 1.0), 0.12).errors
    noisy = pp.build_local_view(pp.phase_fold(synthetic(noise=4e-4, seed=1), 3.0, 1.0), 0.12).errors
    ratio = np.median(noisy) / np.median(quiet)
    assert 3.0 < ratio < 5.5, f"4x the per-point noise should give ~4x the per-bin error, got {ratio:.2f}"


def test_error_channel_maps_missing_to_maximum_uncertainty():
    out = error_channel(np.array([1e-3, 1e-1, 1.0, np.nan, np.inf]))
    assert np.isfinite(out).all()
    assert out[3] == out.max() and out[4] == out.max()
    assert out[0] < out[1] < out[2]


def test_ephemeris_drift_rule():
    eph = pp.Ephemeris(object_id="x", star_id=1, mission="Kepler", period=10.0, epoch=0.0,
                       duration=0.2, label=None, period_err=1e-4, epoch_err=1e-3)
    drift = pp.ephemeris_drift_hours(eph, np.array([0.0, 1000.0]))     # 100 cycles
    assert drift == pytest.approx(np.hypot(1e-3, 100 * 1e-4) * 24)
