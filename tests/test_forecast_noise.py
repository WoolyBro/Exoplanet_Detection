"""Gas-detectability forecast: noise scaling and the transit-count solver (no petitRADTRANS needed)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage_b_atmospheres"))
import detectability_forecast as df  # noqa: E402
import noise_model as nm  # noqa: E402


def test_photon_noise_scales_with_magnitude_and_duration():
    base = nm.ideal_sigma_ppm(3.0, 0.03, 10.0, 3500.0, 2.0)
    assert nm.ideal_sigma_ppm(3.0, 0.03, 12.5, 3500.0, 2.0) / base == pytest.approx(10 ** (0.2 * 2.5))
    assert nm.ideal_sigma_ppm(3.0, 0.03, 10.0, 3500.0, 8.0) / base == pytest.approx(0.5)
    assert nm.ideal_sigma_ppm(3.0, 0.12, 10.0, 3500.0, 2.0) / base == pytest.approx(0.5)


def test_transits_needed_matches_the_analytic_answer_without_a_floor():
    signal = np.array([[30.0, 40.0, 0.0]]) ** 2
    sigma1 = np.array([[50.0, 50.0, 50.0]]) ** 2
    snr1 = np.sqrt((signal / sigma1).sum())
    n = df.solve_n(signal, sigma1, 0.0, 5.0)[0]
    assert n == pytest.approx((5.0 / snr1) ** 2, rel=1e-6)


def test_a_floor_above_the_signal_makes_detection_unreachable():
    signal = np.array([[10.0, 10.0]]) ** 2
    sigma1 = np.array([[50.0, 50.0]]) ** 2
    assert np.isinf(df.solve_n(signal, sigma1, 30.0 ** 2, 5.0)[0])


@pytest.mark.parametrize("instrument,lo,hi,mode", [
    ("Near Infrared Spectrograph (NIRSpec) - PRISM", 0.6, 5.3, "NIRSpec_PRISM"),
    ("Near Infrared Spectrograph (NIRSpec) - G395H", 2.87, 5.17, "NIRSpec_G395"),
    ("Near Infrared Imager and Slitless Spectrograph (NIRISS)", 0.85, 2.83, "NIRISS_SOSS"),
    ("Mid-Infrared Instrument (MIRI) - LRS", 5.0, 11.8, "MIRI_LRS"),
    ("Near Infrared Camera (NIRCam)", 3.9, 5.0, None),
])
def test_observing_mode_is_inferred_from_instrument_and_coverage(instrument, lo, hi, mode):
    assert nm.infer_mode(instrument, lo, hi) == mode


def test_bayes_factor_to_sigma_matches_benneke_and_seager_2013():
    """Their Table 2: ln B = 0.9 -> 2.0 sigma, 2.5 -> 2.7, 5.0 -> 3.6, 11.0 -> 5.0."""
    from retrieval_trappist1e import bayes_sigma

    for ln_b, sigma in [(0.9, 2.0), (2.5, 2.7), (5.0, 3.6), (11.0, 5.0)]:
        assert bayes_sigma(ln_b) == pytest.approx(sigma, abs=0.1)
    assert bayes_sigma(-1.0) == 0.0


def test_transits_for_target_sigma_interpolates_and_flags_extrapolation():
    from retrieval_report import n_for_sigma

    n, ext = n_for_sigma([2, 4, 8, 16], [1.0, 1.5, 3.0, 6.0], 5.0)
    assert 8 < n < 16 and not ext
    n, ext = n_for_sigma([2, 4], [1.0, 1.4], 5.0)
    assert n > 4 and ext
