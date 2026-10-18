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


def test_analytic_contamination_marginalisation_matches_monte_carlo():
    """The real-data retrieval integrates per-visit linear nuisances analytically; check that integral
    against brute-force averaging of the likelihood over the nuisance prior."""
    import retrieval_trappist1e_real as rr

    rng = np.random.default_rng(0)
    n = 8
    depth = np.full(n, 5000.0)
    rr._TPL.update(spot=np.linspace(0.2, 0.8, n), facula=np.linspace(-1.0, -0.1, n))
    sigma = np.full(n, 100.0)
    y = depth + rng.normal(0, 100, n)
    analytic = rr.marginal_loglike(depth, y, sigma)
    lam = np.array([rr.PRIOR_SD["spot"], rr.PRIOR_SD["facula"], rr.PRIOR_SD["offset_ppm"]])
    th = rng.normal(0, 1, (400_000, 3)) * lam
    model = depth * (1 + th[:, :1] * rr._TPL["spot"] + th[:, 1:2] * rr._TPL["facula"]) + th[:, 2:3]
    ll = -0.5 * np.sum(((y - model) / sigma) ** 2, axis=1) - n * np.log(np.sqrt(2 * np.pi) * 100.0)
    mc = np.log(np.mean(np.exp(ll - ll.max()))) + ll.max()
    rr._TPL.clear()
    assert analytic == pytest.approx(mc, abs=0.15)


def _toy(n=60, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1.5, n)
    sig = rng.uniform(20, 40, n)
    s = 50 * np.exp(-0.5 * ((x - 0.8) / 0.05) ** 2)
    B = np.column_stack([np.ones(n), np.sin(3 * x), x ** 2])
    K = 1e4 * np.exp(-0.5 * ((x[:, None] - x[None, :]) / 0.4) ** 2)
    return s, B, sig, K


def test_gls_information_reduces_to_the_template_projection_without_correlated_noise():
    s, B, sig, K = _toy()
    proj = df.project_out(s, B, 1 / sig ** 2)
    q = df.gls_info(s, B, sig, K, np.array([1.0, 2.0]), np.array([0.0]))
    assert q[0, 0] == pytest.approx(np.sum((proj / sig) ** 2), rel=1e-9)
    assert q[1, 0] == pytest.approx(np.sum((proj / sig) ** 2) / 4, rel=1e-9)     # white noise x2 -> info / 4


def test_gls_information_matches_an_explicit_covariance_inverse():
    s, B, sig, K = _toy()
    for A in (0.3, 1.0, 3.0):
        Ci = np.linalg.inv(np.diag(sig ** 2) + A ** 2 * K)
        brute = s @ Ci @ s - s @ Ci @ B @ np.linalg.solve(B.T @ Ci @ B, B.T @ Ci @ s)
        assert df.gls_info(s, B, sig, K, np.array([1.0]), np.array([A]))[0, 0] == pytest.approx(brute, rel=1e-8)
    # correlated noise can only remove information
    q = df.gls_info(s, B, sig, K, np.array([1.0]), np.array([0.0, 0.5, 2.0]))[0]
    assert q[0] >= q[1] >= q[2]


def test_combining_modes_helps_only_through_shared_contamination_terms():
    s, B, sig, _ = _toy()
    P = [{"s": s, "sig": sig, "spot": B[:, 1], "facula": B[:, 2]},
         {"s": 0.6 * s[:45], "sig": sig[:45], "spot": 2 * B[:45, 1] + B[:45, 2], "facula": B[:45, 2] ** 2}]
    one = np.ones((1, 2))
    single = [df.joint_info(P, np.eye(2)[i], one, False)[0] for i in range(2)]
    # no shared terms: information adds linearly in the split, so the best split is the better mode alone
    assert max(df.joint_info(P, f, one, False)[0] for f in df.simplex(2)) == pytest.approx(max(single), rel=1e-9)
    # a one-mode "split" with shared terms is that mode's own template projection
    proj = df.project_out(s, B, 1 / sig ** 2)
    assert df.joint_info(P, np.array([1.0, 0.0]), one, True)[0] == pytest.approx(np.sum((proj / sig) ** 2), rel=1e-9)
    assert len(df.simplex(3)) == 231 and all(abs(f.sum() - 1) < 1e-12 for f in df.simplex(3))
