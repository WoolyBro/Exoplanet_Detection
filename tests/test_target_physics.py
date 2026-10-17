"""Habitable-zone and TSM physics against published values (Kopparapu et al. 2014; Kempton et al. 2018)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage_c_priority_fusion"))
import target_physics as tp  # noqa: E402


def test_solar_hz_matches_kopparapu_2014():
    lim = tp.hz_flux_limits([5780.0])
    assert 1 / np.sqrt(lim["runaway_greenhouse"][0]) == pytest.approx(0.95, abs=0.005)
    assert 1 / np.sqrt(lim["maximum_greenhouse"][0]) == pytest.approx(1.67, abs=0.01)


@pytest.mark.parametrize("name, insol, cons, opt", [("Venus", 1.91, False, False), ("Earth", 1.0, True, True),
                                                    ("Mars", 0.43, True, True), ("Jupiter", 0.037, False, False)])
def test_solar_system_planets(name, insol, cons, opt):
    c, o = tp.in_hz([insol], [5780.0])
    assert (bool(c[0]), bool(o[0])) == (cons, opt), name


def test_hz_is_clamped_outside_the_fit_range():
    assert np.allclose(tp.hz_flux_limits([2000.0])["runaway_greenhouse"], tp.hz_flux_limits([2600.0])["runaway_greenhouse"])


def test_earth_equilibrium_temperature():
    assert tp.teq_kempton(5772.0, 1.0, 1.0) == pytest.approx(278.3, abs=0.5)


def test_tsm_scale_factor_bins():
    _, cut = tp.tsm(np.array([1.0, 2.0, 3.0, 6.0]), np.ones(4), np.ones(4), np.full(4, 300.0), np.zeros(4))
    assert cut.tolist() == [12.0, 92.0, 92.0, 92.0]
    v, _ = tp.tsm(np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([0.0]))
    assert v[0] == pytest.approx(0.190)


def test_chen_kipping_mass_relation():
    assert tp.chen_kipping_mass([1.0])[0] == pytest.approx(0.9718)
    assert tp.chen_kipping_mass([2.0])[0] == pytest.approx(1.436 * 2 ** 1.70)


def test_mass_upper_limit_is_not_used_as_a_measurement():
    import pandas as pd

    row = dict(pl_name="X b", pl_insol=1.0, pl_insolerr1=0.01, pl_insolerr2=-0.01, st_lumerr1=0.0, pl_orbsmax=0.03,
               pl_orbsmaxerr1=1e-4, pl_orbsmaxerr2=-1e-4, st_teff=3000.0, st_tefferr1=10.0, st_tefferr2=-10.0,
               pl_rade=1.4, pl_radeerr1=0.01, pl_radeerr2=-0.01, tran_flag=1, discoverymethod="Transit",
               pl_bmasse=35.0, pl_bmasseerr1=np.nan, pl_bmasseerr2=np.nan, pl_bmassprov="Mass", st_rad=0.2,
               st_raderr1=0.001, st_raderr2=-0.001, sy_jmag=10.0)
    limit, measured = pd.DataFrame([{**row, "pl_bmasselim": 1}]), pd.DataFrame([{**row, "pl_bmasselim": 0}])
    tsm_limit = tp.monte_carlo(limit, n=200).tsm_median[0]
    tsm_meas = tp.monte_carlo(measured, n=200).tsm_median[0]
    # an upper limit falls back to Chen & Kipping (~2.6 M_earth), so its TSM is ~35/2.6 times larger
    assert tsm_limit / tsm_meas == pytest.approx(35.0 / tp.chen_kipping_mass([1.4])[0], rel=0.05)
