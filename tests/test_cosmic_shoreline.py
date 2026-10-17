"""Cosmic shoreline (Zahnle & Catling 2017): definitions, Solar System check, and its use in Stage C."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage_c_priority_fusion"))
import cosmic_shoreline as cs  # noqa: E402
import target_physics as tp  # noqa: E402


def test_escape_velocity_and_xuv_scaling_are_earth_normalised():
    assert cs.v_esc_kms(1.0, 1.0) == pytest.approx(11.186)
    assert cs.cumulative_xuv(1.0, 1.0) == pytest.approx(1.0)                  # Earth around the Sun
    assert cs.cumulative_xuv(1.0, 1e-3) == pytest.approx(1e-3 ** -0.6)       # faint stars: more XUV per bolometric


def test_solar_system_is_sorted_by_the_primary_shoreline():
    from validate_shoreline import solar_system_check

    ss = solar_system_check()
    wrong = ss.loc[~ss["correct_primary_1e-3"], "body"].tolist()
    assert wrong == ["Ganymede"]                     # the only miss, by 0.03-0.2 dex
    assert ss.set_index("body").loc["Earth", "margin_primary_1e-3"] < -1
    assert ss.set_index("body").loc["Mercury", "margin_primary_1e-3"] > 1


def _row(name, rade, masse, insol, lum, tran=1):
    return dict(pl_name=name, pl_insol=insol, pl_insolerr1=0.01 * insol, pl_insolerr2=-0.01 * insol, st_lumerr1=0.01,
                st_lumerr2=-0.01, st_lum=np.log10(lum), pl_orbsmax=0.05, pl_orbsmaxerr1=1e-4, pl_orbsmaxerr2=-1e-4,
                st_teff=3300.0, st_tefferr1=10.0, st_tefferr2=-10.0, pl_rade=rade, pl_radeerr1=0.01, pl_radeerr2=-0.01,
                tran_flag=tran, discoverymethod="Transit", pl_bmasse=masse, pl_bmasseerr1=0.05, pl_bmasseerr2=-0.05,
                pl_bmassprov="Mass", pl_bmasselim=0, st_rad=0.3, st_raderr1=0.001, st_raderr2=-0.001, sy_jmag=9.0)


def test_terrestrial_with_atmosphere_rewards_neither_giants_nor_bare_hot_rocks():
    df = pd.DataFrame([_row("giant", 11.0, 300.0, 1.0, 0.01),        # retains, but not terrestrial
                       _row("hot rock", 1.2, 1.8, 400.0, 0.01),      # terrestrial, far past the shoreline
                       _row("temperate rock", 1.2, 1.8, 0.5, 0.5)])  # terrestrial, quiet star, low fluence
    out, draws = tp.monte_carlo(df, n=500, return_draws=True)
    t_atm = draws["terrestrial_atm"].mean(axis=1)
    assert out.p_retain_primary[0] > 0.9 and t_atm[0] == 0.0
    assert out.p_retain_primary[1] < 0.05 and t_atm[1] < 0.05
    assert t_atm[2] > 0.9
