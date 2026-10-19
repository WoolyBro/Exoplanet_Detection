"""The released JWST molecule screener reproduces the validated register calls from raw spectrum tables."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "jwst_molecule_screener"))
sys.path.insert(0, str(ROOT / "stage_b_atmospheres"))
pytest.importorskip("torch")


def raw_spectra(planet):
    import stage_b_pipeline as sbp
    out = []
    for r in sbp.select_spectra([planet]).itertuples():
        df, _ = sbp.parse_tbl(sbp.SPEC_DIR / "spectra_tbl" / r.file)
        out.append((df.wl.to_numpy(), df.depth_ppm.to_numpy(), df.err_ppm.to_numpy()))
    return out


def test_binning_is_median_centred_with_a_mask():
    from jwst_molecule_screener import bin_spectrum
    wl = np.linspace(2.0, 4.0, 400)
    x = bin_spectrum(wl, 5000 + 10 * np.sin(wl * 7), np.full(400, 30.0))
    assert x.shape == (3, 100) and set(np.unique(x[2])) <= {0.0, 1.0}
    assert np.median(x[0][x[2] > 0]) == pytest.approx(0.0, abs=1e-9)
    assert np.all(x[1][x[2] > 0] > 0)


@pytest.mark.parametrize("planet,gas,call", [("WASP-127 b", "CO2", "present"), ("LHS 1140 b", "CO2", "absent"),
                                             ("GJ 1214 b", "CO2", "absent")])
def test_screener_reproduces_register_calls(planet, gas, call):
    spec_dir = ROOT / "exoplanet_research_data" / "04_atmospheric_spectra"
    if not spec_dir.is_dir():
        pytest.skip("research spectra not available")
    from jwst_molecule_screener import screen
    spectra = raw_spectra(planet)
    if not spectra:
        pytest.skip("no pack spectrum")
    reg = pd.read_csv(ROOT / "outputs" / "stage_b_register" / "molecule_register.csv")
    expected = reg[(reg.planet == planet) & (reg.gas == gas)]
    res = screen(spectra)[gas]
    assert res["call"] == call
    if len(expected) and int(expected.n_spectra.iloc[0]) == res["n_spectra"]:
        assert res["p_gbm"] == pytest.approx(float(expected.p_gbm.iloc[0]), abs=2e-3)
