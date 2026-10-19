# JWST molecule screener

Calibrated **CH₄** and **CO₂** calls for exoplanet transmission spectra (1–5 µm). The detectors were trained only on simulated spectra that include unocculted starspots, faculae and hazes, and were then **validated on real JWST spectra of planets they never saw**. That validation was pre-registered, with labels frozen before scoring.

| Gas | Real labelled JWST planets | ROC-AUC (GBM / 1-D CNN) | Gas-specific? |
|---|---|---|---|
| CH₄ | 41 (11 with CH₄) | 1.00 / 0.97 | Passes a gas-blind feature-strength control |
| CO₂ | 34 (14 with CO₂) | 0.98 / 0.996 | Passes it too; masking 4.20–4.45 µm removes most of the GBM's skill |

## Install and use

```bash
pip install ./jwst_molecule_screener
jwst-molecule-screener visit1.txt visit2.txt --depth-unit ppm      # columns: wavelength (um), depth, uncertainty
jwst-molecule-screener nrs1.txt nrs2.txt --merge                     # two detector halves of ONE spectrum
```

```python
from jwst_molecule_screener import screen, read_table
result = screen([read_table("visit1.txt"), read_table("visit2.txt")])   # several spectra of one planet
result["CO2"]   # {'call': 'present' | 'absent' | 'ambiguous' | 'not observable', 'p_gbm': ..., 'p_cnn': ..., ...}
```

## What a call means

- **Probabilities.** `p_gbm` and `p_cnn` are each model's calibrated probability that a JWST analysis would *detect the gas's feature* in these spectra. They are calibrated on the labelled planets, and they are not an atmospheric abundance.
- **present / absent:** both models agree, at ≥ 0.8 or ≤ 0.2.
- **ambiguous:** anything else, or several spectra of the planet falling on both sides of 0.5. In validation, that pattern marks planets whose literature results depend on analysis choices or on stellar contamination.
- **not observable:** no spectrum has ≥ 3 bins in a band:
  - CH₄: 2.20–2.40 or 3.20–3.45 µm;
  - CO₂: 4.20–4.45 µm.

## Limits

- **Validation sample.** The test planets are mostly hot and warm giants and sub-Neptunes. Nothing here is validated for temperate rocky planets, whose current spectra do not contain a detectable signal.
- **Calibration** rests on 41 and 34 planets. Treat probabilities near 0.5 as uninformative.
- **Version pin.** The models are pickled with scikit-learn 1.9.1, pinned in `pyproject.toml`.

## Provenance

The models are built by `tools/build_screener.py` in the research repository. The build verifies that the package reproduces the validated scores (to within 10⁻⁶).

The validation is written up in:
- `eda/METHANE_EXTENSION_RESULTS.md` (CH₄);
- `eda/CO2_EXTENSION_RESULTS.md` (CO₂);
- `eda/MOLECULE_REGISTER.md`, the frozen calls on 29 unresolved planets.
