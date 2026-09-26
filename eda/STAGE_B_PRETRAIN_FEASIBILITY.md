# Stage B: is synthetic pre-training on INARA viable? (design assessment, nothing trained)

**Recommendation: no. Do not build synthetic-pretrain-then-test-on-real with the data on this machine.** There are three independent blockers, and the first one alone ends it:

1. **The local INARA file contains no spectra.** `psg_models.csv` is 30 rows × 29 columns of planet/star parameters and gas mixing ratios. There is no wavelength axis and no flux or transit-depth array. The same is true of PyATMOS (`datasets/pyatmos_final.csv`, 108,839 rows): concentrations, fluxes, pressure and temperature, no spectra. **There is nothing locally to pre-train a spectrum model on.**
2. **Even if the spectra existed, the planets are the wrong kind by two to three orders of magnitude.** INARA's Earth-like planets would produce features around 1 ppm; the real JWST features are 148–1616 ppm, and the per-bin noise alone is about 56 ppm.
3. **The labels cannot be made comparable.** All 12 gases are present at non-zero abundance in all 30 INARA rows, so any abundance threshold labels everything positive, and any detectability threshold labels everything negative.

Details below. Read-only: nothing in `outputs/stage_b/`, `outputs/labels/` or `eda/STAGE_B_RESULTS.md` was modified.

---

## Step 4: Do the two datasets match in coverage, resolution, depth scale and planet type?

### Wavelength coverage and resolution

| | INARA subset (local) | Real JWST spectra (Stage B) |
|---|---|---|
| Wavelength axis | **none — the file has no spectra** | 1–5 µm grid, 100 bins of 0.04 µm |
| Native sampling | n/a | 1 to 3,213 points inside 1–5 µm (median 100) |
| Native resolving power R = N/ln(λmax/λmin) | n/a | 23 to 5,348 (median 205) |
| Instruments | n/a | NIRCam 17, NIRISS 12, NIRSpec G395H 10, NIRSpec 9, NIRSpec PRISM 2, MIRI LRS 3 |

I could not check INARA's wavelength range and resolution against the 1–5 µm grid, because the local file has no wavelength information to check. The published INARA release does include PSG-generated spectra, but we do not have them here; obtaining them would mean a new download or generating spectra through PSG, which is an external service and a separate data-acquisition decision for you.

### Depth scale and feature amplitude

Measured from our own data, and from INARA's own geometry:

| Quantity | INARA (computed from its parameters) | Real JWST spectra (measured) |
|---|---|---|
| Transit depth (Rp/Rs)² | 53–169 ppm (median 88) | 2,916–29,641 ppm (median per planet) |
| Atmospheric scale height H | 3.8–10.5 km (median 6.1) | — |
| Implied feature amplitude (5H) | **0.45–1.73 ppm** | **148–1,616 ppm** (p95−p5 of median-subtracted depth, median 730) |
| Per-bin uncertainty | — | 25–121 ppm (median 56) |

The INARA planets' absorption features are roughly **400× smaller than the real features and about 50× below the noise floor of the real data**. A model trained on 1 ppm features would see the real spectra as pure noise, and a model trained on the real spectra would see the INARA ones as flat lines. No normalisation fixes this: dividing each spectrum by its own amplitude also divides the real signal-to-noise away, and the shapes it would then match come from different pressure/temperature regimes anyway.

### Planet type

| | INARA subset | The 13 real planets |
|---|---|---|
| Planet radius | 0.87–1.18 R⊕ (median 1.05) | sub-Neptunes to inflated hot Jupiters |
| Surface/equilibrium temperature | 165–350 K (surface) | ~270 K (K2-18 b) up to >1,000 K (WASP-39 b, WASP-17 b, HD 209458 b) |
| Surface pressure | 0.24–6.5 bar, with a defined surface | gas giants and sub-Neptunes with no surface in the observed region |
| Mean molecular weight | 30.9–37.0 | H/He dominated for most (≈2.3) |
| Host star | 5,284–5,910 K, 0.79–1.24 R☉ (Sun-like only) | F through M, including TRAPPIST-1 (M8) and K2-18 (M3) |
| Density | 5.19–6.10 g/cm³ (rocky) | mostly <2 g/cm³ |

Different planet class, different atmospheric chemistry, different mean molecular weight, and no M dwarfs — the very hosts that dominate the real dataset's negative examples. Spectral features would also sit at different pressures and temperatures, so even the band *shapes* an Earth-like model learns are not the shapes present in a hot H/He atmosphere.

## Step 5: Label mismatch

**All five target molecules do appear in INARA:** H2O, CO2, CH4, CO and SO2 are all columns (alongside O2, N2, N2O, O3, NH3, C2H6, NO2, which Stage B excludes). On the surface, overlap is complete. The problem is what the values mean.

| Gas | INARA mixing ratio (volume fraction), 30 rows | Rows with value 0 |
|---|---|---|
| H2O | 1.05e-04 – 1.35e-01 (median 3.6e-02) | 0 |
| CO2 | 1.06e-01 – 6.11e-01 (median 3.5e-01) | 0 |
| CH4 | 4.89e-03 – 1.03e-01 (median 3.7e-02) | 0 |
| CO | 1.65e-04 – 2.77e-02 (median 6.8e-03) | 0 |
| SO2 | 2.03e-04 – 3.04e-02 (median 5.3e-03) | 0 |

**Every gas is present in every model.** There are no negatives, by construction: these are models of Earth-like atmospheres, and each one contains all 12 species.

The real labels mean something different. A literature "1" is *a molecule was detected at ≥3σ in this spectrum, with this instrument, at this noise level*, and a "0" is *a paper reported a non-detection with a stated upper limit*. That is a statement about **detectability**, not abundance. Turning a mixing ratio into a comparable binary label needs the detectability calculation, not a threshold on the number:

- **Threshold on abundance** (e.g. "1 if mixing ratio > 1e-4"): all 30 rows become positive for all five molecules. A single-class training set teaches nothing.
- **Threshold on detectability** (would the band be visible above the noise?): with ~1 ppm features against ~56 ppm per-bin noise, all 30 rows become negative for all five molecules. Also single-class, and also useless.

There is no threshold in between that produces a usable mix, because the detectability gap is about a factor of 50 and the abundance floor is uniformly high. A properly comparable label would require a forward model that injects each synthetic spectrum into a real JWST noise model and runs the same significance test the papers ran — which means having the spectra in the first place (blocker 1).

## Step 6: Recommendation

**Synthetic pre-training on INARA is not viable here.** Not "hard", not "needs tuning" — the local file has no spectra, and if it had, the domain gap (rocky Earth analogues at 1 ppm vs hot giants at 730 ppm, around different stars, with all-positive labels) is larger than the signal we want the model to learn. A model pre-trained on that and tested on the 13 real planets would produce a number, and the number would be meaningless. It would also be worse than the current honest negative result, because it would look like a method.

### What the 13-planet dataset can honestly support

**Report it as a characterisation/benchmark dataset, not a trained classifier.** That is a real, citable contribution and it is already almost finished:

1. **A curated benchmark.** 53 JWST transmission spectra from 13 planets on a common 1–5 µm grid with a coverage mask, each with verified, bibcode-traced molecule labels, plus explicit unknowns and a documented band-observability rule. Nothing like this exists as a tidy, reproducible package, and `stage_b_atmospheres/stage_b_pipeline.py` rebuilds it end to end from the original `.tbl` files.
2. **The negative result itself, stated properly.** Under leave-one-planet-out validation, neither a gradient-boosted model nor a small CNN beats the majority class for any of the five molecules. That is a useful finding about dataset size: it tells the next person how much data this problem needs, and it is honest about the two H2O negatives being confounded with planet type.
3. **A dataset-difficulty analysis rather than a model.** Per molecule: how many planets carry usable labels, how the class balance falls, which molecules are unevaluable, and what the observability rule removes. Concretely: only CH4 has ≥3 planets per class; CO2 and SO2 have one negative planet each.
4. **A physically-motivated non-learned baseline, if you want a "method".** For each molecule, measure a band-vs-continuum statistic (the Δχ² of a Gaussian feature at the band centre against a flat line, using the real per-bin uncertainties) and check whether it correlates with the literature label. This is interpretable, needs no training, and degrades gracefully with 13 planets. It would be a measurement, not a classifier, so its failure mode is an honest error bar rather than an overfit score.

### If you still want synthetic pre-training later

It needs a synthetic set that matches the real domain, not INARA:

- Spectra of **hot Jupiters and sub-Neptunes** (H/He atmospheres, 500–2,000 K), not Earth analogues.
- Generated on the same 1–5 µm grid, then **degraded to each real instrument's coverage and noise** so features land at 100–1,000 ppm with realistic per-bin uncertainties.
- Labelled by **injecting noise and running the same ≥3σ test** the literature uses, so a synthetic label means what a real label means.
- Public grids of this kind exist (for example petitRADTRANS or ATMO forward-model grids, or generating a grid with a radiative-transfer code directly). Any of them is a new data-acquisition step and a decision for you, not something to start silently.

Under those conditions the approach is sound and standard. With `psg_models.csv` as it stands, it is not.
