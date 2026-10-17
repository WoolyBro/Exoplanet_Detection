# Gas detectability forecast — how many JWST transits to detect CO₂, CH₄ and O₃ on the habitable-zone targets

**Date:** 2026-10-17
**Code:**
- `stage_b_atmospheres/noise_model.py`: JWST noise, calibrated and validated.
- `detectability_forecast.py`: the forecast.
- `forward_model.py`: spectra.

**Outputs:** `outputs/forecast/` (`noise_calibration.json`, `noise_validation.csv`, `forecast.csv` (every mode), `forecast_best_mode.csv`, `forecast_summary.json`)
**Figures:** `eda/forecast_noise_model.png`, `eda/forecast_transits_needed.png`

> **What this answers.** For the 11 transiting planets that Stage C places in the optimistic habitable zone, it estimates how many JWST transits would detect each gas at 5σ, *if* the atmosphere looked like a stated scenario. That makes "habitability" concrete in terms of the gases that would show it (CO₂, CH₄, O₃), without claiming any planet has them.
>
> **Bottom line.**
> - An Archean-like CO₂ + CH₄ atmosphere on **TRAPPIST-1 d, e, f or g** is detectable in roughly **2–16 transits**, i.e. within a few years of JWST time.
> - **LHS 1140 b** needs 15–26 transits; **TOI-700 d/e** and **K2-18 b** need 50–150.
> - **Kepler-62 f** and **Kepler-296 f** are out of reach (more than 1,000 transits).
> - **Modern-Earth ozone is out of reach for every target** (440 transits or more).
>
> These are lower bounds (§5).

---

## 1. Noise model — calibrated on what JWST actually delivered

Forecasts usually assume instrument throughputs. This one measures them.

1. **Ideal photon noise.** The star is a blackbody at T_eff, scaled to its 2MASS J magnitude. The telescope area is 25.4 m², and the out-of-transit baseline is 2 × T14. So σ_ideal = √(1.5 / N_in).
2. **Measured precision.** Every JWST transmission spectrum in the NASA Exoplanet Archive whose metadata marks it as **one transit** (`mintranmid == maxtranmid`): 117 spectra of 30 host stars. The noise depends on the star rather than the planet, so hot Jupiters calibrate as well as rocky planets.
3. **Scale factor.** η(λ) = (σ_ideal/σ_measured)², fitted per mode as the median over stars.
   - It absorbs throughput, duty cycle, systematics, real baselines, and the blackbody's error for real stellar spectra.
   - It is **not** a pure throughput. For G395 it exceeds 1, because a J-normalised blackbody underestimates M- and K-dwarf flux at 3–5 µm.
4. **Leave-one-star-out validation.** Each star's measured precision is predicted from the other stars only:

| Mode | Spectra / stars | Median log₁₀(pred/meas) | 68 % error factor |
|---|---|---|---|
| NIRSpec PRISM | 16 / 4 | −0.009 | ×1.24 |
| NIRSpec G395 | 77 / 22 | −0.001 | ×1.46 |
| NIRISS SOSS | 19 / 11 | −0.042 | ×1.48 |
| MIRI LRS | 5 / 3 | −0.160 | ×1.70 (3 stars only) |

The model is unbiased for NIRSpec and NIRISS. The forecast corrects the median bias and propagates the error factor.

## 2. Forecast method

- **Targets.** Transiting planets with P(optimistic HZ) ≥ 0.5 in the Stage C table.
- **Atmospheres.** Isothermal at the Kempton T_eq. Opaque surface at 1 bar ("clear") or an opaque cloud deck at 0.1 bar ("cloudy"). Volume mixing ratios:

| Scenario | Background | Gases |
|---|---|---|
| modern Earth | N₂ | O₂ 0.21, CO₂ 4.2×10⁻⁴, H₂O 10⁻³, CH₄ 1.8×10⁻⁶, O₃ 10⁻⁶ |
| Archean (anoxic biosphere) | N₂ | CO₂ 10⁻², CH₄ 10⁻³, H₂O 10⁻³ |
| CO₂-rich (abiotic) | N₂ | CO₂ 0.9, H₂O 10⁻⁴ |
| H₂-rich (R_p ≥ 1.8 R⊕ only) | H₂/He | CH₄ 10⁻², CO₂ 10⁻², H₂O 10⁻⁴ |

- **Signal per gas.** spectrum(all gases) − spectrum(without that gas), binned at R = 100. A free offset is removed per mode, since a retrieval absorbs it.
- **Detection.** Matched-filter S/N = √Σ(signal/σ)², with σ_N = σ₁/√N. N is solved for 5σ (and 3σ).
- **Modes.**
  - Stars with J ≥ 10.5 use NIRSpec PRISM (brighter stars saturate it, approximately).
  - Brighter stars use NIRSpec G395 and NIRISS SOSS.
  - MIRI LRS is used for every star, for O₃ at 9.6 µm.
  - Each gas is reported for its best mode.
- **Uncertainty.** 4,000 Monte Carlo draws.
  - Planet radius, mass, stellar radius and T_eq vary within their catalogue errors.
  - The signal scales as R_p³T / (M_p R★²).
  - Masses inferred from radius carry Chen & Kipping intrinsic scatter: the σ(log₁₀ R) of 0.0403 / 0.146 divided by the slopes 0.279 / 0.589, giving 0.144 / 0.248 dex in mass.
  - The noise model's validated error is included.
- **Time.** Hours = N × (3 × T14 + 1 h). "In-mission" assumes 27 % of transits are observable (JWST field of regard) over 20 years.
- **Floor case.** Per-bin systematic floors of Greene et al. 2016 (ApJ 817, 17), read from the paper: 20 ppm for NIRISS SOSS, 30 ppm for 2.5–5 µm (used for NIRSpec) and 50 ppm for MIRI LRS.

## 3. Validation

**K2-18 b, a real detection.** Madhusudhan et al. 2023 (ApJL 956, L13) detected CH₄ at 5σ and CO₂ at 3σ from one NIRISS SOSS and one NIRSpec G395H transit, and did not detect H₂O. Taking the H₂-rich scenario:

| Gas | Forecast S/N, 1 SOSS + 1 G395 | Same model signal vs. the **published** noise | Published |
|---|---|---|---|
| CH₄ | 13.4 | 9.3 | 5σ |
| CO₂ | 7.9 | 4.1 | 3σ |
| H₂O | 0.7 | 0.45 | not detected |

The ordering, and the H₂O non-detection, are reproduced. Two optimisms are visible:

- **The noise model** is about 1.4× optimistic for this star, inside its validated ×1.46.
- **The matched filter vs. a full retrieval** accounts for another ×1.4–1.9 in S/N.

Combined, real retrievals may need roughly 2–3.5× the transits forecast here.

**Published forecasts for the same targets:**
- Krissansen-Totton et al. 2018 (AJ 156, 114): about 10 NIRSpec PRISM transits detect CO₂ and constrain CH₄ for an Archean-like TRAPPIST-1 e. Here: CO₂ 9, CH₄ 4.
- Lustig-Yaeger et al. 2019 (AJ 158, 27): CO₂-bearing atmospheres in fewer than 10 transits on clear TRAPPIST-1 planets, and biogenic O₂/O₃ "extremely challenging". Here: CO₂-rich 5–12 transits for d–g, and O₃ 440–560.

## 4. Results — transits for 5σ (median [68 % range]), clear sky, best mode

| Planet | J | Mode | Archean CH₄ | Archean CO₂ | CO₂-rich CO₂ | Modern CO₂ | Modern CH₄ | Modern O₃ (best mode) |
|---|---|---|---|---|---|---|---|---|
| TRAPPIST-1 d | 11.4 | PRISM | 2 [2–4] | 6 [4–9] | 5 [3–8] | 15 [10–23] | 63 [41–98] | 442 [283–679] |
| TRAPPIST-1 e | 11.4 | PRISM | 4 [2–6] | 9 [6–14] | 8 [5–12] | 24 [15–37] | 94 [60–146] | 563 [359–869] |
| TRAPPIST-1 f | 11.4 | PRISM | 5 [3–8] | 13 [8–19] | 10 [7–15] | 33 [22–51] | 112 [73–172] | 534 [349–817] |
| TRAPPIST-1 g | 11.4 | PRISM | 6 [4–9] | 16 [11–25] | 12 [8–18] | 43 [28–66] | 124 [80–189] | 503 [323–769] |
| LHS 1140 b | 9.6 | G395 | 15 [7–32] | 26 [12–57] | 36 [16–78] | 53 [24–115] | 296 [132–641] | 760 [339–1645] |
| LP 890-9 c | 12.3 | PRISM | 30 [8–104] | 74 [21–258] | 59 [16–207] | 190 [53–663] | 780 [217–2722] | 5312 [1477–18548] |
| K2-18 b | 9.8 | G395 | 46 [16–136] | 85 [30–247] | 126 [45–370] | 162 [57–473] | 975 [346–2853] | 3795 [1345–11106] |
| TOI-700 e | 9.5 | G395 | 65 [24–178] | 115 [42–317] | 176 [65–484] | 219 [81–603] | 1459 [537–4021] | 5848 [2154–16116] |
| TOI-700 d | 9.5 | G395 | 86 [31–237] | 146 [53–401] | 218 [80–600] | 284 [104–780] | 1888 [689–5186] | 6343 [2316–17428] |
| Kepler-296 f | 13.4 | PRISM | 1598 [393–6195] | 4412 [1085–17107] | 3256 [801–12626] | >10⁴ | >10⁴ | >10⁴ |
| Kepler-62 f | 12.3 | PRISM | 2796 [844–9620] | 8156 [2460–28058] | 5636 [1700–19390] | >10⁴ | >10⁴ | >10⁴ |

**Reading it.**

1. **TRAPPIST-1 dominates.**
   - Its four HZ planets can show an Archean-like CO₂ + CH₄ pair, the anoxic-biosphere disequilibrium signature, in **2–16 transits**: 8–72 hours of telescope time, about 2 years or less of visible transits.
   - Even modern-Earth CO₂ is within reach (15–43 transits).
   - The reason is the 0.12 R☉ host: the signal scales as R★⁻².
2. **Ozone, the oxygen biosphere's signature, is out of reach everywhere.** Its best mode is NIRSpec, through the O₃ bands near 4.7 µm, with MIRI's 9.6 µm band similar or worse (TRAPPIST-1 e: 563 vs 639 transits). A 2026-10-17 version of this table labelled the column MIRI; that was wrong.
   - TRAPPIST-1 needs 440–560 MIRI transits, which is more than all the transits observable in 20 years for e, f and g (323, 214 and 160).
   - The modern-Earth CH₄ (1.8 ppm) is just as hard.
3. **The bright-star targets are limited by systematics, not photons.**
   - With the Greene floors, LHS 1140 b, TOI-700 d/e and K2-18 b **never** reach 5σ for these N₂ atmospheres (`p_n5_floor_reachable` = 0): their features of tens of ppm sit below a 30 ppm per-bin floor.
   - The faint TRAPPIST-1 planets stay feasible with the floor (Archean CO₂: 7.5–67 transits for d–g).
   - Whether JWST beats the Greene floors is therefore what decides LHS 1140 b's and TOI-700's prospects.
4. **K2-18 b as a hycean / H₂-rich world needs under one transit** for CH₄ and CO₂ (median 0.2 and 0.4). As an N₂ world it needs tens to hundreds of transits. The real data match the H₂-rich case (§3).
5. **Clouds at 0.1 bar** raise the counts by about 10–70 % (Archean CO₂: +45 % for TRAPPIST-1 e, +12 % for LHS 1140 b). A secondary atmosphere's features form above that level.
6. **Kepler-62 f and Kepler-296 f** are habitable-zone planets that transmission spectroscopy cannot characterise: faint hosts with R★ of 0.48–0.64 R☉. Their Stage C rank already reflects that.

## 4b. Stellar contamination (starspots and faculae), added 2026-10-18

**Method.**
- **Setup.** M-dwarf hosts (T_eff < 4000 K) carry unocculted spots and faculae, which imprint stellar spectral features on the transit depth (the transit light-source effect, Rackham et al. 2018). To first order, a covering fraction f of material at temperature T_het adds a template depth × (1 − F_het/F_phot).
- **Templates.** One for spots at 0.86 T_eff and one for faculae at T_eff + 100 K (Rackham's M-dwarf conventions), plus the free offset. Spectra are PHOENIX models, from petitRADTRANS's table.
- **What counts.** These templates are fitted out of the data. Only the component of each gas signal **orthogonal** to them can count towards detection. Because σ_N = σ₁/√N, the surviving fraction does not depend on N.
- **Scope.** Kepler-62 f's host (4925 K) is a K dwarf and is not treated.

| Target, mode | CH₄ signal kept | Archean CH₄: 5σ transits, no TLS → with TLS | CO₂ signal kept | Archean CO₂: → with TLS |
|---|---|---|---|---|
| TRAPPIST-1 d, PRISM | 0.89 | 2.4 → **3.0** | 1.00 | 6.0 → 6.0 |
| TRAPPIST-1 e, PRISM | 0.90 | 3.9 → **4.8** | 1.00 | 9.3 → 9.4 |
| TRAPPIST-1 f, PRISM | 0.90 | 5.0 → **6.2** | 1.00 | 12.6 → 12.7 |
| TRAPPIST-1 g, PRISM | 0.91 | 6.0 → **7.3** | 1.00 | 16.3 → 16.4 |
| LP 890-9 c, PRISM | 0.92 | 30 → 35 | 1.00 | 74 → 74 |
| LHS 1140 b, G395 | 0.77 | 15 → **25** | 0.83 | 26 → 38 |
| TOI-700 d, G395 | **0.56** | 86 → **276** | 0.84 | 146 → 204 |
| TOI-700 e, G395 | **0.56** | 65 → **206** | 0.84 | 115 → 164 |
| K2-18 b (N₂ case), G395 | **0.55** | 46 → **151** | 0.85 | 85 → 118 |

**Reading it.**
- **Wide wavelength coverage protects against starspots.** PRISM's 0.95–5.3 µm range lets the fit separate stellar contamination, which is broad and strongest below 2 µm, from planetary CH₄. TRAPPIST-1 keeps about 90 % of its CH₄ signal, and all of its CO₂.
  - G395 alone (2.9–5.2 µm) cannot separate them. Nearly half the CH₄ signal at 3.3 µm is degenerate with contamination, which triples the transits needed for TOI-700 d/e.
  - Adding NIRISS SOSS (0.85–2.8 µm) for these bright hosts would restore the leverage; that combination is not modelled here.
- **CO₂ at 4.3 µm is almost immune.** Its band shape is unlike any stellar template.
- **This is the optimistic, linear case.** The contamination is a fixed pattern fitted alongside the gases. Real contamination changes between visits (200–300 ppm on TRAPPIST-1; Piaulet-Ghorayeb et al. 2025), and PHOENIX is interpolated on a coarse T_eff grid. The full TRAPPIST-1 e retrievals with stellar contamination (§4c) test it directly.

## 5. Limitations — what would make these numbers wrong

- **Lower bounds.** The matched filter knows the template. A free retrieval needs more data; K2-18 b suggests about 2–3.5× more.
- **Stellar contamination is modelled only to first order** (§4b), as a fixed pattern. Visit-to-visit variability is not modelled, and it dominated the TRAPPIST-1 b/c JWST spectra.
- **No cold trap.** The atmospheres are isothermal with constant mixing ratios. Real H₂O above the tropopause is far lower, so **the H₂O numbers in the output files are optimistic and are not reported here**. Lustig-Yaeger et al. 2019 find habitable-atmosphere H₂O "prohibitively difficult".
- **The scenarios are assumptions.** A thin or absent atmosphere, which JWST has found for TRAPPIST-1 b and c, has no features at all.
- **Noise calibration rests on the archive's single-transit flag** and on the published uncertainties. MIRI LRS is calibrated on only three stars.
- **Visibility.** The 27 % observable fraction and 20-year mission are round numbers, not a schedule.

## 6. Reproducing

```bash
python stage_b_atmospheres/noise_model.py                                    # project .venv, ~1 min
~/micromamba/envs/prt/bin/python stage_b_atmospheres/detectability_forecast.py  # ~10 min idle machine, ~2 GB
```
