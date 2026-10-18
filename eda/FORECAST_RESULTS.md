# Gas detectability forecast — how many JWST transits to detect CO₂, CH₄ and O₃ on the habitable-zone targets

**Date:** 2026-10-17
**Code:**
- `stage_b_atmospheres/noise_model.py`: JWST noise, calibrated and validated.
- `detectability_forecast.py`: the forecast.
- `forward_model.py`: spectra.

**Outputs:** `outputs/forecast/`:
- `noise_calibration.json`, `noise_validation.csv`
- `forecast.csv` (every mode), `forecast_best_mode.csv`
- `forecast_combined.csv`, `forecast_regimes.csv`, `contamination_residuals.json`
- `forecast_summary.json`

**Figures:** `eda/forecast_noise_model.png`, `eda/forecast_transits_needed.png`, `eda/contamination_residuals.png`, `eda/forecast_contamination_regimes.png`

> **What this answers.** For the 11 transiting planets that Stage C places in the optimistic habitable zone, it estimates how many JWST transits would detect each gas at 5σ, *if* the atmosphere looked like a stated scenario. That makes "habitability" concrete in terms of the gases that would show it (CO₂, CH₄, O₃), without claiming any planet has them.
>
> **Bottom line.**
> - An Archean-like CO₂ + CH₄ atmosphere on **TRAPPIST-1 d, e, f or g** is detectable in roughly **2–16 transits**, i.e. within a few years of JWST time.
> - **LHS 1140 b** needs 14–26 transits; **TOI-700 d/e** and **K2-18 b** need 50–150.
> - **Kepler-62 f** and **Kepler-296 f** are out of reach (more than 1,000 transits).
> - **Modern-Earth ozone is out of reach for every target** (440 transits or more).
> - **Stellar contamination (§4b, §4e).**
>   - Calibrated on all 8 real PRISM transits of TRAPPIST-1, the contamination that varies from visit to visit raises the TRAPPIST-1 counts by about 10 % (up to ×1.9 for CH₄), and LHS 1140 b's CH₄ by ×1.7–4.2.
>   - Combining NIRISS SOSS with NIRSpec G395 halves the CH₄ cost for TOI-700 d/e and K2-18 b, but only if the star's contamination is stable between visits.
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

(Refreshed 2026-10-18 with the current Stage C inputs. The equilibrium temperatures moved by up to 0.5 %, which changes the counts by at most a few per cent; no conclusion changed.)

| Planet | J | Mode | Archean CH₄ | Archean CO₂ | CO₂-rich CO₂ | Modern CO₂ | Modern CH₄ | Modern O₃ (best mode) |
|---|---|---|---|---|---|---|---|---|
| TRAPPIST-1 d | 11.4 | PRISM | 2 [2–4] | 6 [4–9] | 5 [3–8] | 15 [10–23] | 64 [40–98] | 444 [282–686] |
| TRAPPIST-1 e | 11.4 | PRISM | 4 [2–6] | 9 [6–15] | 8 [5–12] | 24 [15–38] | 95 [61–148] | 565 [361–882] |
| TRAPPIST-1 f | 11.4 | PRISM | 5 [3–8] | 13 [8–20] | 10 [7–16] | 34 [22–52] | 114 [73–175] | 542 [349–831] |
| TRAPPIST-1 g | 11.4 | PRISM | 6 [4–9] | 16 [10–24] | 12 [8–18] | 42 [27–64] | 122 [79–185] | 494 [320–751] |
| LHS 1140 b | 9.6 | G395 | 14 [7–32] | 26 [12–57] | 35 [16–79] | 52 [24–116] | 291 [135–646] | 747 [346–1659] |
| LP 890-9 c | 12.3 | PRISM | 31 [9–103] | 76 [22–256] | 61 [18–205] | 197 [56–658] | 807 [231–2701] | 5498 [1575–18399] |
| K2-18 b | 9.8 | G395 | 46 [16–136] | 85 [30–248] | 126 [45–370] | 162 [57–473] | 976 [346–2855] | 3790 [1343–11092] |
| TOI-700 e | 9.5 | G395 | 64 [23–180] | 115 [42–322] | 175 [63–491] | 218 [79–611] | 1453 [526–4080] | 5829 [2111–16363] |
| TOI-700 d | 9.5 | G395 | 86 [30–247] | 145 [51–418] | 217 [77–626] | 282 [100–814] | 1877 [664–5417] | 6313 [2234–18219] |
| Kepler-296 f | 13.4 | PRISM | 1625 [381–6502] | 4491 [1054–17974] | 3312 [778–13257] | >10⁴ | >10⁴ | >10⁴ |
| Kepler-62 f | 12.3 | PRISM | 2909 [838–9599] | 8483 [2444–27992] | 5863 [1689–19345] | >10⁴ | >10⁴ | >10⁴ |

**Reading it.**

1. **TRAPPIST-1 dominates.**
   - Its four HZ planets can show an Archean-like CO₂ + CH₄ pair, the anoxic-biosphere disequilibrium signature, in **2–16 transits**: 8–71 hours of telescope time, about 2 years or less of visible transits.
   - Even modern-Earth CO₂ is within reach (15–42 transits).
   - The reason is the 0.12 R☉ host: the signal scales as R★⁻².
2. **Ozone, the oxygen biosphere's signature, is out of reach everywhere.** Its best mode is NIRSpec, through the O₃ bands near 4.7 µm, with MIRI's 9.6 µm band similar or worse (TRAPPIST-1 e: 565 vs 654 transits). A 2026-10-17 version of this table labelled the column MIRI; that was wrong.
   - TRAPPIST-1 needs 440–570 transits, which is more than all the transits observable in 20 years for e, f and g (323, 214 and 160).
   - The modern-Earth CH₄ (1.8 ppm) is just as hard.
3. **The bright-star targets are limited by systematics, not photons.**
   - With the Greene floors, LHS 1140 b, TOI-700 d/e and K2-18 b **never** reach 5σ for these N₂ atmospheres (`p_n5_floor_reachable` ≤ 0.003): their features of tens of ppm sit below a 30 ppm per-bin floor.
   - The faint TRAPPIST-1 planets stay feasible with the floor (Archean CO₂: 7.6–66 transits for d–g).
   - Whether JWST beats the Greene floors is therefore what decides LHS 1140 b's and TOI-700's prospects.
4. **K2-18 b as a hycean / H₂-rich world needs under one transit** for CH₄ and CO₂ (median 0.2 and 0.4). As an N₂ world it needs tens to hundreds of transits. The real data match the H₂-rich case (§3).
5. **Clouds at 0.1 bar** raise the counts by about 10–70 % (Archean CO₂: +44 % for TRAPPIST-1 e, +12 % for LHS 1140 b). A secondary atmosphere's features form above that level.
6. **Kepler-62 f and Kepler-296 f** are habitable-zone planets that transmission spectroscopy cannot characterise: faint hosts with R★ of 0.48–0.64 R☉. Their Stage C rank already reflects that.

## 4b. Stellar contamination (starspots and faculae), added 2026-10-18

**Method.**
- **Setup.** M-dwarf hosts (T_eff < 4000 K) carry unocculted spots and faculae, which imprint stellar spectral features on the transit depth (the transit light-source effect, Rackham et al. 2018). To first order, a covering fraction f of material at temperature T_het adds a template depth × (1 − F_het/F_phot).
- **Templates.** One for spots at 0.86 T_eff and one for faculae at T_eff + 100 K (Rackham's M-dwarf conventions), plus the free offset. Spectra are PHOENIX models, from petitRADTRANS's table.
- **What counts.** These templates are fitted out of the data. Only the component of each gas signal **orthogonal** to them can count towards detection. Because σ_N = σ₁/√N, the surviving fraction does not depend on N.
- **Scope.** Kepler-62 f's host (4925 K) is a K dwarf and is not treated.

| Target, mode | CH₄ signal kept | Archean CH₄: 5σ transits, no TLS → with TLS | CO₂ signal kept | Archean CO₂: → with TLS |
|---|---|---|---|---|
| TRAPPIST-1 d, PRISM | 0.89 | 2.4 → **3.0** | 1.00 | 6.0 → 6.1 |
| TRAPPIST-1 e, PRISM | 0.90 | 3.9 → **4.8** | 1.00 | 9.4 → 9.4 |
| TRAPPIST-1 f, PRISM | 0.90 | 5.1 → **6.2** | 1.00 | 12.8 → 12.9 |
| TRAPPIST-1 g, PRISM | 0.91 | 5.9 → **7.2** | 1.00 | 16.1 → 16.1 |
| LP 890-9 c, PRISM | 0.92 | 31 → 37 | 1.00 | 76 → 77 |
| LHS 1140 b, G395 | 0.77 | 14 → **25** | 0.83 | 26 → 37 |
| TOI-700 d, G395 | **0.56** | 86 → **274** | 0.84 | 145 → 203 |
| TOI-700 e, G395 | **0.56** | 64 → **205** | 0.84 | 115 → 164 |
| K2-18 b (N₂ case), G395 | **0.55** | 46 → **152** | 0.85 | 85 → 118 |

**Reading it.**
- **Wide wavelength coverage protects against starspots.** PRISM's 0.95–5.3 µm range lets the fit separate stellar contamination, which is broad and strongest below 2 µm, from planetary CH₄. TRAPPIST-1 keeps about 90 % of its CH₄ signal, and all of its CO₂.
  - G395 alone (2.9–5.2 µm) cannot separate them. Nearly half the CH₄ signal at 3.3 µm is degenerate with contamination, which triples the transits needed for TOI-700 d/e.
  - Adding NIRISS SOSS (0.85–2.8 µm) restores most of the leverage, but only if the contamination is stable between visits (§4e).
- **CO₂ at 4.3 µm is almost immune.** Its band shape is unlike any stellar template.
- **This is the linear case.** The contamination is a fixed pair of templates fitted alongside the gases. Real contamination changes between visits (200–300 ppm on TRAPPIST-1; Piaulet-Ghorayeb et al. 2025), and PHOENIX is interpolated on a coarse T_eff grid. The full TRAPPIST-1 e retrievals with stellar contamination (§4c) test it directly. §4e adds what the linear model misses, measured on the real TRAPPIST-1 visits.

## 4c. Full Bayesian retrievals for TRAPPIST-1 e (added 2026-10-18)

The numbers above use a known-template matched filter, which is optimistic because it assumes the right spectrum shape is known. Here they are replaced by what an observer would actually do (`stage_b_atmospheres/retrieval_trappist1e.py`):
- **Simulated observation.** The Archean atmosphere (CO₂ 1 %, CH₄ 0.1 %, H₂O 0.1 %, N₂, 1 bar, 250 K) on TRAPPIST-1 e. NIRSpec PRISM, 0.95–5.3 µm at R = 100, with the calibrated noise for N transits.
- **Retrieval.** Nested sampling (dynesty, 250 live points) over log CO₂, CH₄ and H₂O, T, surface pressure and radius.
- **Detection test.** Each gas is tested by Bayesian model comparison against the same retrieval without it, with the Bayes factor converted to sigma (Benneke & Seager 2013).
- **Data.** Each N is an expected (noise-free) dataset, so no single random draw decides the answer.
- **Opacities.** Binned to R = 300 for both the simulated data and the retrieval. That changes the gas signals by < 0.2 % relative to R = 1000.

| Transits | CH₄: ln B (σ) | CO₂: ln B (σ) |
|---|---|---|
| 2 | 1.39 (2.3σ) | 1.17 (2.1σ) |
| 4 | 4.02 (3.3σ) | 2.89 (2.9σ) |
| 8 | 9.68 (4.8σ) | 6.14 (3.9σ) |
| 16 | 22.45 (7.0σ) | 12.89 (5.4σ) |

| | Transits for 3σ | Transits for 5σ: retrieval | Transits for 5σ: matched filter (same data grid) | Retrieval factor |
|---|---|---|---|---|
| CH₄ | 3.4 | **8.7** | 3.9 | **×2.2** |
| CO₂ | 4.3 | **13.4** | 9.6 | **×1.4** |

**Reading it.**
- **A full retrieval needs 1.4–2.2× the transits of the matched filter.** That sits at the low end of the 2–3.5× inferred independently from K2-18 b (§3), so the two checks agree.
- **CO₂ suffers less than CH₄.** Its 4.3 µm band is unique; CH₄'s bands overlap H₂O's.
- **The Archean pair on TRAPPIST-1 e still needs only about 9–13 PRISM transits** without stellar contamination.

**With stellar contamination** (`--tls-linear`):
- **Setup.** Unocculted spots (f = 0.08, 2300 K; about 250 ppm, TRAPPIST-1's observed amplitude) were injected with the full non-linear model. They were then fitted with first-order spot and facula templates plus an offset, marginalised analytically, as on the real data. A fully non-linear 8-parameter retrieval was tried first and abandoned: one run took more than 3 h without converging.

| | 5σ transits, clean | 5σ transits, contamination fitted | Retrieval cost of contamination | First-order forecast (§4b) |
|---|---|---|---|---|
| CH₄ | 8.7 | **9.7** | +11 % | +23 % (CH₄ signal kept 0.90) |
| CO₂ | 13.4 | **13.0** | none | none (CO₂ signal kept 1.00) |

- **The full retrieval confirms §4b.** Stellar contamination costs PRISM a little CH₄ sensitivity and essentially no CO₂ sensitivity. The broad 0.6–5.3 µm coverage separates stellar from planetary features.
- **Remaining optimism.** The contamination is held fixed across stacked visits; the real data (§4d) show it is not.

Multiply the matched-filter counts in §4 by these factors for retrieval-grade estimates. The factors are specific to TRAPPIST-1 e, PRISM and this scenario, but the same order should apply to similar targets.

## 4d. The real TRAPPIST-1 e data (4 JWST visits, Espinoza et al. 2025)

`stage_b_atmospheres/retrieval_trappist1e_real.py` and `real_retrieval_report.py`; figure `eda/retrieval_trappist1e_real.png`.

**Data.** The four NIRSpec PRISM visits as published in the archive: 67 points each, 0.63–5.21 µm.

**Model.**
- **Planet, shared across visits:** an N₂ atmosphere with free CO₂, CH₄ and H₂O, plus temperature, surface pressure and radius; or a bare rock.
- **Per visit:** cool-spot (2200 K) and hot-facula (T★ + 150 K) PHOENIX templates plus an offset, marginalised analytically. The integral was checked against brute-force Monte Carlo (`tests/test_forecast_noise.py`).
- **Sampler:** nested sampling (dynesty).

**Empirical null tests (same star, same mode).**
- TRAPPIST-1 b and c are airless or without a thick atmosphere (JWST thermal emission); d's spectrum is flat. The method finds **no atmosphere** on any of them: ln B(atmosphere vs bare rock) = −0.5 (b), −0.3 (b, ≥ 3 µm only), +0.8 (c), +0.1 (d).
- An earlier run gave b ln B ≈ 511, from a **bug**: data bins narrower than the model's sampling got no model value, and were dropped from the atmosphere models only. It is fixed, and the likelihood now refuses to run if any data point lacks a model value. TRAPPIST-1 e and d had no such bins, which was checked.

**Results.** ln B (Benneke & Seager sigma; negative = the second model is favoured):

| Test | All visits, 0.6–5.2 µm | All visits, ≥ 3 µm only | Quiet visits 1 + 2 only |
|---|---|---|---|
| any atmosphere vs bare rock | +7.4 (4.3σ) | +6.4 (4.0σ) | **+0.4 (none)** |
| CH₄ (with vs without) | +7.8 (4.3σ) | +7.7 (4.3σ) | +1.4 (2.3σ, weak) |
| CO₂ (with vs without) | −1.0 (not needed) | −0.7 (not needed) | — |
| Archean scenario (CO₂ 1 %, CH₄ 0.1 %) vs bare rock | +2.5 | **−1.1 (disfavoured, 2.1σ)** | — |
| Archean scenario vs free atmosphere | **−4.9 (disfavoured, 3.6σ)** | **−7.5 (disfavoured, 4.3σ)** | — |

**Reading it.**
1. **No credible atmosphere detection.** The "atmosphere/CH₄" preference has three red flags:
   - It vanishes when only the two quiet visits are used (ln B +0.4).
   - Adding an atmosphere improves χ² only in visits 3 (+19) and 4 (+11), and worsens visits 1 and 2. A planetary atmosphere is the same in every transit; stellar contamination is not.
   - It needs an implausible ~5 % CH₄.

   It is unexplained, visit-specific stellar contamination: spots or faculae, or flares, which affected TRAPPIST-1 transits in every published visit. That is the same conclusion Glidden et al. (2025) draw: neither a bare rock nor an N₂ atmosphere "fully explains all features".
2. **A CO₂-rich, Archean-like atmosphere is disfavoured.** The 4.3 µm CO₂ band is where contamination matters least (§4b), and the data show no CO₂ feature (95 % upper limit log CO₂ < −5.0). With only ≥ 3 µm data, the Archean scenario is disfavoured even against a bare rock (2.1σ). That is consistent with Glidden et al. (2025), who weakly disfavour CO₂-rich atmospheres (2σ).
3. **It matches the forecast.** The forecast predicted roughly 3σ sensitivity to Archean CO₂ from 4 transits (§4c: 2.9σ at N = 4), and the real data deliver a disfavour of that order.

**Limits.**
- The atmosphere model is clear, isothermal and 1 bar; clouds or a thinner atmosphere would weaken the CO₂ constraint.
- The contamination model is first order with two fixed templates; the null tests show it does not fake atmospheres on b, c or d.
- This is a re-analysis of published, reduced spectra, not of the raw data.

## 4e. Combined instrument modes, and contamination that changes from visit to visit (added 2026-10-18)

**Code:**
- `stage_b_atmospheres/contamination_residuals.py` (calibration)
- `detectability_forecast.py` (columns `n5_vv_*`, files `forecast_combined.csv` and `forecast_regimes.csv`)

**Figures:** `eda/contamination_residuals.png`, `eda/forecast_contamination_regimes.png`

§4b fitted one fixed pair of spot and facula templates. Two things it left out are now modelled.

### What the linear model misses, measured on all 8 public PRISM transits of TRAPPIST-1

The calibration uses every PRISM visit in the research pack: b (1), c (1), d (2) and e (4).

**The model, for each visit:**
- **Planet.** A flat planet depth. None of these planets shows an atmosphere at PRISM precision.
- **Linear contamination.** Exactly the retrieval's marginalised term: spot and facula templates scaled by the depth, plus an offset.
- **Noise.** The published per-bin errors.
- **What the linear model misses.** A Gaussian process in ln λ, with amplitude A × depth. A is fitted separately in each visit; the length scale ℓ is shared by all 8 visits.

| Visit | Bins | χ²/n, linear model only | A (fraction of depth), 68 % | ΔlnL vs A = 0 | p |
|---|---|---|---|---|---|
| b 1 | 362 | 1.31 | 0.002 [0, 0.006] | +0.0 | 0.39 |
| c 1 | 362 | 1.10 | **0.024** [0.015, 0.040] | +6.9 | 1 × 10⁻⁴ |
| d 1 | 24 | 0.84 | 0.021 [0.007, 0.041] | +0.7 | 0.11 |
| d 2 | 24 | 1.30 | **0.046** [0.025, 0.084] | +3.2 | 0.006 |
| e 1 | 67 | 0.96 | 0.008 [0, 0.015] | +0.4 | 0.17 |
| e 2 | 67 | 0.93 | 0.022 [0.010, 0.038] | +1.1 | 0.07 |
| e 3 | 67 | 1.54 | **0.028** [0.019, 0.042] | +16.6 | < 10⁻⁶ |
| e 4 | 67 | 0.93 | **0.022** [0.014, 0.038] | +9.5 | 1 × 10⁻⁵ |

**What the calibration shows.**
- **The contamination left behind varies from visit to visit.** Four of the 8 visits carry significant structure that the linear model cannot absorb, at 2–5 % of the transit depth (about 110–140 ppm for e). The other four are consistent with none.
  - The two worst visits of e are visits 3 and 4. These are exactly the visits that produced the spurious "atmosphere" preference in the real-data retrieval (§4d). The calibration reaches that conclusion independently.
- **The structure is broad.**
  - The best length scale is ℓ = 0.42 in ln λ (68 % range 0.32–0.54). A second, nearly as good maximum sits at ℓ ≈ 0.15 (ΔlnL 0.7). It is used as the sensitivity case, because a shorter scale is more degenerate with gas bands.
  - A stationary kernel fits better than one shaped like the spot contrast (ΔlnL = 7.8).
  - Over all 8 visits the process improves the likelihood by 38.5.

### Two regimes, and why combining modes only helps in one of them

**Visit-varying contamination** (`n5_vv`):
- **What is fitted.** Every visit gets its own free template amplitudes, plus an unmodelled part drawn independently.
- **Where the amplitudes come from.** Each forecast visit takes its amplitude from the 8 measured values, with ℓ = 0.42.
- **How it is computed.** The expected information per visit is computed exactly, by generalised least squares with the templates marginalised. One eigendecomposition per mode keeps the 4,000 Monte Carlo draws cheap.
- **Combining modes cannot help here.** Each visit uses one mode and must be corrected on its own, so the modes' information simply adds. Splitting transits between modes can then never beat the better mode alone. The code checks this numerically for every target: the gain is exactly 1.

**Stable contamination, modes combined** (`n5_tls_joint`):
- **What is fitted.** One spot amplitude and one facula amplitude span every mode and every visit; each mode keeps its own offset.
- **How the transits are split.** Optimally between modes, on a 5 % grid.
- **Why it helps.** The wider wavelength baseline separates contamination from gas bands.

**Archean scenario, clear sky: transits for 5σ (median)**

| Target (host J) | Gas | No contamination | Linear, best single mode | **Modes combined, stable star** (split) | **Visit-varying** (ℓ = 0.42; ℓ = 0.15) |
|---|---|---|---|---|---|
| TRAPPIST-1 d (11.4) | CH₄ | 2.4 | 3.0 | 3.0 (PRISM only) | 3.3; 4.3 |
| TRAPPIST-1 e (11.4) | CH₄ | 3.9 | 4.8 | 4.8 (PRISM only) | **5.3** [3.4–8.1]; 7.7 |
| | CO₂ | 9.4 | 9.4 | 9.4 | **10.4** [6.7–16.1]; 12.1 |
| TRAPPIST-1 f (11.4) | CH₄ | 5.1 | 6.2 | 6.2 | 6.9; 11.1 |
| TRAPPIST-1 g (11.4) | CH₄ | 5.9 | 7.2 | 7.2 | 8.0; 13.6 |
| LP 890-9 c (12.3) | CH₄ | 31 | 37 | 37 | 41; 53 |
| LHS 1140 b (9.6) | CH₄ | 14 | 25 | 21 (G395 0.85 / SOSS 0.15) | **41** [22–80]; 103 (SOSS: 62) |
| | CO₂ | 26 | 37 | 34 | 39; 46 |
| K2-18 b, N₂ case (9.8) | CH₄ | 46 | 146 (SOSS) | **71** (G395 0.70 / SOSS 0.30) | 155; 182 |
| TOI-700 e (9.5) | CH₄ | 64 | 205 | **101** (0.70 / 0.30) | 205; 222 |
| TOI-700 d (9.5) | CH₄ | 86 | 262 (SOSS) | **132** (0.70 / 0.30) | 271; 281 |
| | CO₂ | 145 | 203 | 193 (0.90 / 0.10) | 204; 210 |

The "visit-varying" column takes the best single mode. Mission limits (all observable transits in 20 years): TRAPPIST-1 e 323, LHS 1140 b 80, K2-18 b 60, TOI-700 e 71, TOI-700 d 53.

**Reading it.**
- **Combining SOSS with G395 halves the CH₄ cost, if the star is stable.**
  - The gain is ×2.0 for TOI-700 d and e and K2-18 b, recovering most of what contamination took away.
  - The best split puts about 30 % of the transits on SOSS: its 1.7 and 2.3 µm CH₄ bands and its short-wavelength view of the spots break the degeneracy at 3.3 µm.
  - CO₂ gains only about 5 %, because its 4.3 µm band was never degenerate.
  - For PRISM targets there is nothing to gain: PRISM already spans 0.6–5.3 µm in each visit, and MIRI adds nothing for CH₄ or CO₂.
- **If the contamination varies between visits, as it does on TRAPPIST-1, that gain disappears.**
  - Then broad wavelength coverage must come within each visit. That is an argument for PRISM, or for simultaneous coverage, over separate SOSS and G395 visits.
- **TRAPPIST-1 conclusions survive.**
  - Unmodelled, visit-varying contamination at TRAPPIST-1's own measured level costs about 10 % more transits, or up to ×1.3 (CO₂) and ×1.9 (CH₄) for the shorter length scale.
  - An Archean CO₂ + CH₄ pair on TRAPPIST-1 e still needs only **5–8 (CH₄) and 10–12 (CO₂) transits**.
  - The penalty is small because a faint star's photon noise dominates each visit.
- **Bright hosts with deep transits are hit hardest.**
  - LHS 1140 b's CH₄ in G395 costs ×1.7, or ×4.2 at ℓ = 0.15 (41–103 transits against 80 available). At the short length scale, SOSS becomes the better mode (62).
  - Small photon noise lets the correlated contamination dominate, and contamination scales with the transit depth.
  - TOI-700 d/e barely notice, because their shallow transits (430–550 ppm) carry little contamination.
- **The calibration is pessimistic for most hosts.**
  - TRAPPIST-1 is an active M8 dwarf. LHS 1140 and TOI-700 are older, quieter M4.5 and M2 dwarfs, so the `n5_vv` columns bracket the realistic case from above and `n5_tls` from below.
  - One star's 8 visits are a small sample for the distribution of A.
- **Mission feasibility for CH₄:**
  - **TOI-700 d stays out of reach** even with combined modes and a stable star (132 against 53 available).
  - **TOI-700 e** needs 101 against 71.
  - **K2-18 b as an N₂ world** needs 71 against 60, borderline.

## 5. Limitations — what would make these numbers wrong

- **Lower bounds.** The matched filter knows the template. Full retrievals for TRAPPIST-1 e need 1.4× (CO₂) to 2.2× (CH₄) more transits (§4c). K2-18 b suggested 2–3.5×.
- **Stellar contamination** is modelled as linear templates (§4b) plus visit-varying unmodelled structure calibrated on one star's 8 visits (§4e). Hosts quieter than TRAPPIST-1 are probably better than `n5_vv`. A contamination component that is stable but unmodelled would not average down, and acts like the Greene floors of §4.
- **No cold trap.** The atmospheres are isothermal with constant mixing ratios. Real H₂O above the tropopause is far lower, so **the H₂O numbers in the output files are optimistic and are not reported here**. Lustig-Yaeger et al. 2019 find habitable-atmosphere H₂O "prohibitively difficult".
- **The scenarios are assumptions.** A thin or absent atmosphere, which JWST has found for TRAPPIST-1 b and c, has no features at all.
- **Noise calibration rests on the archive's single-transit flag** and on the published uncertainties. MIRI LRS is calibrated on only three stars.
- **Visibility.** The 27 % observable fraction and 20-year mission are round numbers, not a schedule.

## 6. Reproducing

```bash
python stage_b_atmospheres/noise_model.py                                    # project .venv, ~1 min
~/micromamba/envs/prt/bin/python stage_b_atmospheres/contamination_residuals.py   # ~10 s, the visit-varying calibration
~/micromamba/envs/prt/bin/python stage_b_atmospheres/detectability_forecast.py  # ~8-10 min idle machine, ~2 GB
```
