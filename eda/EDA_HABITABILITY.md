# EDA: habitability / atmosphere data (manifest Stage 2)

Exploratory analysis run before any labeling rule or preprocessing exists. Nothing here is a final rule.

- **Script:** [`eda_habitability.py`](../eda_habitability.py). Its full printed output is in `run_log.txt`.
- **Data access:** only through `data_manifest.py`. PyATMOS and INARA come from `load_stage2_catalogs()`. The Habitable World Catalogue comes from its `STAGE_2_REFERENCE_ONLY` entry.
- **Read-only:** no input, manifest or split file was modified; hashes were checked before and after.
- **Provenance:** the three sources are PyATMOS (VPL), INARA/PSG and the PHL Habitable Worlds Catalogue.

| Figure | Content |
|---|---|
| `pyatmos_gas_distributions.png` | PyATMOS gas mixing-ratio distributions (Steps 1–2) |
| `pyatmos_vs_inara_gases.png` | INARA vs PyATMOS cumulative distributions (Step 3) |
| `mixing_ratio_sums.png` | Per-row sum of gas mixing ratios (Steps 4–5) |
| `pyatmos_o2_co2_scatter.png` | O₂ vs CO₂ coloured by temperature, with illustrative rules (Steps 6–8) |
| `hwc_habitability.png` | Habitable World Catalogue habitability flags and ESI (Step 9) |

---

## 3.1 Gas mixing-ratio distributions

### Columns actually present
- **PyATMOS (108,839 rows):** gas columns are `concentration_{CH4, CO2, H2, H2O, O2}`. The file also has `flux_*` and `input_*` columns for the same five gases, plus `pressure_bar`, `temperature_kelvin` and `hash`.
- **No N₂ column exists in PyATMOS**, so N₂ could not be summarised.
- **INARA (30 rows):** 12 gas columns: H2O, CO2, O2, N2, CH4, N2O, CO, O3, SO2, NH3, C2H6, NO2.

### Step 1: PyATMOS summary statistics

| Gas | min | max | mean | median | std | distinct values |
|---|---|---|---|---|---|---|
| CH4 | 1.63e-6 | 0.13 | 0.0160 | 0.010 | 0.0179 | 19 |
| CO2 | 4.0e-4 | 0.40 | 0.0272 | 0.020 | 0.0285 | 24 |
| H2 | 6.4e-8 | 0.18 | 0.0112 | 8.2e-8 | 0.0225 | 53 |
| H2O | 0.0123 | 0.0124 | 0.0123 | 0.0123 | 5.0e-6 | 2 |
| O2 | 0.02 | 0.90 | 0.212 | 0.21 | 0.0714 | 46 |

**The gas values are the simulation grid's inputs, not simulated outcomes.**
- **CH4, CO2, H2 and O2:** `concentration_X` is identical to `input_X` in all 108,839 rows, and each takes only 19–53 distinct values.
- **H2O:** surface H2O is effectively constant at 0.0123–0.0124. Only 24,815 rows match `input_H2O`, which varies across 34 values from 0.01 to 0.75.

### Step 2: Distribution shape
- **Axis scales:** CH4, CO2 and H2 span four to six orders of magnitude, so they are plotted on a log x-axis. O2 and H2O use a linear axis.
- **Each distinct grid value gets its own bar,** because the values are discrete.
- **CH4, CO2 and H2 each have a large "near-zero / Earth-like" level plus a separate block of higher values:**
  - CH4 is at 1.63e-6 in about 28% of runs.
  - CO2 is at 4e-4 in 25.1% of runs.
  - H2 is below 1e-6 in 70.7% of runs.
- **O2 is centred on Earth's 0.21** (median 0.21; 93.4% of runs between 0.10 and 0.35), with a thin tail up to 0.90.

### Step 3: INARA vs PyATMOS (indicative only, n = 30)

Shared gases: CH4, CO2, H2O and O2. H2 exists only in PyATMOS; N2, N2O, CO, O3, SO2, NH3, C2H6 and NO2 exist only in INARA.

| Gas | PyATMOS median (range) | INARA median (range) | INARA inside PyATMOS range | median percentile of INARA within PyATMOS | KS p-value |
|---|---|---|---|---|---|
| CH4 | 0.010 (1.6e-6–0.13) | 0.037 (0.005–0.10) | 100% | 87 | 4.5e-8 |
| CO2 | 0.020 (4e-4–0.40) | 0.35 (0.11–0.61) | 67% | 100 | 1.5e-56 |
| H2O | 0.0123 (0.0123–0.0124) | 0.036 (1e-4–0.135) | 0% | 100 | 3.7e-21 |
| O2 | 0.21 (0.02–0.90) | 0.28 (0.008–0.56) | 97% | 86 | 7.1e-6 |

**The 30 INARA rows do not look like a draw from PyATMOS's distribution.**
- **CO2 is the clearest difference.** INARA's *minimum* CO2 (0.106) is above 98.7% of PyATMOS runs, and 10 of 30 INARA rows are above PyATMOS's maximum (0.40).
- **Other gases:** INARA's CH4 and O2 sit in the upper part of PyATMOS's range, and INARA's H2O varies while PyATMOS's is fixed.
- **Physical conditions differ too.** INARA spans 165–350 K and 0.24–6.45 bar. PyATMOS spans 251–332 K and 1.01–1.19 bar.
- **How to read the tests:** with 30 rows, the KS p-values and percentiles are indicative only. Still, the CO2 offset is large enough that the sample size can't explain it.
- **INARA isn't very Earth-like:** despite the `inara_earthlike_subset` folder name, mean CO2 is 0.34 and mean N2 is 0.30.

---

## 3.2 Do mixing ratios sum to ~1?

### Step 4: PyATMOS
- **The five `concentration_*` columns sum to 0.03–0.91** per run (median 0.28, 99th percentile 0.49). **All 108,839 runs (100%) deviate from 1.0 by more than 5%.** No run sums above 1.0.
- **The five `input_*` columns** (which use input H2O) sum to 0.03–0.98 (median 0.36). 108,822 runs (99.98%) deviate by more than 5%, and none exceed 1.0.
- **The rest of each atmosphere isn't recorded.** On median, 72% of each atmosphere is unaccounted for (range 8.7%–96.7%). This is presumably a background gas such as N₂, but the file doesn't say so.
- **These deviations are a missing-column issue, not corrupt rows.** Treating the five columns as a complete composition would be wrong.

### Step 5: INARA
- **The 12 gas columns sum to exactly 1.000000 in all 30 rows.** 0 rows deviate by more than 5%.
- **Mean composition:** CO2 0.343, N2 0.304, O2 0.252, H2O 0.038, CH4 0.037; everything else is below 0.01.

---

## 3.3 What a threshold rule would produce

### Step 6: O₂ vs CO₂, coloured by temperature
- **Colour choice:** temperature (251–332 K, std 9.3 K) was used rather than pressure (1.013–1.194 bar, std 0.04).
- **Grid structure:** the 108,839 runs occupy only **505 distinct (CO₂, O₂) cells**, with a median of 37 runs per cell and a maximum of 2,875. The scatter plots one point per cell, sized by run count and coloured by the cell's median temperature. Within a cell, temperature spreads by a median of 14.5 K.
- **O₂ has almost no effect on temperature; CO₂, CH₄ and pressure do.** Spearman correlations with `temperature_kelvin`:

  | Variable | Correlation |
  |---|---|
  | `pressure_bar` | +0.85 |
  | `input_CO2` | +0.63 |
  | `input_CH4` | +0.61 |
  | `input_H2O` | −0.23 |
  | `input_O2` | +0.02 |
  | `input_H2` | −0.00 |

  Visually, the low-CO₂ columns are cooler and the high-CO₂ cells warmer.

### Step 7: Candidate rules (illustrative starting points, not final)

The Earth reference point is O₂ ≈ 0.21 and CO₂ ≈ 4.2e-4.

| Rule | Runs | % of 108,839 |
|---|---|---|
| R1: O₂ ≥ 0.10 AND CO₂ ≤ 0.01 | 42,787 | 39.31% |
| R1′: same, strict CO₂ < 0.01 | 25,700 | 23.61% |
| R2: 0.15 ≤ O₂ ≤ 0.30 AND CO₂ ≤ 0.05 | 71,693 | 65.87% |
| R3: R1 AND 273 ≤ T ≤ 323 K | 42,330 | 38.89% |

- **Results depend on where grid values fall, not on smooth physics.** CO₂ jumps straight from 4e-4 (25.1% of runs) to 0.01 (16.5% of runs). Changing `≤ 0.01` to `< 0.01` moves the positive rate from 39.3% to 23.6%.
- **O₂ ≥ 0.10 does almost no filtering,** because 95.7% of runs already meet it. CO₂ decides the outcome.
- **The temperature condition barely changes anything:** R3 removes only 457 runs from R1. 75.1% of all runs are already within 273–323 K.
- **Positive rates are high (24–66%)** because the grid was designed around Earth-like O₂.

### Step 8
The figure is saved as `pyatmos_o2_co2_scatter.png`. No rule has been finalized.

---

## 3.4 Habitable World Catalogue (reference only)

### Step 9
- **Size:** 5,599 planets and 118 columns.
- **It has no atmosphere or gas columns.** It classifies real confirmed exoplanets by orbital and physical parameters. PyATMOS is a synthetic atmosphere grid, so **no row-level join between them is meaningful.**

The catalogue's own habitability columns:

| Column | Distribution |
|---|---|
| `P_HABITABLE` | 0: 5,529 (98.7%) · 1: 29 (0.5%) · 2: 41 (0.7%) |
| `P_HABZONE_OPT` | 1: 264 (4.7%) |
| `P_HABZONE_CON` | 1: 188 (3.4%) |
| `P_ESI` | 0.024–0.968, median ≈ 0.27 overall; 241 blank; 25 planets ≥ 0.8 |
| `P_TYPE_TEMP` | Hot 4,743 · Cold 358 · Warm 264 · blank 234 |

**What `P_HABITABLE` codes seem to mean (inferred):** the file doesn't document its codes, so the interpretation below is inferred from the data.
- **All 70 planets with `P_HABITABLE` 1 or 2 are in the optimistic habitable zone,** but 9 of the 29 code-1 planets are outside the conservative zone. So the flag is not the habitable-zone flag alone.
- **194 planets in the optimistic habitable zone are still coded 0.**
- **Median ESI is 0.85 for code 1, 0.70 for code 2 and 0.27 for code 0.** This fits code 1 being the stricter class.

### Step 10
**This is a reference comparison only, not a validation join.** The catalogue shows how rare habitable-flagged planets are among known planets (1.25%). It cannot confirm or refute any PyATMOS gas-based label.

---

## Implications for labeling and modeling

Evidence-based takeaways only; no rule is chosen here.

1. **A threshold rule on PyATMOS O₂/CO₂ labels the simulation's inputs, not its results.** A model trained on such labels from those same columns would just learn the rule (trivial leakage). Labels built from these columns need model inputs that exclude them, for example spectra or other derived observables.
2. **Rule outcomes depend on the grid's spacing.** CO₂ is the deciding variable, and moving its boundary across the 4e-4 → 0.01 gap swings the positive rate by 16 percentage points (39.3% vs 23.6%). Any rule should be tested for sensitivity at grid values, and the chosen boundary justified physically, not tuned to a class balance.
3. **Class balance will be much less extreme than among real planets.** Illustrative rules flag 24–66% of PyATMOS runs, while the catalogue flags 1.25% of real planets. PyATMOS is an Earth-centred design grid, not a representative population.
4. **PyATMOS gas columns can't be used as a full composition.** They cover about 28% of the atmosphere on median. Anything that assumes the ratios sum to 1 (normalisation, mean molecular weight, compositional features) needs the missing background gas made explicit first.
5. **Surface H₂O in PyATMOS carries no information.** It is fixed at 0.0123–0.0124, so it should not be a feature or rule term; `input_H2O` is the column that varies.
6. **Temperature follows from pressure, CO₂ and CH₄, not O₂.** A temperature term is almost redundant with CO₂ and pressure: in R3 it changed only 0.4 percentage points.
7. **INARA is not a like-for-like extension of PyATMOS.** It is CO₂-rich, spans far wider pressures and temperatures, uses a 12-gas composition that sums to 1, and has only 30 rows. Pooling it with PyATMOS, or validating a PyATMOS-trained rule on it, would mix different distributions. It is too small for training alone.
8. **The catalogue can support interpretation but not labels.** It has no atmospheric data, and its habitability flag is orbit- and size-based. That fits its `STAGE_2_REFERENCE_ONLY` role.
