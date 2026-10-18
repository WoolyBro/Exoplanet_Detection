# Evidence map for the paper

**Working title:** *Exoplanet detection and habitability factor using CNN and LSTM.*

One row per claim the paper can make, with its number, its figure and the file that proves it. Quote numbers from here and check them against the source file before submission. Rows marked *(pending)* are filled after their pre-registered read.

## 1. Contributions (abstract and introduction)

1. **A leakage that star-grouped splits miss.** 301 of 1,406 Kepler test KOIs had a "signal twin" (same period and epoch on another star) in the training split. Removing them drops the reported ROC-AUC from 0.916 to 0.897. Split v2 groups by star *and* signal family.
2. **A pre-registered, single-read evaluation** of a CNN+BiLSTM vetter. It reaches Kepler 0.924 and TESS 0.815 ROC-AUC, and matches AstroNet on the same KOIs.
3. **Where light-curve vetters fail, measured:**
   - neighbouring-star contamination (centroid-only false positives rejected 49 %);
   - opposite survey priors (the Kepler model rejects real hot Jupiters);
   - not a blind search (arbitrary folds pass 94 %).
4. **Out-of-mission and simulated tests:** a blind K2 test and the Kepler DR25 simulated-data benchmark, both pre-registered *(pending)*, plus registered prospective predictions for 849 open candidates.
5. **A simulation-trained CH₄ detector** that ranks all 40 labelled real JWST planets correctly. It is robust to stellar contamination (false-CH₄ rate 4.1 % → 1.0 %).
6. **TRAPPIST-1 e.**
   - The real JWST data favour no atmosphere once visit-to-visit stellar contamination is modelled.
   - That contamination, calibrated on all 8 TRAPPIST-1 PRISM visits, enters a JWST forecast for the habitable-zone targets.
7. **A physically grounded habitability factor:** habitable-zone membership × rocky-and-retains-atmosphere × observability. It is validated against 20 observed rocky planets, including where it fails.

## 2. Data and splits

| Claim | Number | Source |
|---|---|---|
| Views built | 9,394 Kepler + 5,791 TESS from ~120 GB photometry, streamed and discarded | `README.md`, `run_log.txt` |
| Signal-twin leakage | 301 / 1,406 test KOIs; 0.9158 → 0.8974 | `eda/STAGE_A_V2_FINAL_RESULTS.md` §1 |
| Split v2 | 0 shared stars, 0 cross-split twins | `split_signal_families.py` |
| TESS selection bias | ephemeris filter kept 55.9 % of FPs vs 88.7 % of confirmed | `eda/TESS_AND_CROSS_MISSION_RESULTS.md` |

## 3. Detection (Stage A)

| Claim | Number | Figure | Source |
|---|---|---|---|
| Kepler test ROC-AUC | **0.924** [0.909, 0.937]; CONF-vs-FP 0.971 | 01 | `runs/v2_final/final_eval_v2.json` |
| TESS test ROC-AUC | **0.815** [0.777, 0.853]; lift only +0.098 (86 % planet-like) | 01 | same |
| Beats shape features | +0.034 [0.022, 0.045] Kepler; +0.094 [0.055, 0.136] TESS | 02 | same |
| Architecture (validation) | Kepler: local view +0.016, BiLSTM +0.005, error channel +0.007. TESS: only the local view | 03 | `runs/v2_comparison.csv` |
| Calibration | ECE 0.078 → 0.023 (Kepler), 0.144 → 0.018 (TESS), Platt fitted on validation | 05 | `final_eval_v2.json` |
| Cross-mission | Kepler → TESS 0.811; TESS → Kepler 0.875 | 06 | same |
| Survey priors | Kepler train: 94 % of transits deeper than 10,000 ppm are FPs (TESS: 12 %); ρ(radius) −0.37 vs +0.35 on 41 confirmed planets | 07, 08 | `outputs/stage_c/priority_target_scores_*_v2.csv` |
| Published vetters | 922 common KOIs: ours 0.977, AstroNet 0.976, ExoNet 0.990, ExoMiner 0.996 | 14 | `outputs/stage_a_comparison/` |
| Blind spot | centroid-only FPs rejected 49 % (n = 80) vs eclipsing-binary-only 95 % | 09 | `figures/make_figures.py` (`fooled_table`) |
| Injection (bespoke) | reliable from 40 ppm; null folds pass 94 % | 10 | `outputs/injection_recovery_v2/` |
| DR25 simulated data | completeness, effectiveness, vs Robovetter *(pending)* | 16 | `eda/DR25_SIMULATED_PROTOCOL.md` |
| K2 blind test | P1 and P2, E1–E7 verdicts *(pending)* | 15 | `eda/K2_BLIND_TEST_PROTOCOL.md` |
| Prospective | 564 TESS + 285 Kepler candidates registered; 1 resolved so far | — | `eda/PROSPECTIVE_PREDICTIONS.md` |

## 4. Atmospheres (Stage B)

| Claim | Number | Source |
|---|---|---|
| Real-spectra classifiers cannot learn | 13 labelled planets; no model beats the majority class | `eda/STAGE_B_RESULTS.md` |
| Simulation-trained CH₄ detector | v1 0.97 on 29 new planets (p = 0.0003); the confound control (gas-blind) scores 0.70 | `eda/METHANE_EXTENSION_RESULTS.md` §7 |
| Robust to contamination (v3) | stress-set false-CH₄ 4.1 % → 1.0 %; all 40 labelled planets ranked correctly (10 with CH₄) | same, §9–10 |
| Labels cross-checked | 36/43 agree with exoplanet.eu (κ 0.53); all 7 disagreements explained | same, §11 |
| H₂O / CO₂ / CO | no skill beyond the gas-blind control | `eda/STAGE_B_LEVEL3_RESULTS.md` |

## 5. TRAPPIST-1 e and the JWST forecast

| Claim | Number | Source |
|---|---|---|
| Real retrieval | atmosphere vs bare rock lnB +7.4, but +0.4 on the quiet visits 1+2; the gain comes from visits 3 and 4 | `eda/FORECAST_RESULTS.md` §4d |
| Archean scenario | disfavoured: −3.6σ vs free; CO₂ < 10⁻⁵ (95 %) | same |
| Null tests | b −0.5, c +0.8, d +0.1 | same |
| Visit-varying contamination | significant in 4 of 8 TRAPPIST-1 visits (incl. e3, e4); 2–5 % of depth; ℓ = 0.42 | §4e, `outputs/forecast/contamination_residuals.json` |
| Forecast, Archean CH₄, TRAPPIST-1 e | 3.9 (no star) → 4.8 (linear) → 5.3 (visit-varying) transits | §4e |
| Combined modes | SOSS + G395 halve the CH₄ cost (TOI-700 d 262 → 132) only if the star is stable | §4e |
| Retrieval cost vs matched filter | ×2.2 (CH₄), ×1.4 (CO₂) | §4c |

## 6. Habitability factor (Stage C)

| Claim | Number | Source |
|---|---|---|
| Definition | 0.40 HZ (Kopparapu) + 0.30 P(rocky & atmosphere side) + 0.30 TSM (Kempton); Monte Carlo over uncertainties | `eda/STAGE_C_RESULTS.md` §1 |
| Shoreline validation | airless side 14/14; atmosphere side 1/6; margin AUC 0.55; TOI-561 b (strong atmosphere) 2.4 dex on the airless side | §2b |
| What drives the ranking | without HZ ρ = 0.14; without retention ρ = 0.86 | 12, §3 |
| TRAPPIST-1 e | rank 1 → 4 with retention; depends on the XUV history (P 0.03 / 0.17 / 1.00) | §2b |

## 7. Limitations the paper must state

All are in the write-ups above and in `README.md` (caveats 1–8):
- test sets read once;
- TESS label and selection bias;
- vetter, not search;
- centroid blind spot;
- forecast lower bounds and a calibration from one star;
- 40 labelled CH₄ planets;
- Stage D not modelled (no ground-truth habitability labels exist);
- the shoreline is supported only on its airless side.

## 8. Reproducibility

`REPRODUCE.md`, `environment-prt.yml`, `requirements-lock*.txt`, `MODEL_CARD.md`; 50 tests (`python -m pytest tests`).
