# Evidence map for the paper

**Working title:** *Exoplanet detection and habitability factor using CNN and LSTM.*

One row per claim the paper can make, with its number, its figure and the file that proves it. Quote numbers from here and check them against the source file before submission.

## 1. Contributions (abstract and introduction)

1. **A leakage that star-grouped splits miss.** 301 of 1,406 Kepler test KOIs had a "signal twin" (same period and epoch on another star) in the training split. Removing them drops the reported ROC-AUC from 0.916 to 0.897. Split v2 groups by star *and* signal family.
2. **A pre-registered, single-read evaluation** of a CNN+BiLSTM vetter. It reaches Kepler 0.924 and TESS 0.815 ROC-AUC, and matches AstroNet on the same KOIs.
3. **Where light-curve vetters fail, measured:**
   - neighbouring-star contamination (centroid-only false positives rejected 49 %);
   - opposite survey priors (the Kepler model rejects real hot Jupiters);
   - not a blind search (arbitrary folds pass 94 %).
4. **Out-of-mission and simulated tests, both pre-registered.**
   - **K2, a mission the model never saw:** ROC-AUC 0.863; confirmed vs FP 0.954.
   - **Kepler's own DR25 simulated data:**
     - It keeps more injected planets than the Robovetter (0.935 vs 0.869, p = 3 × 10⁻⁸), most of all for weak, long-period signals.
     - It rejects only 76 % of systematics-born false alarms (Robovetter 99 %), because it was trained on KOIs.
   - **Prospective predictions:** 1,687 open candidates in three missions, locked by hash.
   - **How the network decides, by experiment** (`eda/VETTER_MECHANISMS_RESULTS.md`):
     - seed disagreement predicts errors (AUROC 0.76), and abstaining on half cuts errors from 16.8 % to 5.1 %;
     - occlusion shows the transit core and the secondary eclipse carry the decision;
     - the survey prior is causal through S/N, and S/N-balanced training removes two-thirds of it, at a cost (pre-registered; N2 and N4 failed).
   - **Vetter v3**, which adds real false alarms and Robovetter measurements: V3_SUMMARY_PENDING.
5. **A simulation-trained CH₄ detector** that ranks all 40 labelled real JWST planets correctly. It is robust to stellar contamination (false-CH₄ rate 4.1 % → 1.0 %).
   - A strength-blind CNN + BiLSTM (SpecNet) **failed** its pre-registered tests (0.78 on all 40 planets). The GBM stays the detector.
6. **Three habitable-zone planets, one contamination-aware method, one answer:**
   - **TRAPPIST-1 e:** the apparent atmosphere comes from contaminated visits.
   - **TRAPPIST-1 f:** ln B = −0.70.
   - **LHS 1140 b:** ln B = −0.90; its N₂ hint rests on one spectrum.
   - An H₂ envelope is disfavoured for all three. The contamination, calibrated on all 8 TRAPPIST-1 PRISM visits, enters a JWST forecast.
7. **A habitability factor with a learned component:**
   - HZ membership × rocky-and-retains-atmosphere × observability.
   - Retention is a Bayesian logistic model fitted to 20 observed JWST rocky planets: P(β < 0) = 0.99, magma-ocean exception P = 0.98, leave-one-out validated.
   - Sobol analysis gives a robust core of 7 targets.
   - For candidates, the CNN–LSTM vetter's calibrated P(real) multiplies the score; the 15-candidate shortlist is hash-locked.

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
| DR25 simulated data (registered) | completeness 0.935 vs Robovetter 0.869 (McNemar p = 3e-8; MES 7.1–10: 0.91 vs 0.82); false alarms rejected 0.760 vs 0.992; ROC-AUC 0.952 vs 0.971 | 16 | `eda/DR25_SIMULATED_RESULTS.md` |
| K2 blind test (registered, read once) | Kepler-trained 0.863 [0.840, 0.884], CONF-vs-FP 0.954; TESS-trained 0.843. Deep-transit prior replicated (median 0.72 vs 0.94, p = 2e-6). Platt map does not transfer (ECE 0.36). E1–E3, E5, E7 held; E4 (TESS half) and E6 (EVEREST ≥ K2SFF, −0.031) failed | 15 | `eda/K2_BLIND_TEST_RESULTS.md` |
| Prospective | 564 TESS + 285 Kepler + 838 K2 candidates registered (hash-checked); 1 resolved so far | — | `eda/PROSPECTIVE_PREDICTIONS.md` |
| Selective prediction | seed spread flags errors: AUROC 0.76 (Kepler val), 0.68 (K2), 0.58 (DR25); keep 50 % → error 0.168 → 0.051 | 18 | `eda/VETTER_MECHANISMS_RESULTS.md` §1 |
| Occlusion | planets: transit core −0.18 to −0.25; EBs: phase-0.5 window +0.04 | 18 | same, §2 |
| Survey prior is causal through S/N | K2 deep confirmed 0.72 → 0.95 with 5 % noise; Kepler EBs 0.02 → 0.64 at 20 % | 18 | same, §3 |
| S/N-balanced training (registered) | gap 0.22 → 0.08 (N3 held); deep median 0.84 (N2 failed, needed 0.85); K2 ROC-AUC 0.863 → 0.834 (N4 failed); the prior was partly true base-rate information | 19 | same, §3b |
| Vetter v3 (registered) | V3_ROW_PENDING | 17 | `eda/VETTER_V3_RESULTS.md` |

## 4. Atmospheres (Stage B)

| Claim | Number | Source |
|---|---|---|
| Real-spectra classifiers cannot learn | 13 labelled planets; no model beats the majority class | `eda/STAGE_B_RESULTS.md` |
| Simulation-trained CH₄ detector | v1 0.97 on 29 new planets (p = 0.0003); the confound control (gas-blind) scores 0.70 | `eda/METHANE_EXTENSION_RESULTS.md` §7 |
| Robust to contamination (v3) | stress-set false-CH₄ 4.1 % → 1.0 %; all 40 labelled planets ranked correctly (10 with CH₄) | same, §9–10 |
| Labels cross-checked | 36/43 agree with exoplanet.eu (κ 0.53); all 7 disagreements explained | same, §11 |
| H₂O / CO₂ / CO | no skill beyond the gas-blind control | `eda/STAGE_B_LEVEL3_RESULTS.md` |
| SpecNet (registered) | strength-blind CNN+BiLSTM: 0.80 (29 new), 0.78 (all 40), stress false rate 4.5 %; S1–S4 failed, S5 held | `eda/METHANE_EXTENSION_RESULTS.md` §12 |
| TRAPPIST-1 f (registered) | ln B(N₂ vs bare) −0.70; H₂ −3.1 vs bare; CH₄ < 10^−3.4; R4 held | `eda/RETRIEVAL_HZ_RESULTS.md` §1 |
| LHS 1140 b (registered) | ln B(N₂ vs bare) −0.90; N₂ vs H₂ +1.9; CO₂ < 0.2 %; R2, R3 held; NIRISS χ²/N 1.7 | same, §2 |

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
| Fitted retention (registered) | β −1.27 [−2.64, −0.12], P(β<0) 0.986; γ_magma +2.63, P 0.977; LOO log score −0.53 vs −1.72 (shoreline) vs −0.58 (base rate); A1–A3 held | §2c |
| Sensitivity (Sobol) | total index: observability weight 0.60, HZ 0.47, retention 0.04; robust core of 7; B1 failed, A4 and B2 held | §3b |
| Candidate factor | P(real) × score over 1,064 candidates; shortlist of 15 (11 above the confirmed top-10 cut); C1 held | §3c |

## 7. Limitations the paper must state

All are in the write-ups above and in `README.md` (caveats 1–8):
- test sets read once;
- not a false-alarm filter: 24 % of DR25 INV false alarms pass;
- probabilities and thresholds need recalibration per survey (K2);
- TESS label and selection bias;
- vetter, not search;
- centroid blind spot;
- forecast lower bounds and a calibration from one star;
- 40 labelled CH₄ planets;
- Stage D not modelled (no ground-truth habitability labels exist);
- the shoreline is supported only on its airless side; the fitted retention model has modest discrimination (LOO AUC 0.60);
- the S/N-balance and v3 Kepler-test results are pre-declared second reads, not unbiased estimates;
- temperate rocky-planet spectra cannot yet separate an N₂ atmosphere from bare rock.

## 8. Reproducibility

`REPRODUCE.md`, `environment-prt.yml`, `requirements-lock*.txt`, `MODEL_CARD.md`; 58 tests (`python -m pytest tests`); every pre-registered expectation and outcome in `eda/PREREGISTRATION_SCOREBOARD.md`.
