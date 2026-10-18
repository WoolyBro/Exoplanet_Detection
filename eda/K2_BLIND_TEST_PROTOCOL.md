# K2 blind test of the frozen Stage A models — pre-registered protocol

**Written:** 2026-10-18, before any K2 light curve was folded into a view and before any K2 score existed. It is committed to git before the build; the commit hash is the registration.
**Analysis code:** `tools/k2_ephemerides.py`, `tools/build_k2_views.py`, `stage_a_transit_model/evaluate_k2.py`.
**Guard:** `stage_a_transit_model/runs/k2_blind/TEST_READ.json`. `evaluate_k2.py` scores K2 once and refuses a second read.

## 1. Why

Both v2 test sets were read once on 2026-10-18 and are spent.

K2 is a third, untouched data set:
- **Same telescope, different mission.** It is Kepler's spacecraft and 30-minute cadence, but with a different pointing-noise regime, 80-day campaigns instead of 4 years, and different fields and target selection.
- **Independent dispositions.** Its labels come from the K2 literature, not from the Kepler Robovetter.
- **Never used.** No K2 light curve was used for training, model selection, calibration or thresholds. The K2 split's `split` column is ignored, because no K2 row was ever used anywhere.

## 2. What is frozen (nothing below may change after this commit)

| | Kepler-trained | TESS-trained |
|---|---|---|
| Models | `v2_kepler_dual_cnn_lstm_err` + `_s1 _s2 _s7 _s11` (5-seed mean) | `v2_tess_dual_cnn` + `_s1 _s2 _s7 _s11` |
| Inputs | views + flux-error channel | views only |
| Platt map (from its own validation set, `final_eval_v2.json`) | a = 1.0945, b = −0.9221 | a = 1.0121, b = 1.0341 |
| Recall-90 threshold (validation) | 0.5926 | 0.4793 |

The model is not retrained, recalibrated or thresholded on K2, and no hyperparameter changes.

## 3. Data

- **Rows.** `splits/k2_planets_candidates_split.csv` (mentor, read-only), one row per K2 planet or candidate, with the split's `label_harmonized` labels unchanged (REFUTED → false positive). The one microlensing planet is dropped.
- **Ephemerides.** `tools/k2_ephemerides.py`.
  - Each value comes from the default parameter set, else another published set in the archive's `k2pandc` table (snapshot `external_data/k2/k2pandc_all_2026-10-18.csv`).
  - Missing durations are estimated from P, a/R★, k and b. The estimator, checked on 1,478 catalogue durations, gives a median ratio of 0.999 and a 68 % range of 0.90–1.25.
  - 1,755 rows are complete: 571 confirmed, 241 false positives, 943 candidates.
- **Light curves.**
  - **Primary: EVEREST 2.0** (Luger et al. 2016, 2018), pixel-level-decorrelated K2 photometry. The per-cadence error is EVEREST's FRAW_ERR / FRAW and the quality flags are its own.
    - Scatter/error is 1.23 on the probe star; Kepler PDCSAP gives 1.09 (1.02–1.35). The error channel therefore means what it meant in training.
    - EVEREST is absent for C17 and C19. Signals seen only there drop out of the primary analysis.
  - **Robustness: K2SFF** (Vanderburg & Johnson 2014), with errors and quality from the official K2 light curve on the same cadences. Its scatter is about 2.7× the photometric error, so it tests sensitivity to the detrending and to an understated error channel.
- **Pipeline.** Identical to Kepler and TESS:
  - the Kepler HARD quality bitmask;
  - Savitzky-Golay detrending with every known transit masked;
  - Astronet global and local views;
  - the same ephemeris-reliability rule (rows with propagated timing error > 0.5 × duration are excluded, as for Kepler and TESS).

## 4. Populations

| Population | Definition |
|---|---|
| **Primary** | every K2 row with an EVEREST view, a reliable or unknown-reliability ephemeris, and a catalogue duration (default or other published set) |
| Secondary A | the primary population plus rows whose duration was estimated |
| TESS-model analyses | the same, minus the 187 rows on stars that are in the TESS v2 train or validation split (same star, possibly the same signal) |
| Robustness | rows with both an EVEREST and a K2SFF view (paired) |

## 5. Endpoints and statistics

Everything uses star-bootstrap 95 % intervals (2,000 resamples) and the metric code of `evaluate_v2.py`.

- **P1 (primary).** Kepler-trained ensemble, ROC-AUC, planet-like (candidate + confirmed) vs false positive.
- **P2.** Kepler-trained ensemble, ROC-AUC, confirmed vs false positive.
- **Secondary endpoints.**
  - The TESS-trained ensemble: P1 and P2, and the paired difference to the Kepler ensemble on common rows.
  - Calibration: Brier score and ECE, raw and with the frozen Platt map.
  - Precision and recall at the frozen recall-90 threshold.
  - ROC-AUC by pipeline S/N bin (< 7.1, 7.1–15, 15–50, > 50), as for Kepler and TESS.
  - **Depth prior:** the median score of CONFIRMED planets deeper vs shallower than 10,000 ppm, for each model.
  - The paired EVEREST − K2SFF ROC-AUC difference.
  - Secondary population A.
- **Selection.** Yields by class at every stage (ephemeris, light curve available, reliable ephemeris, view built) are reported, because a class-dependent yield biases every metric (as on TESS).

## 6. Expectations, registered before any K2 view existed

They rest only on earlier, already-public results: Kepler test 0.924, Kepler → TESS 0.811, CONF-vs-FP 0.971 (Kepler) and 0.900 (TESS), and the depth priors of `figures/07`.

| # | Expectation | Reasoning |
|---|---|---|
| E1 | P1 (Kepler ensemble) in **[0.80, 0.92]**, point guess 0.86 | Same telescope and cadence help. Shorter baselines (fewer transits, lower S/N), different noise and an independent labelling hurt. |
| E2 | P2 (confirmed vs FP) in **[0.88, 0.97]**, guess 0.93 | Candidates are the weak label everywhere. |
| E3 | Kepler ensemble > TESS ensemble on K2 | K2 shares Kepler's instrument and cadence. |
| E4 | The Kepler ensemble under-scores deep confirmed planets: median score of CONFIRMED deeper than 10,000 ppm < that of shallower ones; the TESS ensemble does not | Kepler's training prior (94 % of transits deeper than 10,000 ppm are FPs). |
| E5 | ROC-AUC rises with S/N | As on Kepler and TESS. |
| E6 | EVEREST ≥ K2SFF for the Kepler ensemble | K2SFF's error channel understates its scatter about threefold. |
| E7 | The frozen Platt map does **not** transfer: ECE after Platt > 0.05 | K2 is about 86 % planet-like, against 49 % for Kepler validation. |

**Failure criterion.** If P1 < 0.75, the conclusion will be that the Kepler-trained vetter does not transfer to K2.

## 7. Rules after the read

- Results are reported whatever they are.
- Any model change made after seeing K2 is labelled exploratory and is not evaluated on K2 as if it were blind.
- A bug found after the read is fixed, and both the original and corrected numbers are reported.
