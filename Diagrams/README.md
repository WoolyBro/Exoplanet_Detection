# Diagrams

Every graph and diagram made in this project, collected in one folder (copies). The originals stay where the scripts write them (`figures/`, `eda/`), because the write-ups link to those paths and the scripts regenerate them there. If a figure is regenerated, re-copy it here.

**68 diagrams.** File names start with the part of the paper they belong to.

## Paper figures (numbered; the main-text candidates)

| File | What it shows | Original | Write-up |
|---|---|---|---|
| `Paper_Fig01_planet_vs_false_positive_separation.png` | How well does the classifier separate planets from false positives? | `figures/01_planet_vs_false_positive_separation.png` | `figures/README.md` |
| `Paper_Fig02_deep_model_vs_simple_baselines.png` | Does a deep network beat simpler methods? | `figures/02_deep_model_vs_simple_baselines.png` | `figures/README.md` |
| `Paper_Fig03_architecture_ablation.png` | Which architecture parts improve detection? | `figures/03_architecture_ablation.png` | `figures/README.md` |
| `Paper_Fig04_three_class_confusion.png` | Where does it confuse the three dispositions? | `figures/04_three_class_confusion.png` | `figures/README.md` |
| `Paper_Fig05_probability_calibration.png` | Can its output be read as a probability? | `figures/05_probability_calibration.png` | `figures/README.md` |
| `Paper_Fig06_cross_mission_transfer.png` | Does a model from one telescope work on another? | `figures/06_cross_mission_transfer.png` | `figures/README.md` |
| `Paper_Fig07_false_positive_rate_by_transit_depth.png` | Why does a Kepler model distrust deep transits? | `figures/07_false_positive_rate_by_transit_depth.png` | `figures/README.md` |
| `Paper_Fig08_opposite_planet_size_bias.png` | Are the two models biased toward different planets? | `figures/08_opposite_planet_size_bias.png` | `figures/README.md` |
| `Paper_Fig09_which_false_positives_fool_the_model.png` | Which false positives does it fail to catch? | `figures/09_which_false_positives_fool_the_model.png` | `figures/README.md` |
| `Paper_Fig10_smallest_detectable_transit.png` | What is the shallowest detectable transit? | `figures/10_smallest_detectable_transit.png` | `figures/README.md` |
| `Paper_Fig11_detectable_molecules_in_jwst_spectra.png` | Which molecules are detectable in JWST spectra? | `figures/11_detectable_molecules_in_jwst_spectra.png` | `figures/README.md` |
| `Paper_Fig12_priority_ranking_sensitivity.png` | What drives the planet priority ranking? | `figures/12_priority_ranking_sensitivity.png` | `figures/README.md` |
| `Paper_Fig13_stage_a_v2_final.png` | Stage A final evaluation, one panel set | `figures/13_stage_a_v2_final.png` | `figures/README.md` |
| `Paper_Fig14_published_vetters.png` | Against published vetters, same test KOIs | `figures/14_published_vetters.png` | `figures/README.md` |
| `Paper_Fig15_k2_blind_test.png` | Does it work on a mission it never saw? (K2, pre-registered blind test) | `figures/15_k2_blind_test.png` | `figures/README.md` |
| `Paper_Fig16_dr25_simulated.png` | Kepler DR25 simulated data: planets kept, false alarms rejected | `figures/16_dr25_simulated.png` | `figures/README.md` |
| `Paper_Fig17_vetter_v3.png` | Vetter v3: does adding real false alarms and pipeline measurements fix it? | `figures/17_vetter_v3.png` | `figures/README.md` |
| `Paper_Fig18_vetter_mechanisms.png` | How does the vetter decide? | `figures/18_vetter_mechanisms.png` | `figures/README.md` |
| `Paper_Fig19_snr_balance.png` | Does removing the S/N–label link fix the survey prior? | `figures/19_snr_balance.png` | `figures/README.md` |
| `Paper_Fig20_architecture_vs_data.png` | Is the architecture the limit? | `figures/20_architecture_vs_data.png` | `figures/README.md` |

## Detection (Stage A) analyses

| File | What it shows | Original | Write-up |
|---|---|---|---|
| `Detection_kepler_cnn_lstm_results_v1.png` | First CNN+BiLSTM results on Kepler (split v1, history) | `eda/kepler_cnn_lstm_results.png` | `eda/STAGE_A_LEVEL2_RESULTS.md` |
| `Detection_kepler_val_embedding_tsne.png` | t-SNE of the network's learned features on Kepler validation | `eda/kepler_val_embedding_tsne.png` | `eda/STAGE_A_LEVEL2_RESULTS.md` |
| `Detection_injection_recovery_v1.png` | Injected transits: recovery vs depth (v1 models) | `eda/injection_recovery.png` | `eda/STAGE_A_LEVEL2_RESULTS.md` |
| `Detection_injection_recovery_v2.png` | Injected transits: recovery vs depth (final v2 ensemble) | `eda/injection_recovery_v2.png` | `eda/STAGE_A_V2_FINAL_RESULTS.md` |
| `Detection_reliability_kepler_v1.png` | Probability calibration, Kepler model (v1) | `stage_a_transit_model/runs/bn_aug_sched/calibration/reliability.png` | `eda/STAGE_A_LEVEL2_RESULTS.md` |
| `Detection_reliability_tess_v1.png` | Probability calibration, TESS model (v1) | `stage_a_transit_model/runs/tess_bn_aug_sched/calibration/reliability.png` | `eda/TESS_AND_CROSS_MISSION_RESULTS.md` |

## Data and preprocessing

| File | What it shows | Original | Write-up |
|---|---|---|---|
| `Data_preprocessing_sample_views.png` | Example global/local phase-folded views | `eda/preprocessing_sample_views.png` | `README.md` |
| `Data_kepler_test_build_sample.png` | Random sample of built views for this split (v1 build) | `eda/kepler_test_build_sample.png` | `README.md` |
| `Data_kepler_test_detection_views_err_build_sample.png` | Random sample of built views for this split (with the flux-error channel) | `eda/kepler_test_detection_views_err_build_sample.png` | `README.md` |
| `Data_kepler_train_build_sample.png` | Random sample of built views for this split (v1 build) | `eda/kepler_train_build_sample.png` | `README.md` |
| `Data_kepler_train_detection_views_err_build_sample.png` | Random sample of built views for this split (with the flux-error channel) | `eda/kepler_train_detection_views_err_build_sample.png` | `README.md` |
| `Data_kepler_val_build_sample.png` | Random sample of built views for this split (v1 build) | `eda/kepler_val_build_sample.png` | `README.md` |
| `Data_kepler_val_detection_views_err_build_sample.png` | Random sample of built views for this split (with the flux-error channel) | `eda/kepler_val_detection_views_err_build_sample.png` | `README.md` |
| `Data_tess_test_build_sample.png` | Random sample of built views for this split (v1 build) | `eda/tess_test_build_sample.png` | `README.md` |
| `Data_tess_test_detection_views_err_build_sample.png` | Random sample of built views for this split (with the flux-error channel) | `eda/tess_test_detection_views_err_build_sample.png` | `README.md` |
| `Data_tess_train_build_sample.png` | Random sample of built views for this split (v1 build) | `eda/tess_train_build_sample.png` | `README.md` |
| `Data_tess_train_detection_views_err_build_sample.png` | Random sample of built views for this split (with the flux-error channel) | `eda/tess_train_detection_views_err_build_sample.png` | `README.md` |
| `Data_tess_val_build_sample.png` | Random sample of built views for this split (v1 build) | `eda/tess_val_build_sample.png` | `README.md` |
| `Data_tess_val_detection_views_err_build_sample.png` | Random sample of built views for this split (with the flux-error channel) | `eda/tess_val_detection_views_err_build_sample.png` | `README.md` |

## Atmospheres (Stage B), retrievals and the JWST forecast

| File | What it shows | Original | Write-up |
|---|---|---|---|
| `Atmospheres_stage_b_spectra_sample.png` | Sample of real JWST transmission spectra on the model grid | `eda/stage_b_spectra_sample.png` | `eda/STAGE_B_RESULTS.md` |
| `Atmospheres_stage_b_level2_overview.png` | Band statistics and scale heights of the real spectra | `eda/stage_b_level2_overview.png` | `eda/STAGE_B_LEVEL2_RESULTS.md` |
| `Atmospheres_rocky_planet_detection_limits.png` | What JWST precision can and cannot detect on rocky planets | `eda/rocky_planet_detection_limits.png` | `eda/STAGE_B_ROCKY_BENCHMARK.md` |
| `Atmospheres_methane_round1.png` | CH4 detector on 19 new JWST planets (pre-registered round 1) | `eda/methane_extension.png` | `eda/METHANE_EXTENSION_RESULTS.md` |
| `Atmospheres_methane_round2.png` | CH4 detector, round 2: harder negatives and a second grid | `eda/methane_round2.png` | `eda/METHANE_EXTENSION_RESULTS.md` |
| `Atmospheres_methane_contamination_robustness.png` | CH4 detector trained with starspots and hazes (v3) | `eda/methane_tls_robustness.png` | `eda/METHANE_EXTENSION_RESULTS.md` |
| `Atmospheres_methane_specnet_failed.png` | SpecNet (strength-blind CNN+BiLSTM): failed its protocol | `eda/methane_specnet.png` | `eda/METHANE_EXTENSION_RESULTS.md` |
| `Atmospheres_co2_detector_34_planets.png` | CO2 detector on 34 real JWST planets, confound test, band occlusion | `eda/co2_extension.png` | `eda/CO2_EXTENSION_RESULTS.md` |
| `Atmospheres_retrieval_trappist1e_synthetic.png` | Synthetic retrievals for TRAPPIST-1 e | `eda/retrieval_trappist1e.png` | `eda/FORECAST_RESULTS.md` |
| `Atmospheres_retrieval_trappist1e_real.png` | Real TRAPPIST-1 e retrieval: the signal sits in contaminated visits | `eda/retrieval_trappist1e_real.png` | `eda/FORECAST_RESULTS.md` |
| `Atmospheres_retrieval_trappist1f.png` | TRAPPIST-1 f: no atmosphere credited | `eda/retrieval_trappist1f.png` | `eda/RETRIEVAL_HZ_RESULTS.md` |
| `Atmospheres_retrieval_lhs1140b.png` | LHS 1140 b: inconclusive; the N2 hint rests on one spectrum | `eda/retrieval_lhs1140b.png` | `eda/RETRIEVAL_HZ_RESULTS.md` |
| `Atmospheres_contamination_residuals.png` | Visit-to-visit stellar contamination across TRAPPIST-1 visits | `eda/contamination_residuals.png` | `eda/FORECAST_RESULTS.md` |
| `Atmospheres_forecast_noise_model.png` | JWST noise model calibrated on real spectra | `eda/forecast_noise_model.png` | `eda/FORECAST_RESULTS.md` |
| `Atmospheres_forecast_transits_needed.png` | Transits JWST needs to detect each gas on the HZ targets | `eda/forecast_transits_needed.png` | `eda/FORECAST_RESULTS.md` |
| `Atmospheres_forecast_contamination_regimes.png` | Forecast under different stellar-contamination regimes | `eda/forecast_contamination_regimes.png` | `eda/FORECAST_RESULTS.md` |

## Habitability factor (Stage C) and Stage D data preparation

| File | What it shows | Original | Write-up |
|---|---|---|---|
| `Habitability_cosmic_shoreline.png` | The cosmic shoreline against observed rocky planets | `eda/cosmic_shoreline.png` | `eda/STAGE_C_RESULTS.md` |
| `Habitability_retention_model_fit.png` | Bayesian retention model fitted to observed rocky planets | `eda/retention_model.png` | `eda/STAGE_C_RESULTS.md` |
| `Habitability_priority_overview.png` | The habitable-world priority ranking and its components | `eda/stage_c_priority_overview.png` | `eda/STAGE_C_RESULTS.md` |
| `Habitability_sobol_sensitivity_5inputs.png` | Sobol sensitivity of the ranking (5 inputs) and robust core | `eda/stage_c_sensitivity.png` | `eda/STAGE_C_RESULTS.md` |
| `Habitability_sobol_sensitivity_7inputs.png` | Sobol sensitivity with shoreline zero point and mass-radius relation | `eda/stage_c_sensitivity_extended.png` | `eda/STAGE_C_RESULTS.md` |
| `Habitability_hwc_catalogue_eda.png` | Habitable Worlds Catalog exploration (Stage D data prep) | `eda/hwc_habitability.png` | `eda/EDA_HABITABILITY.md` |
| `Habitability_pyatmos_gas_distributions.png` | PyATMOS atmosphere grid: gas distributions (Stage D) | `eda/pyatmos_gas_distributions.png` | `eda/EDA_HABITABILITY.md` |
| `Habitability_pyatmos_o2_co2_scatter.png` | PyATMOS grid: O2 vs CO2 (Stage D) | `eda/pyatmos_o2_co2_scatter.png` | `eda/EDA_HABITABILITY.md` |
| `Habitability_pyatmos_vs_inara_gases.png` | PyATMOS vs INARA gas coverage (Stage D) | `eda/pyatmos_vs_inara_gases.png` | `eda/EDA_HABITABILITY.md` |
| `Habitability_mixing_ratio_sums.png` | Mixing-ratio sanity check of the atmosphere grids (Stage D) | `eda/mixing_ratio_sums.png` | `eda/EDA_HABITABILITY.md` |

## Early exploration (not used in the paper)

| File | What it shows | Original | Write-up |
|---|---|---|---|
| `Early_circumbinary_fold_check.png` | Early exploration: folding a circumbinary planet (not used in the paper) | `eda/circumbinary_fold_check.png` | — |
| `Early_circumbinary_koi_scan.png` | Early exploration: circumbinary KOI scan (not used in the paper) | `eda/circumbinary_koi_scan.png` | — |
| `Early_ph1_fold_check.png` | Early exploration: PH1 (Kepler-64) fold check (not used in the paper) | `eda/ph1_fold_check.png` | — |
