# Post-review analyses (written before any of them was run)

**Written:** 2026-10-31, after the peer review of the paper draft and before any result below existed.
**Status:** every analysis here is **post hoc**. None of them counts toward the 66 registered expectations, and the paper must say so. No new performance claim is made from a spent test set: analyses B and C reuse frozen models and saved test data only to explain an existing result.

## A. Fair architecture comparison (`stage_a_transit_model/nested_architectures.py`)

Why: the registered comparison used the validation split that selected v2, gave the alternatives v2's recipe, and covered Kepler only.

- **Data:** train + validation pooled, Kepler and TESS. Test folders are never opened.
- **Design:** 5 outer folds grouped by star and signal family, stratified by label. Inside each outer-train set, one grouped 20% hold-out chooses the configuration.
- **Equal tuning for all three networks:** learning rates 0.0003, 0.001, 0.003 at dropout 0.3, and learning rate 0.001 at dropout 0.1. The chosen configuration is repeated with one more seed and the two seeds are averaged. (Reduced from 6 configurations and 3 seeds on 2026-10-31 for run time, before any comparison between the networks was computed.)
- **Measure:** pooled out-of-fold ROC-AUC, planet-like vs false positive, with a paired bootstrap of the difference over groups.
- **Reading, fixed now:** an alternative is a material gain if it beats the CNN-BiLSTM by more than 0.01. If the 95% interval of the difference excludes zero but the gain is 0.01 or less, it is a real but small gain, and the paper says so. If the CNN-BiLSTM loses materially on either mission, the thesis wording changes.

## B. Label-shift baseline (`stage_a_transit_model/label_shift_baseline.py`)

Why: the 80-label depth correction was never compared with ordinary recalibration on the same labels.

- **Baselines on the same labeled signals:** Platt refit on the frozen probability, weighted by each depth band's share of the target survey, and a single global prior shift.
- **Reading, fixed now:** the result may be called label shift by depth only if the depth correction's median ECE beats the Platt refit by at least 0.01 in all three cases. Otherwise the paper calls it target-survey recalibration.

## C. Does noise act through S/N or by hiding shape? (`stage_a_transit_model/noise_vs_shape.py`)

Why: full-view noise also hides transit shape and secondary eclipses.

- **Interventions on the K2 deep confirmed planets, frozen Kepler ensemble:** noise of 5% of the depth (i) everywhere, (ii) inside the transit only, (iii) outside the transit and away from phase 0.5, (iv) around phase 0.5 only.
- **Reading, fixed now:** S/N is supported if (iii) alone gives at least two-thirds of the rise from (i). Shape is supported if (ii) or (iv) gives it and (iii) does not. Otherwise the paper says the two are not separated.

## D. CH4 specificity (`stage_b_atmospheres/ch4_band_mask.py`)

Why: only CO2 had a band-masking test.

- **Mask:** CH4 grid bins in 2.15-2.45 um and 3.05-3.65 um on the 41 labeled planets, same strict leave-one-planet-out rule.
- **Reading, fixed now:** specific if masking lowers planets with CH4 more than those without (one-sided Mann-Whitney, p < 0.05).

## E. Retention prior sensitivity (`stage_c_priority_fusion/retention_prior_sensitivity.py`)

Refit with beta priors N(0, 1), N(0, 1.5), N(0, 2), N(0, 3), on 20 and 23 planets. The paper reports the range whatever it is.

## F. Design baseline (`stage_c_priority_fusion/design_baseline.py`)

Why: bright hosts favor both the expert selection and the information ranking.

- **Baselines:** rank the same 1,288 planets by observability alone, without the retention model.
- **Reading, fixed now:** "independently recovers" stays only if the information ranking places the open Rocky Worlds targets more than 10 median ranks better than observability alone. Otherwise the agreement is attributed to shared observability.

## G. Retrieval injection-recovery (`stage_b_atmospheres/injection_recovery_hz.py`)

Why: "cannot separate nitrogen from bare rock" needs a recovery test.

- **Method:** add a model signal to each real spectrum, keeping its real noise and contamination, then rerun the same retrieval and credit rule.
- **Injected:** N2 with 400 ppm CO2; N2 with 1% CO2 and 0.1% CH4; a hydrogen-helium envelope with 0.1% CH4 and 0.1% H2O as a positive control. All 1 bar at the planet's equilibrium temperature, with 0.01% H2O in the two N2 cases.
- **Planets:** TRAPPIST-1 e, TRAPPIST-1 f, LHS 1140 b.

## H. Statistics fixes (`tools/post_review_stats.py`)

- Holm over every p-value the paper prints as a significance claim.
- Newcombe score intervals for the ROC-AUC values.
- Paired bootstrap of detector minus gas-blind control on the strong-feature half.
- CO2 labels against exoplanet.eu, as already done for CH4.

## Not doable by the assistant

A human audit of the molecule labels needs a person. A blind sheet is prepared, but the paper must not claim the audit until it is done.

## Second round (written 2026-10-31, before any of these was run; also post hoc)

### I. Known-truth injection into real spectra (`stage_b_atmospheres/detector_injection.py`)

Why: the labels rest on the same spectra, so the detectors have never been tested against a known truth on real noise.

- **Method:** for real spectra of planets labeled without the gas, add the difference between a model with and without the gas at log mixing ratios -5, -4, -3, -2, then score with the frozen detectors under the same leave-one-planet-out rule.
- **Reading, fixed now:** the detector is sensitive on real noise if injected spectra rank above their own uninjected versions (ROC-AUC at least 0.8) at some abundance. The recovery fraction above the registered boundary is reported at each abundance, whatever it is.

### J. Simple baselines for the molecule tests (`stage_b_atmospheres/molecule_baselines.py`)

Why: CH4 and CO2 detections may track planet temperature or size.

- **Baselines:** ROC-AUC of equilibrium temperature alone and radius alone on the same planets, and the detector ROC-AUC within temperature bands where both classes exist.
- **Reading, fixed now:** if a baseline matches the detector within 0.02, the paper says the planet property explains the ranking as well as the spectrum does.

### K. Retention model comparison (`stage_c_priority_fusion/retention_model_comparison.py`)

- **Models:** intercept only, slope only, magma only, both, under neutral priors, by marginal likelihood and leave-one-out score.
- **Sensitivity:** soft-label mapping, magma temperature threshold (1,000, 1,200, 1,500 K), and the empirical shoreline margin.

### L. Separating priority from habitability (`stage_c_priority_fusion/priority_decomposition.py`)

The 44 planets are also ranked without the observability term, so the two ideas can be read apart.

Correction, 2026-10-31: the first version of `priority_decomposition.py` summed the unweighted `c_*` columns. It now applies the 0.40 and 0.30 weights and also drops the HZ and rocky terms. The ranks and the Spearman value without observability did not change (0.845).

### M. Label evidence (`tools/label_evidence.py`)

Sentences mentioning each gas are extracted by rule from the cited papers, to make a second pass that does not rely on the first labels and to shorten the human check. It does not replace that check.

## O. Learning curve (`stage_a_transit_model/learning_curve.py`), written 2026-10-31 before any run

Why: the claim that data limits vetting rests on better inputs and false alarms. It does not show whether more examples of the same kind would help, which a reader will ask.

- **Method:** the folds, CNN-BiLSTM, tuned configuration, and two seeds of analysis A. Each training set is cut to 12.5%, 25%, and 50% of its star-and-family groups, nested within each fold. The inner validation set and the held-out fold do not change. The 100% point is the run already made for analysis A.
- **Measure:** pooled out-of-fold ROC-AUC on the same 7,990 Kepler and 4,942 TESS signals, with a paired bootstrap over groups for the gain from 50% to 100%. No test set is read.
- **Rule, set now:** the 0.01 limit of analysis A applies. A gain from 50% to 100% below 0.01 means that more examples of this kind are no longer the limit. A gain of 0.01 or more means that they still are. An interval that includes 0.01 is reported as unresolved.

## P. Locked forecast for the open Rocky Worlds targets (`stage_c_priority_fusion/rocky_worlds_predictions.py`), written 2026-10-31

Why: the retention model has been judged on planets already observed, apart from three published afterward. The six open Rocky Worlds targets have no verdict yet, so a forecast made now can only be scored by new data.

- **Forecast:** the mean and 90% interval of the fitted retention probability for each target, from the 23-planet posterior of the observing design, with no refit. The expected number of atmospheres and the probability that none of the six has one are computed per posterior draw.
- **Lock:** the SHA-256 of `outputs/post_review/rocky_worlds_predictions.csv`, printed in the paper.
- **Rule, set now:** a published 15 µm verdict counts as an atmosphere at 3σ or more against a bare rock, and as no atmosphere when a bare rock fits and a thick atmosphere is excluded at 3σ or more. Any other result stays open. When at least three targets are resolved, the mean log score is compared with the 23-planet base rate and the fixed shoreline.

