# Methane classifier — extension to more JWST planets, and calibration (pre-registered)

**Written:** 2026-10-17, **before** any new planet was labelled or scored. This file is not edited after scoring. Deviations, if any, are listed at the end of the results write-up.

## Question

Stage B Level 3 found that a classifier trained only on simulated spectra ranks the 5 CH₄-detected planets above the 4 non-detected ones (planet-level ROC-AUC 1.00, p = 0.007). That result rests on 9 planets. Does it hold on JWST planets the project has never used?

## Test set

- **Candidates.** Every planet with a JWST *transmission* spectrum in the archive metadata that is not one of the 13 Level 1/3 planets: 48 planets.
- **Coverage.** A planet is scorable for CH₄ only if one of its spectra covers a CH₄ band on the 1–5 µm grid (`band_observable` from `stage_b_pipeline.py`, unchanged).

**Labels**, from the published JWST analysis of that planet (abstract, or the paper if the abstract is silent). Each label records its source (bibcode or arXiv ID) and a quote-level note.

| Label | Meaning |
|---|---|
| **1** | CH₄ is reported as detected: ≥ 3σ, or the authors say "detected". |
| **0** | CH₄ is reported as not detected, or ruled out, or given an upper limit. Also used for a spectrum the authors call featureless/flat with no molecule detected (column `featureless` = 1). |
| **unknown** | Not addressed, tentative (< 3σ or "hint"), or conflicting between analyses. Unknowns are not scored. |

**Label rules.** Labels are fixed before scoring: the label file's SHA-256 is recorded in `outputs/stage_b_methane_ext/protocol_hashes.json` before the scoring script runs. No label is changed after seeing a score.

## Model (frozen)

- **Training.** The Level 3 GBM and CNN, with exactly the hyperparameters in `train_synthetic_classifier.py`, trained once on all 3,000 synthetic spectra.
  - No sample is excluded, since none of the new planets anchored or lent noise to the grid.
  - No retraining, tuning or threshold choice uses any new planet.
- **Planet score.** The mean over that planet's spectra, as in Level 3.

## Metrics

| Role | Metric |
|---|---|
| **Primary** | Planet-level CH₄ ROC-AUC of the GBM on the **new** planets, with a one-sided permutation p-value (10⁵ relabellings) and a 95 % bootstrap CI over planets |
| **Secondary** | (a) the same for the CNN; (b) all planets, old plus new; (c) new planets **excluding featureless spectra**, which are easy negatives |
| **Control** | The gas-blind "feature strength" baseline on the same planets |
| **Comparison** | The majority-class Brier score |

## Calibration

Simulations put CH₄ in about a third of spectra, while the literature labels differ, so raw probabilities are mis-calibrated. Calibration is fitted only on *other* planets' labels:

- **Nested leave-one-planet-out Platt scaling.** For each planet, a logistic regression on logit(score) is fitted to every other labelled planet (old plus new), then applied to the held-out planet.
- **Reported.** Brier score and ECE (5 equal-count bins) before and after calibration, at planet level, against the majority baseline. Plus a reliability table.

## What would count as failure

The primary ROC-AUC CI including 0.5, or the control matching the model. Either is reported as such.
