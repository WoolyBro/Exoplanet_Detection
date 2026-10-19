# Survey priors as label shift: a label-free correction (pre-registered)

**Written 2026-10-19, before any corrected score was computed.**
**Code:** `stage_a_transit_model/label_shift.py`. **Output:** `outputs/stage_a_mechanisms/label_shift.json`, `figures/21_label_shift.png`.

## The claim to test

`VETTER_MECHANISMS_RESULTS.md` showed that the survey prior is causal through S/N, and that removing it by retraining costs skill (§3b). The quantitative explanation is **label shift**: a classifier trained on survey A learns p_A(y | x). Transit *shapes* look the same on every survey, but the planet-to-false-positive mix at a given depth does not:
- in Kepler training, 94 % of transits deeper than 10,000 ppm are false positives;
- in TESS training, 12 % are.

If only that mix differs, the Bayes-optimal posterior on survey B follows from A's by a per-stratum prior ratio (Saerens, Latinne & Decaestecker 2002). B's mix itself can be estimated **without B's labels**, by EM on the classifier's own outputs. Then:
- a survey's prior is a predictable, two-number-per-stratum effect, not a defect of the network;
- it can be corrected on a new survey with **no labels and no retraining**.

## Cases (all predictions already read once; this is a pre-declared re-analysis, not a new model)

1. Kepler ensemble → K2 (EVEREST primary set; scores saved in `runs/k2_blind/k2_predictions.csv`).
2. Kepler ensemble → TESS v2 test (the cross-mission number, 0.811).
3. TESS ensemble → Kepler v2 test (0.875). The prior is opposite here, so the correction must work in the other direction.

## Method (fixed now)

- **Strata:** measured transit depth, < 1,000 / 1,000–3,000 / 3,000–10,000 / > 10,000 ppm (view metadata `depth_ppm`).
- **Source posterior:** each ensemble's frozen Platt-calibrated planet-like probability (maps fitted on the source's validation split).
- **Source prior per stratum:** the planet-like fraction in the source's own validation split in that stratum.
- **Label-free correction:** per stratum, EM (Saerens et al. 2002) on the target's unlabelled calibrated probabilities. It gives the target prior π_B and the corrected posteriors, starting from π_A, with tolerance 10⁻⁶. **No target label is used.**
- **Oracle (reported only as an upper bound):** the same correction using the true target fraction per stratum.
- **Evaluation:** uses the target labels:
  - ROC-AUC, planet-like vs FP;
  - ECE (10 equal-count bins);
  - for case 1, the median of K2 confirmed planets deeper than 10,000 ppm against the shallower ones.

## Expectations

| ID | Expectation |
|---|---|
| L1 | In every case, the label-free correction at least halves the ECE of the frozen Platt map |
| L2 | Case 1: the median score of K2 confirmed planets deeper than 10,000 ppm rises to ≥ 0.85, the deep-vs-shallow gap at least halves, and ROC-AUC does not fall by more than 0.005. S/N-balanced retraining (N4) cost 0.03 |
| L3 | The EM estimate of the target prior lies within 0.15 of the true fraction in ≥ 75 % of (case, stratum) cells with ≥ 20 signals |
| L4 | ROC-AUC improves in at least 2 of the 3 cases |

**If L1–L3 hold,** the paper can state the survey prior as a law: training base rates by depth set the cross-survey bias, and the bias is removed by the label-free per-stratum correction. That makes it predictable for any vetter, not specific to this network.

## Addendum (2026-10-19, written after L1–L4 were read and failed; committed before the analysis below)

**What L1–L4 showed.** The label-free EM estimate misjudged the target base rates by up to 0.5 and lowered ROC-AUC in all three cases. The *oracle* correction, with the true base rate per depth stratum, fixed calibration in every case without hurting ROC-AUC (K2 ECE 0.358 → 0.099; Kepler → TESS 0.199 → 0.068; TESS → Kepler 0.266 → 0.053). So label shift explains the prior, but the pure label-shift assumption EM needs does not hold.

**The practical question.** How many labelled target signals are needed?

**Test (fixed now).** For m = 5, 10, 20 and 50 labelled signals per depth stratum:
1. Draw them at random (200 repetitions, seed 20261019).
2. Estimate each stratum's target base rate from them; use the source prior where a stratum has fewer than m signals.
3. Apply the per-stratum correction to the remaining signals, and evaluate ECE and ROC-AUC on those.

| ID | Expectation |
|---|---|
| L5 | At m = 20 per stratum (≤ 80 labels per survey), the median ECE is within 0.03 of the oracle's in all three cases, and the median ROC-AUC is within 0.01 of the frozen map's |

This is an addendum after an observed result, and is reported as such.
