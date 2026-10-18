# Closing the loop on the survey prior: S/N-balanced training (critique 2, pre-registered)

**Written and committed before any model of this protocol is trained.**
**Code:** `stage_a_transit_model/vetter_v3.py --config N`; evaluation `stage_a_transit_model/evaluate_snr_balance.py`.

## Why

`VETTER_MECHANISMS_RESULTS.md` §3 found, by intervention, that the Kepler vetter's survey prior ("deep and clean = eclipsing binary") acts through signal-to-noise. Adding noise at 5 % of the depth raised deep K2 confirmed planets from 0.72 to 0.95.

An explanation is stronger if it predicts a fix. If the prior is learned from the label distribution of the training set, then removing the dependence of the label on S/N from that distribution should remove most of the prior. This should cost little in-survey skill.

**The training data.** In the Kepler v2 training split, the planet-like fraction is:
- 4.7 % in the top S/N decile (S/N > 1,390);
- 27 % in the ninth decile;
- 50–65 % in the middle deciles.

## The single fix (model N)

**Unchanged from v2:**
- the network: dual CNN + BiLSTM + error channel, 40,451 parameters;
- the recipe and hyper-parameters (`vetter_v3.HP`);
- the training and validation data (Kepler v2 train/val);
- the seeds (42, 1, 2, 7, 11);
- the model selection (best validation ROC-AUC epoch).

**The only change: per-example loss weights.**
- **Bins.** The training set is cut into 10 equal-count bins of log₁₀(depth S/N), using the view metadata `depth_snr`.
- **Weight.** An example of binary class c (planet-like or false positive) in bin b gets w = P(c) / P(c | b), clipped to [0.2, 10] and normalised to mean 1.
- **Effect.** In the weighted training distribution, S/N carries (almost) no information about the label.
- **Loss.** The 3-class cross-entropy, with v2's inverse-frequency class weights, is multiplied by w.

No other configuration is trained. No hyper-parameter is tuned for N.

## Evaluation

**Comparator:** the frozen v2 five-seed Kepler ensemble. Scores are the five-seed mean planet-like probability (CANDIDATE + CONFIRMED).

**Validation** (Kepler v2 validation, already used for model selection):
- ensemble ROC-AUC (planet-like vs FP);
- the fraction of high-S/N (> 100) false positives rejected at each model's own validation recall-90 threshold.

**K2, a pre-declared second read.**
- **What was read before.** The K2 EVEREST views were read once, for the v2 blind test (`K2_BLIND_TEST_RESULTS.md`, guard `runs/k2_blind/TEST_READ.json`).
- **This read.** They are read a second time, only for model N, only for the quantities below. The guard file is `runs/snr_balance/K2_READ.json`.
- **This is not an unbiased performance estimate.** The fix was motivated by the K2 finding. No parameter of N is fitted to K2, but the deep-planet subset was chosen *because* v2 failed on it. The read tests the mechanism, not the model's general skill.

**Descriptive only:** TESS light curves of the 41 Stage C targets, not a test set.

## Expectations (fixed now)

| ID | Expectation | v2 value |
|---|---|---|
| N1 | Validation ensemble ROC-AUC ≥ v2's − 0.01 (≥ 0.922): the fix costs little in-survey skill | 0.932 |
| N2 | Median N score of K2 confirmed planets deeper than 10,000 ppm ≥ 0.85 | 0.72 (n = 27) |
| N3 | The K2 confirmed deep-vs-shallow median gap (shallow = < 10,000 ppm) shrinks by at least half (≤ 0.11) | 0.22 |
| N4 | K2 P1 ROC-AUC (planet-like vs FP, all K2 dispositions) ≥ v2's − 0.01 (≥ 0.853): deep planets are not bought by losing false-positive rejection | 0.863 |
| N5 | High-S/N validation false positives rejected at the recall-90 threshold: N ≥ v2 − 0.05 | measured |

**Interpretation, fixed now.**
- **N2 and N3 hold:** the prior was learned from the S/N–label dependence of the training set, and balancing removes it. This is the mechanism claim, confirmed by a fix.
- **N2 fails while N1 holds:** S/N is not the whole story. The views also carry shape cues correlated with depth (for example, the ingress fraction of large planets), and the claim in `VETTER_MECHANISMS_RESULTS.md` is narrowed accordingly.
- **N1 or N4 fails:** the fix trades in-survey skill for transfer, and is reported as a trade-off, not a fix.
