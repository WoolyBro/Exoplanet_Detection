# TESS model and cross-mission transfer

**Date:** 2026-09-26
**Data:** TESS views built from the TOI split — train 4,061 / val 874 / test 856 (5,791 total),
star-disjoint, on top of the 9,394 Kepler views.
**Code:** `stage_a_transit_model/{cnn_lstm,baselines,cross_mission}.py` (`--mission TESS`)
**Runs:** `stage_a_transit_model/runs/{tess_bn_aug_sched, tess_baselines, cross_mission}`

---

## 1. Headline: the TESS task is far less informative than Kepler's

| model, on TESS val (874 rows) | acc (3-class) | ROC-AUC | PR-AUC | P@R90 |
|---|---|---|---|---|
| majority class | **0.6648** | 0.5000 | 0.8696 | 0.870 |
| logistic regression on views | 0.4611 | 0.5859 | 0.8900 | 0.877 |
| 11 shape features + GBM | 0.5892 | 0.6952 | 0.9225 | 0.898 |
| CNN+LSTM (`tess_bn_aug_sched`) | 0.4700 | **0.7701** | **0.9506** | **0.908** |

**Two things must be said before the PR-AUC of 0.95 is quoted anywhere.**

**(a) 87 % of TESS val is planet-like.** The no-skill PR-AUC floor is 0.870, so 0.9506 is a lift
of **+0.08**, not a strong result. On Kepler the same statistic is a lift of +0.41. The TESS
number *looks* better and is in fact five times weaker.

**(b) The CNN's 3-class accuracy (0.470) is worse than always predicting the majority class
(0.665).** Inverse-frequency class weighting makes it over-predict the rare FALSE POSITIVE
class — it answered FP 365 times on a set containing 114. It buys balanced recall at the cost
of accuracy. On the ranking task it is the best model here; on the decision task it is worse
than a constant. Both are true and both are reported.

The root cause is the TOI split's composition: **66 % CANDIDATE**, a class that means "not yet
decided". Kepler's split was ~50 % FALSE POSITIVE, a genuine negative. A classifier can learn
"is this real?" from Kepler; on TESS it is largely being asked to guess which unconfirmed
candidates will later be confirmed, which is a different and much harder question.

---

## 2. Cross-mission transfer (guide 7.2)

Every model on every mission's held-out test set. The views are mission-agnostic by
construction (2001-bin global + 201-bin local, depth-normalised), so no adaptation is needed.

| trained on | evaluated on | ROC-AUC | PR-AUC | base rate | **PR-AUC lift** | acc3 |
|---|---|---|---|---|---|---|
| Kepler | Kepler | **0.9158** | 0.9060 | 0.494 | **+0.412** | 0.730 |
| TESS | Kepler | 0.8802 | 0.8667 | 0.494 | +0.372 | 0.625 |
| Kepler | TESS | 0.7586 | 0.9439 | 0.860 | +0.084 | 0.353 |
| TESS | TESS | 0.7843 | 0.9526 | 0.860 | +0.093 | 0.501 |

**Transfer works better than expected.** The penalty for using the wrong mission's model is
only **0.026 ROC-AUC** on TESS and **0.036** on Kepler. Both models carry genuinely
mission-independent structure — which is the point of normalising the views.

**The TESS-trained model scores 0.8802 on Kepler**, which beats Kepler's own 11-feature
shape baseline (0.8800). A model that never saw a Kepler light curve is competitive with a
hand-built Kepler feature set.

**Reading the columns together is essential.** Ranked by PR-AUC, the two TESS rows (0.944,
0.953) look like the best results in the table. Ranked by lift over base rate, they are the
two worst (+0.08, +0.09) and the Kepler-on-Kepler row (+0.41) is five times better than
anything measured on TESS. The same numbers support opposite conclusions depending on whether
the base rate is shown, which is why it is in the table.

---

## 3. A prediction that was checked before training, not after

Before spending hours on the TESS model, the question was whether it would inherit Kepler's
short-period/deep-transit blind spot (Stage A §6d, Stage C §1b). That is answerable from the
training distributions alone:

| | Kepler train | TESS train |
|---|---|---|
| median transit depth | 410 ppm | **4,851 ppm** |
| FP rate among depths > 10,000 ppm | **93.4 %** (base 49.9 %) → **+43.5 pp** | **11.5 %** (base 17.2 %) → **−5.7 pp** |
| FP rate among periods < 2 d | 83.2 % → +33.3 pp | 33.3 % → +16.1 pp |
| FP rate among deep **and** short | 95.7 % → +45.8 pp | 15.7 % → **−1.5 pp** |

**TESS does not merely lack the bias — it is inverted.** In Kepler a deep transit is 93 %
likely to be a false positive, which is precisely what taught the model to reject hot Jupiters.
In TESS deep transits are *less* likely than average to be false positives, and the
deep-and-short combination sits at the base rate.

This is the mechanism behind the cross-mission numbers: the Kepler model loses 0.026 ROC-AUC on
TESS partly because it distrusts exactly the deep short-period signals TESS is full of.

---

## 4. Honest limitations

- **Selection bias in the TESS build.** 71 % yield overall, but class-dependent: FALSE POSITIVE
  55.9 % kept, CANDIDATE 70.6 %, CONFIRMED 88.7 % (+4.3 pp shift). Confirmed planets have
  followed-up ephemerides; candidates and false positives have provisional ones that drift and
  are refused by the reliability check. The surviving false positives are the ones with good
  ephemerides, which may differ systematically from those lost. Class weights do not fix this.
- **The TESS binary task is close to degenerate** (86–87 % positive). Precision at fixed recall
  is the only operating-point statistic worth quoting there, and even that starts at 0.87.
- The TESS model was trained with the configuration that won on Kepler. It was **not**
  independently swept, so it is a fair comparison but not necessarily TESS's best possible model.
- TESS test has been evaluated once, here. It is now spent for this model.

---

## 5. What this means for Stage C

The 54 priority-table planets can now be scored by a **TESS-trained** model rather than the
Kepler one, which §1b of the Stage C report showed was systematically wrong for them
(it rejected KELT-9 b, HD 189733 b, WASP-19 b and the rest of the canonical hot Jupiters).
Given §3 above, the TESS model should not repeat that failure. That re-scoring is the next
step, and it is the one that finally makes `transit_ML_probability` a real model output.

---

## 6. Reproducing

```bash
python stage_a_transit_model/cnn_lstm.py --mission TESS --batchnorm --augment --scheduler \
       --out stage_a_transit_model/runs/tess_bn_aug_sched
python stage_a_transit_model/baselines.py --train-views detection_views/tess_train \
       --val-views detection_views/tess_val --out stage_a_transit_model/runs/tess_baselines
python stage_a_transit_model/cross_mission.py
```
