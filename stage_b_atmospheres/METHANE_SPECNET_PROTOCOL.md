# SpecNet — a strength-blind CNN + BiLSTM for CH₄ (pre-registered; critique 4)

**Written:** 2026-10-18, before SpecNet existed. The commit adding this file is the registration.
**Code:** `stage_b_atmospheres/methane_specnet.py`

## Why

The paper is titled "... using CNN and LSTM". Its CH₄ detector, however, is a gradient-boosted model (GBM). Its earlier CNN failed the pre-registered confound test (round 2, METHANE_EXTENSION_RESULTS §7):
- On the strong-feature half of the planets it scored 0.79 (n.s.), against 0.97 for the GBM.
- It was probably reading overall feature strength ("big features → giant planet → CH₄") rather than CH₄ bands.

## The design, fixed now

**Input, strength-blind by construction.** For each spectrum:
1. Take the noise-whitened residual about its weighted mean, r = (d − μ) / σ.
2. Divide it by its own RMS. That RMS is exactly √(feature strength), the gas-blind confound (`train_synthetic_classifier.feature_strength`), so the network cannot see overall strength at all.
3. Add two more channels: the coverage mask, and the noise *pattern*, log₁₀(σ / median σ). The noise pattern carries no absolute S/N.

**Network: CNN + BiLSTM.**
- **Convolution:** Conv1d(3→32, k5) → ReLU → Conv1d(32→32, k5) → ReLU → MaxPool(2) → Conv1d(32→64, k3) → ReLU → MaxPool(2).
- **Recurrent part:** a BiLSTM (64 → 2 × 32) over the 25-step feature sequence, then the mean over the sequence.
- **Head:** Dropout 0.3 → Linear(64→32) → ReLU → Linear(32→5).
- **Output:** multi-label sigmoid for H₂O, CH₄, CO₂, SO₂ and CO.

**Training:**
- **Data:** the contamination + haze grid (`outputs/stage_b_synthetic_realistic`, the v3 grid).
- **Optimiser:** AdamW, lr 2 × 10⁻³, weight decay 10⁻³, batch 64, 40 epochs.
- **Augmentation:** noise jitter (0.3 σ) applied before the whitening.
- **Ensemble:** 5 seeds (42, 1, 2, 7, 11), averaged.
- **Real-planet LOPO:** the original 9 planets use strict leave-one-planet-out, exactly as for v3 (anchored or noise-borrowing samples excluded).

No other architecture or setting will be tried.

## Tests (one read)

All with the frozen label files, the same planet-level aggregation and permutation p-values as before:
1. the 29 planets of rounds 1–2;
2. their strong-feature half (the confound test);
3. all labelled planets, with the original 9 by LOPO and the round-3 planets;
4. the 800-spectrum stress set;
5. the Spearman correlation between the CH₄ score and feature strength over the labelled planets.

## Expectations

| # | Expectation | v3 GBM / v3 CNN today |
|---|---|---|
| S1 | ROC-AUC ≥ 0.95 on the 29 planets | 1.00 / 0.97 |
| S2 | ROC-AUC ≥ 0.90 on the strong-feature half, permutation p < 0.05 (passes the confound test) | GBM passes; v1 CNN 0.79 |
| S3 | ROC-AUC ≥ 0.95 on all labelled planets | 1.00 / 0.94 |
| S4 | False-CH₄ rate ≤ 3 % on the stress set's no-CH₄ spectra (P ≥ 0.5) | 1.0 % / 3.2 % |
| S5 | |Spearman(score, feature strength)| over labelled planets below the v3 CNN's | measured |

**Decision rule.**
- **If S1–S4 hold,** SpecNet is reported as the CH₄ detector alongside the GBM, which keeps the title's CNN/LSTM claim honest.
- **If any fails,** SpecNet is reported as a failed attempt and the GBM stays the headline.
