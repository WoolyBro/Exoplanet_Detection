# Methane classifier — round 2 (pre-registered addendum)

**Written:** 2026-10-17, after round 1 was scored, but **before any of the planets below was scored.** Round 1 scored only planets with a known label, and its script never computed a score for an "unknown" planet. This file is not edited after scoring.

## Why a round 2

The round-1 negatives were mostly small, flat-spectrum planets, which are easy for any classifier. The decisive test of *gas identification* rather than *feature strength* is a planet with strong absorption features but no methane. Most such planets were left "unknown" in round 1 because their abstracts do not mention CH₄. Their full papers often do.

## Which planets

The 21 round-1 "unknown" planets whose status was **not addressed in the abstract** and that have a spectrum on the 1–5 µm grid:

TOI-270 b, TOI-776 b, TRAPPIST-1 b, TRAPPIST-1 e, TrES-4 b, WASP-121 b, WASP-15 b, WASP-94 A b, NGTS-2 b, GJ 486 b, HAT-P-14 b, HAT-P-30 b, HIP 67522 b, KELT-7 b, L 98-59 b, L 98-59 d, L 168-9 b, LHS 1140 c, GJ 9827 d, WASP-96 b, WASP-127 b.

**Not reopened.**
- The four labelled unknown because the evidence was *tentative* (GJ 1214 b, TOI-1130 b, Kepler-51 d, LTT 9779 b). The evidence threshold does not change.
- K2-22 b and WASP-43 b, which have no data on the grid.

## Labels (same rules as round 1, applied to the full text)

| Label | Meaning |
|---|---|
| **1** | The paper reports CH₄ at ≥ 3σ, or as "detected". |
| **0** | The paper explicitly reports CH₄ as not detected, as an upper limit, as not required by the retrieval, or as absent/depleted. |
| **unknown** | CH₄ is still not mentioned, or the result is tentative. Silence is never read as absence. |

Each row cites the paper and section. The file `outputs/labels/ch4_labels_round2.csv` is frozen by SHA-256 before scoring.

## Analysis (fixed now)

- **Model.** The same frozen Level 3 GBM/CNN, trained on the original synthetic grid.
- **Primary.** GBM planet-level ROC-AUC on **all new planets with a label (round 1 + round 2)**, with a permutation p-value and a bootstrap CI.
- **Key secondary, the confound test.** The same metric restricted to planets whose spectra have **strong features** (gas-blind feature strength above the median of the labelled new planets), compared with the gas-blind control on that subset.
- **Robustness.** The whole round-1 and round-2 evaluation is repeated with the model retrained on a **second, independently generated synthetic grid** (seed 7770001, same generator). Both are reported.

## What would count as failure

- The primary CI including 0.5.
- On the strong-feature subset, the model failing to beat the gas-blind control.
- A large drop (more than 0.1 in ROC-AUC) between the two independent grids.
