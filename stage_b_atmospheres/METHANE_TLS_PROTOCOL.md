# Methane classifier vs stellar contamination and hazes (pre-registered)

**Written:** 2026-10-18, before the contaminated grids were generated or any model was trained on them. This file is not edited after the results exist.

## Why

The CH₄ classifier (Stage B Level 3) was trained on clean simulated spectra. Real M-dwarf spectra carry unocculted spots and faculae, whose stellar molecular bands (H₂O, and CH₄-like absorption in the coolest spots) can mimic or mask planetary gases. Hazes add slopes too. The classifier's one confident false positive, TOI-260 b, sits on an M dwarf. Two questions:

- **H1 (robustness of the current model).** Does the current classifier (v1, trained on the clean grid) produce false CH₄ calls when contamination and hazes are present?
- **H2 (the fix).** Does training on a grid with contamination and hazes (v3) reduce that without losing skill on real planets?

## Data (generated after this file)

| Set | Size | Seed | Flags | Use |
|---|---|---|---|---|
| **Training grid v3** | 3,000 | 31415001 | `--tls --haze` | training v3 |
| **Stress-test set** | 800 | 27182001 | `--tls --haze` | testing only, never trained on |

Both use the same generator as before. Contamination: 70 % of samples, spots (0.70–0.95 T_eff, 60 %) or faculae (T_eff + 50–500 K, 40 %), coverage U(0, 0.25), PHOENIX spectra at the anchor star's T_eff. Haze: 50 % of samples, κ₃₅₀ log-U(0.01, 30) cm²/g, slope U(−6, −2).

## Models (hyperparameters frozen: those of `train_synthetic_classifier.py`)

- **v1:** GBM and CNN trained on the original clean grid (as already reported).
- **v3:** the same, trained on training grid v3.

## Metrics

**Stress test (H1, H2)**, on the 800 held-out contaminated spectra, CH₄ label as defined by the generator (detectable):
- ROC-AUC of v1 vs v3.
- False-positive rate at P ≥ 0.5 among spectra with **no CH₄ present**: overall, and in the strongly contaminated subset (f ≥ 0.1).
- ROC-AUC within the contaminated-only subset.

**Real planets**, with the frozen labels:
- the 29 new planets (rounds 1 + 2), GBM and CNN, v1 vs v3;
- the original 9 by strict leave-one-planet-out, retraining v3 on the v3 grid minus each planet's anchored and noise-borrowing samples.

**Diagnostic** (labels unknown, reported only): scores for the contamination-dominated TRAPPIST-1 b and e and for TOI-260 b.

## Decision rule (fixed now)

- **v3 becomes the headline CH₄ model only if** it lowers the stress-test false-positive rate at P ≥ 0.5 (no-CH₄ spectra), **and** its ROC-AUC on the 29 new planets is no more than 0.03 below v1's.
- Otherwise v1 stays the headline, and v3 is reported as a robustness check.
- Either way, both are reported. The 29 planets' v1 scores have already been seen, so the real-planet comparison is a non-inferiority check, not a selection.
