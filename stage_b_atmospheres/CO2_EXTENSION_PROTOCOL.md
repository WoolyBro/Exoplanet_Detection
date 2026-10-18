# CO₂ detector on real JWST spectra (pre-registered; Pillar B)

**Written:** 2026-10-19, before any CO₂ label beyond the original 13 was entered and before any CO₂ score of the v3-grid models was computed. The commit adding this file is the registration.
**Code (to be written after this file):** `stage_b_atmospheres/co2_extension.py`.

## Why

Level 3 reported "no skill beyond the gas-blind control" for CO₂ (`eda/STAGE_B_LEVEL3_RESULTS.md` §4). That test had **8 planets, 7 with CO₂ and 1 without**, so it could not show skill or its absence. Its own verdict was "one negative: uninformative".

CO₂ has the strongest band in the 1–5 µm range (4.3 µm), and the archive now has 54 planets with JWST transmission spectra covering it. A proper test is possible.

## Planets

All planets in the archive's Atmospheric Spectroscopy table (refresh of 2026-10-18) with a JWST transmission spectrum of ≥ 10 points spanning 4.20–4.45 µm: **54 planets**.

## Labels

One label per planet: **was a CO₂ absorption feature detected in its JWST transmission spectrum?** This is the property a transmission classifier can see. It is not the same as "the atmosphere contains CO₂": a flat spectrum is labelled 0 even if a high-mean-molecular-weight CO₂ atmosphere cannot be excluded.

| Label | Rule |
|---|---|
| **1** | A JWST paper reports CO₂ detected: ≥ 3σ, or Bayes-factor equivalent, or stated "detected" with a clear 4.3 µm feature |
| **0** | A JWST paper with 4.3 µm coverage reports CO₂ not detected, or an upper limit, or a spectrum that is flat or featureless at 4.3 µm |
| **unknown** (excluded) | Tentative (< 3σ, "possible", "hint"); papers that conflict; or no statement about CO₂ |

**Sources.**
- The first-author JWST paper(s) behind the archive spectra, read at the abstract and results level.
- For the 13 Level 3 planets, `outputs/labels/verified_gas_labels.csv` is the starting point, re-applied under this rule. Any change is listed with its reason.

**Freezing.** Labels go into `outputs/labels/co2_labels.csv`, each with a source and a quote or significance. They are frozen by SHA-256 in `outputs/stage_b_co2/protocol_hashes.json` **before any CO₂ score is computed** for the v3-grid models.

## Models (fixed now; no tuning on real planets)

1. **GBM-CO₂:** the v3 CH₄ GBM recipe (`train_synthetic_classifier.fit_gbm`) trained on the CO₂ column of the contamination + haze grid (`outputs/stage_b_synthetic_realistic`; 3,000 spectra, CO₂ detectable in 32 %).
2. **CNN-CO₂:** the CO₂ output of the v3 multi-label 1-D CNN (`train_synthetic_classifier.train_cnn`, the same grid).
3. **Control:** the gas-blind feature strength (`feature_strength`), as before.

**Leakage.** For the 13 grid-anchor planets, the models are retrained without the synthetic spectra anchored on that planet or borrowing its noise (strict leave-one-planet-out, as for CH₄). All other planets are scored by the full-grid models.

## Scoring

- Stage B preprocessing, unchanged.
- Only spectra with ≥ 3 bins inside 4.20–4.45 µm are scored.
- A planet's score is the mean over its eligible spectra.

## Tests (one read)

All are planet level, with a permutation p-value (10⁵ relabellings) and a bootstrap 95 % CI.

1. **Primary:** GBM-CO₂ ROC-AUC on all labelled planets.
2. **Confound test:** the same on the **strong-feature half** (feature strength above the median of the labelled planets), against the gas-blind control on that half. It is computed only if the half has ≥ 3 planets of each label; otherwise it is reported as untestable.
3. **Band occlusion, the gas-specificity test.** Re-score every eligible spectrum with the 4.20–4.45 µm bins masked (mask = 0, depth = 0). If the model reads the CO₂ band, its score must fall more for CO₂ planets than for the others (one-sided Mann–Whitney on the per-planet drop).
4. **Secondary:** CNN-CO₂ ROC-AUC.

## Expectations

| ID | Expectation |
|---|---|
| K1 | GBM-CO₂ ROC-AUC ≥ 0.85 on all labelled planets, permutation p < 0.01 |
| K2 | Strong-feature half: GBM-CO₂ ROC-AUC ≥ 0.80 and ≥ the control's + 0.10 |
| K3 | Band occlusion lowers CO₂ planets' scores more than the others' (p < 0.05) |
| K4 | CNN-CO₂ ROC-AUC ≥ 0.80 on all labelled planets |

**Failure, reported as such:**
- the K1 CI includes 0.5;
- or the GBM fails to beat the control on the strong-feature half;
- or K3 fails. Then the "skill" is not CO₂-specific.

**If K1–K3 hold,** CO₂ joins CH₄ as a validated simulation-trained detector. If not, the paper states that CO₂ detection on real spectra is not shown, and gives the reason found.
