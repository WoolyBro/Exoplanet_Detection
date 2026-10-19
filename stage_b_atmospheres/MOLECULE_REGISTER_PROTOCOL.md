# A prospective molecule register: frozen, validated detectors on every unresolved JWST planet (pre-registered)

**Written 2026-10-19, before any score below was computed.** The commit adding this file is the registration.
**Code (written after this file):** `stage_b_atmospheres/molecule_register.py`.
**Output:** `outputs/stage_b_register/molecule_register.csv`, hash-locked in `eda/MOLECULE_REGISTER.md`.

## Why

The CH₄ and CO₂ detectors were validated on real JWST spectra:
- **CH₄:** 41 labelled planets, all ranked correctly by the v3 GBM;
- **CO₂:** 34 labelled planets (GBM 0.982, CNN 0.996), passing the confound test and with band occlusion showing gas specificity.

Every JWST transmission planet whose CH₄ or CO₂ status the literature leaves unresolved now gets an independent call from these frozen detectors. The calls are locked by hash and checked against future publications, as the 1,687 transit candidates were. Their results do not depend on the retrieval choices that make several of these planets contested.

## Planets (all of them, fixed now)

- **CH₄ register:** the 22 archive JWST transmission planets whose CH₄ label is "unknown" in the frozen label files (`verified_gas_labels.csv`, `ch4_labels_extended.csv`, `_round2.csv`, `_round3.csv`). They are GJ 1214 b, GJ 486 b, HAT-P-18 b, HAT-P-26 b, HAT-P-30 b, K2-22 b, KELT-7 b, Kepler-51 d, L 98-59 b, LTT 9779 b, TOI-1130 b, TOI-3235 b, TRAPPIST-1 b, TRAPPIST-1 e, WASP-121 b, WASP-127 b, WASP-15 b, WASP-17 b, WASP-43 b, WASP-52 b, WASP-94 A b and WASP-96 b. A planet is scored only if it has a spectrum with ≥ 3 grid bins in a CH₄ band (the Stage B rule).
- **CO₂ register:** the 17 planets labelled "unknown" in `co2_labels.csv`. They are GJ 1214 b, GJ 3090 b, HAT-P-30 b, KELT-7 b, NGTS-2 b, HATS-6 b, TOI-3235 b, K2-18 b, TOI-270 d, LHS 1140 b, WASP-127 b, GJ 486 b, L 98-59 b, L 98-59 d, TRAPPIST-1 b, TRAPPIST-1 c and TOI-270 b. A planet is scored only if it has a spectrum with ≥ 3 grid bins in 4.20–4.45 µm (the CO₂ protocol rule).

## Models (frozen; nothing is retrained except the leave-one-planet-out folds)

- **CH₄:** the v3 GBM (the headline) and the v3 1-D CNN, trained on the contamination + haze grid.
- **CO₂:** the GBM-CO₂ and the CNN-CO₂ of `co2_extension.py`, same grid.
- **Grid-anchor planets** (HAT-P-18 b, HAT-P-26 b, WASP-17 b, WASP-52 b, K2-18 b, TOI-270 d, TRAPPIST-1 c) are scored by models retrained without their anchored and noise-borrowing samples (strict leave-one-planet-out), as in every earlier test.
- **Planet score:** the mean over the planet's eligible spectra. The spread across spectra is reported.

## Calibration and the decision rule (fixed now)

**Calibration.** For each gas and model, a Platt map (L2-regularised logistic regression on logit(score), C = 1) is fitted on the labelled planets' scores:
- **CH₄:** the 41 labelled planets (the original 9 by leave-one-planet-out, rounds 1–2, round 3, round 4);
- **CO₂:** the 34 scored labelled planets.

It turns each score into P(feature detectable). The labels are about detected features, so P means "a JWST analysis would detect this gas's feature", not "the gas is present".

**Call per planet and gas.** Each model gives a probability, P_GBM and P_CNN:
- **present:** P_GBM ≥ 0.8 and P_CNN ≥ 0.8;
- **absent:** P_GBM ≤ 0.2 and P_CNN ≤ 0.2;
- **ambiguous:** anything else, including the two models disagreeing.

**Spectrum disagreement.** If a planet's individual spectra fall on both sides of 0.5 (GBM, calibrated), the call is ambiguous and flagged "spectra disagree". The disagreement may be stellar contamination or a reduction difference.

## What will be reported

- **The register:** score, calibrated probability, call and spectrum spread for every scored planet.
- **Hash locking:** the CSV's SHA-256 goes in `eda/MOLECULE_REGISTER.md`.
- **Comparison with the literature, descriptive only:** for the contested planets (K2-18 b, TOI-270 d, GJ 1214 b, LHS 1140 b, WASP-127 b, HATS-6 b, TRAPPIST-1 e), the register's call is set beside the published claims.

## How it will be checked (prospective)

1. When a JWST paper resolves a register entry (≥ 3σ detection, or a non-detection with 4.3 µm or CH₄-band coverage, under the frozen labelling rules), it is scored as a hit or a miss.
2. Ambiguous calls are counted separately, never as hits.
3. `tools/register_check.py` (to be written with the register) recomputes the hash and lists the entries still open.

## Expectations

| ID | Expectation |
|---|---|
| P1 | The calibration is not degenerate: on the labelled planets, leave-one-planet-out calibrated probabilities beat the base rate in log score, for both gases (GBM) |
| P2 | At least half of the scored register planets get a definite call (present or absent) for at least one gas |

No expectation is set on *which* calls come out. That is the point of a register.
