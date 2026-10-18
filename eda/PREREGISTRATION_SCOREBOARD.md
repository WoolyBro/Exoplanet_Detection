# Pre-registration scoreboard

Every expectation this project registered **before** seeing the result, with its outcome. Outcomes are taken verbatim from the result files listed; nothing was re-scored for this table. For the paper this is Table 7 (`PAPER_OUTLINE.md` §7).

**How a row was registered.**
- **Commit:** the protocol was committed to git before the build, training or read it governs. The hash is that commit.
- **Hash file:** the labels were frozen by SHA-256, with a timestamp, before scoring. The protocol entered git together with the results, so the evidence for the order is the local hash file, not git. This applies to methane rounds 1–2, which predate the rule of committing protocols first.

**Totals:** TOTALS_PLACEHOLDER

---

## Act I — Detection

### Stage A v2 model selection (rule in `stage_a_transit_model/sweep.py`, commit `0bb4e65`; validation only, then one test read)

| ID | Registered rule | Outcome |
|---|---|---|
| V-K | Kepler: simplest architecture within one seed-std of the best; error channel only if it adds > 1 std | dual CNN + BiLSTM + error channel chosen (0.9282 vs the 0.9237 needed); test ROC-AUC **0.924** |
| V-T | TESS: the same rule | dual CNN chosen; error channel not used (0.7892 vs 0.7919 needed); test ROC-AUC **0.815** |

### K2 blind test (`eda/K2_BLIND_TEST_PROTOCOL.md`, commit `fbc60d8`; result `stage_a_transit_model/runs/k2_blind/k2_eval.json`)

| ID | Expectation | Outcome |
|---|---|---|
| E1 | P1 (Kepler ensemble) in [0.80, 0.92] | **held**: 0.863 |
| E2 | P2 (confirmed vs FP) in [0.88, 0.97] | **held**: 0.954 |
| E3 | Kepler ensemble > TESS ensemble on K2 | **held** |
| E4 | The Kepler ensemble under-scores deep confirmed planets; the TESS ensemble does not | **failed** on the TESS half (0.75 vs 0.80, p = 0.20, n = 11). The Kepler half held strongly: 0.72 vs 0.94, p = 2 × 10⁻⁶ |
| E5 | ROC-AUC rises with S/N | **held** |
| E6 | EVEREST ≥ K2SFF | **failed**: K2SFF higher by 0.031 [0.018, 0.045] |
| E7 | The frozen Platt map does not transfer (ECE > 0.05) | **held**: 0.358 |
| F | Failure criterion: P1 < 0.75 | not met |

### DR25 simulated benchmark (`eda/DR25_SIMULATED_PROTOCOL.md`, commit `6c92294`; result `outputs/dr25_simulated/dr25_eval.json`)

| ID | Expectation | Outcome |
|---|---|---|
| D1 | Completeness in [0.85, 0.97] and ≥ the Robovetter's | **held**: 0.935 vs 0.869 (McNemar p = 3 × 10⁻⁸) |
| D2 | Effectiveness in [0.40, 0.85], far below the Robovetter's | **held**: 0.760 vs 0.992 |
| D3 | ROC-AUC in [0.80, 0.95] and below the Robovetter's | **failed**: 0.952, just above the range (below the Robovetter's 0.971 as expected) |
| D4 | Completeness rises with MES | **held** |

### S/N-balanced training, the survey-prior mechanism test (`eda/SNR_BALANCE_PROTOCOL.md`, commit `47a6930`; result `outputs/stage_a_mechanisms/snr_balance.json`)

| ID | Expectation | Outcome |
|---|---|---|
| N1 | Validation ROC-AUC ≥ v2 − 0.01 | **held**: 0.925 vs 0.932 |
| N2 | K2 confirmed deeper than 10,000 ppm: median ≥ 0.85 | **failed**, narrowly: 0.72 → 0.84 |
| N3 | Deep-vs-shallow gap shrinks by at least half | **held**: 0.22 → 0.077 |
| N4 | K2 P1 ROC-AUC ≥ v2 − 0.01 | **failed**: 0.863 → 0.834 |
| N5 | High-S/N validation FPs rejected: ≥ v2 − 0.05 | **held**: 0.985 → 0.940 |

### Vetter v3 (`eda/VETTER_V3_PROTOCOL.md`, commit `ae91f4e`; result `stage_a_transit_model/runs/v3_final/v3_eval.json`)

| ID | Expectation | Outcome |
|---|---|---|
| H1 | C rejects ≥ 0.95 of fresh INV | **held**: 0.991 |
| H2 | C keeps ≥ 0.90 of fresh INJ1 | **failed**: 0.842 (Robovetter 0.859; an operating-point trade, ranker AUC 0.989 vs 0.940) |
| H3 | C rejects INJ2 ≥ v2 + 0.15 and ≥ the Robovetter | **failed**: +0.152 over v2, but 0.20 vs 0.53 |
| H4 | C rejects INJ3 ≥ v2 | **held**: 0.228 vs 0.142 |
| H5 | C rejects ≥ 0.95 of the real NTP test | **held**: 0.999 |
| H6 | C within 0.01 of ExoMiner (confirmed vs FP, common KOIs) | **held**: 0.9961 vs 0.9963 |
| H7 | C rejects ≥ 0.80 of centroid-only FPs | **held**: 0.800 |
| H8a | NTP drives the INV gain | **failed**: the diagnostics do (B 0.948 > A 0.837) |
| H8b | The diagnostics drive the INJ2 gain | **held** |

---

## Act II — Atmospheres

### CH₄ classifier, rounds 1–2 (`METHANE_EXTENSION_PROTOCOL.md`, `_ROUND2.md`; hash file `outputs/stage_b_methane_ext/protocol_hashes.json`, frozen 2026-10-17 21:18 and 21:37)

| ID | Registered failure criterion or test | Outcome |
|---|---|---|
| M1 | Primary GBM ROC-AUC CI on the new planets includes 0.5 → failure | not met: 0.97 [0.88, 1.00], p = 0.0003 (29 planets) |
| M2 | Strong-feature half: the model fails to beat the gas-blind control → failure | not met for the GBM: 0.97 vs control 0.70. **Met for the CNN** (0.79 and 0.73, not significant), so no claim is made for the v1 CNN |
| M3 | Drop of more than 0.1 in ROC-AUC between two independent training grids → failure | not met: 0.97 vs 0.98 |
| M4 | Nested leave-one-planet-out Platt calibration improves Brier score and ECE | **failed**: Brier 0.110 → 0.126, ECE 0.136 → 0.145 |

### Contamination-trained CH₄ model (`METHANE_TLS_PROTOCOL.md`, commit `ff7b73e`; result `outputs/stage_b_methane_ext/tls_robustness.json`)

| ID | Decision rule | Outcome |
|---|---|---|
| T1 | v3 becomes the headline only if it lowers the stress-test false-CH₄ rate **and** loses ≤ 0.03 ROC-AUC on the 29 planets | **held**: false rate 4.1 % → 1.0 %; ROC-AUC 0.97 → 1.00. v3 is the headline |

### Round 3, planets new to the archive (`METHANE_EXTENSION_PROTOCOL_ROUND3.md`, commit `7a21b85`; labels hash-frozen 2026-10-18 13:33)

| ID | Expectation | Outcome |
|---|---|---|
| R3-1 | Each labelled planet falls on the correct side of the v3 class separation (0.08–0.17) | **held**: HATS-6 b 0.982 (CH₄), TRAPPIST-1 f 0.032 (none), HAT-P-12 b 0.017 (none) |

### CO₂ detector on 34 real JWST planets (`CO2_EXTENSION_PROTOCOL.md`, commit `4893248`; labels hash-frozen in commit `fb66170`; result `outputs/stage_b_co2/co2_results.json`)

| ID | Expectation | Outcome |
|---|---|---|
| K1 | GBM ROC-AUC ≥ 0.85, p < 0.01 | **held**: 0.982 [0.932, 1.00] |
| K2 | Strong-feature half: GBM ≥ 0.80 and ≥ control + 0.10 | **held**: 0.942 vs 0.673 (CNN 0.981) |
| K3 | Masking 4.20–4.45 µm lowers CO₂ planets' scores more (p < 0.05) | **held**: median drop 0.87 vs −0.005, p = 3 × 10⁻⁶ |
| K4 | CNN ROC-AUC ≥ 0.80 | **held**: 0.996 |

### SpecNet, the strength-blind CNN–BiLSTM (`METHANE_SPECNET_PROTOCOL.md`, commit `3f5b744`; result `outputs/stage_b_methane_ext/specnet.json`)

| ID | Expectation | Outcome |
|---|---|---|
| S1 | ROC-AUC ≥ 0.95 on the 29 new planets | **failed**: 0.80 [0.46, 1.00] |
| S2 | ≥ 0.90 on the strong-feature half, p < 0.05 | **failed**: 0.76, p = 0.11 |
| S3 | ≥ 0.95 on all 40 labelled planets | **failed**: 0.78 [0.61, 0.94] |
| S4 | False-CH₄ rate ≤ 3 % on the stress set | **failed**: 4.5 % |
| S5 | Less tied to feature strength than the v3 CNN | **held**: Spearman 0.34 vs 0.53 |
| Rule | If any of S1–S4 fails, SpecNet is reported as failed and the GBM stays the headline | applied |

### Retrievals of habitable-zone planets (`RETRIEVAL_HZ_PROTOCOL.md`, commit `4b6b107`; results `outputs/retrieval_trappist1_f/`, `outputs/retrieval_lhs1140_b/`)

| ID | Expectation | Outcome |
|---|---|---|
| R1 | Credit rule: ln B ≥ 3 and a χ² gain in ≥ ⌈2/3⌉ of spectra | applied: no atmosphere credited for either planet |
| R2 | LHS 1140 b: ln B(N₂ vs H₂) > 0 | **held**: +1.93 |
| R3 | LHS 1140 b: atmosphere vs bare inconclusive (\|ln B\| < 3) | **held**: −0.90 |
| R4 | TRAPPIST-1 f: no atmosphere credited | **held**: ln B = −0.70 |

---

## Act III — Habitability factor (`stage_c_priority_fusion/HABITABILITY_FACTOR_PROTOCOL.md`, commit `6e58c74`)

| ID | Expectation | Outcome (result file) |
|---|---|---|
| A1 | P(β < 0) ≥ 0.90: the data support the shoreline's direction | **held**: 0.986 (`outputs/stage_c/retention_model.json`) |
| A2 | Leave-one-out log score beats the fixed shoreline and the base rate | **held**: −0.534 vs −1.724 and −0.580 |
| A3 | P(γ > 0) ≥ 0.80: magma-ocean worlds keep atmospheres more often | **held**: 0.977 |
| A4 | Ranking with the fitted retention: Spearman ≥ 0.8 with the registered one | **held**: 0.841, top 10 unchanged (`sensitivity.json`) |
| B1 | The HZ weight has the largest total Sobol index | **failed**: the observability weight does (0.60 vs 0.47) |
| B2 | ≥ 5 targets in the top 10 in ≥ 90 % of samples | **held**: 7 |
| C1 | At least one candidate with P(real) ≥ 0.5 and P(optimistic HZ) ≥ 0.5 | **held**: 15 (`candidate_factor.json`) |

---

## What the failures say

FAILURES_PLACEHOLDER
