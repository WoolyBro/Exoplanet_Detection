# Pre-registration scoreboard

Every expectation this project registered **before** seeing the result, with its outcome. Outcomes are taken verbatim from the result files listed; nothing was re-scored for this table. For the paper this is Table 7 (`PAPER_OUTLINE.md` §7).

**How a row was registered.**
- **Commit:** the protocol was committed to git before the build, training or read it governs. The hash is that commit.
- **Hash file:** the labels were frozen by SHA-256, with a timestamp, before scoring. The protocol entered git together with the results, so the evidence for the order is the local hash file, not git. This applies to methane rounds 1–2, which predate the rule of committing protocols first.

**Totals: 59 directional expectations, 45 held (76 %) and 14 failed.**
- **Also registered:**
  - 2 model-selection rules (V-K, V-T);
  - 1 test failure criterion (F, not met);
  - 3 CH₄ failure criteria (M1–M3; none met for the GBM, M2 met for the v1 CNN);
  - 2 decision rules (SpecNet, R1), both applied.
- **The 76 % rate is itself evidence.** Expectations were set to be informative, not safe; a scoreboard with every row "held" would suggest they were.

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

### Architecture comparison (`eda/ARCHITECTURE_PROTOCOL.md`, commit `a528555`; result `outputs/stage_a_mechanisms/architectures.json`)

| ID | Expectation | Outcome |
|---|---|---|
| X1 | Neither a 1-D ResNet nor a Transformer beats v2 by more than 0.01 (validation ensemble) | **held**: −0.006 and −0.008 |
| X2 | The best architecture gain is less than a quarter of the data gain | **held**: −0.006 vs +0.040 |

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
| R4-1 | Round 4 (commit `944f3dc`): TOI-1231 b, published after the models were frozen, falls on the CH₄ side (≥ 0.125) | **held**: 0.639 (NIRSpec 0.98–0.99) |

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
| B3 | Addendum (commit `225b269`): ≥ 5 targets stay in the top 10 in ≥ 90 % of samples when the zero point and the M–R relation also vary | **held**: the same 7 (`sensitivity_extended.json`) |
| B4 | The zero point's total index < the observability weight's | **held**: 0.06 vs 0.57 |
| B5 | The mass–radius relation's total index < 0.05 | **held**: 0.005 |
| A5 | Addendum 2 (commit `2f3582c`): on 3 planets published after the fit, the frozen model beats the base rate and the fixed shoreline | **held**: log score −0.59 vs −1.00 and −4.75 (`retention_update.json`) |
| A6 | Refit on 23: P(β < 0) ≥ 0.90 and P(γ > 0) ≥ 0.95 | **held**: 0.989 and 0.997 |
| A7 | Refit on 23: leave-one-out beats the fixed shoreline and the base rate | **held**: −0.50 vs −1.87 and −0.63; AUC 0.77 |
| S1 (design) | Shoreline design (commit `d1ecaab`): the 15 µm noise model fits the 8 published eclipses within 0.2 dex | **held**: 0.107 dex (`shoreline_design.json`) |
| S2 (design) | ≥ 6 of the top 10 by bits per hour have 0.2 ≤ P(retain) ≤ 0.8 | **held**: 9 of 10 |
| S3 (design) | The 6 Rocky Worlds DDT targets without a verdict have a median rank in the top quarter of 1,288 | **held**: median 29.5, all within the top 45 |

## Significance claims, Holm-adjusted as one family

Every positive significance claim in the main text is listed, with its raw p-value. All are adjusted together (Holm, m = 12), as if they were one family of tests, which is stricter than adjusting each analysis separately.

| Claim | Raw p | Holm-adjusted p | Source |
|---|---|---|---|
| v3 rejects more fresh INV false alarms than v2 (McNemar) | 2e-86 | 2e-85 | `eda/VETTER_V3_RESULTS.md` |
| v3 rejects more real NTP false alarms than v2 (McNemar) | 2e-64 | 2e-63 | `eda/VETTER_V3_RESULTS.md` |
| v3 rejects more off-target INJ2 than v2 (McNemar) | 1e-33 | 1e-32 | `eda/VETTER_V3_RESULTS.md` |
| v3 rejects more EB-mimic INJ3 than v2 (McNemar) | 1e-09 | 1e-08 | `eda/VETTER_V3_RESULTS.md` |
| DR25: v2 keeps more injected planets than the Robovetter (McNemar) | 3e-08 | 2e-07 | `eda/DR25_SIMULATED_RESULTS.md` |
| S/N balance raises deep K2 planets (Wilcoxon) | 7e-07 | 5e-06 | `outputs/stage_a_mechanisms/snr_balance.json` |
| K2: Kepler model under-scores deep confirmed planets (Mann-Whitney) | 2e-06 | 1e-05 | `eda/K2_BLIND_TEST_RESULTS.md` |
| CO2 band occlusion is gas-specific (Mann-Whitney) | 3e-06 | 1e-05 | `eda/CO2_EXTENSION_RESULTS.md` |
| CO2 GBM on 34 planets (permutation, < 1e-5) | 1e-05 | 4e-05 | `eda/CO2_EXTENSION_RESULTS.md` |
| CH4 v1 GBM on 29 new planets (permutation) | 0.0003 | 0.0009 | `eda/METHANE_EXTENSION_RESULTS.md §7` |
| CH4 v3 GBM on the strong-feature half (permutation) | 0.003 | 0.006 | `outputs/stage_b_methane_ext/specnet.json` |
| CO2 GBM on the strong-feature half (permutation) | 0.003 | 0.006 | `eda/CO2_EXTENSION_RESULTS.md` |

**All 12 survive** (largest adjusted p = 0.006).
- **Not in the family:**
  - The Bayesian results (retention P(β < 0) = 0.986; the retrieval Bayes factors) are reported as posteriors and evidence ratios, not p-values.
  - The two *equivalence* readings (v3 = Robovetter on INV, p = 0.17; on INJ1, p = 0.14) are non-significance statements, so a multiplicity correction would only strengthen them.
- **The 14 failed expectations above are not p-values.** They are point or interval predictions, reported as stated.

---

## What the failures say

| Failed | What it taught |
|---|---|
| E4 (TESS half) | The TESS-trained model also under-scores deep K2 planets slightly; the survey prior is not unique to Kepler |
| E6 | K2SFF beat EVEREST; the reasoning about understated errors misleading the error channel was wrong |
| D3 | The vetter ranked DR25 simulations better than expected (0.952); its weakness is the operating point, not the ranking |
| M4 | Platt calibration on 28 planets adds more variance than it removes |
| N2, N4 | The survey prior carried real base-rate information; removing it is a trade-off, not a fix |
| H2 | v3's completeness fell to the Robovetter's operating point (an operating-point effect; as a ranker it is better) |
| H3 | Faint off-target transits are outside the KOI false positives' distribution; rule-based thresholds still win there |
| H8a | The pipeline diagnostics, not the NTP examples, drive the false-alarm gain |
| S1–S4 | Hiding feature strength from a CNN also hides band significance; the confound must be controlled by evaluation |
| B1 | The observability weight, not the HZ weight, shapes the full ranking most; HZ decides the top group |

**The failures do not concentrate in one place.** Each was reported where it occurred, and five of them (E6, H8a, N2/N4, S1–S4, B1) changed a claim in the text.
