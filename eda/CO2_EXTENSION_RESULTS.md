# CO₂ detection on real JWST spectra: a pre-registered test (Pillar B)

**Protocol, committed before the labels:** `stage_b_atmospheres/CO2_EXTENSION_PROTOCOL.md` (commit `4893248`).
**Labels:** `outputs/labels/co2_labels.csv`, frozen by SHA-256 before any score was computed (commit `fb66170`).
**Code:** `stage_b_atmospheres/co2_extension.py` (commit `d53a550`, written after the freeze).
**Output:** `outputs/stage_b_co2/` (`co2_results.json`, `co2_scores.csv`, `co2_spectra.csv`).
**Figure:** `eda/co2_extension.png`.

> **Result.** A CO₂ detector trained only on simulated spectra, which include stellar contamination and hazes, identifies CO₂ in real JWST transmission spectra of 34 planets it never saw.
> - **The 1-D CNN:** ROC-AUC 0.996 [0.979, 1.00].
> - **The GBM:** 0.982 [0.932, 1.00].
> - **It is not feature strength.** On the strong-feature half, where the gas-blind control drops to chance (0.67, p = 0.17), the CNN scores 0.98 (p = 0.0008) and the GBM 0.94 (p = 0.003).
> - **It is the CO₂ band.** Masking 4.20–4.45 µm lowers CO₂ planets' scores by a median 0.87 and leaves the others unchanged (p = 3 × 10⁻⁶).
>
> **All four registered expectations held.** Level 3's "no CO₂ skill" was a test with one negative, not a failure of the model.

## 1. Why this test

Level 3 had 8 CO₂-labelled planets, 7 positive and 1 negative (`STAGE_B_LEVEL3_RESULTS.md` §4), so it could not show skill or its absence. The archive's 2026-10-18 refresh has **54 planets** with a JWST transmission spectrum spanning the 4.3 µm band.

## 2. Labels (frozen before scoring)

**The rule.** "Was a CO₂ feature detected in the JWST transmission spectrum?"
- **1:** ≥ 3σ, or stated "detected".
- **0:** not detected, an upper limit, or a flat or featureless spectrum.
- **unknown:** tentative, conflicting or not stated.

Each label carries its paper and the quoted evidence.

| | n | Examples |
|---|---|---|
| CO₂ detected | 14 | WASP-39 b (28σ), V1298 Tau b (35σ), WASP-166 b (14.7σ), WASP-94 A b (~11σ), WASP-15 b (8.9σ), HIP 67522 b (≥ 7σ), TrES-4 b (4.0σ), TOI-1130 b (3.3σ) |
| Not detected | 23 | 16 flat or featureless small planets; 7 with other gases but no CO₂: TOI-421 b, TOI-199 b, TOI-5205 b, LTT 3780 c, LP 791-18 c, HAT-P-14 b, WASP-121 b |
| Unknown, excluded | 17 | 8 tentative or model-dependent (e.g. GJ 1214 b, HAT-P-30 b 2.1σ, KELT-7 b, NGTS-2 b, HATS-6 b, LHS 1140 b); K2-18 b and TOI-270 d conflicting; 7 with no CO₂ statement (WASP-127 b, GJ 486 b, L 98-59 b and d, TRAPPIST-1 b and c, TOI-270 b) |

**Changes from the Level 3 labels:** K2-18 b goes from 0 to unknown, because the 2023 detection (3σ) and the 2025 reanalysis conflict.

**Judgement calls**, flagged in the file:
- WASP-121 b = 0: CO₂ was a free retrieval species, but the paper's detected species are H₂O, CO and SiO.
- TrES-4 b = 1: "evidence for … small amounts of CO₂" at 4.0σ.

**Not scored.** Three labelled negatives have no spectrum with ≥ 3 grid bins inside the band: LP 791-18 c, TRAPPIST-1 d and e (coarse PRISM binning). That leaves **34 planets** (14 with CO₂, 20 without) and 139 spectra.

## 3. Models (fixed in the protocol)

- **GBM-CO₂:** the v3 CH₄ GBM recipe, trained on the CO₂ column of the contamination + haze grid (3,000 spectra).
- **CNN-CO₂:** the CO₂ output of the multi-label 1-D CNN, same grid.
- **Control:** the gas-blind feature strength.
- **Leakage.** The six labelled grid-anchor planets were scored by models retrained without their anchored or noise-borrowing samples (2,101–2,676 training spectra per fold): GJ 3470 b, HAT-P-26 b, HD 189733 b, HD 209458 b, WASP-107 b, WASP-39 b.

## 4. Results

| | GBM | 1-D CNN | Gas-blind control |
|---|---|---|---|
| **All 34 planets** | **0.982** [0.932, 1.00], p < 10⁻⁵ | **0.996** [0.979, 1.00], p < 10⁻⁵ | 0.925 [0.821, 0.993] |
| **Strong-feature half** (17: 13 with CO₂, 4 without) | **0.942** [0.80, 1.00], p = 0.003 | **0.981** [0.885, 1.00], p = 0.0008 | 0.673 [0.37, 0.93], p = 0.17 |
| After masking 4.20–4.45 µm (all 34) | 0.664, p = 0.06 | 0.868 | — |

**Band occlusion** (per-planet drop in P(CO₂) when the band is masked):

| | CO₂ planets (median drop) | Others (median drop) | Mann–Whitney p |
|---|---|---|---|
| GBM | **0.87** | −0.005 | 3 × 10⁻⁶ |
| CNN | **0.50** | −0.008 | 3 × 10⁻⁵ |

**Expectations:**

| ID | Expectation | Outcome |
|---|---|---|
| K1 | GBM ROC-AUC ≥ 0.85, p < 0.01 | **held** (0.982) |
| K2 | Strong half: GBM ≥ 0.80 and ≥ control + 0.10 | **held** (0.942 vs 0.673) |
| K3 | Occlusion lowers CO₂ planets' scores more (p < 0.05) | **held** (p = 3 × 10⁻⁶) |
| K4 | CNN ROC-AUC ≥ 0.80 | **held** (0.996) |

No failure criterion was met.

## 5. Reading it

1. **The skill is CO₂-specific.**
   - The control's 0.93 on all planets is the expected confound: CO₂ is detected mostly in giants, and the negatives are mostly flat small planets.
   - On the half where every planet has strong features, the control falls to chance and the detectors do not.
   - Masking the band removes most of the GBM's skill (0.98 → 0.66), so the GBM reads the 4.3 µm band.
   - The CNN keeps 0.87 after masking. It also uses information outside the band, plausibly the 2.0 µm CO₂ band and the band edges, so its gas specificity is shown by the strong-half test rather than by occlusion alone.
2. **The masking is informative in both directions.** For two negatives with strong features elsewhere, masking the band *raises* P(CO₂) by about 0.4: Kepler-51 d (haze slope) and TOI-5205 b (CH₄ and H₂S). The flat 4.3 µm region was evidence *against* CO₂, and without it the model is less sure.
3. **The errors are where the labels are hardest.**
   - **TrES-4 b**, labelled 1 for "small amounts of CO₂" at 4.0σ, scores 0.03. A weak feature on a spectrum dominated by H₂O and CO is the hardest case, like WASP-107 b for CH₄.
   - **WASP-121 b**, labelled 0 by the most judgement-based rule, scores 0.50 and loses 0.48 when the band is masked. The model sees absorption at 4.2–4.45 µm in this ultra-hot Jupiter, where CO and SiO overlap the band edge. Either CO is being read as CO₂, or the paper's non-report understates a weak CO₂ signal. Neither is resolved here.
4. **CNNs in Stage B.** The 1-D CNN passes the CO₂ confound test (0.98 on the strong half, against 0.67 for the control). Contrast the strength-blind CNN–BiLSTM, which failed for CH₄ (`METHANE_EXTENSION_RESULTS.md` §12). A plain CNN, trained on contamination-aware simulations and evaluated against the confound, is a validated real-spectrum detector for CO₂. The GBM stays the CH₄ detector.

## 6. Limits

- **One labeller,** working from the published papers under a fixed rule, with the evidence quoted. Labels are about detected features, not atmospheric composition.
- **The strong half has only 4 negatives,** so its CI is wide (GBM [0.80, 1.00]).
- **The grid's CO₂ is independent of temperature and composition by construction,** so the model cannot have learned "hot giant ⇒ CO₂". The real CO₂ planets are still mostly warm giants, and a cold or small CO₂ planet is not in the test.
- **This is CO₂ in giants and sub-Neptunes.** It says nothing about CO₂ on temperate rocky planets: see `RETRIEVAL_HZ_RESULTS.md` for why their spectra do not yet contain a detectable signal.
