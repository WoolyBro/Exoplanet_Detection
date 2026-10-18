# Methane classifier on 19 new JWST planets — a pre-registered test

**Date:** 2026-10-17
**Protocol, written before any labelling or scoring:** `stage_b_atmospheres/METHANE_EXTENSION_PROTOCOL.md`
**Code:**
- `stage_b_atmospheres/methane_extension.py`
- `tools/add_ch4_label.py` (label entry)
- `tools/methane_progress.py` (progress)

**Labels:** `outputs/labels/ch4_labels_extended.csv`: 49 planets, each with its published source. Frozen by SHA-256 in `outputs/stage_b_methane_ext/protocol_hashes.json` before scoring.
**Outputs:** `outputs/stage_b_methane_ext/` (`new_planet_scores.csv`, `all_planet_scores.csv`, `metrics.json`, `new_spectra_used.csv`)
**Figure:** `eda/methane_extension.png`
**Tests:** `tests/test_methane_extension.py`

> **Bottom line (rounds 1 + 2, and the contamination test of §9).** A CH₄ classifier trained only on simulated spectra identifies methane in real JWST spectra of planets it never saw.
> - **The headline model, since 2026-10-18,** is the one trained on simulations that **include unocculted starspots, faculae and hazes** (v3; §9). It ranks all 9 CH₄ planets above all 29 non-CH₄ planets (ROC-AUC 1.00 on the 29 new planets and on all 38). It cuts false CH₄ calls under strong stellar contamination from 7.5 % to 2.5 %, and removes the one confident real false positive (TOI-260 b: 0.85 → 0.04).
> - **The clean-trained model (v1)** scored 0.97 on the 29 new planets (CI 0.88–1.00, p = 0.0003) and 0.97 on the strong-feature half, where a gas-blind control scores 0.70. A second independent clean grid gave 0.98.
> - **Probabilities are conservative, not calibrated.** v3 separates the classes at P ≈ 0.1–0.15, not 0.5. The pre-registered Platt calibration did not help v1 (§4).
> - We found no prior study that tests a simulation-trained molecule classifier on real JWST spectra of a planet population against literature labels (§8).

---

## 1. Why this test

Level 3 rested on 9 planets, so one atypical planet could have changed its headline. The only convincing check is planets the model and its developer had never touched. The labels and the model were therefore fixed before any of these planets was scored.

## 2. Labels

All 49 JWST transmission-spectrum planets in the archive outside the original 13 were labelled from their published JWST papers:

| Label | Planets | Rule |
|---|---|---|
| **1** | 5 | TOI-5205 b (~10σ), V1298 Tau b (6σ), LP 791-18 c (4.5σ), LTT 3780 c (3.0–4.6σ), TOI-199 b (Bayes factor ~700) |
| **0** | 17 | Reported non-detection or upper limit (5), or a spectrum the authors call featureless/flat (12) |
| unknown | 27 | Not addressed (most hot Jupiters), tentative (GJ 1214 b, TOI-1130 b, Kepler-51 d, LTT 9779 b) or unverified (WASP-127 b) |

Every row cites a bibcode or arXiv ID. The judgement calls:
- **WASP-121 b** is *unknown*, because its CH₄ comes from a separate emission study, not the transmission spectrum.
- **LHS 1140 b** is *0*: the CH₄/CO₂ features an H₂-rich atmosphere would show are absent.

**Scorable.** 19 of the 22 labelled planets have a spectrum with ≥ 3 grid bins in a CH₄ band (2.2–2.4 or 3.2–3.45 µm), the unchanged Stage B rule. LP 791-18 c and TRAPPIST-1 d are published only as coarse PRISM spectra (24–40 points), and GJ 341 b's NIRCam data start at 3.9 µm.

## 3. Results (planet level; the score is the mean over that planet's CH₄-observable spectra)

| Test set | n (CH₄ / none) | GBM ROC-AUC [95 % CI] (p) | CNN | Gas-blind control |
|---|---|---|---|---|
| **New planets (primary)** | 19 (4 / 15) | **0.95 [0.82, 1.00] (0.002)** | 0.87 [0.67, 1.00] (0.014) | 0.73 [0.33, 1.00] (0.09) |
| New, featureless excluded | 9 (4 / 5) | 1.00 (0.007) | 1.00 (0.007) | 0.70 (0.21) |
| All planets, original + new | 28 (9 / 19) | 0.91 [0.76, 1.00] (0.0001) | 0.85 [0.68, 0.98] (0.001) | 0.70 [0.48, 0.89] (0.05) |

**New planets, by GBM score.** Asterisked rows are CH₄-detected.

| Planet | Label | P(CH₄) | Note |
|---|---|---|---|
| V1298 Tau b* | 1 | 0.995 | young warm giant, CH₄ 6σ |
| TOI-5205 b* | 1 | 0.892 | CH₄ ~10σ despite heavy stellar contamination |
| TOI-260 b | 0 (flat) | 0.849 | **the one confident false positive.** The paper reports a possible 3.17 µm feature, next to the CH₄ band, that the authors suspect is systematic |
| LTT 3780 c* | 1 | 0.502 | CH₄ degenerate with higher hydrocarbons |
| GJ 357 b | 0 (flat) | 0.368 | |
| TOI-199 b* | 1 | 0.354 | Its CNN score is 0.97 |
| TOI-1685 b … HAT-P-12 b | 0 | 0.001–0.19 | the remaining 13 negatives, every one below every CH₄ planet. The only mis-ordered pairs are TOI-260 b and GJ 357 b above TOI-199 b, and TOI-260 b above LTT 3780 c: 3 of 60 pairs, hence 0.95 |

**Post-hoc (not pre-registered).** The two planets excluded only for coarse binning were scored with the same frozen model. LP 791-18 c (CH₄ detected) scores 0.895, and TRAPPIST-1 d (flat) scores 0.003. Both are correct.

## 4. Calibration (nested leave-one-planet-out Platt scaling, all 28 planets)

| | Brier | ECE | Accuracy at 0.5 | ROC-AUC |
|---|---|---|---|---|
| GBM, raw | **0.110** | 0.136 | **0.857** | 0.912 |
| GBM, Platt-calibrated | 0.126 | 0.145 | 0.821 | 0.877 |
| Base rate (nested) | 0.235 | — | 0.679 | — |

**The pre-registered calibration failed to improve anything.** With 28 planets, fitting even two Platt parameters adds more variance than it removes bias. The raw GBM probability is already twice as good as the base rate in Brier score.

The reliability table shows mild *under*-confidence in the middle range: planets scored around 0.45 have CH₄ about 80 % of the time. Treat the raw score as a usable but rough probability.

What changed since Level 3, where calibration looked poor? Level 3's 9 planets were mostly CH₄-positive with mid-range scores. The 15 new negatives are mostly small, flat-spectrum planets, which the model scores near zero.

## 5. Limits

- **Small n is still small.** Four new CH₄ planets, and CIs of about ±0.15. The result is "clearly better than chance and than the control", not a precise skill estimate.
- **Labels are the literature's,** so they inherit its detection thresholds. "Not addressed" is not "absent", which is why 27 planets are unknown and unscored.
- **Featureless negatives are easy.** Without them (9 planets), the model still separates perfectly, but that rests on 4 positives and 5 negatives.
- **One reader labelled everything.** A second independent labelling would strengthen the label set; the per-row sources make that check possible.

## 6. Reproducing

```bash
python stage_b_atmospheres/methane_extension.py     # ~5 min; refuses to run if the frozen label file changed
python tools/methane_progress.py                    # progress tracker used while this ran
python -m pytest tests/test_methane_extension.py
```

---

## 7. Round 2 — harder negatives and a second training grid (pre-registered: `METHANE_EXTENSION_PROTOCOL_ROUND2.md`)

**Why.** Round 1's negatives were mostly flat spectra, which are easy. The 21 planets whose abstracts were silent on CH₄ were re-read **in full text**, under the same rules. Silence still counts as unknown; only explicit statements count.

- **10 new negatives.** Each is an explicit non-detection, upper limit, or "CH₄ feature not observed": TrES-4 b, NGTS-2 b, HIP 67522 b, HAT-P-14 b, TOI-270 b, TOI-776 b, L 168-9 b, L 98-59 d, GJ 9827 d, LHS 1140 c. Several are feature-rich giants.
- **11 stay unknown.** These include WASP-94 A b and HAT-P-30 b, whose papers contradict themselves (text vs table), and WASP-15 b, WASP-121 b and WASP-96 b, where CH₄ appears only in models or is "unconstrained".

These labels were frozen by hash before scoring. **No round-2 planet had ever been scored:** round 1's script only scored labelled planets.

| Test | n (CH₄ / none) | GBM, grid A | GBM, grid B | CNN, grid A | Gas-blind control |
|---|---|---|---|---|---|
| **All new labelled planets (primary)** | 29 (4 / 25) | **0.97** [0.88, 1.00] p = 0.0003 | **0.98** [0.92, 1.00] p = 0.0002 | 0.91 p = 0.003 | 0.71 p = 0.10 |
| **Strong-feature half (confound test)** | 14 (3 / 11) | **0.97** p = 0.006 | **0.97** p = 0.006 | 0.79 p = 0.09 | 0.70 p = 0.18 |
| Featureless spectra excluded | 17 (4 / 13) | 1.00 p = 0.0005 | 1.00 | 0.98 | 0.65 |

**The decisive cases** are feature-rich planets without CH₄. The model ignores feature strength when it decides about CH₄:

| Planet | Feature strength | P(CH₄) |
|---|---|---|
| HIP 67522 b | 38 | 0.019 |
| TrES-4 b | 9.4 | 0.002 |
| WASP-166 b | 6.9 | 0.033 |
| NGTS-2 b | 4.9 | 0.015 |
| HAT-P-12 b | 4.3 | 0.001 |

**The two training grids agree.** Spearman 0.90 between scores, ROC-AUC within 0.01. The largest disagreement is TOI-836 c (0.15 vs 0.41).

**Pre-registered failure criteria.**
- None is met for the GBM.
- **The CNN does not pass the confound test** (0.79 and 0.73, not significant). The claim is therefore made for the GBM. The CNN's skill on all planets (0.87–0.91) may partly reflect feature strength.

## 8. Related work — what this adds

- **Márquez-Neila et al. 2018** (Nature Astronomy 2, 719; arXiv:1806.03944). A random forest trained on a synthetic grid, applied to *one* real HST spectrum (WASP-12 b). That is retrieval of parameters, not classification across planets.
- **arXiv:2407.19167** (MNRAS 539, 1528). Classifiers for CH₄/O₃/H₂O trained on synthetic TRAPPIST-1e-like spectra, tested **only on simulated** JWST observations.
- **Garvin et al. 2024** (A&A; arXiv:2405.13469). Machine learning for molecule detection on synthetic planets injected into **real instrument noise**, for direct imaging.
- **This work.** A simulation-trained molecule classifier evaluated on **real JWST transmission spectra of 38 planets** (9 original + 29 new), with literature labels, a gas-blind confound control, pre-registration, and two independent training grids. We found no prior study doing this. That is a search result, not proof of absence.

---

## 9. Stellar contamination and hazes (pre-registered: `stage_b_atmospheres/METHANE_TLS_PROTOCOL.md`)

**Question.** M-dwarf spectra carry unocculted spots and faculae, whose stellar molecular bands can mimic planetary gases, and hazes add slopes. Does the clean-trained classifier (v1) hallucinate CH₄ under these (H1)? Does training with them (v3) fix it without losing real-planet skill (H2)?

**Data** (`synthetic_grid.py --tls --haze`; generated after the protocol was written):
- A 3,000-spectrum training grid and an **independent 800-spectrum stress-test set**.
- 70 % of spectra carry contamination: spots at 0.70–0.95 T_eff or faculae at T_eff + 50–500 K, covering 0–25 % of the disc, with PHOENIX spectra at each anchor star's own T_eff.
- 50 % carry a power-law haze (κ₃₅₀ 0.01–30 cm²/g, slope −6 to −2).
- Labels stay "detectable at this noise", computed on the contaminated, hazy spectra.
- With the flags off, the generator reproduces the original grid exactly (checked).

**Stress test** (800 held-out spectra; false CH₄ rate = P ≥ 0.5 among spectra containing no CH₄):

| Model | ROC-AUC | False CH₄: no contamination | False CH₄: all | False CH₄: strong contamination (f ≥ 0.1) | ROC-AUC, contaminated only |
|---|---|---|---|---|---|
| v1 GBM (clean-trained) | 0.938 | 1.0 % | 4.1 % | **7.5 %** | 0.928 |
| v1 CNN | 0.921 | 1.9 % | 7.3 % | 12.5 % | 0.914 |
| **v3 GBM** | **0.953** | **0.0 %** | **1.0 %** | **2.5 %** | **0.947** |
| v3 CNN | 0.948 | 1.9 % | 3.2 % | 5.8 % | 0.947 |

**H1 is confirmed.** Contamination multiplies v1's false-CH₄ rate about sevenfold, and the CNN is the more vulnerable model.

**H2 is confirmed.** Training with contamination cuts the false rate by 3–4× and raises ROC-AUC.

**Real planets** (frozen labels; the original 9 by strict leave-one-planet-out on the v3 grid):

| | v1 GBM | **v3 GBM** | v1 CNN | v3 CNN |
|---|---|---|---|---|
| 29 new planets | 0.97 [0.88, 1.00] | **1.00** | 0.91 | 0.97 [0.89, 1.00] |
| all 38 labelled planets | 0.94 [0.84, 1.00] | **1.00** | 0.87 | 0.94 [0.86, 1.00] |

v3 places every CH₄ planet (lowest: GJ 3470 b, 0.17) above every non-CH₄ planet (highest: L 98-59 d, 0.08).
- **It is more cautious.** The CH₄ planets' scores fall (TOI-5205 b 0.89 → 0.58, GJ 3470 b 0.62 → 0.17), so at P ≥ 0.5 it flags only 4 of the 9.
- **The hardest case improves:** WASP-107 b, a depleted-CH₄ detection, rises from 0.05 to 0.22.

**Diagnostic spectra** (GBM; not labelled tests):

| Planet | v1 | v3 | Context |
|---|---|---|---|
| **TOI-260 b** | **0.849** | **0.039** | flat spectrum, possible 3.17 µm systematic: v1's one confident false positive is gone |
| TRAPPIST-1 b | 0.001 | 0.001 | airless; contamination-dominated spectrum |
| TRAPPIST-1 e | 0.094 | 0.184 | no CH₄ indicated, consistent with the real-data retrieval null tests (FORECAST_RESULTS.md §4d) |

**Pre-registered decision.** v3's stress-test false rate (1.0 %) is below v1's (4.1 %), and its ROC-AUC on the 29 new planets (1.00) is not below v1's (0.97). **v3 becomes the headline CH₄ model.**

**Caveats.**
- A perfect ranking on 9 positives has a wide true uncertainty, and the bootstrap CI of [1.00, 1.00] understates it.
- The 29 planets' v1 scores had been seen before v3 was built. The comparison was a pre-registered non-inferiority check, not a selection, but it is not a fully blind test of v3.
- PHOENIX's T_eff grid is coarse below 3,000 K (2002, 2314, 2642 K, …), and real M-dwarf contamination is not fully described by two PHOENIX templates. Visits 3 and 4 of TRAPPIST-1 e keep unexplained structure (FORECAST_RESULTS.md §4d), although null tests on TRAPPIST-1 b, c and d show the fitting method does not fake atmospheres.

**Reproducing**
```bash
~/micromamba/envs/prt/bin/python stage_b_atmospheres/synthetic_grid.py --n 3000 --seed 31415001 --tls --haze --out outputs/stage_b_synthetic_realistic
~/micromamba/envs/prt/bin/python stage_b_atmospheres/synthetic_grid.py --n 800 --seed 27182001 --tls --haze --out outputs/stage_b_synthetic_stress
python stage_b_atmospheres/methane_tls_robustness.py
```

---

## 10. Round 3: planets new to the archive (refresh of 2026-10-18; pre-registered: `METHANE_EXTENSION_PROTOCOL_ROUND3.md`)

**The refresh.** `tools/refresh_archive_spectra.py` downloaded the archive's current Atmospheric Spectroscopy table: 1,875 spectra, against the pack's 1,826. It also downloaded the 49 new spectrum files into `external_data/archive_spectra_2026-10-18/`, after checking that a re-downloaded pack file is byte-identical. Of the 49, 13 are JWST transmission spectra, covering three planets new to this project and new spectra of HAT-P-12 b. Labels were frozen by hash before scoring.

| Planet | CH₄ label (source) | v3 P(CH₄) | v1 P(CH₄) | Outcome |
|---|---|---|---|---|
| **HATS-6 b** | **1**: "strong evidence", log X = −5.38 ± 0.19 (Guzmán Caloca et al. 2026, arXiv:2608.16990) | **0.982** | 0.987 | correct (above the class boundary) |
| **TRAPPIST-1 f** | **0**: pure CH₄ 1 bar rejected at 6.8σ, never detected (Lim et al. 2026, arXiv:2608.17207); 6 NIRISS spectra | **0.032** | 0.006 | correct |
| HAT-P-12 b (new NIRISS spectrum) | 0 (round 1) | 0.017 | 0.002 | consistent |
| TOI-3235 b | unknown: CH₄ constrained only with a deterministic contamination model (Ko et al. 2026, arXiv:2608.04194) | 0.899 | 0.943 | prediction: CH₄ present |

All planets with a label fall on the correct side of the v3 class boundary from rounds 1–2 (CH₄ planets ≥ 0.17, others ≤ 0.08). TOI-3235 b is a genuine prediction: the contamination-trained model calls CH₄, as does the paper's deterministic retrieval. HAT-P-12 b's two new MIRI spectra have no CH₄ band on the 1–5 µm grid.

**The CH₄ test set now totals 40 labelled planets: 10 with CH₄, 30 without.** Every one is ranked correctly by v3.

## 11. An independent check of the labels (added 2026-10-18; in place of a second human labeller)

**Code and output:** `tools/label_crosscheck.py` → `outputs/labels/ch4_label_crosscheck_exoplanet_eu.csv` and its summary.

**What was compared.**
- **Ours:** the CH₄ labels of all 43 labelled planets.
- **Theirs:** the molecules listed by *The Extrasolar Planets Encyclopaedia* (exoplanet.eu), a compilation kept by a different team. Its `species` field was read through the Paris Observatory EPN-TAP service (`exoplanet.epn_core`, snapshot `external_data/exoplanet_eu/exoplanet_eu_species_2026-10-18.csv`).

The two definitions differ by design:
- **exoplanet.eu** lists every species ever reported for a planet (detections, claims, tentative mentions), from any instrument or epoch, without a significance threshold.
- **Ours** require a JWST transmission paper that states a detection or a non-detection.

| | exoplanet.eu: no CH₄ | exoplanet.eu: CH₄ |
|---|---|---|
| Ours: no CH₄ (32) | 30 | 2 |
| Ours: CH₄ (11) | 5 | 6 |

- **Agreement:** 36 of 43 (84 %), Cohen's κ = 0.53. On the 18 planets exoplanet.eu lists species for, κ = 0.66.
- **All 7 disagreements trace to that difference in definition and date. None comes from a disputed reading of a JWST paper:**
  - **HD 189733 b and HD 209458 b.** exoplanet.eu keeps pre-JWST CH₄ claims (HST and ground high-resolution). JWST later set a 5σ upper limit of 0.1 ppm (Fu et al. 2024, Nature) and found "no evidence of CH₄" (Xue et al. 2024, ApJL).
  - **TOI-270 d, TOI-5205 b, LP 791-18 c, LTT 3780 c and HATS-6 b.** JWST CH₄ detections published 2024–2026 that exoplanet.eu does not list yet. Four have no species entry at all; TOI-270 d is listed with H₂O only.
- **What this does and does not establish.**
  - No disagreement involves the two sources reading the same JWST result differently.
  - It cannot replace a blind second reading of the papers. The planets where the literature itself is split are flagged `unknown` and are not in the test set.

## 12. SpecNet, a strength-blind CNN + BiLSTM: failed (pre-registered: `METHANE_SPECNET_PROTOCOL.md`, commit `3f5b744`)

**Code:** `stage_b_atmospheres/methane_specnet.py`
**Output:** `outputs/stage_b_methane_ext/specnet.json` and `specnet_scores.csv`
**Figure:** `eda/methane_specnet.png`

**The attempt.** The paper's title promises CNNs and LSTMs, but its CH₄ detector is a GBM. SpecNet tried to earn the CNN/LSTM claim properly:
- Its input is the whitened residual divided by its own RMS, so the network **cannot** see overall feature strength, the confound that the v1 CNN failed on (§7).
- It is a CNN + BiLSTM, trained on the contamination + haze grid, as a 5-seed ensemble.
- The original 9 planets were scored by strict leave-one-planet-out.

| # | Expectation | SpecNet | GBM v3 | CNN v3 |
|---|---|---|---|---|
| S1 | ROC-AUC ≥ 0.95, 29 new planets | **0.80** [0.46, 1.00], p = 0.03 — failed | 1.00 | 0.97 |
| S2 | ≥ 0.90 on the strong-feature half, p < 0.05 | **0.76**, p = 0.11 — failed | 1.00 (p = 0.003) | 0.97 (p = 0.006)* |
| S3 | ≥ 0.95 on all 40 labelled planets | **0.78** [0.61, 0.94] — failed | 1.00 | 0.94 (n = 38) |
| S4 | False-CH₄ rate ≤ 3 % on the stress set | **4.5 %** (6.7 % under strong contamination) — failed | 1.0 % | 3.2 % |
| S5 | Less tied to feature strength than the v3 CNN | **held**: Spearman 0.34 vs 0.53 | 0.25 | 0.53 |

\* Computed in this run as a comparison column. It was **not** a registered criterion for the v3 CNN.

**Decision (fixed in advance).** SpecNet failed S1–S4, so it is reported as a failed attempt, and **the GBM stays the CH₄ detector.**

**Why it failed.** The figure's right panel shows it. Dividing by the RMS removes what a detector needs: whether a band is *significant*, not just what shape it has.
- The two strongest CH₄ spectra get low scores: V1298 Tau b 0.15 (GBM 0.97) and WASP-107 b 0.15. Their normalised residual is dominated by H₂O and haze slopes, so the CH₄ bands become a small part of a unit-variance curve.
- Weak, featureless spectra are inflated to unit variance, which turns noise into "features". Hence the 4.5 % false rate.

Hiding strength from the input is therefore not how the confound is controlled. It is controlled by *evaluating* against it (the strong-feature half, the gas-blind control).

**Where that leaves CNNs in Stage B.** The contamination-trained **1D CNN (v3)** of §9 remains a working CNN detector:
- 0.97 on the 29 new planets (p = 0.0003) and 0.94 on all 38 it scored;
- 0.97 on the strong-feature half (p = 0.006). This last number is post hoc, so it is not a confirmed pass of the confound test.

The headline stays with the GBM, which is the most accurate and also the least tied to feature strength (Spearman 0.25). The CNN–LSTM claim of the title rests on detection (Stage A) and on the candidate habitability factor (`STAGE_C_RESULTS.md` §3c), not on atmospheres.

## 13. Round 4: a planet published after the models were frozen (pre-registered: `METHANE_EXTENSION_PROTOCOL_ROUND4.md`, commit `944f3dc`)

**Code:** `stage_b_atmospheres/methane_round4.py`. **Output:** `outputs/stage_b_methane_ext/round4.json`. **Data:** `external_data/round4_toi1231b/` (the authors' spectra from OSF, osf.io/mryd6).

**Why so few planets.** Every JWST transmission planet in the archive's 2026-10-18 snapshot has been considered in rounds 1–3, and the archive's query service was down on 2026-10-19. A literature search found two CH₄ planets published since; only one has a public reduced spectrum:
- **TOI-1231 b**, a temperate Neptune (T_eq ≈ 330 K), cooler than every CH₄ planet in rounds 1–3. CH₄ was detected at ln B = 54.5–69.6 (arXiv:2607.00973).
- **TOI-5293 A b** has strong CH₄, but its spectrum is not public, so it is not scored.

| Spectrum (two independent reductions) | v3 GBM P(CH₄) |
|---|---|
| NIRSpec G395H, JexoRes / JexoPipe | **0.984 / 0.993** |
| NIRISS SOSS, JexoRes / JexoPipe | 0.101 / 0.479 |
| **Planet (mean of 4)** | **0.639**: on the CH₄ side of the 0.125 boundary, so **R4-1 held** |

**Reading it.**
- **A blind check outside the training regime.** The frozen model correctly identifies CH₄ in a planet published after it was frozen, and colder than anything it was tested on.
- **The 3.3 µm band carries the detection,** as expected for a cool atmosphere: the NIRSpec spectra are nearly certain. NIRISS sees only the weaker 1.7 and 2.3 µm bands, and the two reductions disagree there (0.10 vs 0.48).
- **The CH₄ test set now has 41 labelled planets, 11 with CH₄.** Every one is ranked correctly by v3: TOI-1231 b's 0.639 is above every non-CH₄ planet (≤ 0.08).
