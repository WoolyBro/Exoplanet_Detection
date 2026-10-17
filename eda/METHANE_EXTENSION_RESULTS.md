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

> **Bottom line (rounds 1 + 2).** The CH₄ classifier is trained only on simulated spectra; no real spectrum was ever used to train it. On **29 JWST planets this project had never used**, it identifies methane with **ROC-AUC 0.97** (95 % CI 0.88–1.00, permutation p = 0.0003). The gas-blind control scores 0.71.
> - **It passes the confound test.** On the half of planets with the strongest spectral features, where "big features" cannot stand in for "CH₄", it scores 0.97 (p = 0.006) against 0.70 for the control.
> - **It is robust to the training set.** A second, independently generated simulated grid gives 0.98, with Spearman 0.90 between the two models' scores.
> - **Calibration did not help.** The pre-registered Platt calibration made the probabilities slightly worse; the raw score is usable but rough (§4).
> - **Prior work.** We found no prior study that tests a simulation-trained molecule classifier on real JWST spectra of a planet population against literature labels (§8).

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
