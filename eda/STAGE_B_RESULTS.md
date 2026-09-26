# Stage B: molecule classification from real JWST transmission spectra

**Bottom line:** 53 usable JWST spectra from 13 planets aren't enough to train a molecule classifier that can be trusted. Neither a gradient-boosted tree model nor a small 1-D CNN beats a constant majority-class prediction at planet level for any of the five molecules. Nothing here supports claims that a model "detects" H2O, CH4, CO2, SO2 or CO. This is a negative result, reported as one.

Code: `stage_b_atmospheres/stage_b_pipeline.py`. Outputs: `outputs/stage_b/` (index, dataset `.npz`, dropped-spectra list, LOPO predictions, metrics CSV, log, summary JSON) and `eda/stage_b_spectra_sample.png`.
Labels: `outputs/labels/verified_gas_labels.csv`. Every label traces to a bibcode. It includes the two user-approved updates: K2-18 b CO2 = 0 and HD 209458 b SO2 = 0.

---

## 1. Dataset construction

| Step | Result |
|---|---|
| JWST transmission spectra listed for the 13 labelled planets | 64, matching the expected per-planet counts |
| Source | Only the original IPAC `.tbl` files. `05_final_ML_dataset_DO_NOT_USE/` was never read. |
| Raw rows parsed | 31,747 |
| Artifact rows dropped | 1 all-null row, 0 all-zero rows, 0 partially-null rows |
| Depth column | `PL_TRANDEP` (%) in every file, converted to ppm. The `PL_RATROR` fallback was never needed. |
| Uncertainty | Mean of \|err1\| and \|err2\|, in ppm. No missing errors. |
| Spectra with **no data in 1–5 µm** (dropped, listed in `stage_b_dropped_spectra.csv`) | **11**: 8 MIRI LRS spectra (WASP-17 b ×2 Grant 2023; WASP-39 b ×3 Powell 2024; WASP-107 b ×1 Welbanks 2024; K2-18 b ×2 Madhusudhan 2025) and 3 short NIRISS segments covering only 0.63–0.95 µm (WASP-39 b ×1 Carter 2024; WASP-17 b ×2 Louie 2025) |
| **Usable spectra** | **53** |

Four planets have fewer usable spectra than listed: WASP-39 b 13/17, WASP-107 b 11/12, WASP-17 b 2/6 and K2-18 b 2/4. All nine other planets kept every spectrum.

**Resampling.** The grid is 100 equal-width bins from 1 to 5 µm (0.04 µm each). Points in a bin are combined by inverse-variance weighted mean, and the bin uncertainty is 1/√Σw. Each spectrum becomes a 3×100 tensor:

1. depth minus the spectrum's median depth over covered bins (ppm, then ÷1000 as model input)
2. binned uncertainty (ppm ÷1000)
3. coverage mask (1 = bin has data, 0 = no data)

Coverage varies a lot: the fewest bins covered is 1, the median 45 and the most 100.

**Why a coverage mask instead of interpolation.** The instruments cover very different wavelength ranges: NIRISS SOSS reaches about 2.8 µm, NIRCam F322W2 covers about 2.4–4.0 µm, and NIRSpec G395H about 2.8–5.2 µm. Interpolating across a gap would invent absorption features where nothing was observed. Empty bins are set to 0 and flagged with mask 0, so the model sees "no data" as a separate state rather than as flat depth. Median subtraction removes the planet-to-planet baseline (radius ratio) and the offsets between instruments, but keeps feature shapes.

**Band-observability label masking (my addition; not in the brief).** A label only applies to a spectrum if that spectrum covers the molecule's absorption bands. Example: K2-18 b's CO2 = 0 comes from a NIRSpec detection limit and says nothing about a NIRISS spectrum that stops at 2.8 µm. I kept a label only if the spectrum covers **at least 3 bins** in at least one of the molecule's main bands:

| Molecule | Bands used (µm) |
|---|---|
| H2O | 1.30–1.50, 1.80–2.00, 2.60–3.00 |
| CH4 | 2.20–2.40, 3.20–3.45 |
| CO2 | 1.95–2.10, 4.20–4.45 |
| SO2 | 3.95–4.15 |
| CO | 4.50–4.90 |

Otherwise the label is treated as unknown. This removed 60 of the 189 known spectrum-level labels. Planet-level "unknown" labels stay unknown. Unknowns are **masked out of the loss** in both models (for GBM, those rows are left out of that molecule's fit) and are **never treated as 0**.

| Molecule | Spectrum labels used | positive | negative | band-masked |
|---|---|---|---|---|
| H2O | 31 | 26 | 5 | 13 |
| CH4 | 36 | 17 | 19 | 9 |
| CO2 | 26 | 24 | 2 | 12 |
| SO2 | 17 | 15 | 2 | 14 |
| CO | 19 | 17 | 2 | 12 |

## 2. Why leave-one-planet-out cross-validation

There are only 13 planets, and several have many spectra: WASP-39 b has 13 and WASP-107 b has 11. Spectra of the same planet share its atmosphere, and its labels are the same by construction. A random or even stratified split would put a planet's spectra on both sides, and the model could pass by recognising the planet. Grouping by planet is required, and with 13 groups the only split that leaves any training data is leave-one-planet-out (13 folds). Training samples are weighted 1/(number of spectra for that planet), so WASP-39 b doesn't dominate.

A molecule can only be evaluated on a fold if the held-out planet has at least one usable known label for it:

| Molecule | Evaluable folds (planets) | Training folds with only one class |
|---|---|---|
| H2O | 11 (9 pos / 2 neg) | 0 |
| CH4 | 9 (5 pos / 4 neg) | 0 |
| CO2 | 8 (7 pos / 1 neg) | 1 |
| SO2 | 5 (4 pos / 1 neg) | 1 |
| CO | 5 (3 pos / 2 neg) | 0 |

If a training fold has only one class (for CO2 and SO2, holding out the single negative planet leaves only positives), both models just predict that class, since there's nothing to learn from.

## 3. Models

Both models are deliberately small, because there are 53 spectra.

- **GBM:** one `GradientBoostingClassifier` per molecule (50 trees, depth 2, learning rate 0.1, subsample 0.8), trained on the flattened 300 features with planet weights.
- **1-D CNN:** conv(3→8, k=7) → ReLU → maxpool 4 → conv(8→8, k=5) → ReLU → global average pool → dense(8→5) → sigmoid. About 550 parameters.
  - Loss: masked, planet-weighted multi-label BCE plus L2 1e-3.
  - Training: Adam at lr 3e-3, 300 full-batch epochs, predictions averaged over 5 seeds.
  - Written in NumPy with a numerical gradient check (worst relative error 1.9e-8).
  - One network covers all five molecules; the mask means unknown labels contribute no gradient.

**Baselines.**

- **LOO-majority:** the weighted majority class of the training planets in each fold.
- **Global majority:** a constant equal to the majority class across all evaluable planets for that molecule.

The LOO-majority baseline has a known artifact on near-balanced data. For CH4 (5 vs 4), removing a positive planet leaves 4 vs 4, which ties to negative, and removing a negative leaves 5 vs 3, which gives positive. It is wrong on every held-out planet and scores 0%. That's an artifact, not a weak bar to clear. **The fair reference is the global majority**, and it is the one used below.

## 4. Results

**Primary metrics are at planet level.** Spectrum probabilities are averaged per held-out planet, and each planet counts once. ROC-AUC is 0.5 for constant predictors.

| Molecule | Planets (pos/neg) | Reliability | Model | Acc | Prec | Rec | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| **H2O** | 9 / 2 | exploratory | GBM | 0.818 | 0.818 | 1.000 | 0.900 | 0.500 |
| | | | CNN | 0.818 | 0.889 | 0.889 | 0.889 | 0.667 |
| | | | majority (always 1) | 0.818 | 0.818 | 1.000 | 0.900 | 0.500 |
| **CH4** | 5 / 4 | meaningful (≥3 per class) | GBM | 0.556 | 0.600 | 0.600 | 0.600 | 0.600 |
| | | | CNN | 0.556 | 0.571 | 0.800 | 0.667 | 0.450 |
| | | | majority (always 1) | 0.556 | 0.556 | 1.000 | 0.714 | 0.500 |
| | | | *LOO-majority (artifact)* | *0.000* | – | – | – | – |
| **CO2** | 7 / 1 | **NOT MEANINGFUL** | GBM | 0.875 | 0.875 | 1.000 | 0.933 | 0.000 |
| | | | CNN | 0.875 | 0.875 | 1.000 | 0.933 | 0.143 |
| | | | majority (always 1) | 0.875 | 0.875 | 1.000 | 0.933 | 0.500 |
| **SO2** | 4 / 1 | **NOT MEANINGFUL** | GBM | 0.600 | 0.750 | 0.750 | 0.750 | 0.000 |
| | | | CNN | 0.800 | 0.800 | 1.000 | 0.889 | 0.000 |
| | | | majority (always 1) | 0.800 | 0.800 | 1.000 | 0.889 | 0.500 |
| **CO** | 3 / 2 | exploratory | GBM | 0.600 | 0.600 | 1.000 | 0.750 | 0.167 |
| | | | CNN | 0.600 | 0.667 | 0.667 | 0.667 | 0.667 |
| | | | majority (always 1) | 0.600 | 0.600 | 1.000 | 0.750 | 0.500 |

**Molecule by molecule, compared with majority:**

- **H2O:** neither model beats majority. The CNN has the same accuracy and a lower F1. Its AUC of 0.667 rests on ranking only 2 negative planets, and the GBM gives every planet ≥0.93.
- **CH4:** neither model beats majority. Both match the 55.6% of a constant "always CH4" prediction, and AUC is 0.60 (GBM) and 0.45 (CNN), close to chance with 9 planets. The GBM misses WASP-107 b (true 1, predicted 0.02) and puts WASP-39 b at 0.71 (true 0). **Not learning.**
- **CO2:** identical to majority on accuracy and F1, and worse on AUC: the only negative, K2-18 b, gets 1.00 from both models. **Not learning, and the data can't show otherwise.**
- **SO2:** the GBM is *worse* than majority and the CNN equals it. The only negative, HD 209458 b, gets 1.00 from both models. **Not learning.**
- **CO:** the GBM equals majority on accuracy and has AUC 0.17. The CNN has the same accuracy, a lower F1 and AUC 0.67 from 5 planets. **No evidence of learning.**

**Spectrum-level metrics (secondary).** Each spectrum counts once, but spectra of one planet aren't independent, so these overstate the evidence.

| Molecule | n (pos/neg) | GBM acc / AUC | CNN acc / AUC | Majority acc |
|---|---|---|---|---|
| H2O | 31 (26/5) | 0.839 / 0.554 | 0.903 / 0.808 | 0.839 |
| CH4 | 36 (17/19) | 0.528 / 0.560 | 0.389 / 0.254 | 0.472 (planet-majority constant) |
| CO2 | 26 (24/2) | 0.923 / 0.000 | 0.923 / 0.396 | 0.923 |
| SO2 | 17 (15/2) | 0.706 / 0.000 | 0.882 / 0.067 | 0.882 |
| CO | 19 (17/2) | 0.789 / 0.176 | 0.842 / 0.765 | 0.895 |

The only spectrum-level number above its baseline is the CNN on H2O: 0.903 vs 0.839, which is 2 more spectra correct out of 31, with AUC 0.81. At planet level this disappears, and it rests on 5 negative spectra from just 2 planets. On CH4, the one balanced molecule, the CNN is clearly *below* chance (AUC 0.25).

## 5. What these results can and cannot support

**H2O and CH4 are the only molecules balanced enough for supervised evaluation to mean anything, and even they are marginal:**

- **CH4** is the only molecule with ≥3 planets in each class (5/4). Both models only match a constant prediction, so **this is evidence they are not learning a CH4 signal**.
- **H2O** has just 2 negative planets, K2-18 b and TRAPPIST-1 c, and they are confounded. Both are small planets around M dwarfs with flat, low-amplitude or stellar-contaminated spectra, while every H2O-positive planet is a puffy gas giant or sub-Neptune with large features. A model that separates them could be learning "small feature amplitude" or "M-dwarf host", not water. With n_neg = 2, no H2O accuracy or AUC here can separate those explanations.

**CO2, SO2 and CO are exploratory at best.** CO2 and SO2 each have one negative planet, so their scores are determined by how one planet is predicted and **are not meaningful**. Both models got that planet wrong for both molecules. CO has 3/2 planets, and its numbers move by 0.2 when one planet changes.

**The results cannot support:**

- any claim that either model detects any of the five molecules in JWST spectra
- any choice between GBM and CNN (their differences are one or two planets)
- the CNN's spectrum-level H2O number (0.903 / AUC 0.81) as evidence of skill

**The results can support:**

- The dataset build is sound and reproducible. It uses only the original `.tbl` files, drops artifact rows explicitly, and handles coverage and label observability explicitly.
- The LOPO evaluation protocol and masked-loss models run correctly: gradient check passed, and no planet leaks across folds.
- The honest conclusion: **13 labelled planets are too few for supervised molecule classification from observed spectra.** This is a data-size limit, not a tuning problem, and bigger or better-tuned models on these 53 spectra would overfit further.

**Practical next steps (not done):**

- Pre-train on synthetic spectra with known compositions (for example the INARA/PyATMOS forward models already in this project, after resampling onto the same 1–5 µm grid and mask) and use the 13 real planets only as an out-of-domain test set.
- Add more JWST planets with verified labels as they're published.
- Treat each molecule as a band-level feature test, such as a Δχ² for a Gaussian feature at the band, instead of an end-to-end classifier.

## 6. Figure

`eda/stage_b_spectra_sample.png` shows four model inputs: WASP-39 b (NIRSpec PRISM, 96/100 bins), K2-18 b (NIRSpec, 55/100), HAT-P-18 b (NIRISS, 46/100) and WASP-80 b (NIRCam, 42/100). Each panel shows median-subtracted depth, the ±1σ binned uncertainty, the coverage gaps in grey, and the verified labels ("?" = unknown). The WASP-39 b panel shows the 4.3 µm CO2 feature and the 2.7–2.8 µm H2O/CO2 structure. K2-18 b, one of the two H2O "negatives", is flat to about ±100 ppm, which illustrates the amplitude confound described above.
