# Stage B, Level 3 — a molecule classifier trained only on simulated JWST spectra

**Date:** 2026-10-17
**Code:**
- `stage_b_atmospheres/forward_model.py`: petitRADTRANS 3.4.
- `synthetic_grid.py`: the training set.
- `train_synthetic_classifier.py`: training and the real-planet test.

**Outputs:** `outputs/stage_b_synthetic/` (`synthetic_grid.npz`, `params.csv`, `grid_summary.json`, `level3_predictions.csv`, `level3_metrics.csv`, `level3_summary.json`)

> **Bottom line.** Trained on 3,000 simulated spectra and tested on the 13 real JWST planets it had never seen, the classifier identifies **methane**.
> - It ranks all five CH₄-detected planets above all four CH₄-non-detected ones: planet-level ROC-AUC 1.00 for the GBM (permutation p = 0.007) and 0.95 for the CNN (p = 0.015).
> - A gas-blind baseline cannot do this (ROC-AUC 0.40, p = 0.72).
> - Stage B Level 1, trained on the real spectra themselves, managed 0.60.
>
> For H₂O, CO₂ and CO, the near-perfect real-planet scores are **not** evidence of gas identification: the gas-blind baseline scores just as well. For SO₂ there is too little data to say anything.

---

## 1. Why this level exists

Level 1 trained on 53 real spectra from 13 planets and had no skill above a constant (`STAGE_B_RESULTS.md`). The limiting factor was the number of labelled examples, not the model. Physics can supply unlimited labelled examples. The question is whether a model trained on them transfers to real data, which is the simulation-to-reality gap.

## 2. Training set (`synthetic_grid.py`)

| Choice | Value | Why |
|---|---|---|
| Planets | each of 3,000 samples anchored on one of the 13 real planets' (R_p, M_p, R★, T_eq), perturbed: radius ×U(0.8, 1.25), mass ×U(0.7, 1.4), T ×U(0.7, 1.1) | spans the real population, unlike Earth-analogue grids such as INARA |
| Background | H₂/He; N₂ with probability 0.7 for anchors below 1.8 R⊕ | no hydrogen envelopes on rocky planets |
| Gases | each of H₂O, CO₂, CH₄, CO, SO₂ present with probability 0.6, log VMR ~ U(−7, −2.5), **independent of temperature** | no chemistry prior the model could exploit (see §5) |
| Clouds | grey deck, log P ~ U(−5, 0) bar | |
| Coverage and noise | borrowed from a randomly chosen **real** spectrum, then Gaussian noise of that size added | the model never sees a cleaner or fuller spectrum than JWST delivers |
| Label | 1 only if removing the gas changes the noise-free spectrum by χ² ≥ 9 over the covered bins (offset removed) | "detectable at this noise", which is what a literature label means |
| Input | 1–5 µm, 100 bins, depth minus median, channels (depth, σ, mask) | identical to Stage B |

**Rates.** Of the gases present, the fraction detectable was H₂O 0.38, CH₄ 0.55, CO₂ 0.56, SO₂ 0.39, CO 0.14.

**Forward-model check.** The WASP-39 b spectrum with no composition fitting had correlation 0.95 with the data. The CO₂ band was 916 ppm in the model vs 906 ppm measured.

## 3. Protocol

- **Strict leave-one-planet-out.** To score real planet P, every synthetic sample anchored on P, *or* that borrowed the noise of one of P's spectra, is removed from training. That leaves 2,095–2,737 training samples per fold.
- **Labels.** Verified literature labels (`outputs/labels/verified_gas_labels.csv`). "Unknown" labels are not scored.
- **Models.**
  - GBM: histogram gradient boosting on depth, σ, mask and depth/σ.
  - CNN: 1-D, three conv layers, masked multi-label BCE, 40 epochs, noise-jitter augmentation.
  - No hyperparameter was tuned on the real planets.
- **Confound control, the "feature strength" baseline.** For each spectrum, the reduced χ² of its departure from flat. It knows nothing about which gas is present; it only measures how strong a spectrum's features are. Giants have strong features and small planets weak ones, so for any gas whose detections fall mostly on giants, this baseline will look skilful.
- **Significance.** Planet-level one-sided permutation p-value of the ROC-AUC (10⁵ relabellings).

## 4. Results (planet level; the score is the mean over that planet's spectra)

| Gas | n (pos/neg) | GBM ROC-AUC (p) | CNN ROC-AUC (p) | Gas-blind baseline (p) | Level 1 GBM / CNN | Reading |
|---|---|---|---|---|---|---|
| **CH₄** | 9 (5/4) | **1.00 (0.007)** | **0.95 (0.015)** | 0.40 (0.72) | 0.60 / 0.45 | **real, gas-specific skill** |
| H₂O | 11 (9/2) | 1.00 (0.018) | 0.94 (0.037) | 1.00 (0.018) | 0.50 / 0.67 | explained by feature strength |
| CO₂ | 8 (7/1) | 1.00 (0.12) | 1.00 (0.12) | 1.00 (0.12) | 0.00 / 0.14 | one negative: uninformative. **Since addressed:** on 34 planets, 0.98 / 1.00, and gas-specific (`CO2_EXTENSION_RESULTS.md`) |
| CO | 5 (3/2) | 1.00 (0.10) | 1.00 (0.10) | 1.00 (0.10) | 0.17 / 0.67 | explained by feature strength |
| SO₂ | 5 (4/1) | 1.00 (0.20) | 1.00 (0.20) | 0.50 (0.60) | 0.00 / 0.00 | one negative: uninformative |

**In-domain check.** ROC-AUC on the held-out *synthetic* samples is 0.89–0.97 for every gas (`level3_summary.json`). The models learn the simulated task well; what limits them is transfer and the size of the real test set.

**CH₄, planet by planet (GBM probability, mean over spectra):**

| Planet | CH₄ (literature) | P(CH₄) | Feature strength |
|---|---|---|---|
| WASP-80 b | detected | 0.96 | 3.5 |
| TOI-270 d | detected | 0.76 | 3.4 |
| GJ 3470 b | detected | 0.62 | 3.4 |
| K2-18 b | detected | 0.41 | 1.8 |
| WASP-107 b | detected | 0.05 | 30.2 |
| TRAPPIST-1 c | not detected | 0.02 | 1.3 |
| HD 189733 b | not detected | 0.02 | 27.8 |
| HD 209458 b | not detected | 0.02 | 13.5 |
| WASP-39 b | not detected | 0.01 | 25.2 |

The model separates CH₄ by the gas, not by planet type. It rejects the three hot giants whose spectra have the strongest features (WASP-39 b, HD 189733 b, HD 209458 b), and accepts three moderate-feature planets.

**Its weakest case is WASP-107 b,** where the literature CH₄ is a depleted, ~5σ detection on a spectrum dominated by other gases. The model's probability there is low (0.05), although it still ranks above every negative.

## 5. What this does and does not show

**Supports.**
- A model trained only on physics-based simulations, with realistic coverage and noise, can recognise CH₄ in real JWST spectra of planets it never saw. It does so better than a model trained on the real spectra, and better than a confound control.
- The grid assigned gases independently of temperature, so the model cannot have learned "cool planet ⇒ CH₄". The only thing it can use is the spectrum's shape.

**Does not support.**
- **Calibrated probabilities.** At a 0.5 threshold, CH₄ accuracy is 0.78, and SO₂/CO accuracy (0.40) is below the majority class. The synthetic prior is about 0.1–0.35 detectable per gas, whereas the real labels are mostly positive, since papers report what they find. Ranking transfers; calibration does not.
- **H₂O, CO₂ or CO identification.** Feature strength explains those scores.
- **Biosignature detection.** These are 13 mostly giant, hot planets. Nothing here concerns O₂ or O₃, or rocky habitable-zone planets; for those, see `FORECAST_RESULTS.md`.

**Small n — since addressed.** A pre-registered test on 19 JWST planets never used before gave CH₄ ROC-AUC 0.95 (p = 0.002; control 0.73), see `METHANE_EXTENSION_RESULTS.md`. The original statement: CH₄ rests on 9 planets. A p-value of 0.007 means the ranking is unlikely to arise by chance, but one more mislabelled or atypical planet would change the headline number.

## 6. Reproducing

```bash
~/micromamba/envs/prt/bin/python stage_b_atmospheres/synthetic_grid.py --n 3000 --workers 3   # ~40 min, ~2.2 GB per worker
python stage_b_atmospheres/train_synthetic_classifier.py                                         # ~10 min
```

> **Memory warning, a lesson from this run.** An earlier version of `forward_model.py` cached one
> opacity object per gas combination. That exhausted 16 GB of RAM and froze the machine twice on
> 2026-10-17. The cache is now one object per wavelength segment, with absent gases at zero
> abundance; this was verified to give identical spectra (0.0 ppm difference).
