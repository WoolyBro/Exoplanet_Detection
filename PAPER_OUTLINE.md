# Paper outline: one narrative, three acts (critique 6: "it sprawls")

**The rule that removes the sprawl.** Every result in the main text supports the central thesis below. Everything else goes to an appendix. Numbers come from `PAPER_EVIDENCE_MAP.md`.

**Title** (keeps the user's terms and leads with what is unique):
- **Main:** *Exoplanet detection and a habitability factor with CNN–LSTM networks: data, not architecture, sets the limits, from Kepler light curves to real JWST atmospheres*
- **Shorter:** *Data, not architecture: pre-registered CNN–LSTM pipelines from transit detection to JWST atmospheres and habitable-world priorities*

**Central thesis.** Across an end-to-end pipeline from transit signals to habitable-world priorities, what a deep network can learn is set by its data, not its architecture. Pre-registered tests measure each data limit and show which fixes remove it.

**Abstract draft** (≈ 250 words; numbers from `PAPER_EVIDENCE_MAP.md`). It **opens with the two unique results** and puts the vetter numbers after them:

> Machine learning for exoplanets is usually judged by test scores on held-out catalogues. We instead pre-registered every expectation (66 in all; 47 held) and tested a CNN–BiLSTM pipeline from transit signals to habitable-world priorities on data it never saw.
>
> **Atmospheres.** Molecule detectors trained only on simulated spectra with stellar contamination and hazes identify CH₄ in all 41 and CO₂ in 34 labelled real JWST transmission spectra (ROC-AUC 1.00 and 0.996). They pass a gas-blind confound test, and masking the 4.3 µm band removes most of the CO₂ skill. Frozen, they make hash-locked calls on 29 planets whose CH₄ or CO₂ status is unresolved; CO₂ is absent in K2-18 b, GJ 1214 b and LHS 1140 b.
>
> **Detection.** Signal twins inflate Kepler test scores by 0.018 even in star-grouped splits. On the same data, a ResNet and a Transformer with 5–12× more parameters do no better than the CNN–BiLSTM, while adding pipeline measurements and real false alarms raises it to ExoMiner's level (0.996), with 99 % of false alarms rejected. Survey priors are label shift by depth: about 80 labelled signals per survey fix the cross-survey calibration without retraining.
>
> **Habitability.** Contamination-aware retrievals credit no atmosphere to TRAPPIST-1 e, TRAPPIST-1 f or LHS 1140 b. A Bayesian retention model fitted to 20 observed rocky planets predicts three published later. Refitted on all 23, it drives an experimental design that independently recovers the JWST Rocky Worlds targets.
>
> Data, labels and the observing record, not network design, set the limits at every stage.

**Contributions, in the order the introduction lists them (most unique first):**
1. **Real-spectrum molecule detection with a confound test, and a prospective register.** CH₄ (41 planets) and CO₂ (34); the band occlusion; hash-locked calls on 29 unresolved planets. *Unique.*
2. **Data, not architecture, measured.**
   - Signal-twin leakage.
   - ResNet and Transformer nulls against +0.04 from information.
   - Survey priors as label shift by depth, fixed with ~80 labels.
   - v3 at ExoMiner's level with a 44k-parameter network. *Unique as a set.*
3. **What to observe next.** A posterior-driven design for the cosmic shoreline that recovers the Rocky Worlds selection and adds bright K dwarfs and lava worlds. *Unique.*
4. **Contamination-aware verdicts** on three habitable-zone planets, and a fitted retention model validated out of sample.
5. **Pre-registration throughout:** 66 expectations, 19 failures reported, 12 significance claims Holm-adjusted together.

**Target journal.** AJ or MNRAS (methods + science), about 9,000 words, 10 main figures, appendices A–F.

---

## 1. Introduction (≈ 900 words)

- **Problem.** ML vetters report test scores near 0.99, but their evaluation is rarely independent. Habitability rankings rarely carry uncertainty. Atmospheric ML is mostly tested on simulations.
- **Contributions:** the five above, in that order. Open with contribution 1, the result no earlier paper has. Note the related work it must be set against:
  - Duque-Castaño et al. 2025 (synthetic-only tests);
  - Márquez-Neila et al. 2018 (one real HST spectrum);
  - the ESS Open Archive CNN preprint, to read before submission;
  - ExoMiner and ExoMiner++ for vetting;
  - Meni-Gallardo & Pallé 2025 and Ih et al. 2025 for the shoreline.

## 2. Data and the leakage problem (≈ 900 words)

- **Catalogues and light curves:** Kepler, TESS, K2 (blind test only); DR25 simulated data; JWST spectra.
- **Signal twins.** 301 of 1,406 Kepler test KOIs had a twin in training; 0.916 → 0.897. Split v2 groups by star and signal family. **Fig. 1:** the pipeline overview (new) with a twin example.

## 3. Act I — Detection (≈ 2,400 words)

1. **Vetter v2.**
   - Kepler 0.924, TESS 0.815, single test read. Calibration and cross-mission transfer.
   - **Fig. 2:** test ROC plus the architecture ablation (merge figures 01 and 03).
2. **External tests, pre-registered.**
   - K2 0.863 (never seen).
   - DR25: more complete than the Robovetter, 0.935 vs 0.869, but rejects only 76 % of false alarms.
   - **Fig. 3:** figures 15 and 16 combined.
3. **What the network learned** (critique 2).
   - Survey priors are causal through S/N (noise-injection test), measured against the training data.
   - S/N-balanced training removes two-thirds of the prior but costs K2 skill, because the prior carries real base rates. **Fig. 4b:** figure 19.
   - Seed disagreement flags errors, so the vetter can abstain.
   - Occlusion maps.
   - **Architecture is not the limit:** a ResNet and a Transformer with 5–12× the parameters do no better on the same data (−0.006, −0.008), while adding pipeline information gives +0.040. **Fig. 4c:** figure 20.
   - **Fig. 4:** figure 18, with figure 08's opposite biases as an inset.
4. **The blind spots and their fix** (critique 1).
   - Centroid-only false positives rejected 49 %; false alarms 76 %.
   - Vetter v3 adds real false alarms and diagnostics. Results on fresh INJ1, INV, INJ2 and INJ3, real NTP and the Kepler test; the training-data ablation; ExoMiner on common KOIs.
   - **Numbers:**
     - false alarms rejected 0.70 → 0.99;
     - neighbouring-star FPs 0.49 → 0.80;
     - equal to ExoMiner (0.9961 vs 0.9963).
   - **Remaining gap:** faint simulated off-target/EB signals (0.20 / 0.23 vs the Robovetter's 0.53 / 0.63), explained by the training-FP distribution.
   - **Ablation:** the diagnostics, not the NTP examples, drive most gains (H8a failed).
   - **Fig. 5:** figure 17.
5. **Prospective register:** 1,687 open candidates locked by hash.

## 4. Act II — Atmospheres (≈ 2,000 words)

1. **Why supervised learning on real spectra fails:** 13 labelled planets.
2. **Simulation-trained detectors validated on real data.**
   - The GBM, plus **SpecNet**, the strength-blind CNN–BiLSTM, which failed S1–S4 (critique 4): hiding feature strength also hides band significance.
   - 40 labelled planets; the confound test; the stress set; the exoplanet.eu label cross-check.
   - **CO₂ (Pillar B):** 1-D CNN 0.996 and GBM 0.982 on 34 planets; the confound test is passed; masking the 4.3 µm band removes the GBM's skill.
   - **The prospective molecule register:** the frozen detectors call CH₄ and CO₂ on 29 unresolved planets (hash-locked). CO₂ is absent for K2-18 b, GJ 1214 b and LHS 1140 b, and present for WASP-127 b; the analysis-dependent cases are flagged ambiguous. These are falsifiable predictions for future JWST papers.
   - **Fig. 6:** co2_extension + methane_tls_robustness (methane_specnet in an appendix).
3. **Stellar contamination is the limiting systematic.**
   - TRAPPIST-1 e: the atmosphere "signal" lives in visits 3–4.
   - Calibrated on 8 visits: significant in 4.
   - **Fig. 7:** retrieval_trappist1e_real + contamination_residuals.
4. **New verdicts** (critique 5): TRAPPIST-1 f (ln B −0.70) and LHS 1140 b (ln B −0.90; the N₂ hint rests on the NIRISS spectrum), by the same method with a pre-registered credit rule. **Fig. 8:** retrieval_trappist1f + retrieval_lhs1140b.

## 5. Act III — The habitability factor (≈ 1,800 words)

1. **Definition:** HZ (Kopparapu), rocky and retains an atmosphere, and observability (TSM), each with Monte Carlo uncertainty.
2. **Retention fitted to 20 observed rocky planets** (critique 3).
   - The shoreline direction is confirmed (P(β < 0) = 0.99).
   - Magma-ocean worlds are the exception (TOI-561 b, 55 Cnc e).
   - Leave-one-out validation.
3. **Robustness.**
   - Sobol indices.
   - A robust core of 7 targets (TOI-700 d, Kepler-62 f, TRAPPIST-1 e/f/g, LP 890-9 c, LHS 1140 b).
   - The retention treatment barely matters (total index 0.04).
   - **Fig. 9:** figure 12 + stage_c_sensitivity + retention_model.
4. **What to observe next:** the experimental design ranks 1,288 rocky planets by information about the shoreline per JWST hour. It independently recovers the Rocky Worlds DDT selection (all six open targets in the top 45) and adds bright K-dwarf planets and lava worlds. **Fig. 10:** shoreline_design.
5. **What JWST would need for gases:** the forecast with contamination regimes (forecast_contamination_regimes, appendix E).
6. **From detection to habitability:** HZ candidates weighted by the vetter's calibrated probability; the shortlist goes into the prospective register.

## 6. Discussion (≈ 800 words)

- **What limits each act:** labels, false-alarm examples, stellar contamination, the small number of observed rocky planets.
- **Comparison with ExoMiner, AstroNet and the Robovetter,** and with published atmosphere claims.
- **Limitations:** spent test sets; circularity of the Robovetter-derived labels; one star calibrating the contamination; Stage D not modelled because no ground-truth habitability label exists.

## 7. Pre-registration scoreboard (table)

Every registered expectation, its registration commit and its outcome:
- the K2 test (E1–E7);
- DR25 (D1–D4);
- vetter v3 (H1–H8);
- S/N balance (N1–N5);
- SpecNet (S1–S5);
- CO₂ (K1–K4);
- retrievals (R1–R4);
- habitability (A1–A4, B1–B2, C1);
- methane rounds 1–3.

Reporting the failures (E4, E6, D3, M4, N2, N4, S1–S4, B1, ...) is part of the evidence. The table is `eda/PREREGISTRATION_SCOREBOARD.md`.

## 8. Reproducibility

`REPRODUCE.md`, the environments, the model card, 63 tests, the installable JWST molecule screener (`pip install ./jwst_molecule_screener`), and the data and code DOIs (to create at submission).

---

## Appendices (everything else)

| | Content |
|---|---|
| A | Preprocessing and views; the error channel; build yields and selection bias (TESS) |
| B | Baselines, sweep, calibration, three-class confusion, by-S/N tables (figures 02, 04, 05, 06, 07) |
| C | Published-vetter comparison details (figure 14); failure analysis by flag (figure 09); bespoke injection test (figure 10) |
| D | Stage B Level 2 band statistics (figure 11); methane rounds 1–3 labels and protocols; the rocky-planet benchmark |
| E | Forecast method, noise calibration, K2-18 b validation, retrieval factors |
| F | Stage C formulas, the shoreline definition and validation table, the ablation |
