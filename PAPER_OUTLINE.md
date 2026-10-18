# Paper outline: one narrative, three acts (critique 6: "it sprawls")

**The rule that removes the sprawl.** Every result in the main text supports the central thesis below. Everything else goes to an appendix. Numbers come from `PAPER_EVIDENCE_MAP.md`.

**Working title** (keep the user's terms):
- *Exoplanet detection and a habitability factor with CNN–LSTM networks: a pre-registered, end-to-end evaluation*
- Alternative: *From transit signals to habitable-world priorities: what deep networks can and cannot learn from Kepler, TESS, K2 and JWST*

**Central thesis (abstract, first sentence).** In an end-to-end pipeline from transit signals to habitable-world priorities, the limits come from the data — leakage, survey priors, missing false-alarm examples, stellar contamination — more than from the architecture. Pre-registered external tests expose each limit, and targeted data fixes remove most of them.

**Target journal.** AJ or MNRAS (methods + science), about 9,000 words, 10 main figures, appendices A–F.

---

## 1. Introduction (≈ 900 words)

- **Problem.** ML vetters report test scores near 0.99, but their evaluation is rarely independent. Habitability rankings rarely carry uncertainty. Atmospheric ML is mostly tested on simulations.
- **Contributions** (numbered, one per act, plus the method):
  1. Signal-twin leakage in star-grouped splits.
  2. A pre-registered evaluation of a CNN–BiLSTM vetter: external K2 and DR25 tests, a diagnosed blind spot and its fix (v3).
  3. A strength-blind CNN–BiLSTM CH₄ detector validated on 40 real JWST planets.
  4. Contamination-aware verdicts on three habitable-zone planets.
  5. A fitted habitability factor that is robust to its assumptions.
- **The method itself:** pre-registration and single reads, with a scoreboard of all registered expectations (§7).

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
   - Seed disagreement flags errors, so the vetter can abstain.
   - Occlusion maps.
   - **Fig. 4:** figure 18, with figure 08's opposite biases as an inset.
4. **The blind spots and their fix** (critique 1).
   - Centroid-only false positives rejected 49 %; false alarms 76 %.
   - Vetter v3 adds real false alarms and diagnostics. Results on fresh INJ1, INV, INJ2 and INJ3, real NTP and the Kepler test; the training-data ablation; ExoMiner on common KOIs.
   - **Fig. 5:** figure 17.
5. **Prospective register:** 1,687 open candidates locked by hash.

## 4. Act II — Atmospheres (≈ 2,000 words)

1. **Why supervised learning on real spectra fails:** 13 labelled planets.
2. **Simulation-trained detectors validated on real data.**
   - The GBM and **SpecNet**, the strength-blind CNN–BiLSTM (critique 4).
   - 40 labelled planets; the confound test; the stress set; the exoplanet.eu label cross-check.
   - **Fig. 6:** methane_specnet + methane_tls_robustness.
3. **Stellar contamination is the limiting systematic.**
   - TRAPPIST-1 e: the atmosphere "signal" lives in visits 3–4.
   - Calibrated on 8 visits: significant in 4.
   - **Fig. 7:** retrieval_trappist1e_real + contamination_residuals.
4. **New verdicts** (critique 5): LHS 1140 b and TRAPPIST-1 f, by the same method with a pre-registered credit rule. **Fig. 8.**

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
4. **What JWST would need:** the forecast with contamination regimes. **Fig. 10:** forecast_contamination_regimes.
5. **From detection to habitability:** HZ candidates weighted by the vetter's calibrated probability; the shortlist goes into the prospective register.

## 6. Discussion (≈ 800 words)

- **What limits each act:** labels, false-alarm examples, stellar contamination, the small number of observed rocky planets.
- **Comparison with ExoMiner, AstroNet and the Robovetter,** and with published atmosphere claims.
- **Limitations:** spent test sets; circularity of the Robovetter-derived labels; one star calibrating the contamination; Stage D not modelled because no ground-truth habitability label exists.

## 7. Pre-registration scoreboard (table)

Every registered expectation, its registration commit and its outcome:
- the K2 test (E1–E7);
- DR25 (D1–D4);
- vetter v3 (H1–H8);
- SpecNet (S1–S5);
- retrievals (R1–R4);
- habitability (A1–A4, B1–B2, C1);
- methane rounds 1–3.

Reporting the failures (E4, E6, D3, B1, ...) is part of the evidence.

## 8. Reproducibility

`REPRODUCE.md`, the environments, the model card, 50+ tests, and the data and code DOIs (to create at submission).

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
