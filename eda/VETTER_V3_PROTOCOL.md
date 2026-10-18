# Vetter v3 — pre-registered protocol (critique 1: "it is not a complete vetter")

**Written:** 2026-10-18, before any v3 model was trained. The commit that adds this file is the registration.
**Data:** `tools/build_v3_dataset.py` → `external_data/kepler_dr25_simulated/v3/`
**Views:** `tools/build_dr25_simulated_views.py` (sets OBS, INJ1, INJ2, INJ3, INV)
**Training and evaluation:** `stage_a_transit_model/vetter_v3.py`, `stage_a_transit_model/evaluate_v3.py`

## 1. The three measured weaknesses of v2

| Weakness | Measured | Source |
|---|---|---|
| Systematics false alarms pass | 24 % of DR25 inverted-light-curve TCEs pass the threshold (Robovetter 0.8 %) | `DR25_SIMULATED_RESULTS.md` |
| Neighbouring-star contamination | Kepler test FPs with only a centroid flag are rejected 49 % of the time | `figures/09` |
| Gap to the best vetter | −0.019 ROC-AUC vs ExoMiner, confirmed vs FP, 922 common test KOIs | `STAGE_A_V2_FINAL_RESULTS.md` §5b |

**Diagnosis.** v2 saw only folded light curves of KOIs.
- KOIs had already passed the pipeline's transit-like triage, so v2 never met a raw false alarm.
- A folded light curve carries no centroid information.

## 2. What changes, and what does not

The architecture and hyperparameters are the v2 Kepler model's: dual CNN + BiLSTM with the flux-error channel, the same optimiser, class weights, augmentation, scheduler and early stopping. Only two things are added.

1. **Real false alarms (NTP):** 6,000 DR25 TCEs that never became KOIs, labelled FALSE POSITIVE.
   - They come only from stars that host no KOI.
   - They are split by star **and** signal family, built over KOIs and NTP together: 4,200 train / 900 val / 900 test.
   - 507 NTP whose family chained KOIs of different v2 splits were excluded.
   - Verified: 0 cross-split twins, 0 shared stars.
2. **A diagnostic branch.** A small MLP (two layers of 32 units) reads 27 raw Robovetter measurements: centroid score, ghost/halo ratio, odd/even and secondary-eclipse significances, Modelshift, LPP, CHASES, SES/MES, the number of good transits, period, duration and depth. Its output is concatenated with the view features.
   - Robovetter **decision** flags (`ephem_disp`, `cent_disp`, `cflag*`, `iflag*`) are excluded.
   - The 1,373 KOIs without a DR25 TCE get zeros plus a missing-indicator.
   - Features are robust-standardised on the training set and clipped at ±5.

| Config (5 seeds each) | NTP in training | Diagnostic branch | Role |
|---|---|---|---|
| **C — v3 full** | yes | yes | **primary, declared now** |
| A — views + NTP | yes | no | ablation |
| B — views + diagnostics | no | yes | ablation |

No other configuration will be trained.
- **Threshold:** recall 90 % on validation (KOI + NTP val, planet-like vs not).
- **Calibration:** a Platt map fitted on validation.
- **Ensemble:** the mean of the 5 seeds, as in v2.

## 3. Tests, each read once after all training

| | Data | Truth | Compared |
|---|---|---|---|
| T1 | Fresh DR25 simulated signals on held-out stars (no KOI host, no NTP train/val family, none in the 2026-10-18 benchmark): 1,000 INJ1, 1,000 INV, 1,000 INJ2 (transit on a neighbouring star), 1,000 INJ3 (eclipsing-binary mimic) | by construction | v2, A, B, C, Robovetter |
| T2 | Real NTP test (900, held-out families) | DR25 non-KOI TCEs | v2, A, B, C |
| T3 | Kepler v2 test KOIs: a **second, pre-declared read** of that split, for C (A and B as ablations) | archive dispositions | v2's single read; ExoMiner on the 922 common KOIs; exclusive-flag rejection rates |

Multiple comparisons: Holm correction over H1–H7.

## 4. Expectations, registered now

| # | Expectation | v2 today |
|---|---|---|
| H1 | C rejects ≥ 0.95 of INV false alarms | 0.760 |
| H2 | C keeps ≥ 0.90 of INJ1 planets | 0.935 |
| H3 | C rejects INJ2 off-target transits at least 0.15 more often than v2, and at least as often as the Robovetter (≈ 0.53) | measured in T1 |
| H4 | C rejects INJ3 eclipsing-binary mimics at least as often as v2 | measured in T1 |
| H5 | C rejects ≥ 0.95 of real NTP test TCEs | measured in T2 |
| H6 | C's confirmed-vs-FP ROC-AUC on the common KOIs is within 0.01 of ExoMiner's 0.996 (≥ 0.986) | 0.977 |
| H7 | C rejects ≥ 0.80 of the centroid-only false positives on the Kepler test | 0.49 |
| H8 | **Attribution:** NTP training drives the INV gain (A ≈ C, B lower); the diagnostic branch drives the INJ2 and centroid gains (B ≈ C, A lower) | ablation |

## 5. Caveats stated in advance

- **Circularity on real labels.** Kepler's FALSE POSITIVE dispositions, and the NTP set, come largely from the Robovetter, which used these same measurements. A model given them can partly re-learn the Robovetter, which inflates T2 and T3. This is the same circularity ExoMiner has.
- **The non-circular evidence is T1.** The truth there is set by the injection, not by any vetter.
- **The second read of the Kepler test split (T3)** is declared here, before training, and is reported with v2's original single read.
- **Diagnostics exist only for Kepler.** The v3 model cannot be applied to K2 or TESS; v2 remains the cross-mission model.
