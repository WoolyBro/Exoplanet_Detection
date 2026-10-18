# Model card — Stage A transit vetters (final, split v2)

Two 5-seed ensembles that score how planet-like a transit signal is. They were frozen on 2026-10-18.

| | Kepler-trained | TESS-trained |
|---|---|---|
| Checkpoints | `stage_a_transit_model/runs/v2_kepler_dual_cnn_lstm_err{,_s1,_s2,_s7,_s11}/best.pt` | `stage_a_transit_model/runs/v2_tess_dual_cnn{,_s1,_s2,_s7,_s11}/best.pt` |
| Architecture | Global (2001-bin) and local (201-bin) CNN branches, then a BiLSTM; 40,451 parameters per seed | Global and local CNN branches; 16,051 parameters per seed |
| Inputs | Phase-folded, depth-normalised Astronet views, plus a per-bin flux-error channel | The same views, without the error channel |
| Output | p(false positive), p(candidate), p(confirmed). The planet-like score is p(candidate) + p(confirmed), averaged over the 5 seeds. | The same |
| Training | Adam, lr 1e-3, batch 64, up to 60 epochs with early stopping (patience 8), ReduceLROnPlateau (factor 0.5, patience 3), batch norm, dropout 0.3, gradient clip 5, random left-right flip of both views, inverse-frequency class weights. Best epoch by validation ROC-AUC. | The same |
| Selection | Rule fixed before any v2 run (`sweep.py`); validation only | The same |
| Calibration | Platt map fitted on validation: a = 1.0945, b = −0.9221 | a = 1.0121, b = 1.0341 |
| Operating point | Recall-90 threshold from validation: 0.593 | 0.479 |

## Data

- **Kepler.** Labels are the cumulative KOI table's dispositions (`koi_cumulative_split_v2.csv`): CONFIRMED, CANDIDATE, FALSE POSITIVE. The light curves are Kepler PDCSAP, 30-minute cadence.
- **TESS.** Labels are TOI dispositions (`toi_tess_candidates_split_v2.csv`): CP/KP → confirmed, PC/APC → candidate, FP/FA → false positive. The light curves are TESS SPOC 2-minute data.
- **Split v2.** Grouped by host star **and** by signal family: no star and no cross-split signal twin is shared between train, validation and test (`split_signal_families.py`). The earlier v1 split leaked through 301 twins.

## Performance (single test read, 2026-10-18; 95 % star-bootstrap intervals)

| | Kepler test (1,404 signals, 1,227 stars) | TESS test (867 signals, 829 stars) |
|---|---|---|
| ROC-AUC, planet-like vs false positive | **0.924** [0.909, 0.937] | **0.815** [0.777, 0.853] |
| ROC-AUC, confirmed vs false positive | 0.971 [0.961, 0.980] | 0.900 [0.859, 0.935] |
| Precision / recall at the validation recall-90 threshold | 0.792 / 0.873 | 0.915 / 0.901 |
| ECE, raw → Platt | 0.078 → 0.023 | 0.144 → 0.018 |
| Against published vetters, same 922 Kepler test KOIs (confirmed vs FP) | 0.977, the same as AstroNet (0.976); below ExoNet (0.990) and ExoMiner (0.996) | — |
| On the other mission's test set | Kepler → TESS 0.811 | TESS → Kepler 0.875 |
| On K2, never seen (pre-registered blind test) | *filled in after the single read* | *filled in after the single read* |

Details: `eda/STAGE_A_V2_FINAL_RESULTS.md` and `eda/K2_BLIND_TEST_PROTOCOL.md`.

## Intended use

Rank transit-like signals **that a detection pipeline has already found** (TCEs, TOIs, catalogued K2 candidates) by how planet-like they are. This prioritises follow-up and vetting effort.

## Not intended for

- **A blind transit search.** Folding a star's light curve at an arbitrary period already scores a median 0.68, and 94 % of such folds pass the threshold (`figures/10`).
- **Confirming planets.** A high score is not a confirmation, and the CANDIDATE class means undecided.
- **Reading the raw score as a probability.** Use the Platt-calibrated score, and only on data like the validation set: base rates differ between surveys.

## Known failure modes (measured)

- **Light from a neighbouring star.** A false positive whose only warning sign is a centroid offset is rejected 49 % of the time, against 95 % for eclipsing binaries. The folded light curve carries no centroid information (`figures/09`). This is where vetters with centroid diagnostics win.
- **Survey priors.**
  - The Kepler-trained model distrusts deep transits: 94 % of Kepler training transits deeper than 10,000 ppm are false positives. On TESS photometry it rejects real hot Jupiters (HD 189733 b scores 0.09).
  - The TESS-trained model has the opposite bias (`figures/07`, `08`).
- **Weak signals.** Near chance below a pipeline S/N of about 7 (Kepler 0.60, TESS 0.65).
- **TESS label and selection bias.** The TESS build kept 56 % of false positives but 89 % of confirmed planets (the ephemeris-reliability filter).

## Ethical and practical notes

- **Data.** Public archive data only: NASA Exoplanet Archive and MAST.
- **No personal data.**
- **Compute.** Training all 60 sweep runs took about 9 h on a 10-core laptop CPU.
