# Prospective predictions: candidates scored before anyone knew what they are

**Registered:** 2026-10-18.
- **Predictions:** written by the single v2 test read in commit `ccd2eed` (02:23 IST).
- **This document and its checker:** committed afterwards.

**Checker:** `python tools/prospective_check.py`. It verifies the hashes, downloads the current dispositions and scores the resolved candidates. Each check is kept in `outputs/prospective/`.

## What is registered

The final v2 ensembles scored every test-split signal once, including the CANDIDATES. A candidate is a signal nobody has yet confirmed or ruled out.

| File (SHA-256) | Candidates | Their disposition today |
|---|---|---|
| `stage_a_transit_model/runs/v2_final/tess_test_predictions.csv` (`99206c08…433e`) | 564 TOIs (PC/APC) | TFOPWG: 518 PC, 45 APC, 1 FP (TOI 6287.01, scored 0.44, below the 0.48 threshold) |
| `stage_a_transit_model/runs/v2_final/kepler_test_predictions.csv` (`3d897473…14b2`) | 285 KOIs | all still CANDIDATE |
| `stage_a_transit_model/runs/k2_blind/k2_predictions.csv` (`94fd3b1d…f5a1`), Kepler-trained ensemble score, from the single K2 read (2026-10-18 21:13) | 838 K2 candidates | all still CANDIDATE (`k2pandc`) |

**In total: 1,687 candidates in three missions.** The K2 scores come from a mission the model never trained on, so their eventual resolution tests transfer as well.

## How it will be scored

The rule is fixed now:
- A candidate counts as **resolved** when its disposition becomes CP or KP (TESS) or CONFIRMED (Kepler, K2), labelled planet, or FP, FA or REFUTED, labelled false positive.
- The score is the ROC-AUC of the registered `planet_like_score` on resolved candidates, reported once at least 10 have resolved each way.
- **Expectation:** ROC-AUC > 0.70. The test-set CONF-vs-FP AUC was 0.90 on TESS. Candidates resolve slowly and selectively (easy ones first), so the first numbers will be optimistic.

## Why it matters

The model cannot have been tuned on these answers, which did not exist when it was frozen. The archive changes all the time: between the 2026-09-08 snapshot and 2026-10-18, 24 TOI dispositions changed (10 PC → FP, 4 CP → FP). A prospective test is the strongest evidence a classifier can offer. It costs nothing to keep running, and each re-check is a dated, falsifiable result.
