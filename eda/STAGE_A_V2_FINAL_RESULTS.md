# Stage A — final evaluation on the leakage-free split (v2), test read once

**Date:** 2026-10-18 (test read 01:11; guard `stage_a_transit_model/runs/v2_final/TEST_READ.json` = 1 read)
**Code:**
- `split_signal_families.py` (split v2)
- `stage_a_transit_model/sweep.py --v2` (60 runs)
- `stage_a_transit_model/evaluate_v2.py` (selection, calibration, single test read)
- `stage_a_transit_model/report_v2.py` (figure)

**Outputs:** `stage_a_transit_model/runs/v2_final/` (`final_eval_v2.json`, `*_test_predictions.csv`), `runs/v2_comparison.csv`
**Figure:** `figures/13_stage_a_v2_final.png`

> **Bottom line.** On a split with no shared host stars and no cross-split signal twins, the final models score:
>
> | Mission | Model | Test ROC-AUC (95 % star bootstrap) | Confirmed vs false positive only |
> |---|---|---|---|
> | Kepler | 5-seed CNN+BiLSTM ensemble with the flux-error channel | **0.924** [0.909, 0.937] | **0.971** |
> | TESS | 5-seed dual-CNN ensemble | **0.815** [0.777, 0.853] | 0.900 |
>
> Both beat a gradient-boosted model on transit-shape features from the same split: Kepler by **+0.034** [0.022, 0.045], TESS by **+0.094** [0.055, 0.136]. Every choice was fixed on validation before test was opened.

---

## 1. Why a v2 evaluation

The 2026-10-17 audit found that the original star-grouped split (v1) leaked. 301 of 1,406 Kepler test KOIs had a **signal twin** in another split: the same period and epoch on a different star, mostly eclipsing-binary contamination copies. Removing them dropped the v1 test ROC-AUC from 0.9158 to 0.8974.

Split v2 groups by star **and** signal family (|ΔP|/P < 10⁻⁴, folded epoch within 0.5 × duration; tolerance chosen against a scrambled-epoch null). It has 0 shared stars and 0 cross-split twins, with class balance unchanged. The v1 test set was spent, so the v2 test set is the one honest reading.

## 2. Model selection (validation only; rule pre-registered in `sweep.py` before any v2 run)

**Kepler** (5 seeds per configuration):

| Configuration | Validation ROC-AUC |
|---|---|
| global CNN | 0.9003 ± 0.0049 |
| dual CNN | 0.9162 ± 0.0016 |
| dual CNN + BiLSTM | 0.9210 ± 0.0027 |
| **dual CNN + BiLSTM + error channel** | **0.9282 ± 0.0046** (needed > 0.9237) → **used** |

**TESS** (5 seeds per configuration):

| Configuration | Validation ROC-AUC |
|---|---|
| global CNN | 0.7728 ± 0.0118 |
| **dual CNN** | **0.7893 ± 0.0025** |
| dual CNN + BiLSTM | 0.7890 ± 0.0048 |
| dual CNN + error channel | 0.7892 ± 0.0062 (needed > 0.7919) → not used |

**The rule:** the simplest architecture within one seed-std of the best. Then the error channel only if it beats the same architecture without it by more than one std.

**The result is a finding in itself.**
- The recurrent layer earns its place on Kepler but not on TESS, where the simpler dual CNN is as good.
- The flux-error channel helps Kepler (+0.007) but not TESS.

Each final model is the **5-seed ensemble**. Platt calibration and the recall-90/95/99 thresholds were fitted on validation and carried to test unchanged.

## 3. Test results

| | Kepler (1,404 KOIs, 1,227 stars) | TESS (867 TOIs, 829 stars) |
|---|---|---|
| ROC-AUC [95 % CI] | **0.924** [0.909, 0.937] | **0.815** [0.777, 0.853] |
| PR-AUC (lift over base rate) | 0.922 (+0.428) | 0.961 (+0.098) |
| Confirmed vs false positive | **0.971** [0.961, 0.980] | 0.900 [0.859, 0.935] |
| 3-class accuracy | 0.760 | 0.533 |
| Precision / recall at the validation-fixed recall-90 threshold | 0.792 / 0.873 | 0.915 / 0.901 |
| Precision / recall at the validation-fixed recall-95 threshold | 0.748 / 0.952 | 0.893 / 0.939 |
| Per-seed test ROC-AUC (mean ± std) | 0.919 ± 0.002 | 0.811 ± 0.005 |
| Brier score, raw → Platt (fitted on validation) | 0.120 → **0.110** | 0.123 → **0.098** |
| ECE, raw → Platt | 0.078 → **0.023** | 0.144 → **0.018** |

**Reading it.**
- **Test agrees with validation.** Kepler's validation was 0.928 and test 0.924, so the validation-only selection did not overfit.
- **Thresholds and calibration transfer.** The validation-fixed thresholds land within 3 points of their target recall on test. Platt scaling fitted on validation cuts test ECE by a factor of 3–8.
- **The weak label is CANDIDATE, not the model.** On confirmed planets vs false positives alone, Kepler reaches 0.971. The 3-class confusion shows CANDIDATE is where errors concentrate (precision 0.50).
- **TESS's high PR-AUC is mostly its base rate.** The lift is only +0.098 at an 86 % positive rate, so ROC-AUC is the informative number there, as in v1.
- **Comparison with v1.**
  - v1 test: a single model scored 0.9158 on a leaky split, or 0.8974 with the twins removed.
  - v2: the ensemble with the error channel reaches 0.924 on a split with no leakage at all.
  - The gain comes from the ensemble and the error channel, not from leakage. A single v2 seed averages 0.919.

## 4. Performance by signal strength (pipeline-measured transit S/N)

| S/N bin | Kepler n | Kepler ROC-AUC | TESS n | TESS ROC-AUC |
|---|---|---|---|---|
| < 7.1 | 98 | 0.60 [0.49, 0.71] | 29 | 0.65 (CI undefined) |
| 7.1–15 | 373 | 0.81 [0.76, 0.85] | 108 | 0.83 [0.74, 0.91] |
| 15–50 | 524 | 0.95 [0.93, 0.97] | 368 | 0.87 [0.82, 0.91] |
| > 50 | 402 | 0.96 [0.94, 0.98] | 354 | **0.79** [0.71, 0.85] |

- **Signal strength drives performance:** near chance below S/N ≈ 7, excellent above 15.
- **TESS gets worse on its strongest signals (0.79).** This is the training prior measured earlier. In TESS's training split, deep transits are rarely false positives (11.5 % vs 17.2 % overall), so deep eclipsing binaries get planet-like scores. Kepler's model, trained on the opposite prior, shows no such drop.

## 5. Cross-mission transfer (each mission's ensemble on the other's test set)

| Trained on → evaluated on | ROC-AUC |
|---|---|
| Kepler → Kepler | 0.924 |
| Kepler → TESS | **0.811** |
| TESS → Kepler | 0.875 |
| TESS → TESS | 0.815 |

- **The Kepler-trained model is as good on TESS as TESS's own model** (0.811 vs 0.815).
- **The TESS-trained model transfers to Kepler better than it does on TESS itself** (0.875).
- Both point the same way: TESS's labels and class balance, not its light curves, limit what can be learned from it.

## 6. Limits

- One test reading per mission. The CIs are star-bootstrap intervals and do not include the selection procedure's own variance. The validation–test agreement suggests that is small.
- Labels are the archive dispositions. CANDIDATE is undecided, which is why confirmed vs false positive is reported alongside.
- The TESS build kept 55.9 % of false positives but 88.7 % of confirmed planets (ephemeris-reliability filter): selection bias that no split fixes.
- Views are reused from the error-channel rebuild (`detection_views_v2` links into `detection_views_err`). No new download was needed for v2.

## 7. Reproducing

```bash
python split_signal_families.py
python tools/make_v2_views.py
python stage_a_transit_model/sweep.py --v2 --jobs 3                  # 60 runs, ~9 h on a 10-core laptop
python stage_a_transit_model/evaluate_v2.py --select-only            # selection, no test data
python stage_a_transit_model/evaluate_v2.py --dry-run-on-val         # pipeline check, no test data
python stage_a_transit_model/evaluate_v2.py                          # the one test read (guarded)
python stage_a_transit_model/report_v2.py
```
