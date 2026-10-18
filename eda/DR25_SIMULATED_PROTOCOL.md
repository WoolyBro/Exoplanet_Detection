# Kepler DR25 simulated-data benchmark — pre-registered protocol

**Written:** 2026-10-18, before any of these signals was scored. The commit that adds this file is the registration. One builder smoke test came first (5 INJ1 and 3 INV signals): the views were checked for construction only (depth, S/N, coverage) and never scored.
**Code:**
- `tools/dr25_simulated_sample.py` (the samples)
- `tools/build_dr25_simulated_views.py` (the views)
- `stage_a_transit_model/evaluate_dr25_simulated.py` (the analysis)

## 1. Why

The Kepler mission measured its own vetting with simulated data (Thompson et al. 2018; Christiansen 2017, KSCI-19110; Coughlin 2017, KSCI-19114). Every DR25 vetter is judged on it.

- **Injected transits (INJ1)** are real planets by construction. The fraction a vetter keeps is its **completeness**.
- **TCEs found in inverted light curves (INV)** are false alarms by construction: systematics and stellar variability that the Kepler pipeline mistook for transits. The fraction a vetter rejects is its **effectiveness** against false alarms.

Our own injection test (`figures/10`) was bespoke and could not be compared with anything. It also showed that arbitrary folds already pass the threshold 94 % of the time. INV measures directly whether that makes the vetter easy prey for systematics.

## 2. Data

| Set | Source | Sample (`tools/dr25_simulated_sample.py`, seed 20261018) |
|---|---|---|
| INJ1 | `kplr_dr25_inj1_tces.txt` (45,377 recovered injections) and the archive's injected PDC light curves (`injected-light-curves-dr25-inj1/q<N>/`). Verified on KIC 891916: the injected file differs from the real one only by the injected transit (2,400 ppm at BKJD 1042.0). | 1,000 TCEs on 1,000 stars; MES median 10.0; period median 198 d; Robovetter PC 86.9 % |
| INV | `kplr_dr25_inv_tces.txt` (19,531 TCEs). The inverted light curves are not distributed, so they are rebuilt from the real Kepler PDCSAP flux as 2 − f, with f the flux normalised per quarter (the DR25 INV data are the inverted light curves). | 1,000 TCEs on 925 stars; MES median 9.9; period median 33 d; Robovetter PC 0.8 % |

- **Eligible stars.** Only stars that host no KOI in any Kepler split, so neither model saw any light curve of these stars.
- **Pipeline.** Unchanged: the same quality bitmask, detrending (every TCE of the set on the star is masked), views and error channel as the Kepler training data.

## 3. Frozen model and metrics

**Model:** the frozen Kepler v2 ensemble (`MODEL_CARD.md`), with its validation recall-90 threshold of 0.593. Nothing is fitted on these data.

| Metric | Definition | Compared with |
|---|---|---|
| **Completeness** | fraction of INJ1 TCEs with score ≥ 0.593 | the Robovetter PC fraction on the same TCEs |
| **Effectiveness** | fraction of INV TCEs with score < 0.593 | the same, for the Robovetter |
| **ROC-AUC, INJ1 vs INV** | our score against the Robovetter `Score` on the same 2,000 TCEs | paired star-bootstrap difference |
| By signal strength | completeness and effectiveness in MES bins 7.1–10, 10–15, 15–25, > 25 | the same, for the Robovetter |

**Caveat, stated in advance:** the Robovetter's thresholds were tuned on these very simulated sets. It has an in-sample advantage here that our model does not.

## 4. Expectations, registered before any view was built

| # | Expectation | Reasoning |
|---|---|---|
| D1 | Completeness in **[0.85, 0.97]**, guess 0.92; ≥ the Robovetter's on the same TCEs | Validation recall at this threshold is 0.90, and injected transits are clean planets. |
| D2 | Effectiveness in **[0.40, 0.85]**, guess 0.65, far below the Robovetter's (≈ 0.99) | 94 % of arbitrary folds pass. Our training FPs were KOIs that had already passed the pipeline's transit-like triage. |
| D3 | ROC-AUC (INJ1 vs INV) in **[0.80, 0.95]**, and below the Robovetter score's | Same reasoning. The Robovetter is in-sample. |
| D4 | Completeness rises with MES | As in every vetter. |

## 5. Rules

- Results are reported whatever they are.
- No model is changed in response to them. If one were, it would be labelled exploratory and need new data.
