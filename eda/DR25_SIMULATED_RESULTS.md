# Kepler DR25 simulated-data benchmark — results

**Scored:** 2026-10-18, as registered in `eda/DR25_SIMULATED_PROTOCOL.md` (commit `6c92294`, evaluator `b46206e`). The run is deterministic: a second run gave a byte-identical result.
**Outputs:** `outputs/dr25_simulated/{dr25_eval.json, dr25_predictions.csv}`
**Figure:** `figures/16_dr25_simulated.png`

> **Bottom line.** On Kepler's own simulated vetting data, on stars no model saw, the frozen Kepler ensemble:
> - **keeps 93.5 % [91.8, 94.9] of injected planets**, against 86.9 % for the Robovetter on the same 1,000 signals. That is significantly more (exact McNemar p = 3 × 10⁻⁸), and the gain is largest for weak and long-period signals.
> - **rejects only 76.0 % [73.3, 78.5] of false alarms** made from inverted light curves, against 99.2 % for the Robovetter (p = 10⁻⁶⁵). The ones that get through are long-period, few-transit, low-MES signals.
>
> Over both sets it separates injected planets from false alarms at **ROC-AUC 0.952** [0.944, 0.960], against 0.971 for the Robovetter score (difference −0.019 [−0.028, −0.009]). The Robovetter was tuned on these very sets.

## 1. Data

| Set | Sample | Views | What it is |
|---|---|---|---|
| INJ1 | 1,000 recovered injections on 1,000 stars | 1,000 | the archive's injected PDC light curves; planets by construction |
| INV | 1,000 TCEs on 925 stars | 1,000 | TCEs the Kepler pipeline found in inverted light curves; false alarms by construction. Rebuilt here as 2 − f from the real PDCSAP flux. |

No star hosts a KOI in any Kepler split. The pipeline and the threshold are the frozen ones (0.593, Kepler validation recall 90 %).

## 2. Results

| | Ours | Robovetter (same TCEs) | Paired (exact McNemar) |
|---|---|---|---|
| **Completeness, INJ1** | **0.935** [0.918, 0.949] | 0.869 [0.847, 0.889] | ours alone keeps 104, Robovetter alone 38; p = 3 × 10⁻⁸ |
| MES 7.1–10 (n = 494) | 0.909 | 0.818 | p = 2 × 10⁻⁶ |
| MES 10–15 (415) | 0.959 | 0.916 | p = 0.011 |
| MES 15–25 (49) / > 25 (42) | 0.959 / 0.976 | 0.898 / 0.976 | n.s. |
| Period 10–100 d (250) | 0.992 | 0.948 | p = 0.007 |
| Period 100–300 d (437) | 0.938 | 0.872 | p = 3 × 10⁻⁴ |
| Period > 300 d (296) | 0.892 | 0.797 | p = 5 × 10⁻⁴ |
| Period < 10 d (17) | 0.765 | 0.882 | n.s. (n = 17) |
| **Effectiveness, INV** | **0.760** [0.733, 0.785] | 0.992 [0.984, 0.996] | ours alone rejects 3, Robovetter alone 235; p = 10⁻⁶⁵ |
| MES 7.1–10 (521) | 0.685 | 0.985 | p = 2 × 10⁻⁴³ |
| MES 10–15 (316) | 0.817 | 1.000 | p = 7 × 10⁻¹⁸ |
| MES 15–25 (112) / > 25 (51) | 0.866 / 0.941 | 1.000 / 1.000 | p = 6 × 10⁻⁵ / 0.25 |
| **ROC-AUC, INJ1 vs INV** | **0.952** [0.944, 0.960] | 0.971 (score) | −0.019 [−0.028, −0.009] |

**The false alarms that pass** have a median period of 126 d and MES 8.9. Those rejected have a median period of 4 d. Only 0.8 % of passing false alarms have P < 1 d, against 18 % of all INV TCEs.

## 3. The registered expectations

| # | Expectation | Verdict |
|---|---|---|
| D1 | Completeness in [0.85, 0.97] and ≥ Robovetter | **held** (0.935 vs 0.869) |
| D2 | Effectiveness in [0.40, 0.85], far below Robovetter | **held** (0.760 vs 0.992; guess 0.65) |
| D3 | ROC-AUC in [0.80, 0.95] and below Robovetter | **failed**: 0.952 is just *above* the range, while below the Robovetter as expected |
| D4 | Completeness rises with MES | **held** (0.909 → 0.976) |

## 4. What it means

1. **The network is the better planet keeper.** It retains 6.6 points more injected planets than the Robovetter.
   - The gain is in exactly the regime that matters most for small, temperate planets: weak signals (MES < 10: +9 points) and long periods (> 300 d: +9.5 points).
   - It is not an artefact of a lenient threshold. The threshold was fixed on Kepler validation at 90 % recall, and the network still rejects 76 % of false alarms.
2. **It is not a false-alarm filter, and the reason is its training data.**
   - It learned from KOIs, which had already survived the pipeline's transit-like triage, so it never saw the systematics-born TCEs that the Robovetter's not-transit-like tests remove.
   - This is the same fact behind the 94 % pass rate of arbitrary folds (`figures/10`). It is the main difference from TCE-trained vetters such as AstroNet and ExoMiner, whose training sets include non-transiting phenomena.
3. **The weakness sits where habitable-zone planets live.** False alarms that pass are long-period with few transits. Used on raw TCEs, the network must sit behind a false-alarm filter, or be retrained with INV/SCR false alarms (a training set the Kepler mission already provides).
4. **Overall separation is close to the Robovetter's** (0.952 vs 0.971) even though the Robovetter is in-sample here. The difference of −0.019 is the same size as the gap to ExoMiner on real KOIs (§5b of `STAGE_A_V2_FINAL_RESULTS.md`).

## 5. Limits

- **Our INV light curves are rebuilt.** They invert the final PDCSAP flux; the archive does not distribute its inverted light curves.
  - Our detrending differs from TPS's, so 136 INV TCEs show no dip in our views (S/N < 0).
  - The model rejects 105 of them but passes 31 that have no dip at all, the same behaviour as on arbitrary folds (`figures/10`).
- **SCR** (scrambled-quarter) false alarms were not tested: the scrambling order is not published with the data.
- **Sample size.** 1,000 + 1,000 of 45,377 + 19,531. The intervals above reflect it.

## 6. Reproducing

```bash
python tools/dr25_simulated_sample.py
python tools/build_dr25_simulated_views.py --set INJ1 --workers 4     # about 30 min
python tools/build_dr25_simulated_views.py --set INV --workers 4      # about 55 min
python stage_a_transit_model/evaluate_dr25_simulated.py
```
