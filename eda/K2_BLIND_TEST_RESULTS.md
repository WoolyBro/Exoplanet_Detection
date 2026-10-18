# K2 blind test — results of the single pre-registered read

**Read:** 2026-10-18 21:13 IST, once.
- **Guard:** `stage_a_transit_model/runs/k2_blind/TEST_READ.json`.
- **Protocol:** `eda/K2_BLIND_TEST_PROTOCOL.md`, registered in commit `fbc60d8` before any K2 view existed.
- **Evaluator:** committed before the read (`16db11a`).

**Outputs:** `stage_a_transit_model/runs/k2_blind/{k2_eval.json, k2_predictions.csv}`
**Figure:** `figures/15_k2_blind_test.png`

> **Bottom line.** On K2, a mission neither model saw, the frozen Kepler-trained ensemble scores **ROC-AUC 0.863 [0.840, 0.884]** (planet-like vs false positive; 1,462 signals on 1,282 stars) and **0.954 [0.936, 0.968]** for confirmed planets vs false positives.
> - The pre-registered point guess was 0.86 and the failure line was 0.75.
> - Five of the seven registered expectations held. The two that did not are reported in §3.
> - The ranking transfers to a new mission; the absolute probabilities and thresholds do not.

## 1. What was scored

| Stage | Candidate | Confirmed | False positive |
|---|---|---|---|
| In the K2 split | 976 | 579 | 250 |
| Complete ephemeris (`tools/k2_ephemerides.py`) | 943 | 571 | 241 |
| View built (EVEREST light curves) | 838 (89 %) | 519 (91 %) | 237 (98 %) |
| ... of which primary (catalogue durations) | 780 | 471 | 211 |

- **Exclusions.**
  - 144 rows were excluded for an unreliable ephemeris (94 candidates, 48 confirmed, 2 false positives), by the same rule as Kepler and TESS.
  - 17 had no EVEREST light curve.
  - Unlike the TESS build, the yield does not favour planets.
- **Light curves.** The EVEREST error channel behaves like Kepler's: scatter/error median 1.13 (68 % range 0.76–2.06), against 1.09 for Kepler PDCSAP.

## 2. Results

| | Kepler-trained ensemble | TESS-trained ensemble (187 TESS-seen stars removed) |
|---|---|---|
| n (stars) | 1,462 (1,282) | 1,297 (1,160) |
| **ROC-AUC, planet-like vs FP (P1)** | **0.863** [0.840, 0.884] | 0.843 [0.816, 0.871] |
| **ROC-AUC, confirmed vs FP (P2)** | **0.954** [0.936, 0.968] | 0.938 [0.916, 0.957] |
| PR-AUC (base rate) | 0.975 (0.856; lift +0.119) | 0.964 (0.840; lift +0.124) |
| Per-seed ROC-AUC | 0.851–0.865 | 0.820–0.850 |
| At the frozen recall-90 threshold: precision / recall | 0.984 / **0.651** | 0.950 / 0.738 |
| ECE, raw → frozen Platt map | 0.264 → **0.358** (worse) | 0.256 → **0.087** (better) |
| 3-class accuracy | 0.486 | 0.420 |
| ROC-AUC by S/N: < 7.1 / 7.1–15 / 15–50 / > 50 | 0.72 / 0.91 / 0.89 / 0.81 | 0.80 / 0.95 / 0.94 / 0.76 |

**Further numbers:**
- **Paired, on common rows:** Kepler minus TESS = +0.010 [−0.015, +0.036], not significant.
- **Secondary population A** (adding the 154 estimated durations): Kepler 0.854 [0.831, 0.875]; confirmed vs FP 0.943.
- **Robustness to the detrending** (1,450 common primary rows): EVEREST minus K2SFF = **−0.031** [−0.045, −0.018]. The K2SFF light curves score *higher*.

## 3. The registered expectations

| # | Expectation | Verdict | Detail |
|---|---|---|---|
| E1 | P1 in [0.80, 0.92] | **held** | 0.863 (guess 0.86) |
| E2 | P2 in [0.88, 0.97] | **held** | 0.954 |
| E3 | Kepler > TESS on K2 | **held as registered** (point estimate) | +0.010, but the interval includes 0 |
| E4 | The Kepler ensemble under-scores deep confirmed planets, the TESS ensemble does not | **failed** (TESS half) | Kepler half replicated strongly: median 0.72 (27 planets deeper than 10,000 ppm) vs 0.94 (444 shallower), p = 2 × 10⁻⁶. TESS deep 0.75 vs shallow 0.80 (p = 0.20, n = 11), which breaks the "does not" as written. |
| E5 | ROC-AUC rises with S/N | **held** (first bin 0.72 < last bin 0.81) | not monotonic: the peak is at S/N 7–50 and it falls above 50 |
| E6 | EVEREST ≥ K2SFF | **failed** | K2SFF higher by 0.031 [0.018, 0.045]. The reasoning (K2SFF's understated errors would mislead the error channel) was wrong. |
| E7 | The frozen Platt map does not transfer (ECE > 0.05) | **held** | 0.358 |
| — | Failure criterion P1 < 0.75 | not met | |

## 4. What it means

1. **The vetter generalises to a new mission.** On K2 it ranks signals at 0.863, close to its Kepler-to-TESS transfer (0.811) and below its own Kepler test score (0.924).
   - The shortfall is expected. K2's 80-day campaigns hold fewer transits, its noise differs, and its labels come from different teams.
   - Confirmed vs false positive stays high (0.954).
2. **The Kepler survey prior travels with the model.** Deep confirmed K2 planets are under-scored (0.72 vs 0.94, p = 2 × 10⁻⁶), the same bias seen on TESS photometry (`figures/08`). It also explains the drop at S/N > 50, where K2's hot Jupiters sit.
3. **Rankings transfer; probabilities and thresholds do not.**
   - The Kepler validation threshold, which gave 87 % recall on Kepler test, gives 65 % on K2.
   - The Platt map makes calibration worse, because K2 is 86 % planet-like against 49 % for Kepler validation.
   - The TESS map transfers (ECE 0.087) because TESS's base rate (86 %) matches K2's.
   - A new survey needs its own calibration: a small labelled set suffices, the ranking does not.
4. **The light-curve product matters at the 0.03 level.** K2SFF beat EVEREST by 0.031. The pre-registered EVEREST number is the conservative one. The reason is open; two candidates are EVEREST's wider scatter/error spread and its pixel-level decorrelation.
5. **Context.** AstroNet-K2 (Dattilo et al. 2019, AJ 157, 169; doi:10.3847/1538-3881/ab0e12) was trained on K2 itself and reports 98 % accuracy on its own K2 test set.
   - Ours never saw K2, uses a different label source (the archive's K2 dispositions) and is measured by ROC-AUC, so the numbers are not directly comparable.
   - The comparison that matters here is transfer: a Kepler-trained model scored without any K2 training.

## 5. What is now registered

The 838 K2 candidates' scores (`k2_predictions.csv`) join the prospective register (`eda/PROSPECTIVE_PREDICTIONS.md`).

## 6. Reproducing

```bash
python tools/k2_ephemerides.py
python tools/build_k2_views.py                                   # EVEREST, about 1.2 h
python tools/build_k2_views.py --author K2SFF --cache-dir ~/lightkurve_cache_k2sff
python stage_a_transit_model/evaluate_k2.py --figure-only        # the read itself is spent (guarded)
```
