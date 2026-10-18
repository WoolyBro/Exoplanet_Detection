# A fitted, tested habitability factor — pre-registered (critique 3)

**Written:** 2026-10-18, before any of the models below was fitted. The commit adding this file is the registration.

## Why

Critique 3: the "habitability factor" is a weighted sum of published formulas, and its key physical term fails. Extended to 20 observed rocky planets, the cosmic shoreline's margin does not separate atmospheres from bare rock (ROC-AUC 0.55; `STAGE_C_RESULTS.md` §2b). Three changes answer it.

## (a) A retention model fitted to the observed planets — `retention_model.py`

**Data.** The 20 planets of `rocky_atmosphere_observations.csv`, with each one's shoreline-margin draws (4,000, `target_physics.monte_carlo`) so its uncertainty is carried.

**Soft labels** (fixed now, from the evidence strength recorded in the file):

| Recorded as | Label |
|---|---|
| atmosphere, strong or moderate | 1.0 |
| atmosphere, tentative | 0.6 |
| no thick atmosphere, tentative | 0.25 |
| no thick atmosphere, moderate or strong | 0.0 |

**Model.** logit P(atmosphere) = α + β · margin + γ · magma.
- `magma` = 1 if T_eq ≥ 1,200 K, fixed a priori: a tidally locked dayside without heat redistribution reaches about 1.3 T_eq, above the ~1,500 K silicate solidus.
- The likelihood averages over each planet's margin draws.

**Priors** (theory, before data):
- α ~ N(0, 1.5²): roughly 50 % on the shoreline;
- β ~ N(−2, 1²): one dex past the shoreline lowers the odds about sevenfold;
- γ ~ N(0, 2²): agnostic in sign.

**Inference.** Importance sampling from the prior (200,000 draws). Exact leave-one-out by reweighting.

**Use.** For the ranked targets (all temperate, so magma = 0), P(retain) is the posterior-predictive mean over their margin draws. It replaces the 0/1 shoreline indicator in `terrestrial_atm` as a sensitivity ranking. The registered ranking is unchanged.

## (b) Global sensitivity of the ranking — `sensitivity.py`

**Inputs, independent:**
- the three weights, each log-uniform in [½, 2] × the default, then normalised;
- the retention treatment, one of {shoreline primary, empirical, bolometric, fitted (a)};
- the credit given to the optimistic-only HZ band, U(0, 1) (default 0.5).

**Outputs:** the Spearman correlation with the registered ranking, and each target's top-10 membership.

**Method:** Saltelli sampling with the Jansen estimators, 4,096 base samples, giving first-order and total Sobol indices.

## (c) The factor for unconfirmed candidates, using the vetter — `candidate_factor.py`

The candidates are those with an unbiased vetter score:
- the 564 TESS v2-test candidates (TESS ensemble, Platt map fitted on TESS validation);
- the 838 K2 candidates (TESS ensemble score with the TESS Platt map, which transferred to K2: ECE 0.087, `K2_BLIND_TEST_RESULTS.md`).

**Factor** = P(real, the calibrated vetter probability) × Stage C score, using the same physics and weights. Candidates with P(optimistic HZ) ≥ 0.5 form a shortlist, which is added to the prospective register.

## Expectations

| # | Expectation |
|---|---|
| A1 | P(β < 0 \| data) ≥ 0.90: the data support the shoreline's direction |
| A2 | The leave-one-out log score of (a) beats both the fixed shoreline (P = `p_retain_primary`) and a constant base rate |
| A3 | P(γ > 0 \| data) ≥ 0.80: magma-ocean worlds keep atmospheres more often |
| A4 | The ranking with the fitted retention keeps Spearman ≥ 0.8 with the registered ranking |
| B1 | The habitable-zone weight has the largest total Sobol index for the ranking's Spearman correlation |
| B2 | At least 5 targets are in the top 10 in ≥ 90 % of the Sobol samples (a robust core) |
| C1 | At least one candidate has P(real) ≥ 0.5 and P(optimistic HZ) ≥ 0.5 (the shortlist is not empty) |

## Addendum (2026-10-19, committed before the extended run): two more uncertain assumptions

The Sobol analysis of part (b) varied:
- the weights;
- the retention treatment (primary XUV, empirical or bolometric shoreline, or the fitted model);
- the optimistic-HZ credit.

Two physical assumptions behind the score were held fixed. They are now added as inputs; x1–x5 are unchanged.
- **x6, the shoreline zero point.** A shift δ ~ U(−1, +1) dex is added to every margin, for all treatments. The primary shoreline already marginalises log I₀ over 7–10 × 10⁻⁴ and the XUV history over ×3, so δ tests a much larger, order-of-magnitude error.
- **x7, the mass–radius relation for planets without a measured mass.** Either Chen & Kipping 2017 (the registered choice, with its scatter) or Otegi et al. 2020 rocky, R = 1.03 M^0.29, with 0.05 dex scatter in mass. Mass enters both the TSM (observability) and the escape velocity (retention).

| ID | Expectation |
|---|---|
| B3 | With x6 and x7 also varied, at least 5 targets still stay in the top 10 in ≥ 90 % of samples |
| B4 | The shoreline zero point (x6) has a smaller total Sobol index than the observability weight |
| B5 | The mass–radius relation (x7) has a total Sobol index < 0.05 |

**Not pre-registered, reported as a consistency check:** the fitted retention model applied to 8 Solar System bodies, none of which were in the fit (Venus, Earth, Mars and Titan have atmospheres; Mercury, the Moon, Ganymede and Callisto do not). I estimated its outcome by hand before writing this addendum, so it cannot count as a blind test.
