# Stage C — transmission-follow-up priority for 54 known planets

**Rebuilt:** 2026-10-17, after an audit found the earlier version scientifically unsound (§4).
**Code:** `stage_c_priority_fusion/target_physics.py` (the physics), `build_priority_table.py` (the ranking)
**Outputs:** `outputs/stage_c/final_priority_table.csv`, `ablation.csv`, `stage_c_summary.json`
**Figure:** `eda/stage_c_priority_overview.png`; `figures/12_priority_ranking_sensitivity.png`
**Tests:** `tests/test_target_physics.py` (published solar-system values reproduced)

> **What this is and is not.** This is a *ranking of transiting planets for transmission-spectroscopy
> follow-up*, favouring ones that are in the habitable zone and terrestrial in size, with stated
> weights. It is **not** a probability that any planet is habitable, and it says nothing about
> atmospheres, water, or life. Being in the habitable zone (HZ) means receiving a flux at which
> liquid surface water is *possible* for a suitable atmosphere — no more.

---

## 1. Method

Every input comes from the NASA Exoplanet Archive composite table (`pscomppars`), and every formula
from a published source. Measurement errors are propagated by Monte Carlo (4,000 draws, split-normal
on the catalogue's upper and lower errors).

| Component | Weight | Definition | Source |
|---|---|---|---|
| habitable zone | 0.40 | P(in conservative HZ) + ½ × P(in optimistic HZ only) | Kopparapu et al. 2014, ApJL 787 L29 |
| terrestrial size **and able to keep an atmosphere** (since 2026-10-18) | 0.30 | P(R_p < 1.5 R⊕ **and** on the atmosphere side of the cosmic shoreline) | Kempton et al. 2018 bin; Zahnle & Catling 2017 (§2b) |
| observability | 0.30 | Transmission Spectroscopy Metric ÷ its follow-up cut, capped at 1 | Kempton et al. 2018, PASP |

**Habitable zone.**
- Effective-flux boundaries: S_eff(T★) = S☉ + aT + bT² + cT³ + dT⁴, with T = T_eff − 5780 K. The 1 M⊕ coefficients of Kopparapu et al. (2014) were checked against the authors' calculator on 2026-10-17.
- Conservative zone: runaway greenhouse to maximum greenhouse. Optimistic zone: recent Venus to early Mars.
- Applied to the planet's incident flux (`pl_insol`). Where its error is missing (20 of 54), the error is propagated from L★ and a.
- Check: for the Sun this gives 0.95–1.67 AU, and places Venus outside and Earth and Mars inside (tested).

**Transmission Spectroscopy Metric.**
- TSM = scale × R_p³ T_eq / (M_p R★²) × 10^(−m_J/5).
- T_eq is recomputed under **one** convention, zero albedo and full redistribution: T_eq = T★ √(R★/a) (¼)^¼.
- Scale factors are 0.190 / 1.26 / 1.28 / 1.15 for the four radius bins, and the follow-up cuts are 12 (R < 1.5 R⊕) and 92 (larger). Both were read from the paper.
- Unmeasured masses use the Chen & Kipping relation as given there.
- Recomputing T_eq replaces the catalogue `pl_eqt`, which for these 54 planets came from 45 different papers with inconsistent assumptions.

**Eligibility.** Transmission spectroscopy needs a transit. **10 of the 54 do not transit:** radial-velocity planets GJ 1132 c, L 98-59 f, GJ 1061 d, GJ 1002 b and GJ 251 c, and imaged planets HR 8799 b–e and 51 Eri b. They are listed with their HZ physics but not ranked; their radii are estimates from mass, not measurements.

**Ranks with uncertainty.** The score is computed for every draw, so each planet gets a median rank and a 90 % rank interval. Ties share the better rank: three planets tied at the top are all rank 1, never 1/2/3 by row order.

---

## 2. Result

> The ranks in this section are the 2026-10-17 ranking, before atmosphere retention. The current ranking, with the cosmic shoreline, is in §2b; the planet-level physics below is unchanged.

**44 transiting planets are ranked.** Eleven planets are in the conservative HZ with P ≥ 0.5, and fifteen in the optimistic one; this counts transiting and non-transiting planets together.

| Rank (90 % interval) | Planet | P(cons. HZ) | P(opt. HZ) | P(R < 1.5 R⊕) | TSM / cut | T_eq (K) |
|---|---|---|---|---|---|---|
| 1 (1–1) | TRAPPIST-1 e | 1.00 | 1.00 | 1.00 | 20.2 / 12 | 250 |
| 1 (1–1) | TRAPPIST-1 f | 1.00 | 1.00 | 1.00 | 17.2 / 12 | 218 |
| 1 (1–1) | TRAPPIST-1 g | 1.00 | 1.00 | 1.00 | 15.4 / 12 | 197 |
| 4 (4–7) | LP 890-9 c | 0.69 | 1.00 | 0.99 | 7.9 / 12 | 271 |
| 5 (4–5) | TRAPPIST-1 d | 0.00 | 1.00 | 1.00 | 25.9 / 12 | 286 |
| 6 (5–14) | TOI-700 d | 0.82 | 1.00 | 1.00 | 3.6 / 12 | 268 |
| 7 (6–15) | Kepler-62 f | 1.00 | 1.00 | 0.90 | 0.40 / 12 | 224 |
| 8 (7–13) | LHS 1140 b | 1.00 | 1.00 | 0.00 | 67.5 / 92 | 225 |
| 9 (7–15) | TOI-700 e | 0.01 | 0.95 | 1.00 | 4.3 / 12 | 295 |
| 10 (8–11) | GJ 1132 b, GJ 486 b, LHS 475 b, TRAPPIST-1 b, TRAPPIST-1 c | 0.00 | 0.00 | 1.00 | above cut | 340–696 |

**Reading it.**
- **TRAPPIST-1 e, f and g are the only planets that satisfy all three criteria at once.** Each is in the conservative HZ, terrestrial, and above Kempton's follow-up cut. The reason is physical: the host is 0.119 R☉, and the transmission signal scales as R★⁻².
- **The next tier is in the HZ but harder to observe.** LP 890-9 c (TSM 7.9), TOI-700 d (3.6) and Kepler-62 f (0.40) are all below the cut; Kepler-62 f because its host is faint (J = 12.3) and 0.64 R☉. Their wide rank intervals reflect exactly that trade.
- **LHS 1140 b is in the conservative HZ with certainty but is not terrestrial-size by this criterion** (1.73 R⊕).
- **The hot rocky planets at rank 10** (GJ 1132 b, GJ 486 b, LHS 475 b, TRAPPIST-1 b/c) are excellent transmission targets but not in the HZ. The ranking reflects that rather than hiding it.

**Caveats carried in `physics_notes`.**
- TRAPPIST-1's T_eff = 2,566 K is 34 K below the HZ fit's validity range, so the boundaries are evaluated at 2,600 K.
- Six transiting planets' masses come from the Chen & Kipping mass–radius relation, not a measurement (LP 890-9 c, TOI-700 d, TOI-700 e, Kepler-62 f, LHS 475 b, Kepler-296 f). Two of them, Kepler-62 f and LP 890-9 c, have only catalogue mass *upper limits* (< 35 and < 25.3 M⊕). A version of 2026-10-17 used those limits as masses, which understated their TSM about tenfold (corrected the same day; `tests/test_target_physics.py` guards it).
- Planets above 10 R⊕ use the 4–10 R⊕ TSM scale factor, an extrapolation.

## 2b. Can they keep an atmosphere? The cosmic shoreline (added 2026-10-18)

**Why.** The habitable zone says whether liquid water is *possible* under an atmosphere. It does not say whether the planet can keep one. JWST (and Spitzer) have now found no thick atmosphere on 14 rocky planets, all around M dwarfs (table below; extended to 20 planets on 2026-10-18). The best-known explanation is the **cosmic shoreline** (Zahnle & Catling 2017, ApJ 843, 122): bodies with low escape velocity and high cumulative XUV exposure lose their atmospheres.

**Definition** (`stage_c_priority_fusion/cosmic_shoreline.py`).
- **Escape velocity:** v_esc = 11.19 km/s × √(M_p/R_p).
- **Cumulative XUV fluence**, relative to Earth: I_XUV = I · (L★/L☉)^−0.6. This is Zahnle & Catling's eq. 27; it encodes that faint stars stay active for longer.
- **The shoreline:** I_XUV = I₀ v_esc⁴, with log I₀ uniform between log 7×10⁻⁴ (through Mars) and −3, as in arXiv:2508.08253.
- **Monte Carlo** over mass (Chen & Kipping masses carry their intrinsic scatter), radius, instellation and luminosity. Also included: the XUV scaling's own uncertainty, 0.48 dex (it agrees with age-based estimates to a factor of ~3; arXiv:2508.08253), and the zero point.

**Validation, before it touched the ranking** (`validate_shoreline.py`; figure `eda/cosmic_shoreline.png`; sources in `stage_c_priority_fusion/rocky_atmosphere_observations.csv`):

| Shoreline definition | JWST/Spitzer "no thick atmosphere" planets on its airless side | Atmosphere-evidence planets on its atmosphere side | Solar System correct |
|---|---|---|---|
| **XUV, slope 4 (used)** | **14 / 14** | 1 / 6 | **7 / 8** (Ganymede misses by 0.03–0.2 dex) |
| empirical, slope 5.89 (arXiv:2508.12865) | 11 / 14 (puts LTT 1445 A b, LHS 1140 c, TOI-1468 b on the wrong side) | 1 / 6 | 7 / 8 (Titan wrong) |
| bolometric I = 10⁻⁴ v_esc⁴ | 12 / 14 (TRAPPIST-1 c, LTT 1445 A b wrong) | 1 / 6 | 5 / 8 |

**Additions on 2026-10-18** (each checked against the paper itself):
- **GJ 3473 b** (Hot Rocks Survey V, Holmberg et al. 2026): thick CO₂ excluded, bare rock not settled; entered as tentative.
- **TOI-561 b** (Teske et al. 2025, ApJL): the first *strong* atmosphere detection on a rocky planet. The dayside is far cooler than bare rock allows.
- **Better sources** for LTT 1445 A b (Wachiraphan et al. 2025, AJ 169, 311: pure CO₂ disfavoured at 4.2σ at 1 bar) and GJ 3929 b (the four-visit Rocky Worlds result, 2026).

- **The XUV shoreline is contradicted by no bare-rock result.** The four "atmosphere" planets on its airless side are not counter-examples to *retention*:
  - 55 Cnc e's atmosphere is thought to be resupplied by a magma ocean (Hu et al. 2024).
  - L 98-59 b and d and LHS 1478 b are tentative (2.6–3.6σ), and two of them are proposed volcanic atmospheres.
- **Limit:** almost every observed planet lies on the airless side. The validation tests the shoreline's "airless" predictions far better than its "keeps an atmosphere" predictions. Only LHS 1140 b, with a tentative atmosphere, sits on the atmosphere side.
- **TOI-561 b breaks the shoreline as a general rule.**
  - It has a strong atmosphere detection, yet sits 2.4 dex deep on the airless side.
  - Across all 20 planets, the shoreline margin does not separate atmosphere evidence from its absence: ROC-AUC 0.55, or 0.32 counting only the moderate and strong cases.
  - Both strong counter-examples, 55 Cnc e and TOI-561 b, are ultra-short-period lava worlds whose magma oceans resupply volatiles (Hu et al. 2024; Teske et al. 2025). Cumulative escape, the shoreline's only physics, does not describe them.
- **What this means for the ranking.**
  - The targets ranked here are temperate (T_eq ≈ 200–300 K), with no magma ocean, so that resupply route is closed. The shoreline remains the relevant prior for them.
  - It is a prior supported only on its airless side: no temperate rocky planet yet has a confirmed atmosphere to test the other side.
  - This is why `rank_without_retention` is kept in the table, and why the ablation (§3) reports the ranking without it.

**How it enters the score.** It is combined with size: the component is P(R_p < 1.5 R⊕ *and* atmosphere side), and the weights are unchanged. A separate fourth term was tried first and rejected: it put hot Jupiters in the top 15 for trivially keeping their H/He envelopes.

**What changed.** These are median ranks. `rank_without_retention` in the table is the previous ranking.

| Planet | Rank now (90 %) | Before | P(atmosphere side) | Shoreline margin (dex) | Empirical / bolometric P |
|---|---|---|---|---|---|
| TRAPPIST-1 g | 2 (1–5) | 1 | 0.42 | +0.10 | 0.87 / 1.00 |
| TOI-700 d | 3 (1–10) | 6 | **0.72** | −0.34 | 0.93 / 0.90 |
| Kepler-62 f | 3 (1–9) | 7 | **0.98** | −1.51 | 1.00 / 0.98 |
| TRAPPIST-1 f | 3 (1–5) | 1 | 0.19 | +0.41 | 0.63 / 1.00 |
| TRAPPIST-1 e | 4 (2–5) | 1 | **0.03** | +0.89 | 0.17 / 1.00 |
| LP 890-9 c | 4 (1–9) | 4 | 0.47 | +0.05 | 0.80 / 0.93 |
| LHS 1140 b | 6 (5–8) | 8 | 0.99 | −1.05 | 1.00 / 1.00 (but 1.73 R⊕) |
| TRAPPIST-1 d | 8 (7–10) | 5 | 0.00 | +1.50 | 0.01 / 0.00 |
| GJ 1132 b, GJ 486 b, LHS 475 b, TRAPPIST-1 b, c | 12 | 9 | 0.00–0.02 | +1.0 to +1.5 | — |

**Reading it.**
1. **TRAPPIST-1 e loses the top spot.** Its cumulative XUV is ~58× Earth's, because a faint, long-active star gives ~90× more XUV per unit of bolometric light. That puts it 0.9 dex past the shoreline.
   - **This verdict depends on the XUV-history assumption.** The empirical shoreline gives 0.17, and the bolometric one, which ignores M-dwarf activity, gives 1.00.
   - Read it as "the TRAPPIST-1 planets are at risk", not as "TRAPPIST-1 e is bare". The tentative JWST hints for e (Espinoza et al. 2025; Glidden et al. 2025) are the direct test.
2. **TRAPPIST-1 g is the best TRAPPIST-1 target** for an atmosphere (0.42). It is the farthest out and the most massive of the three, the classic shoreline expectation.
3. **The habitable-zone planets of quieter stars rise:** TOI-700 d (M2, 0.72) and Kepler-62 f (K2, 0.98). They are much harder to observe, though (TSM 3.6 and 0.4).
4. **The hot rocky planets fall** (GJ 1132 b, GJ 486 b, LHS 475 b, TRAPPIST-1 b, c). JWST found no thick atmosphere on GJ 1132 b, GJ 486 b and TRAPPIST-1 b and c, which agrees with the shoreline.

## 2c. Retention fitted to the observations (added 2026-10-18; pre-registered in `stage_c_priority_fusion/HABITABILITY_FACTOR_PROTOCOL.md`)

**Code:** `retention_model.py`. **Figure:** `eda/retention_model.png`.

The fixed shoreline was replaced, as a tested alternative, by a model fitted to the 20 observed rocky planets:
- **The model:** logit P(atmosphere) = α + β · margin + γ · magma (T_eq ≥ 1,200 K, fixed a priori).
- **Labels:** soft, from the evidence strength.
- **Priors:** from theory.
- **Uncertainty:** each planet's margin uncertainty is averaged over.
- **Inference:** importance sampling (effective sample size 15,400), with exact leave-one-out.

| | Posterior median [95 %] | Reading |
|---|---|---|
| β (margin, per dex) | **−1.27** [−2.64, −0.12]; P(β < 0) = **0.99** | the shoreline's direction is supported by the data (A1 held) |
| γ (magma-ocean dayside) | **+2.63** [+0.05, +5.45]; P(γ > 0) = **0.98** | hot lava worlds keep (resupplied) atmospheres more often (A3 held) |
| α | −0.34 [−1.85, +1.25] | ~42 % on the shoreline itself |

**Leave-one-out mean log score:**
- fitted **−0.53**;
- fixed shoreline −1.72 (it gives TOI-561 b and 55 Cnc e near-zero probability);
- constant base rate −0.58.

A2 held, but the discrimination is modest: leave-one-out AUC 0.60 against 0.55.

**For the ranked targets** (all temperate, magma = 0) the fitted retention is more cautious and, unlike the 0/1 shoreline, carries honest uncertainty:

| Planet | P(retain): fitted [5–95 %] | Fixed shoreline |
|---|---|---|
| LHS 1140 b | 0.69 [0.28, 0.95] | 0.99 |
| Kepler-62 f | 0.76 [0.32, 0.98] | 0.98 |
| TOI-700 d | 0.51 [0.22, 0.78] | 0.72 |
| TRAPPIST-1 g / f / e | 0.40 / 0.33 / 0.23 | 0.42 / 0.19 / 0.03 |

**Conclusion.** The data back the shoreline's direction but not its sharpness. For temperate planets nothing in the data can test it: no temperate rocky planet has a confirmed atmosphere. The registered ranking keeps the fixed shoreline; the fitted one is the sensitivity case (§3b).

## 3b. Global sensitivity of the ranking (Sobol; added 2026-10-18)

**Code:** `sensitivity.py`. **Figure:** `eda/stage_c_sensitivity.png`.

**Inputs varied together:**
- the three weights, each ×½ to ×2;
- the retention treatment (primary, empirical, bolometric or fitted);
- the credit given to the optimistic-only HZ band, from 0 to 1.

There are 4,096 Saltelli base samples. The output is the Spearman correlation with the registered ranking (median 0.84, 5th percentile 0.51).

| Input | First-order index | Total index |
|---|---|---|
| observability weight | 0.43 | **0.60** |
| HZ weight | 0.27 | 0.47 |
| optimistic-HZ credit | 0.12 | 0.19 |
| rocky + retained weight | 0.02 | 0.11 |
| **retention treatment** | −0.01 | **0.04** |

- **The robust core.** Seven targets are in the top 10 in ≥ 90 % of all samples:
  - TRAPPIST-1 e, f and g (1.00);
  - LP 890-9 c (1.00);
  - LHS 1140 b (0.996);
  - TOI-700 d (0.98);
  - Kepler-62 f (0.94).
  
  Next come TRAPPIST-1 d (0.89), TOI-700 e (0.73) and Kepler-296 f (0.58). B2 held.
- **The retention term barely matters (total index 0.04).** With the fitted model the top 10 is unchanged (10/10) and Spearman = 0.84 (A4 held). The term attacked by the critique is not load-bearing for the ranking.
- **B1 failed.** Across plausible weights, the *observability* weight shapes the full 44-planet order more than the HZ weight. The HZ term decides who is in the top group (§3: removing it collapses the order); TSM decides the order within the long tail.

## 3c. Candidates: where the CNN–LSTM vetter enters the factor (added 2026-10-18)

**Code:** `candidate_factor.py`. **Output:** `outputs/stage_c/candidate_shortlist.csv`, hash-locked in `PROSPECTIVE_PREDICTIONS.md`.

For unconfirmed candidates the factor is **P(real) × Stage C score**:
- **P(real)** is the vetter's calibrated planet-like probability: the TESS ensemble with its validation Platt map, which also transferred to K2 (ECE 0.087).
- **The Stage C score** uses the physics and weights of the confirmed targets.
- **Missing parameters:** stellar parameters missing from the archive tables come from the TIC, and insolation is derived where absent.
- **Period aliases:** catalogue insolations inconsistent with the period by more than ×3 (two-transit candidates whose listed period is an alias) are flagged, and the insolation is trusted.

**Coverage:** 1,064 of the 1,360 candidates (486 TESS, 578 K2).

**The shortlist (C1 held):**
- **15 candidates** have P(real) ≥ 0.5 and P(optimistic HZ) ≥ 0.5.
- **11 of them score above the 10th-ranked confirmed target.**
- **Leading entries:**
  - EPIC 246711015.01 (K2): 1.23 R⊕ at 0.60 S⊕ around a 3,245 K M dwarf, P(real) 0.86. Its sibling .02 is 0.51 R⊕ at 0.27 S⊕.
  - EPIC 216876207.01: 3.3 R⊕, 0.48 S⊕.
  - TOI 4353.01: 1.37 R⊕, 0.43 S⊕, period flagged as an alias.

These are catalogued candidates, not discoveries. The contribution is a ranked, uncertainty-aware follow-up list in which the detection network's verdict is part of the habitability factor.

## 3. Ablation — what the order depends on

| Criterion removed | Spearman vs full ranking | Top-10 overlap |
|---|---|---|
| habitable zone | 0.140 | 2 / 10 |
| retention (back to size only) | 0.865 | 9 / 10 |
| observability (TSM) | 0.940 | 10 / 10 |
| terrestrial-with-atmosphere | 0.987 | 9 / 10 |

Habitable-zone membership drives the ranking most. The weights are a stated preference; the ablation shows what changes if they are argued with.

---

## 4. What was wrong with the previous version, and why it changed

The version of 2026-09-25 to 2026-10-16 had four defects, found in the 2026-10-17 audit:

1. **The "temperate" term was a 400 K-wide Gaussian around 300 K**, applied to catalogue equilibrium temperatures from 45 different papers. A 600 K planet scored 0.76. There was no habitable-zone physics.
2. **20 % of the weight came from Stage B molecule probabilities.** Stage B's own conclusion is that the model has no skill above a constant, and three planets scored ≥ 0.5 for molecules that the literature records as *not* detected.
3. **Ten planets that do not transit were given transmission-observability scores.** GJ 1002 b ranked 11th.
4. **The planet list was read from the excluded `05_final_ML_dataset_DO_NOT_USE/` folder.** It now lives in `stage_c_priority_fusion/target_list.csv`.

The molecule model is no longer an input. Literature detections and non-detections from `outputs/labels/verified_gas_labels.csv` are shown in `lit_detected` / `lit_not_detected` for information.

---

## 5. Vetting scores of the 54 (informational — not in the ranking)

All 54 are confirmed planets, so the vetting models' scores on their TESS light curves are a **sanity check on the vetters**, not evidence about the planets. 44 have a complete ephemeris. Of those, 41 produced views; 2 were excluded for an unreliable ephemeris and 1 has no TESS light curve.

**Updated 2026-10-18 to the final v2 models.**
- The scorer is `score_priority_targets.py --with-errors --tag-suffix _v2 --runs …`.
- The views were rebuilt with the flux-error channel in `detection_views_err/stage_c_targets/`.
- The planet list now comes from `target_list.csv`. The script used to read it from the excluded `DO_NOT_USE` folder; that has been fixed.

| | Kepler-trained ensemble (cross-mission) | TESS-trained ensemble |
|---|---|---|
| Model | 5-seed CNN+BiLSTM + error channel | 5-seed dual CNN |
| Scored ≥ 0.5 | 71 % | 98 % |
| Median planet-like score | 0.70 | 0.83 |
| Spearman with planet radius (41 planets) | **−0.37** | **+0.35** |
| v1 single models, for comparison | 71 %, ρ = −0.57 | 93 %, ρ = +0.30 |

- **The opposite size biases survive the change of model and split.** The Kepler-trained ensemble still rejects the canonical hot Jupiters: HD 189733 b scores 0.09, KELT-9 b 0.23 and WASP-17 b 0.31.
  - The cause is measured in its training data. In the v2 Kepler training split, 94 % of KOIs deeper than 10,000 ppm are false positives, against a 50 % base rate.
  - The TESS training split shows the opposite: 12 % against 17 % (`figures/07`).
- **The Kepler bias is weaker in v2** (ρ −0.37 vs −0.57).
- **Small planets still score lowest with the TESS ensemble.** The TRAPPIST-1 planets score 0.54–0.58, and LHS 475 b, at 0.46, is the only one below 0.5. These are shallow transits in TESS photometry.
- Both columns are kept in the table (`vetting_score_*_model`, now read from the `_v2` files). Neither feeds the ranking. Rebuilding the table changed only these two columns.

---

## 6. Limitations

- **The weights are a preference, not a measurement.** The ablation (§3) and the rank intervals are how the ranking should be read.
- **HZ boundaries depend on the climate-model assumptions** of Kopparapu et al.: a 1 M⊕ planet, an N₂–CO₂–H₂O atmosphere, no clouds. They are *not* habitability.
- **The TSM assumes a cloud-free atmosphere** of fixed composition per size bin. Clouds or a high mean molecular weight can suppress the signal by large factors.
- **Fifty-four planets are a mentor-chosen list**, not a complete or unbiased sample. Bands and ranks are relative to this list only.

## 7. Reproducing

```bash
# optional, informational vetting columns (section 5): fetch the 54 TESS light curves and score them
R=stage_a_transit_model/runs
python stage_c_priority_fusion/score_priority_targets.py --with-errors --tag-suffix _v2 \
       --runs $R/v2_tess_dual_cnn $R/v2_tess_dual_cnn_s1 $R/v2_tess_dual_cnn_s2 $R/v2_tess_dual_cnn_s7 $R/v2_tess_dual_cnn_s11
python stage_c_priority_fusion/score_priority_targets.py --score-only --with-errors --tag-suffix _v2 \
       --runs $R/v2_kepler_dual_cnn_lstm_err $R/v2_kepler_dual_cnn_lstm_err_s1 $R/v2_kepler_dual_cnn_lstm_err_s2 \
              $R/v2_kepler_dual_cnn_lstm_err_s7 $R/v2_kepler_dual_cnn_lstm_err_s11
python stage_c_priority_fusion/build_priority_table.py
python -m pytest tests/test_target_physics.py
```
