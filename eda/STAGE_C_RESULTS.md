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
| terrestrial size | 0.30 | P(R_p < 1.5 R⊕) | radius bin of Kempton et al. 2018 |
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

## 3. Ablation — what the order depends on

| Criterion removed | Spearman vs full ranking | Top-10 overlap |
|---|---|---|
| habitable zone | 0.711 | 6 / 10 |
| terrestrial size | 0.867 | 8 / 10 |
| observability (TSM) | 0.960 | 9 / 10 |

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

All 54 are confirmed planets, so the vetting models' scores on their TESS light curves are a **sanity check on the vetters**, not evidence about the planets. These findings, from 2026-09-26, still stand:

- The **Kepler-trained** vetter rejected the canonical hot Jupiters (KELT-9 b 0.06, HD 189733 b 0.12, WASP-19 b 0.17 …). Only 71 % of these confirmed planets scored ≥ 0.5.
  - The cause is measured in its training data: 93.4 % of Kepler KOIs deeper than 10,000 ppm are false positives (base rate 49.9 %).
- The **TESS-trained** vetter, whose training data show the opposite (11.5 % against a 17.2 % base rate), scores 93 % of them ≥ 0.5.
- The two models' biases are opposite across these 41 planets: Spearman with planet radius is −0.570 (Kepler model) vs +0.302 (TESS model).
- Both columns are kept in the table (`vetting_score_*_model`). Neither feeds the ranking.

---

## 6. Limitations

- **The weights are a preference, not a measurement.** The ablation (§3) and the rank intervals are how the ranking should be read.
- **HZ boundaries depend on the climate-model assumptions** of Kopparapu et al.: a 1 M⊕ planet, an N₂–CO₂–H₂O atmosphere, no clouds. They are *not* habitability.
- **The TSM assumes a cloud-free atmosphere** of fixed composition per size bin. Clouds or a high mean molecular weight can suppress the signal by large factors.
- **Fifty-four planets are a mentor-chosen list**, not a complete or unbiased sample. Bands and ranks are relative to this list only.

## 7. Reproducing

```bash
python stage_c_priority_fusion/build_priority_table.py
python -m pytest tests/test_target_physics.py
```
