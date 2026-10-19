# Which rocky planet should JWST observe next to learn the most about the cosmic shoreline? (pre-registered)

**Written 2026-10-19, before any target was scored.** The commit adding this file is the registration.
**Code:** `stage_c_priority_fusion/shoreline_design.py`.
**Output:** `outputs/stage_c/shoreline_design.csv`, `shoreline_design.json`, `eda/shoreline_design.png`.

## Why

The habitability factor ranks planets by how promising they are. A separate question, for an observatory, is **which observation would teach the most about whether and where rocky planets keep atmospheres**. The fitted retention model makes this answerable. Its posterior over (α, β, γ) says what is uncertain, and each candidate observation is valued by how much it is expected to reduce that uncertainty, per hour of JWST time. This is Bayesian experimental design. Ih et al. 2025 simulate survey strategies for testing the shoreline; here the design is driven by the posterior actually fitted to 23 observed planets.

## Pool (fixed now)

Catalogue planets (`pscomppars_confirmed_planets.csv`) that meet all of these:
- transiting;
- R ≤ 1.8 R⊕;
- K magnitude, transit duration, T_eq and stellar parameters known;
- **no JWST atmosphere verdict yet**, meaning not among the 23 planets of the retention fit.

## Model

**1. Prior knowledge.**
- The retention posterior refitted on the 23 planets (`retention_update.py`; importance samples, same seed).
- Each candidate's shoreline-margin draws (`target_physics.monte_carlo`) and magma flag (T_eq ≥ 1,200 K).

**2. The observation: MIRI 15 µm secondary-eclipse photometry** (the Hot Rocks and Rocky Worlds method).
- **Bare rock:** dayside T = T_eff √(R_s / a) (2/3)^(1/4), zero albedo and no redistribution.
- **Thick atmosphere:** the same with (1/4)^(1/4), full redistribution.
- **Depths:** (R_p / R_s)² B_15µm(T_day) / B_15µm(T_eff). The discriminating signal Δ is the bare-rock depth minus the atmosphere depth.

**3. Noise per visit.**
- σ = σ₀ · 10^(0.2 (K − 8)) · √(2 h / T₁₄).
- σ₀ is fitted, in log space, on the per-visit 15 µm uncertainties of 8 published planets:
  - GJ 357 b, 12.7 ppm (1 visit);
  - GJ 3473 b, 45 ppm combined over 4 visits;
  - LHS 1478 b, 56 ppm (1 visit);
  - TRAPPIST-1 b, 99 ppm over 5;
  - TRAPPIST-1 c, 94 ppm over 4;
  - GJ 3929 b, 21 ppm over 3;
  - TOI-1468 b, 31 ppm over 3;
  - LTT 3780 b, 38 ppm over 2.
- The fit's scatter is reported and propagated as a ×10^(±rms) range on the hours.

**4. Programme per target.**
- **Visits:** n = the visits needed for a 3σ separation, Δ / (σ / √n) ≥ 3, capped at 10.
- **Time:** each visit costs 3 T₁₄ + 1 h.
- **Verdict error:** the reported verdict is wrong with probability ε = Φ(−(Δ√n / σ) / 2).

**5. Value.**
- **Information:** the mutual information I(θ; verdict) in bits, between the noisy binary verdict and the retention parameters θ, computed over the posterior samples.
- **Also reported:** the expected reduction in the posterior variance of β (the shoreline slope), by importance re-weighting on each verdict.
- **Ranking:** **bits per JWST hour.**

## Expectations

| ID | Expectation |
|---|---|
| S1 | The noise model is usable: the 8 calibration planets' per-visit σ are fitted with log-rms ≤ 0.2 dex (a factor ≤ 1.6) |
| S2 | Information concentrates near the line: at least 6 of the top 10 by bits per hour have a fitted P(retain) between 0.2 and 0.8 |
| S3 | The six Rocky Worlds DDT targets without a verdict yet (LTT 1445 A c, TOI-198 b, TOI-1452 b, TOI-771 b, HD 260655 c, TOI-244 b) have a median rank in the top quarter of the pool. This is an independent check against an expert selection |

**Interpretation, fixed now.** If S3 fails, the disagreement is reported as a finding: the expert selection and the information-optimal selection differ, and the reasons are listed (observability, the magma regime, the position on the line).
