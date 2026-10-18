# Contamination-aware retrievals of LHS 1140 b and TRAPPIST-1 f — pre-registered (critique 5)

**Written:** 2026-10-18, before any retrieval of these planets was run. The commit adding this file is the registration.
**Code:** `stage_b_atmospheres/retrieval_real_general.py`

## Why

Both are habitable-zone targets in the Stage C top 10. Their JWST spectra are public, and claims about them are open:
- **LHS 1140 b.** Cadieux et al. 2024 (NIRISS SOSS) favour an N₂-dominated atmosphere over an H₂-rich one, with a 2.3σ Rayleigh-slope hint. Damiano et al. 2024 (NIRSpec G235H + G395H) also discuss it.
- **TRAPPIST-1 f.** Lim et al. 2026 published six NIRISS SOSS transits. The median depth varies by ~400 ppm between visits.

The method that showed TRAPPIST-1 e's apparent atmosphere to be stellar contamination (visit-by-visit linear contamination marginalised analytically, plus a per-visit χ² check; `FORECAST_RESULTS.md` §4d–4e) is applied here unchanged.

## Data

| Target | Spectra used | Notes |
|---|---|---|
| LHS 1140 b | NIRISS SOSS (Cadieux 2024; 0.65–2.65 µm), NIRSpec G235H (1.68–3.05 µm) and G395H (2.9–5.15 µm) (Damiano 2024) | three separate visits, each with its own contamination and offset |
| TRAPPIST-1 f | NIRISS SOSS visits 1–5 (Lim 2026; 0.73–2.80 µm) | file 6 (no transit time, the smallest errors) is the combined spectrum and is excluded to avoid double counting |

## Model, fixed now

- **Forward model:** petitRADTRANS, isothermal, opaque surface, R = 300 correlated-k opacities. The mass and stellar radius are from pscomppars.
- **Contamination per spectrum:** depth × (1 + a S_spot + b S_fac) + c.
  - PHOENIX templates at the host's T_eff: spots at 0.86 T_eff, faculae at T_eff + 150 K.
  - Priors a, b ~ N(0, 0.5²) and c ~ N(0, 500 ppm²), marginalised analytically.
- **Models (dynesty, nlive 400, dlogz 0.3):**
  - `bare_rock`: R_p only.
  - `n2`: N₂ background with free CO₂, CH₄, H₂O (log VMR U(−10, −0.5)), T ~ U(150, 400) K, log P_s ~ U(−3, 1).
  - `h2`: H₂/He background with the same free gases.
- **Read-outs:**
  - ln B for atmosphere vs bare rock (the better of `n2` and `h2`) and for `n2` vs `h2`;
  - the best-fit χ² change for every spectrum separately.

## Pre-registered rules and expectations

- **R1, the credit rule.** An atmosphere is credited only if ln B(atmosphere vs bare) ≥ 3 **and** the χ² improves in at least ⌈2/3⌉ of the spectra (4 of 5 for f, 2 of 3 for LHS 1140 b). A preference carried by a minority of visits is classified as contamination, as for TRAPPIST-1 e.
- **R2 (LHS 1140 b).** ln B(n2 vs h2) > 0: a clear H₂-rich envelope is disfavoured, as published.
- **R3 (LHS 1140 b).** Atmosphere vs bare rock is inconclusive (|ln B| < 3) once each spectrum has its own contamination.
- **R4 (TRAPPIST-1 f).** No atmosphere is credited under R1. The visit-to-visit depth variations are expected to be absorbed by the per-visit contamination terms.

All results are reported whatever they are.
