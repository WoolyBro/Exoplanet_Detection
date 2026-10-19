# The prospective molecule register

**Protocol:** `stage_b_atmospheres/MOLECULE_REGISTER_PROTOCOL.md` (commit `d7f1f6b`, before any score).
**Code:** `stage_b_atmospheres/molecule_register.py` (commit after the protocol).
**Register:** `outputs/stage_b_register/molecule_register.csv`.
**SHA-256:** `66c65066e4397b045224011847337eb91dbd23e0703024403df8fdca2ac653fb`.
**Check:** `python tools/register_check.py` recomputes the hash and scores any entry resolved in `outputs/stage_b_register/resolutions.csv`.

> **What it is.** The frozen CH₄ and CO₂ detectors, validated on real JWST spectra, make an independent call on every archive JWST planet whose CH₄ or CO₂ status the literature leaves unresolved:
> - **CH₄:** 20 planets, from a detector that ranks all 41 labelled planets correctly (GBM; CNN 0.967);
> - **CO₂:** 17 planets, from one that scores 0.996 (CNN) and 0.982 (GBM) on 34.
>
> **How a call is made.** It needs both models to agree, after calibration on the labelled planets, and the planet's individual spectra not to contradict each other. The calls are locked by hash, to be scored against future JWST papers.
>
> **Result: 19 of 29 planets get a definite call** (P2 held). The calibration beats the base rate under leave-one-out (P1 held): CH₄ log score −0.05 vs −0.61, CO₂ −0.23 vs −0.71.

## Calls

**CO₂ (17 planets):**
- **present (1):** WASP-127 b (P = 0.995 / 1.000);
- **absent (11):** GJ 1214 b, GJ 3090 b, GJ 486 b, K2-18 b, KELT-7 b, L 98-59 b, L 98-59 d, LHS 1140 b, TOI-270 b, TRAPPIST-1 b, TRAPPIST-1 c;
- **ambiguous (5):** HAT-P-30 b, HATS-6 b, NGTS-2 b (its spectra disagree), TOI-270 d (spectra disagree), TOI-3235 b.

**CH₄ (20 planets scored; K2-22 b and WASP-43 b have no CH₄-band spectrum):**
- **present (0);**
- **absent (12):** GJ 486 b, HAT-P-18 b, HAT-P-26 b, KELT-7 b, L 98-59 b, LTT 9779 b, TRAPPIST-1 b, WASP-121 b, WASP-127 b, WASP-15 b, WASP-52 b, WASP-96 b;
- **ambiguous (8):**
  - spectra disagree: HAT-P-30 b, TRAPPIST-1 e, WASP-94 A b;
  - models disagree: Kepler-51 d, TOI-3235 b, WASP-17 b;
  - near the boundary: GJ 1214 b, TOI-1130 b.

Per-planet scores, calibrated probabilities and the spread across spectra are in the CSV.

## The contested planets beside the literature (descriptive, not scored)

| Planet, gas | Published claim | Register |
|---|---|---|
| **K2-18 b, CO₂** | 3σ (Madhusudhan et al. 2023); not robust in the 2025 reanalysis (arXiv:2501.18477) | **absent** (P = 0.13 / 0.09): no detectable 4.3 µm feature in the archive's G395H spectrum |
| **TOI-270 d, CO₂** | 2.9–3.9σ depending on the analysis | **ambiguous**: its 7 spectra range from P = 0.04 to 0.95 |
| **GJ 1214 b, CO₂** | "possible" (3.3–3.6σ for CO₂ + CH₄, Schlawin et al. 2024) | **absent** (P = 0.04 / 0.01) |
| **LHS 1140 b, CO₂** | "possibly N₂-dominated with … CO₂, modest confidence" (Damiano et al. 2024) | **absent** (P = 0.005); agrees with our retrieval (CO₂ < 0.2 %, `RETRIEVAL_HZ_RESULTS.md`) |
| **WASP-127 b, CO₂** | no detection statement; CO₂-L index 2.64 ± 0.39 scale heights (Fu et al. 2025) | **present** (P = 0.995) |
| **HATS-6 b, CO₂** | ln B = 4.24, a "minor result", not a detection | **ambiguous** (P = 0.32 / 0.51) |
| **NGTS-2 b, CO₂** | 1.9–5.4σ depending on the model | **ambiguous** (spectra disagree) |
| **TRAPPIST-1 e, CH₄** | no detection; visit-to-visit stellar contamination | **ambiguous**: the 4 visits range from P = 0.00 to 0.99, the same contaminated-visit pattern found by the retrieval (`FORECAST_RESULTS.md` §4d) |
| **TOI-3235 b, CH₄** | constrained only with a deterministic contamination model; a GP gives no robust constraint (Ko et al. 2026) | **ambiguous**: the GBM says 0.999 and the CNN 0.39 |

**Reading it.**
- **The rule flags the disputed planets as ambiguous.** It marks as ambiguous the planets whose published results depend on analysis choices (TOI-270 d, NGTS-2 b, TOI-3235 b, TRAPPIST-1 e), because their individual spectra or the two detectors disagree. That is the behaviour wanted from an independent screen.
- **Where the register is definite and the literature hedges, it sides with the null.** K2-18 b, GJ 1214 b and LHS 1140 b get "CO₂ absent". These calls will be the register's most informative tests as new JWST data arrive.

## Limits

- **The calls are about detectable features** in the archive's spectra, not about atmospheric composition.
- **Calibration rests on 41 (CH₄) and 34 (CO₂) planets.** The probability bands (≥ 0.8, ≤ 0.2) are coarse by design.
- **Registered before resolution, not before publication.** Some register planets have published claims that were read while labelling. The register is therefore an independent screen of those claims, not a blind prediction for them. It is a prospective test for the outcomes still to come.
