# CH₄ round 4: planets published after the models were frozen (pre-registered)

**Written 2026-10-19, before any spectrum below was downloaded or scored.**

**How the planets were found.** The NASA archive's query service was down on 2026-10-19 (cx_Oracle error), and every JWST transmission planet in its 2026-10-18 snapshot has already been considered (rounds 1–3). A literature search on 2026-10-19 looked for JWST transmission spectra published but not in that snapshot, with a public reduced spectrum.

| Planet | CH₄ label (source) | Public spectrum | Scored? |
|---|---|---|---|
| **TOI-1231 b** (temperate Neptune, T_eq ≈ 330 K) | **1**: "strong detection of CH₄ (ln B = 54.5–69.6)" (arXiv:2607.00973; NIRISS SOSS + NIRSpec G395H) | OSF, https://osf.io/mryd6 | yes |
| TOI-5293 A b | 1: "strong evidence for CH₄", ln B = 39.7 in visit 2 (arXiv:2603.16464; NIRSpec PRISM) | none found (MAST raw data only) | no; listed so it is not silently dropped |

**CO₂ for both is "unknown"** under the CO₂ protocol's rule: TOI-1231 b's ln B = 2.9–6.6 depends on the analysis, and TOI-5293 A b's ln B = 3.2 is called a "low-significance detection". Neither is scored for CO₂.

**Model.** The frozen v3 CH₄ GBM, trained on the contamination + haze grid, exactly as in round 3 (`methane_round3.py`). Stage B preprocessing is unchanged, and only spectra with a CH₄ band on the grid are scored.

**Expectation (R4-1).** Each scored CH₄ planet falls on the correct side of the class separation seen in rounds 1–3: CH₄ planets ≥ 0.17, others ≤ 0.08, so the midpoint 0.125 is the boundary. One planet is a confirmation check, not a statistic.
