# Methane classifier — round 3: planets new to the archive since the pack snapshot (pre-registered)

**Written:** 2026-10-18, after the archive refresh (`tools/refresh_archive_spectra.py`) and **before** these planets were labelled or scored.

**Planets.** JWST transmission spectra added to the NASA Exoplanet Archive since the research pack's snapshot, for planets not in any earlier round: **HATS-6 b, TOI-3235 b, TRAPPIST-1 f**. HAT-P-12 b's new spectra (Heinke et al. 2026) belong to a planet already labelled CH₄ = 0 in round 1. They are scored as additional spectra of that planet and reported separately.

**Labels.** Same rules as rounds 1–2 (abstract, then full text). Each label is frozen by SHA-256 in `outputs/stage_b_methane_ext/protocol_hashes.json` before scoring. Unknown stays unknown.

**Model.** The headline v3 GBM, plus v1 for reference, frozen. A spectrum is scored only if the unchanged ≥ 3-bin CH₄-band rule holds; otherwise it is reported as unscorable.

**Analysis.** Per-planet P(CH₄) against the label. With three planets this is a confirmation check, not a statistic: success means each labelled planet falls on the correct side of the class separation seen in rounds 1–2 (v3: between 0.08 and 0.17).
