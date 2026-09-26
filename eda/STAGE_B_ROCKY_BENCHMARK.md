# Rocky-planet benchmark: current detection limits for rocky-exoplanet atmospheres

**What this measures.** How large an atmospheric absorption band the existing JWST transmission spectra of rocky planets could have detected, and how large a band they rule out. The answer is given in ppm and in atmospheric scale heights.

**What it is not.** This is not a classifier and not a biosignature search. No rocky-planet atmosphere has a confirmed detection (Kreidberg & Stevenson 2025: "no atmospheres have been definitively detected"), so there are zero positive labels. Nothing here feeds the Stage B classifier. `outputs/labels/verified_gas_labels.csv`, `outputs/stage_b/`, the Kepler build, Stage 1, PyATMOS/INARA and `05_final_ML_dataset_DO_NOT_USE/` were not touched.

**Result in one paragraph.**
- **Sample:** 9 rocky planets with 10 spectra (LHS 1140 b has two instruments). TRAPPIST-1 b and h are excluded because their provenance could not be confirmed.
- **Band statistic:** no molecule band is detected in any spectrum. Two "CH4" excesses above 3σ (GJ 1132 b and GJ 486 b) turned out to be the literature's known, reduction-dependent "water or starspots" rise at the blue end of NIRSpec, which the CH4 box happens to overlap. They are not CH4.
- **Sensitivity (μ = 28, an N2 atmosphere):** the best spectra (GJ 1132 b, GJ 486 b, L 98-59 c) reach about 1–2 scale heights of band precision. So they could see a strong (~5 H) band in a clear N2 atmosphere, and their 3σ upper limits reach 4–6 H. The worst (TOI-836 b, LHS 1140 b/c) sit at 4–8 H per band, where only implausibly large features would show.
- **The real limit is systematic.** Across reductions and visits of the same planet, depths shift by about 15 ppm and band significances by about 1σ. That is comparable to one to four scale heights.

---

## 1. Prerequisite check

- **Level 2 code:** none exists. The Stage B "Level 2" idea (scale-height normalisation plus band-index features) appears only as a paragraph in `exoplanet_research_data/06_docs/ML_GUIDE.md` and in my earlier feasibility note. There is no code to reuse, so the statistic is implemented here in `stage_b_rocky_benchmark/rocky_benchmark.py`.
- **Band definitions:** the ones specified for this task, **not** Stage B's `BANDS`:
  - H2O: 2.5–2.9 µm
  - CH4: 2.2–2.4 and 2.6–3.5 µm
  - SO2: 3.85–4.05 µm
  - CO2: 4.05–4.55 µm
  - CO: 4.35–4.85 µm
- **Bands that overlap:** CH4 overlaps H2O at 2.6–2.9 µm, and CO2 overlaps CO at 4.35–4.55 µm. Tests on those pairs are not independent, and the CH4/H2O overlap drives the finding in §6.

## 2. Parsing the Zenodo files (Steps 1–5)

Code: `stage_b_rocky_benchmark/zenodo_io.py`. Output: `outputs/rocky_benchmark/zenodo_parse_index.csv`. The Stage B `parse_tbl` / `select_spectra` / `resample` functions are reused unchanged.

- **Units.** Every file declares ppm, so no assumption was needed that they match the `.tbl` files (those are in %). I confirmed ppm numerically: each file's median depth is **0.88–1.35×** the catalogue transit depth, which rules out % (would be 10⁻⁴×) and fraction (10⁻⁶×). All are normalised to ppm; the `.tbl` depths used in the cross-check are converted from %.
- **LHS1140b-NIRSpec.txt.** Columns 1–2 are bin **start and end**, not centre and half-width. This is detected generically: column 2 is about the size of a wavelength, and every row has end > start. Converted to centre = (start + end)/2 and half-width = (end − start)/2.

  | row | start | end | → centre | half-width |
  |---|---|---|---|---|
  | 1 | 1.6652 | 1.6908 | 1.6780 | 0.0128 |
  | 2 | 1.6908 | 1.7168 | 1.7038 | 0.0130 |
  | 3 | 1.7168 | 1.7433 | 1.7301 | 0.0132 |

  The centres run 1.678–5.150 µm, inside the G235H and G395H ranges. The median resolving power λ/Δλ is **66**, against the file header's "R = 65". The G235H and G395H segments overlap at 2.88–3.07 µm (one wavelength restart).
- **GJ486b.txt.** The 21 comment lines (the pasted CDS journal-table header) are stripped. The remaining 110 rows all parse with exactly 4 numeric columns. Every file is checked for 4 columns on every row and fails loudly otherwise.
- **Zero half-widths (GJ1132b.txt, L98-59c.txt).** Each half-width is replaced with half the mean spacing to neighbouring centres and flagged (`half_width_derived`). This cannot cause a division by zero: binning assigns points to grid bins by centre and weights them by 1/σ², so widths never appear in a denominator. Points with non-positive errors are dropped, because they would give infinite weights; none exist. The derived widths give R ≈ 101 and 205, matching the other G395H files.
- **All 12 files parse.**

| File | Planet | Instrument (from local match) | λ range (µm) | Points | 1–5 µm grid bins |
|---|---|---|---|---|---|
| GJ1132b | GJ 1132 b | NIRSpec G395H/M | 2.77–5.16 | 64 | 53 |
| GJ341b | GJ 341 b | NIRCam | 3.91–4.99 | 55 | 28 |
| GJ486b | GJ 486 b | NIRSpec G395H | 2.87–5.16 | 110 | 52 |
| L98-59c | L 98-59 c | NIRSpec G395H | 2.87–5.07 | 106 | 52 |
| L98-59d | L 98-59 d | NIRSpec G395H | 2.88–5.15 | 56 | 49 |
| LHS1140b-NIRISS | LHS 1140 b | NIRISS SOSS | 0.65–2.65 | 142 | 42 |
| LHS1140b-NIRSpec | LHS 1140 b | NIRSpec G235H+G395H | 1.68–5.15 | 76 | 66 |
| LHS1140c | LHS 1140 c | NIRISS SOSS | 0.65–2.65 | 142 | 42 |
| LHS475b | LHS 475 b | NIRSpec G395H | 2.83–5.14 | 56 | 51 |
| TOI836b | TOI-836 b | NIRSpec G395H | 2.87–5.07 | 106 | 52 |
| TRAPPIST1b | TRAPPIST-1 b | **unconfirmed** (no local match) | 0.71–2.81 | 50 | 46 |
| TRAPPIST1h | TRAPPIST-1 h | **unconfirmed** | 0.69–5.38 | 87 | 76 |

All 12 overlap the 1–5 µm grid. **None is MIRI-only.** This differs from the earlier Stage B run, where 11 of 64 spectra fell outside the grid.

## 3. Data-provenance validation (Steps 6–8)

Each Zenodo file was compared with every local JWST `.tbl` spectrum of the same planet. For each pair, the finer spectrum is binned onto the coarser one's bins, depths are compared in units of the per-point error, and all files from the same paper are also tried in combination. Output: `outputs/rocky_benchmark/zenodo_vs_local_matches.csv` and `zenodo_best_match.csv`.

| Zenodo file | Local counterpart | Verdict |
|---|---|---|
| GJ341b | `GJ_341_b_..._5705_1.tbl` (Kirk 2024) | **exact copy** (median diff 0.002σ, r = 1.000) |
| GJ486b | `GJ_486_b_..._5146_1.tbl` (Moran 2023) | **exact copy** (0.000σ) |
| L98-59d | `L_98-59_d_..._5499_1.tbl` (Gressier 2024) | **exact copy** (0.004σ) |
| LHS1140b-NIRSpec | `LHS_1140_b_..._5435_1.tbl` + `_5435_2.tbl` (Damiano 2024) | **exact copy** of the two files together |
| LHS1140c | `LHS_1140_c_..._5422_2.tbl` (Cadieux 2024) | **exact copy** (0.000σ) |
| LHS475b | `LHS_475_b_..._5239_4.tbl` (Lustig-Yaeger 2023) | **exact copy** (0.001σ) |
| TOI836b | 5 Alderson 2024 files combined | **combination** (0.17σ, r = 0.965) |
| L98-59c | 4 Scarsdale 2024 files combined | **combination with a constant −10 ppm offset** (0.20σ after the offset) |
| GJ1132b | closest `_5867_14.tbl` (Bennett 2025); 21 candidates | **same observations, different reduction** (0.51σ, r = 0.84, −11 ppm) |
| LHS1140b-NIRISS | `_5422_1.tbl` (Cadieux 2024) | **different version**: −86 ppm offset, 0.37σ after the offset, r = 0.74. See the reduction-variation notes below. |
| TRAPPIST1b | only local JWST spectrum is Rathcke 2025 NIRSpec PRISM | **not a copy**: −150 ppm offset, r = −0.18, different wavelength coverage |
| TRAPPIST1h | none | no local JWST spectrum |

**Step 7 decision (yours): Zenodo versions throughout.** Each planet gets one expert-curated spectrum, under one citable DOI, and the benchmark is reproducible. The local `.tbl` files are used only for this cross-check and the reduction-sensitivity check (§6). They never contribute benchmark data, so no planet is counted twice.

**What these comparisons say about reduction-to-reduction variation.** These numbers indicate how much of any weak signal depends on the reduction rather than the planet.

- **L 98-59 c, −10 ppm.** A constant offset between the compilation's spectrum and the combination of the four local Scarsdale spectra. This is a clean measure of reduction/combination variation for one planet.
- **Across all planets.** Single local reductions and visits differ from the Zenodo version by a median of about **15 ppm** (GJ 1132 b: −40 to +53 ppm over 21 files; GJ 341 b: −11 to +23 ppm over 12 files; TOI-836 b: ±7 ppm over 5 files).
- **Band significance moves too.** The same band on different reductions or visits of the same planet shifts by a **median 1.1σ** in z (22 planet–molecule pairs with ≥3 local versions; range 0.2–5.2σ). Individual visits are noisier than combined spectra, so this is an upper bound on pure reduction dependence.
- **LHS 1140 b NIRISS, −86 ppm: correction to how this was framed.** The −86 ppm offset should *not* be read as pure reduction variation. The compilation paper states: "The spectra for TRAPPIST-1b and LHS 1140b have been corrected for stellar contamination" (Fig. 3 caption). Cadieux et al. 2024 describe that correction as "dividing out the best-fitting unocculted stellar contamination model". Most of the 86 ppm is therefore most likely the size of the stellar-contamination correction, which is itself a measurement of how strongly the star can imprint on a rocky-planet spectrum. I haven't confirmed that the local `_5422_1` file is the uncorrected version, so this is a strong inference, not a certainty.

**TRAPPIST-1 provenance: both remain EXCLUDED.** Here is what was checked.

1. **The Zenodo record.** There is one version (15084226) and no README or metadata file; the 12 `.txt` files are the only files. The `related_identifiers`, `references` and `notes` fields are empty. The description says only that the spectra were "derived from published or submitted work (see references in the paper)". Each file header is just column names.
2. **The paper (arXiv:2507.00933).** I searched the full text, figure captions, data-availability statement and reference list. There is no table of per-planet data sources.
   - *TRAPPIST-1 b:* the paper says only that its spectrum is contamination-corrected ("The contamination-corrected featureless spectrum of TRAPPIST-1b in Figure 3…"). Its one JWST TRAPPIST-1 b transmission reference, [59] Lim et al. 2023 (NIRISS), is cited for statements about stellar contamination, not as the source of the plotted spectrum.
   - *TRAPPIST-1 h:* the only occurrence in the paper is the title of reference [13], Garcia et al. 2022, "HST/WFC3 transmission spectroscopy of the cold rocky planet TRAPPIST-1h". HST/WFC3 covers about 1.1–1.7 µm, so it cannot have produced the file's 0.69–5.38 µm spectrum. The only cited source is inconsistent with the data.
3. **Wavelength range alone** would suggest NIRISS for b and PRISM for h. Per the instructions, that was not treated as provenance.

To resolve either planet, compare the file point by point with a candidate source's published data (e.g. the Lim et al. 2023 NIRISS spectrum), or ask the authors. That would be a new download or contact, so it's your decision.

**Final benchmark sample: 9 planets, 10 spectra.** The task text expected "10 if both TRAPPIST-1 planets stay excluded". That counts spectra; the planet count is 9, because 11 planets minus 2 excluded leaves 9, and LHS 1140 b contributes two spectra.

## 4. Literature upper limits (Steps 9–11)

**All 11 compilation planets:** GJ 1132 b, GJ 341 b, GJ 486 b, L 98-59 c, L 98-59 d, LHS 1140 b, LHS 1140 c, LHS 475 b, TOI-836 b, TRAPPIST-1 b, TRAPPIST-1 h.
- **Already reviewed** during the 13-planet label work and excluded for having no per-molecule result: GJ 1132 b, GJ 486 b, LHS 475 b, TRAPPIST-1 b. Carried over, not re-checked.
- **Unreviewed (7):** GJ 341 b, L 98-59 c, L 98-59 d, LHS 1140 b, LHS 1140 c, TOI-836 b, TRAPPIST-1 h.
- **Checked (6):** all the unreviewed planets except TRAPPIST-1 h, whose spectrum has no identifiable source paper.

Saved to **`outputs/rocky_benchmark/rocky_planet_upper_limits.csv`** (37 rows). This is a separate file and is never merged with `verified_gas_labels.csv`. Numbers were read from arXiv; check them against the journal PDFs before citing (a `verify` column records this).

| Planet (source) | Per-molecule constraints found |
|---|---|
| **LHS 1140 b** (Cadieux+2024, 2406.15136) | log H2O < −2.94 and log CH4 < −2.78 (**2σ**, cloud-free). Pure CO2 needs T < 233 K (2σ), a conditional limit. H2-rich atmospheres rejected at >10σ. N2 favoured at 2.3σ (Rayleigh scattering, not a target molecule). Damiano+2024 (NIRSpec) gives no numerical limits and says a ~20 ppm CO2 feature at 4.2 µm isn't yet detectable. |
| **LHS 1140 c** (Cadieux+2024) | log CH4 < −2.25 (**2σ**). Flat line favoured at 2.1σ. NIRISS stops at 2.8 µm, so CO2, CO and SO2 aren't covered. |
| **L 98-59 d** (Gressier+2024, 2408.15855) | Free-chemistry upper limits: log H2O < −5.0, log CH4 < −6.0, log CO2 < −2.5, log CO < −2.5. **The sigma convention isn't stated** in the retrieved text. SO2 is unconstrained (log −5.64, −4.12/+4.41). |
| **L 98-59 c** (Scarsdale+2024, 2409.07552) | Pure CH4 ruled out at 2.8σ; clear CH4 disfavoured at 3.2σ. CO2 fractions ≲10% ruled out at 3σ, which excludes *low*-CO2 H2 mixtures and is **not** an upper limit on CO2. Metallicities ≲300× solar ruled out at 3σ. |
| **GJ 341 b** (Kirk+2024, 2401.06043) | 1-bar CH4 atmospheres disfavoured at 1–3σ, depending on the reduction. Water-dominated atmospheres allowed. Metallicities <350× solar ruled out at ≥3σ. |
| **TOI-836 b** (Alderson+2024, 2404.00093) | **No per-molecule limit.** Bulk constraint only: metallicities <250× solar ruled out at a 0.1 bar opaque level (mean molecular weight ≲6). |

**Positive entries: none.** As expected, no source claims a molecule detection on any rocky planet. Seven rows are explicit numerical per-molecule upper limits (`nondetection_equivalent = 0`): 4 for L 98-59 d, 2 for LHS 1140 b and 1 for LHS 1140 c. Everything else is a bulk-composition exclusion or "not stated".

**Two unexpected items** (both recorded, neither a detection):
1. **GJ 341 b.** One reduction's retrieval gives "a weak detection of CO2 (3.1σ)". The authors reject it: the candidate features sit at different wavelengths in different reductions (4.3 vs 4.7 µm), "suggesting that they are not real astrophysical signals."
2. **L 98-59 d.** The spectrum "deviates from a flat line by 2.6 to 5.6σ, depending on the data reduction and retrieval setup". The signal is carried mostly by H2S, which is not one of the five targets. The paper's title presents it as "Hints". So the upper end of the range exceeds 3σ, but it depends on the reduction and retrieval, and there is an independent analysis (Banerjee+2024, 2408.15707).

## 5. Scale heights and theoretical detectability (Step 12)

**Formulas.** H = k·T_eq / (μ·m_u·g), with g = G·M/R², and the feature amplitude per scale height is A_H = 2·R_p·H / R_s². Parameters come from `pscomppars`. Output: `outputs/rocky_benchmark/rocky_scale_heights.csv` and `rocky_noise_vs_scale_height.csv`.

**Mean molecular weight.**
- **Primary μ = 28 (N2-dominated, Earth-like) for every planet.** It is the value Kreidberg & Stevenson use to put these same spectra into scale heights (their Fig. 3), so the numbers are directly comparable. One common value also keeps the planets comparable with each other.
- **Bracket:** μ = 18 (steam, the most favourable high-μ case) and μ = 44 (CO2, Venus-like, the least favourable). Hydrogen-dominated μ ≈ 2.3 is not a rocky-planet assumption, and it is already ruled out for most of these planets (§4).
- L 98-59 d's suggested sulfur-rich atmosphere (H2S, μ = 34) falls inside the bracket.

**Mass caveats.**
- **GJ 341 b:** the catalogue's 4.0 M⊕ is flagged as an **upper limit**, and taken literally it would imply about 32 g/cm³ at 0.88 R⊕. I used an Earth-like rocky mass–radius relation, M = R^3.7, which gives 0.62 M⊕. This is flagged in the CSV. Using the upper limit would shrink A_H by 6.4×.
- **LHS 475 b:** mass from the catalogue's own mass–radius relation.

| Planet | T_eq (K) | 1 H (ppm), μ = 18 / 28 / 44 | Median per-point σ (ppm) | σ in H (μ = 28) |
|---|---|---|---|---|
| GJ 1132 b | 584 | 13.6 / **8.8** / 5.6 | 26 | **3.0** |
| L 98-59 d | 416 | 13.6 / **8.7** / 5.6 | 44 | 5.1 |
| GJ 486 b | 696 | 6.3 / **4.1** / 2.6 | 21 | 5.1 |
| L 98-59 c | 526 | 7.7 / **4.9** / 3.1 | 26 | 5.2 |
| LHS 475 b | 586 | 9.7 / **6.2** / 4.0 | 33 | 5.4 |
| GJ 341 b | 629 | 3.3 / **2.1** / 1.4 | 17 | 8.0 |
| LHS 1140 b (NIRSpec) | 226 | 5.6 / **3.6** / 2.3 | 31 | 8.6 |
| LHS 1140 c | 422 | 12.1 / **7.8** / 4.9 | 77 | 9.9 |
| LHS 1140 b (NIRISS) | 226 | 5.6 / **3.6** / 2.3 | 45 | 12.7 |
| TOI-836 b | 871 | 2.7 / **1.7** / 1.1 | 25 | **14.7** |

**One scale height is 1.7–8.8 ppm (μ = 28), while each point is uncertain by 17–77 ppm.** So no single point can show an atmosphere; a signal would only appear once many points are averaged over a band. After that averaging (§7), the best spectra reach about 1–2 H of band precision, which is enough in principle for a strong band in a clear atmosphere. The worst stay above 4 H.

## 6. Band statistic and comparison with the literature (Steps 13–14)

**The test.** For each spectrum and molecule, model a flat local continuum plus a box offset δ inside the band.
- The continuum is every point outside all bands and within 0.5 µm of the tested band.
- δ is the weighted-mean depth in the band minus the weighted-mean continuum depth. Its uncertainty σ_δ comes from the per-point errors, and Δχ² = (δ/σ_δ)².
- **Significance** is z = δ/σ_δ, signed so that positive means extra absorption. It is quoted after inflating the errors by √(χ²_red) wherever the local scatter exceeds the quoted errors.
- A test needs at least 3 band points and 3 continuum points. The 3σ upper limit is max(δ, 0) + 3σ_δ.
- **Validation:** injecting a box of 5σ into every one of the 35 evaluable tests recovers 4.97–5.03σ.

Outputs: `rocky_band_statistics.csv` and `rocky_reduction_sensitivity.csv`.

**Not evaluable.** H2O cannot be tested for any NIRSpec G395H spectrum: they start at 2.77–2.88 µm, so the 2.5–2.9 µm box has almost no points and no continuum within 0.5 µm. The NIRISS spectra can't test SO2, CO2 or CO. GJ 341 b (NIRCam, 3.9–5.0 µm) can test only CO2 and CO. That leaves 35 of 50 spectrum–molecule tests evaluable.

**Significance z** (error-inflated), and **3σ upper limit in scale heights** (μ = 28) in brackets:

| Spectrum | H2O | CH4 | SO2 | CO2 | CO |
|---|---|---|---|---|---|
| GJ 1132 b | — | **3.39** [6.7] | −0.96 [4.4] | 0.10 [5.0] | −0.76 [7.4] |
| GJ 486 b | — | **3.14** [10.2] | 0.83 [7.5] | 0.23 [5.6] | −0.37 [8.1] |
| L 98-59 c | — | 0.39 [5.2] | −0.75 [6.4] | 1.49 [8.1] | 0.46 [11.2] |
| L 98-59 d | — | −1.15 [8.2] | 0.77 [10.4] | 0.12 [7.7] | 1.73 [19.1] |
| LHS 475 b | — | −0.37 [6.3] | −0.95 [8.0] | −1.05 [8.2] | −0.64 [12.4] |
| GJ 341 b | — | — | — | −0.39 [12.5] | −0.01 [13.1] |
| LHS 1140 b NIRSpec | 0.66 [17.5] | 1.29 [14.8] | −0.95 [18.9] | 0.46 [28.2] | 0.72 [34.9] |
| LHS 1140 b NIRISS | −1.29 [31.2] | 2.28 [35.9] | — | — | — |
| LHS 1140 c | 1.54 [44.2] | −0.57 [16.5] | — | — | — |
| TOI-836 b | — | −0.11 [13.0] | −0.13 [17.0] | −0.18 [18.7] | 1.75 [50.5] |

**The two results above 3σ were investigated before reporting, as the bug-check clause requires** (`diagnose_high_sigma.py`, `rocky_high_sigma_diagnostics.txt`). Both are GJ 1132 b and GJ 486 b in the CH4 box. They are not CH4, and they are not a bug in the statistic:
- **Location.** The excess is broad and strongest at the blue edge: in the 2.6–3.0 µm sub-band, z = 3.59 (GJ 1132 b) and 3.48 (GJ 486 b). In the CH4 3.3 µm band core (3.2–3.45 µm) it is below 3σ: 2.75 and 2.38. Binned in 0.1 µm steps, depth rises steadily toward the blue from about 3.5 µm.
- **Shape of the continuum.** For G395H, the CH4 continuum lies only on the red side (3.5–3.85 µm), so a slope looks like band absorption. Measured against a linear continuum fitted to all out-of-band points, both fall below 3σ: 2.78 and 2.96.
- **Reduction dependence.** GJ 486 b reaches z = 3.55 in only 1 of 3 reductions; the other two give 0.64 and 0.71, and the Zenodo file is a copy of the high one. For GJ 1132 b, none of the 21 local reductions or visits exceeds 2.28, and only the compilation's own reduction reaches 3.39.
- **Interpretation.** A rise at the blue end of NIRSpec (λ < 3.5 µm) is exactly the published ambiguous signal for both planets: "water vapor … source is unclear" and "water in planet vs unocculted starspots indistinguishable". The CH4 box (2.6–3.5 µm) overlaps the 2.7 µm water region, so the signal lands in "CH4".
- **Verdict:** the statistic **recovers the literature's weak, contamination-ambiguous signals at a similar level**, and mislabels the molecule because of the band definition. There is no new detection. With H2O untestable in G395H, these two rows are best read as "blue-end excess (water or starspots)".

**Comparison with the literature's hints:**
- **L 98-59 d, sulfur gases.** SO2 box z = 0.77 (δ = 18 ± 24 ppm); nothing significant. The published signal is carried mostly by H2S (not tested) and ranges from 2.6 to 5.6σ depending on the reduction and retrieval. The box statistic doesn't recover it, which is consistent with it being a weak, model-dependent result rather than a strong band.
- **GJ 1132 b and GJ 486 b, water.** The H2O box is not testable here, but the blue-end excess above reproduces the published hint at about 3σ, including its dependence on the reduction.
- **LHS 1140 b, nitrogen.** N2 has no band in 1–5 µm, so this can't be tested; the hint comes from Rayleigh scattering. The NIRISS "CH4" box gives z = 2.28, consistent with the paper's "tentative evidence of residual spectral features". In the local NIRISS version (`_5422_1`, likely uncorrected), the same box gives **3.69**, compared with 2.84 for the Zenodo version before error inflation. The correction lowers the band significance by about 0.9σ and moves it from above 3σ to below it, which is direct evidence that uncorrected stellar contamination can produce a fake >3σ band.
- **GJ 341 b, CO2.** The Zenodo version gives z = −0.39, but one of the 12 local reductions (`_5705_5`) gives **z = 3.14**. That reproduces the authors' rejected, reduction-dependent "CO2 at 3.1σ", which is a useful check that the statistic behaves the way their analysis did.
- **One more outlier, not in the benchmark.** A single LHS 475 b local file (`_5239_2`) gives CO at z = 9.5. The cause is an instrumental trend of about 470 ppm across 3.5–5.3 µm in that one visit. It is not an atmosphere; for scale, one scale height is 6.2 ppm. The benchmark's Zenodo version is flat (CO z = −0.64).

## 7. What sensitivity would detect an Earth-like atmosphere? (forward-looking result)

This is the result the data can actually support. For a band spanning N_H scale heights to be detected at 5σ, the band precision must reach σ_δ ≤ N_H · A_H / 5. The tables compare that with the precision in hand, with the improvement expressed as a multiple of the current data volume. That multiple assumes photon-limited noise; because the systematics floor in §3 (about 15 ppm and about 1σ between reductions) doesn't average down, **treat these as lower bounds**. Output: `outputs/rocky_benchmark/rocky_sensitivity_requirements.csv`.

**Assumption, stated plainly:** a strong molecular band in a clear, high-μ atmosphere spans about 2–5 H. Clouds, hazes and refraction push real features toward the low end. This range is a benchmark convention, not a measurement.

**The CO2 4.05–4.55 µm band.** It is the strongest band expected in N2/CO2 secondary atmospheres and is covered by every NIRSpec or NIRCam spectrum.

| Spectrum | CO2 band σ (ppm) | σ in H (μ = 28) | Data × for a 5 H band at 5σ | Data × for a 2 H band | 5 H band, μ = 18 / μ = 44 |
|---|---|---|---|---|---|
| GJ 1132 b | 14.1 | 1.6 | ×3 | ×16 | ×1 / ×6 |
| GJ 486 b | 7.1 | 1.7 | ×3 | ×19 | ×1 / ×8 |
| L 98-59 c | 8.9 | 1.8 | ×3 | ×20 | ×1 / ×8 |
| L 98-59 d | 21.6 | 2.5 | ×6 | ×38 | ×3 / ×15 |
| LHS 475 b | 17.0 | 2.7 | ×8 | ×47 | ×3 / ×19 |
| GJ 341 b | 8.9 | 4.2 | ×17 | ×109 | ×7 / ×43 |
| TOI-836 b | 10.7 | 6.3 | ×39 | ×245 | ×16 / ×96 |
| LHS 1140 b (NIRSpec) | 29.1 | 8.2 | ×66 | ×415 | ×27 / ×164 |

**What this means for Earth-like signatures in these systems:**
- **Hot, small M-dwarf planets (GJ 1132 b, GJ 486 b, L 98-59 c).** With 1 H = 4–9 ppm, a clear N2/CO2 atmosphere with a strong CO2 band would already be marginally detectable. Three times the current data would detect a 5 H band at 5σ, and 16–20 times would detect a weak 2 H band. The current 3σ limits of about 5–6 H already disfavour strong clear-atmosphere CO2 bands on these planets, which agrees with the literature's "flat line / no thick atmosphere" results.
- **LHS 1140 b, the habitable-zone target most relevant here.** At T_eq = 226 K and 5.6 M⊕, one scale height at μ = 28 is only **3.6 ppm**. A CO2 band at 5σ needs about **66×** the current NIRSpec data volume for a strong band, or about **415×** for a modest one. That is consistent with Damiano+2024's estimate that a ~20 ppm CO2 feature would need many more transits. Cold, massive habitable-zone planets are the hardest case, and the band statistic confirms it quantitatively.
- **TOI-836 b** (large K-dwarf host, 1 H = 1.7 ppm) and **GJ 341 b** (1 H = 2.1 ppm) need roughly 17–245× more data. These targets are effectively out of reach for atmosphere detection by band depth.
- **The systematic floor sets the practical limit.** Band significance moves by about 1σ between reductions, and one visit's instrumental trend reached about 470 ppm. For the best targets, the limit is therefore how stable reductions and stellar-contamination corrections are, not photon noise. Averaging more transits alone won't reach the 2 H regime unless the ~10–15 ppm reduction-to-reduction variation is controlled. For TRAPPIST-1-like hosts, the contamination correction alone is the largest term (the compilation's LHS 1140 b correction is of order 86 ppm, about 24 H).
- **Biosignature pairs (O2/O3 with CH4) are out of scope.** O2 has no strong 1–5 µm band, and O3's strongest band (9.6 µm) is outside this grid. Nothing here speaks to them, except that the CH4 limits above (6–36 H) are far above what an Earth-like CH4 abundance would produce.

## 8. What this benchmark can and cannot support

**It can support:**
- A quantitative statement of current detection limits: best-band precision of about 1–7 H and 3σ band limits of about 4–50 H at μ = 28, planet by planet.
- A validated, reproducible dataset: 10 spectra from one DOI, each tied to its archive counterpart.
- A measured reduction-to-reduction systematic of about 15 ppm and about 1σ in band significance, plus a stellar-contamination correction of order 86 ppm (LHS 1140 b).
- Evidence that simple band statistics reproduce the literature's tentative signals and their reduction dependence.

**It cannot support:**
- Any detection claim.
- Any per-molecule presence/absence label: no positives, and the 7 literature non-detections are conditional on each paper's retrieval assumptions.
- Any classifier.
- Conclusions about TRAPPIST-1 b or h.
- Molecule attribution in regions where the band boxes overlap (CH4/H2O at 2.6–2.9 µm, CO2/CO at 4.35–4.55 µm).

## Files

| File | Contents |
|---|---|
| `stage_b_rocky_benchmark/zenodo_io.py`, `run_parse_and_match.py` | Parser and cross-match (Steps 1–6) |
| `stage_b_rocky_benchmark/rocky_benchmark.py` | Scale heights, band statistic, injection-recovery, reduction sensitivity (Steps 12–13) |
| `stage_b_rocky_benchmark/diagnose_high_sigma.py` → `rocky_high_sigma_diagnostics.txt` | Investigation of the >3σ results (Step 14) |
| `stage_b_rocky_benchmark/build_rocky_upper_limits.py` → `rocky_planet_upper_limits.csv` | Literature constraints (Steps 9–11) |
| `stage_b_rocky_benchmark/sensitivity_requirements.py` → `rocky_sensitivity_requirements.csv` | §7 |
| `stage_b_rocky_benchmark/plot_detection_limits.py` → `eda/rocky_planet_detection_limits.png` | Step 16 figure |
| `outputs/rocky_benchmark/*.csv`, `rocky_benchmark_config.json` | All tables and settings |

**Figure (`eda/rocky_planet_detection_limits.png`).** Each spectrum is plotted in scale heights (μ = 28) on one shared y-axis, ordered from best to worst noise. The blue envelope is the per-point ±1σ noise; the orange strip is the assumed 2–5 H size of a strong band in a clear N2 atmosphere; molecule bands are labelled bars in a neutral track. Even for the best spectrum, the per-point noise (±3 H) is as wide as the whole expected feature, and for the worst (±15 H) the feature is lost in the noise.
