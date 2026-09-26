# Scoping: Stevenson & Kreidberg JWST rocky-planet compilation (Zenodo 10.5281/zenodo.15084225)

**Bottom line:** the data is real, small, open and easy to parse. It adds 11 rocky planets that aren't in the labelled set. But it can't produce supervised gas labels for this project. By the source paper's own conclusion, **no rocky-planet atmosphere, and so no molecule, has been definitively detected**. Under the verification rule used for the 13 planets (1 = a detection at ≥3σ), no rocky planet can have a positive label for any of the five target molecules. A rocky-only training set would have zero positive examples.

This is a scoping report. Nothing was merged. `outputs/labels/verified_gas_labels.csv`, the Stage B outputs, the Kepler build, PyATMOS/INARA and `05_final_ML_dataset_DO_NOT_USE/` were not touched.

## Step 1: What the record contains

| Item | Value |
|---|---|
| DOI | 10.5281/zenodo.15084225 is the *concept* DOI. It redirects to the only version, record 15084226, published 2025-03-25. |
| Title and authors | "JWST Rocky Exoplanet Spectra", by Stevenson, K. and Kreidberg, L. |
| Access and licence | Open, CC-BY-4.0 |
| Content | **Reduced transmission spectra**: derived from published or submitted work and binned to a common wavelength scale. No raw data, no code. |
| Files | 12 plain-text `.txt` files, **50 KB in total** |
| Planets | 11. LHS 1140 b has two files, one from NIRISS and one from NIRSpec. |
| Companion paper | Kreidberg & Stevenson 2025, PNAS, arXiv:2507.00933 |

## Step 2: Overlap with local holdings

None of the 11 planets is in the verified 13-planet label set. The only rocky planet in that set is TRAPPIST-1 c, and it isn't in this compilation. Against the local 940-spectrum transmission collection (`04_atmospheric_spectra`, which holds 261 JWST transmission spectra for 62 planets):

| Planet | JWST transmission spectra already local | Label status |
|---|---|---|
| GJ 1132 b | 21 (May 2023; Bennett 2025) | **already reviewed, excluded**: the two visits disagree, and the best fit is a flat line |
| GJ 341 b | 12 (Kirk 2024) | never reviewed |
| GJ 486 b | 3 (Moran 2023) | **already reviewed, excluded**: water in the planet can't be told apart from starspots |
| L 98-59 c | 4 (Scarsdale 2024) | never reviewed |
| L 98-59 d | 1 (Gressier 2024) | never reviewed |
| LHS 1140 b | 3 (Cadieux 2024 NIRISS; Damiano 2024 NIRSpec) | never reviewed |
| LHS 1140 c | 1 (Cadieux 2024 NIRISS) | never reviewed |
| LHS 475 b | 4 (Lustig-Yaeger 2023) | **already reviewed, excluded**: no per-molecule limits |
| TOI-836 b | 5 (Alderson 2024) | never reviewed |
| TRAPPIST-1 b | 1 (Rathcke 2025 PRISM) | **already reviewed, excluded**: "bare rock or thin atmosphere" only |
| **TRAPPIST-1 h** | **0** (only HST/WFC3 locally, Gressier 2022) | never reviewed |

- **New to the labelled set:** all 11 planets.
- **New as JWST spectra at all:** only **TRAPPIST-1 h**. Its Zenodo file spans 0.69–5.38 µm in 87 points, which looks like a NIRSpec PRISM spectrum. The record doesn't say which paper it comes from; the PNAS paper's own TRAPPIST-1 h reference is HST/WFC3 (Garcia 2022), which can't reach 5 µm. So its provenance still needs checking before it's used for anything. For the other 10 planets, the compilation is a re-binned copy of data we already hold.
- The value of the compilation is **curation**: one consistent, binned file per planet, chosen by the people who assembled the rocky-planet review. It isn't new observations.

## Step 3: Compatibility with the Stage B pipeline

The format isn't compatible with the `.tbl` files, so it needs a small new parser (about 20 lines). Everything after parsing (resampling, the coverage mask, median subtraction) works unchanged.

| | IPAC `.tbl` (Stage B) | Zenodo `.txt` |
|---|---|---|
| Structure | IPAC table with named columns | whitespace or tab text; `#` comments |
| Wavelength | `CENTRALWAVELNG` (µm) | column 1 (µm) |
| Depth | `PL_TRANDEP` (**%**), fallback `PL_RATROR` | column 3 (**ppm**, already) |
| Errors | asymmetric `ERR1` / `ERR2` | one symmetric 1σ (ppm) |

The parser would need to handle three quirks, all found by reading the files rather than trusting the record's description:
- `LHS1140b-NIRSpec.txt` has **bin start and bin end** in columns 1–2, not centre and half-width. Reading column 1 as the centre would shift every point by half a bin (about 0.02 µm, half a Stage B bin). Its G235H and G395H segments overlap between 2.88 and 3.07 µm.
- `GJ1132b.txt` and `L98-59c.txt` have an all-zero half-width column. That's harmless, because Stage B bins on the centre.
- `GJ486b.txt` has a 20-line CDS header.

Coverage on the 1–5 µm, 100-bin grid is 28–76 of 100 bins (median about 50). That's similar to the real spectra already used (median 45). Most of these spectra are NIRSpec G395H/PRISM, covering 2.8–5.2 µm; the NIRISS spectra of LHS 1140 b/c and TRAPPIST-1 b cover 0.65–2.8 µm.

## Step 4: Download decision

Downloaded, because all three conditions were met:
- (a) all 11 planets are new to the labelled set, and one has a new JWST spectrum;
- (b) the whole record is 50 KB;
- (c) the files are real reduced spectra.

The files are in `datasets/zenodo_15084226_rocky_spectra/`, kept separate from the research pack, with a provenance README. All 12 md5 checksums match.

## Step 5: What it would add

**New rocky planets:** 11, in 12 spectra.

**Gas labels.** The Zenodo record contains none; its files are spectra only. The paper's own summary rules out positive labels:

- "despite these advances, no atmospheres have been definitively detected".
- Hints are "marginally significant" and may reflect stellar contamination.

What the paper says per planet (these are the paper's statements, not yet checked against the primary papers):

| Planet | What the paper reports | Possible label under our rules |
|---|---|---|
| LHS 475 b | featureless; H2-rich atmospheres ruled out | unknown (already reviewed) |
| TRAPPIST-1 b | featureless after stellar correction; consistent with an airless body | unknown (already reviewed) |
| GJ 1132 b | "evidence for water vapor", but the "source is unclear" | unknown (already reviewed: visits disagree) |
| GJ 486 b | water evidence; planet vs starspots unclear | unknown (already reviewed) |
| L 98-59 d | tentative H2S/SO2 "preferred over a flat line" | at most tentative, below ≥3σ, so **not a 1** |
| LHS 1140 b | tentative N2-rich atmosphere; cloudy H2 fits comparably | N2 isn't a target; H2O/CH4/CO2/CO/SO2 probably unknown |
| L 98-59 c, TOI-836 b, GJ 341 b | flat line, "inconclusive" | unknown unless a primary paper gives per-molecule limits |
| LHS 1140 c, TRAPPIST-1 h | not discussed in detail | needs review |

- **Positive labels (1):** **zero are possible** for any of the five molecules on any rocky planet in this compilation, per the paper's summary.
- **Negative labels (0):** only where a primary paper states an explicit per-molecule limit. Four of these planets were already reviewed and none had one. The other seven haven't been reviewed; the same bibcode-verification pass would decide them. Based on the four reviewed and the paper's "flat line / inconclusive" wording, expect few.

**Combined rocky-only sample if integrated:**

| | Planets with spectra | Planets with any known label | Positives for any molecule |
|---|---|---|---|
| Now (verified 13, rocky only) | 1 (TRAPPIST-1 c) | 1 (H2O=0, CH4=0, CO=0) | 0 |
| + this compilation | **12** | 1 + however many of the 7 unreviewed planets yield explicit limits | **0** |

A rocky-only set with no positive examples can't train or evaluate a presence/absence classifier. That's true however many planets are added, as long as the literature reports non-detections. The limit isn't this dataset. It reflects the state of the field: rocky-planet atmospheres haven't been detected yet.

## Recommendation (your decision)

- **For the Stage B classifier: not worth integrating.** It adds spectra but no usable labels.
- **Worth keeping for:**
  1. the benchmark/characterisation framing proposed in `STAGE_B_PRETRAIN_FEASIBILITY.md`, as a rocky-planet section documenting that JWST rocky spectra are consistent with flat lines at the 20–170 ppm noise level;
  2. a per-molecule **upper-limit** analysis rather than a classifier, since non-detections with limits are what the rocky data actually supports;
  3. a check of the TRAPPIST-1 h spectrum's provenance, since it's the only genuinely new observation here.
- If you want the seven unreviewed planets checked for explicit per-molecule limits, that's the same bibcode-verification process used for the 13. Its likely result is a handful of 0 labels and no 1s.
