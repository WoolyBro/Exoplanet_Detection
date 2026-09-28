# Data boundary: Stage 1 vs Stage 2

This project has two modelling stages. They must not share training data:

- **Stage 1: transit detection.** A CNN+LSTM that finds planet transits in light curves.
- **Stage 2: habitability and atmospheres.** Models that assess atmospheres, for example detecting O₂ and CO₂.

Every data file the project uses is assigned exactly one role in
[`data_manifest.py`](data_manifest.py). That file is the source of truth, and this page explains it.
Scripts load training data only through `load_stage1_catalogs()` or `load_stage2_catalogs()`.
Each loader returns only its own stage's training files and stops with an error if one is missing.
When the manifest is imported, it also checks that no file has two roles.

## What the roles mean

| Role | Meaning |
|---|---|
| **Training** | The model learns from this data. Only these files are returned by the stage's loader. |
| **Supplementary** | Extra information joined onto training rows (for example star or planet properties). It is not a source of training examples or labels by itself. |
| **Validation-only** | Used only to measure how well a finished model works. It is never trained on or used for tuning, so results on it stay honest. |
| **Reference-only** | Used for context, cross-checks or interpreting results (for example "is this planet in a habitable-zone list?"). It is never a model input or a label source. |
| **Excluded** | Must not be used by any stage. |

**Provenance:** each file is also tagged by where it came from.
- **mentor**: from the mentor-approved pack in `exoplanet_research_data/`, or derived only from it.
- **external archive**: obtained directly from the named public archive.

## Files by role

Relative paths are inside this `Research` folder. The `D:/Files/...` paths are raw data kept outside it and only read, never modified.

### Stage 1: Training
| File | Provenance | Role |
|---|---|---|
| `splits/koi_cumulative_split.csv` | mentor | Kepler KOI catalog with harmonized labels and a star-grouped 70/15/15 `split` column; this is the Kepler source of Stage 1 labels. |
| `splits/k2_planets_candidates_split.csv` | mentor | K2 catalog filtered to one row per planet (`default_flag == 1`), with harmonized labels and a star-grouped `split` column. **Scope: K2 light-curve modeling is out of scope** (K2 light curves need pointing-drift correction that the detection pipeline does not do), so this file is used for EDA and statistics only; `preprocessing_pipeline.build_dataset` raises an error for K2 rows. |
| `splits/toi_tess_candidates_split.csv` | mentor | TESS TOI catalog with harmonized labels and a star-grouped `split` column. |

### Stage 1: Supplementary
| File | Provenance | Role |
|---|---|---|
| `exoplanet_research_data/01_candidate_catalogs/pscomppars_confirmed_planets.csv` | mentor | Best-available parameters for every confirmed planet, used to add planet properties to catalog rows. |
| `exoplanet_research_data/01_candidate_catalogs/ps_transiting_default.csv` | mentor | Default published solutions for transiting planets, used for transit parameters such as period, depth and duration. |
| `exoplanet_research_data/03_stellar_parameters/stellarhosts.csv` | mentor | Host-star properties (temperature, radius, mass, gravity), used as star-level context features. |
| `exoplanet_research_data/02_lightcurves/` | mentor | 36 real light curves (22 Kepler Quarter 9, 14 TESS single sectors) with `lightcurve_manifest.csv`, used to test the detection preprocessing on real files and verify per-KOI folding. Not the primary training source: each covers one quarter or sector, so `build_dataset` pulls full-mission light curves via the API by default (`use_local_files_first=False`) to keep data depth consistent across KOIs. |

### Stage 2: Training
| File | Provenance | Role |
|---|---|---|
| `datasets/pyatmos_final.csv` | external archive | PyATMOS simulated atmospheres, one row per simulation run (108,839 runs whose folders exist under `D:/Files/Dataset/`, built by `build_pyatmos_final.py`), with surface-level gas concentrations, fluxes, temperature and pressure. It is used as a reference population of surface compositions rather than to generate spectra from vertical profiles. It replaces `D:/Files/Dataset/run_summary_final.csv`, which was missing all of dir_0 (7,828 runs) and had blank temperature/pressure for 62,040 runs. |
| `D:/Files/inara_earthlike_subset/psg_models.csv` | external archive | INARA/PSG Earth-like planet models (30 rows) with atmospheric composition and planet and star parameters. |

### Stage 2: Validation-only
| File | Provenance | Role |
|---|---|---|
| `exoplanet_research_data/04_atmospheric_spectra/` | mentor | Real observed spectra (JWST/HST/Spitzer) from the mentor's pack, used only to test Stage 2 models on real data after training on simulations. |

### Stage 2: Reference-only
| File | Provenance | Role |
|---|---|---|
| `D:/Files/Habitable World Catalogue.csv` | external archive | Catalogue of potentially habitable worlds, used to sanity-check and interpret Stage 2 rankings, never as labels. |

### Excluded
| File | Provenance | Role |
|---|---|---|
| `exoplanet_research_data/05_final_ML_dataset_DO_NOT_USE/` | mentor | Pre-built ML tables that must not be used (renamed to say so); `role_of()` refuses any path inside it. |

## Derived outputs

These folders are built by scripts from manifest files. They are not manifest entries, so `role_of()` does not recognise them.

| Output | Built by | From | Notes |
|---|---|---|---|
| `outputs/habitability_prepared/train.csv`, `val.csv`, `test.csv`, `label_rule.json` | `prepare_habitability_data.py` | INARA `psg_models.csv` (Stage 2 training, via `load_stage2_catalogs()`) | Adds `habitability_label` from an exploratory **placeholder** rule based on INARA's own O₂/CO₂ quantiles, a derived `star_group_id`, and a host-star-grouped 70/15/15 split. `label_rule.json` records the rule, split sizes and the columns to exclude from features. **Not meaningful for training or evaluation until INARA is expanded**; re-run the script once a larger file is in place. |

## Rules enforced in code

1. A file can hold only one role. A file inside a directory that belongs to another role is rejected when `data_manifest.py` is imported.
2. Loaders return only their stage's **training** files. If a listed file is missing, they stop with an error instead of skipping it.
3. `role_of(path)` rejects files that are unlisted or excluded. `assert_role(path, "STAGE_1_TRAINING", ...)` lets a script refuse data that has the wrong role.
4. In Stage 1 splits, all rows of a host star are in the same split (train/val/test), so no star's light curve can leak between them. See `split_catalogs.py` and `run_log.txt`.

## Not in the manifest (deliberately)

These files exist but have no role. `role_of()` rejects them until someone assigns one:
- `D:/Files/NASA Exoplanet Archive (Kepler KOI DR25).csv`, `D:/Files/NASA Exoplanet Archive (PS  PSCompPars).csv`, `D:/Files/TESS Project Candidates.csv`: external archive archive exports that overlap the mentor pack.
- `D:/Files/psg_models.csv`: a byte-identical duplicate of the INARA file listed above.
- `Downloads/pyatmos_summary.csv`: the full PyATMOS summary (124,314 runs). It is the source of `datasets/pyatmos_final.csv`, which keeps only the runs present on disk.
- `D:/Files/Dataset/run_summary_final.csv`: superseded PyATMOS summary (missing dir_0, incomplete temperature/pressure); do not use.
