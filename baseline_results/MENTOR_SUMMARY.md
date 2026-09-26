# Level-1 tabular baseline: summary

A RandomForest trained on catalogue features predicts each transit signal's disposition in two ways:
- **3-class:** false positive, candidate or confirmed.
- **Binary planet-like probability:** candidate or confirmed vs false positive. It is computed as P(candidate) + P(confirmed) from the same model.

The verdict columns `koi_fpflag_*`, `koi_score` and `koi_pdisposition` are removed, as you asked. Unless stated otherwise, all numbers are on the **held-out test split** (Kepler 1,434 KOIs, TESS 1,223 TOIs).

## 1. Kepler result

| Feature set | 3-class accuracy | Binary accuracy | Binary ROC-AUC |
|---|---|---|---|
| **Chosen: all catalogue measurements, including centroid and odd/even diagnostics** | **0.848** | **0.899** | **0.971** |
| Same, but centroid and odd/even diagnostics removed (26 columns) | 0.803 | 0.847 | 0.934 |
| Always guess the most common class | 0.506 | 0.506 | 0.500 |

**Decision: keep the centroid and odd/even diagnostics** (e.g. `koi_dicco_msky`, `koi_dikco_msky`, `koi_fwm_stat_sig`, `koi_bin_oedp_sig`).
- **Why keep them:** they are raw measurements of the light curve and the star's image: how far the light centre shifts during transit, and whether alternating transits have different depths. They are not a human or pipeline judgement, and a model built on real data could compute them itself.
- **Why it matters:** they are the measurements the Kepler Robovetter uses to set its centroid and eclipsing-binary flags (`koi_fpflag_co`, `koi_fpflag_ec`), which we removed. They are worth about 4–5 points of accuracy. If you would rather treat them as too close to the flags, the stricter result is 0.803 / 0.847 / 0.934.

**Other columns also removed** because they carry the verdict:
- `kepler_name`: only filled in for confirmed planets.
- `koi_comment`: the Robovetter's reason codes.
- Vetting dates, vetting status and provenance fields.

A version that keeps the verdict columns reaches 0.927 / 0.986 / 0.997, which confirms that removing them was necessary.

## 2. Star-grouped vs random split

The original starter script splits **rows** at random. Many stars host several KOIs, so a random split can put KOIs from the same star in both training and test data. The model then gets tested partly on stars it has already seen.

I reproduced the original script's protocol on the same catalogue data, with the verdict flags removed and the same 14 features:

| Protocol (same 14 features) | 3-class accuracy | Binary accuracy | Binary ROC-AUC |
|---|---|---|---|
| Original: random row split, rows with any missing value dropped | 0.815 | **0.871** | 0.942 |
| Star-grouped split: every KOI of a star in the same split | 0.753 | **0.818** | 0.913 |

- **The effect:** under the random split, **334 stars appeared in both train and test**. Switching to the star-grouped split lowers binary accuracy from 0.871 to 0.818.
- **What the gap measures:** the original protocol also dropped rows with missing values, so the ~5-point gap combines the two differences. Most of the "mid-0.80s" level seen with the original script comes from this setup, not from the features.
- **Which to trust:** the star-grouped numbers are the honest estimate of performance on stars the model has never seen.

## 3. TESS (TOI) result

| | 3-class accuracy | Binary accuracy | Binary ROC-AUC |
|---|---|---|---|
| **All catalogue measurements** | **0.747** | 0.846 | **0.836** |
| Always guess the most common class | 0.652 | 0.827 | 0.500 |

- **Class imbalance:** 83% of TESS test signals are planet-like, so always guessing "planet-like" already scores 0.827. The **binary accuracy of 0.846 is barely above that**, and on its own says little.
- **The meaningful numbers:** **ROC-AUC 0.836** (how well the model ranks planet-like signals above false positives) and **3-class accuracy 0.747 against a 0.652 baseline**.
- **Harder than Kepler:** the TOI table has fewer and noisier measurements, and no centroid or odd/even diagnostics.

**Disclosed time leak (kept):**
- **The leak:** the transit epoch (`pl_tranmid` and its errors) partly encodes *when* a TOI was found. Older TOIs have had more time for follow-up, so they are more likely to be resolved as confirmed or false positive.
- **Its size:** removing these columns lowers ROC-AUC from 0.836 to 0.819, about 0.02.
- **Why it's kept:** the effect is small and the epoch is a genuine measurement, but the leak is disclosed here.

## 4. Code and data used

- **Rebuilt script:** `Research/tools/train_transit_baseline.py`.
  - Loads the Kepler (`koi_cumulative_split.csv`) and TESS (`toi_tess_candidates_split.csv`) catalogues through `load_stage1_catalogs()`, using the existing star-grouped train/val/test assignment. It never re-splits the data.
  - Run `--toi` for TESS, `--features mentor` for the original 14 or 10 features, and `--include-leaky` only as a control.
- **Model:** RandomForest, 300 trees, `class_weight="balanced_subsample"`, `random_state=42`, the same settings as the original script.
- **Left untouched:** the original script, `exoplanet_research_data/tools/train_transit_baseline.py`, and its data source, `kepler_transit_ML.csv`, were **not modified or deleted**. That file sits in the quarantined folder `05_final_ML_dataset_DO_NOT_USE/` because it has no star-grouped split, and it was not used.
- **Where the numbers are:**
  - `baseline_results/kepler_all.json`, `kepler_mentor.json`, `toi_all.json`, `toi_mentor.json` and `kepler_all_LEAKY_CONTROL.json` hold the per-class reports and confusion matrices.
  - The two comparison runs (diagnostics removed, original protocol) and the epoch check are recorded in `run_log.txt`.
