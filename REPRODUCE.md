# Reproducing every result, in order

This is the complete run sheet: every number in `eda/` and every figure comes from one of the commands below.

**Each step has three tiers:**
- **Read:** the committed outputs, with nothing to run.
- **Re-derive:** recompute from committed intermediate files, in minutes.
- **Rebuild:** start from the archives, from hours to about a day.

## 0. Environments and inputs

| What | How |
|---|---|
| Main environment (Python 3.14) | `python3.14 -m venv .venv && .venv/bin/pip install -r requirements-lock.txt` (on macOS use `requirements-lock-macos.txt`; `requirements.txt` lists the direct dependencies) |
| petitRADTRANS environment (Python 3.12, Fortran) for spectra, the forecast and the retrievals | `micromamba create -f environment-prt.yml`, which pins petitRADTRANS 3.4.0, dynesty 2.1.5 and exo_k 1.3.2. Exact build as run: `environment-prt-lock-osx-arm64.yml`. Opacities (about 1.7 GB) go to `~/petitRADTRANS/input_data`; petitRADTRANS fetches each one on first use (see the `forward_model.py` docstring). |
| The mentor's research pack `exoplanet_research_data/` | Not redistributed. It provides catalogues, JWST spectra and pscomppars. Read-only, and its `05_final_ML_dataset_DO_NOT_USE` folder is never read (`DATA_BOUNDARY.md`). |
| Light-curve views | v1 views: GitHub release `v1.0-views`. The v2/error-channel and K2 views are rebuilt with section 2 (not yet released). |

Everything below runs with `.venv/bin/python` unless marked **prt** (`~/micromamba/envs/prt/bin/python`).

## 1. Data and splits (minutes)

```bash
python split_signal_families.py                 # split v2: star + signal-family grouping (the leakage fix)
python tools/k2_ephemerides.py                  # K2 blind-test ephemerides (archive k2pandc snapshot in external_data/k2/)
```

The mentor's v1 splits in `splits/` are read, never recomputed.

## 2. Light-curve views (network; about 30 h for Kepler + TESS, 1–3 h for K2)

```bash
python tools/rebuild_with_errors.py              # every Kepler and TESS split with the flux-error channel ->
                                                 # detection_views_err/ (unattended, resumes itself; one split by hand:
                                                 # python preprocessing_pipeline.py build-split val --mission TESS
                                                 #        --discard-raw --s3 --views-root detection_views_err)
python tools/make_v2_views.py                    # detection_views_v2/ links the error-channel views into split v2
python tools/build_k2_views.py                   # K2, EVEREST light curves (primary)
python tools/build_k2_views.py --author K2SFF --cache-dir ~/lightkurve_cache_k2sff   # K2 robustness
```

- **Behaviour.** Builds checkpoint per star and resume when rerun. `--discard-raw` keeps the raw cache to the stars in flight.
- **Progress.** `python tools/status.py`, or `tail -1` of the `build_progress.txt` in each output folder.

## 3. Stage A — transit vetting

```bash
python stage_a_transit_model/sweep.py --v2 --jobs 3                 # 60 runs, about 9 h on a 10-core laptop
python stage_a_transit_model/evaluate_v2.py --select-only           # pre-registered selection, validation only
python stage_a_transit_model/evaluate_v2.py                         # THE single test read (guarded; already done)
python stage_a_transit_model/report_v2.py                           # figures/13
python stage_a_transit_model/compare_published_vetters.py           # figures/14 (ExoMiner table in external_data/exominer/)
python stage_a_transit_model/injection_recovery.py --stars 50 --seed 0 --split-version v2 \
       --runs stage_a_transit_model/runs/v2_kepler_dual_cnn_lstm_err{,_s1,_s2,_s7,_s11} \
       --out outputs/injection_recovery_v2 --threshold 0.5925728678703308   # about 25 min
python stage_a_transit_model/evaluate_k2.py --dry-run               # K2 evaluation, code check on Kepler validation
python stage_a_transit_model/evaluate_k2.py                         # THE single K2 read (guarded; protocol in eda/)
python tools/dr25_simulated_sample.py                               # DR25 simulated-data benchmark (protocol in eda/)
python tools/build_dr25_simulated_views.py --set INJ1 --workers 4   # about 30 min; also --set INV (about 55 min)
python stage_a_transit_model/evaluate_dr25_simulated.py             # figures/16
python tools/prospective_check.py                                   # registered predictions (TESS, Kepler, K2) vs today's dispositions
```

The guarded reads refuse to run a second time.
- **To re-derive without reading anything again:** the predictions they wrote (`runs/v2_final/*_test_predictions.csv`, `runs/k2_blind/k2_predictions.csv`) are committed, and `figures/make_figures.py` uses them.

## 4. Stage B — atmospheres

```bash
python stage_b_atmospheres/stage_b_pipeline.py          # observed JWST spectra -> dataset (research pack)
python stage_b_atmospheres/stage_b_level2.py            # band statistics, scale heights
prt:  stage_b_atmospheres/synthetic_grid.py --n 3000 --workers 3                                     # clean grid, about 40 min
prt:  stage_b_atmospheres/synthetic_grid.py --n 3000 --seed 7770001 --out outputs/stage_b_synthetic_seed2   # grid B
prt:  stage_b_atmospheres/synthetic_grid.py --n 3000 --seed 31415001 --tls --haze --out outputs/stage_b_synthetic_realistic
prt:  stage_b_atmospheres/synthetic_grid.py --n 800 --seed 27182001 --tls --haze --out outputs/stage_b_synthetic_stress
python stage_b_atmospheres/train_synthetic_classifier.py
python stage_b_atmospheres/methane_extension.py         # round 1 (frozen labels, hash-checked)
python stage_b_atmospheres/methane_round2.py            # round 2
python stage_b_atmospheres/methane_tls_robustness.py    # contamination-trained v3 classifier, stress test
python tools/refresh_archive_spectra.py                 # archive refresh (network)
python stage_b_atmospheres/methane_round3.py            # round 3: planets new to the archive
python tools/label_crosscheck.py                        # labels vs exoplanet.eu (independent compilation)
```

## 5. Forecast and retrievals

```bash
python stage_b_atmospheres/noise_model.py                                   # JWST noise calibration, about 1 min
prt:  stage_b_atmospheres/contamination_residuals.py                        # visit-varying starspot calibration, about 10 s
prt:  stage_b_atmospheres/detectability_forecast.py                         # about 8-10 min
prt:  stage_b_atmospheres/retrieval_trappist1e.py --workers 3 [--tls-linear]   # synthetic retrievals, hours
prt:  stage_b_atmospheres/retrieval_trappist1e_real.py --workers 4          # real TRAPPIST-1 e; RETRIEVAL_PLANET=TRAPPIST-1 b|c|d,
                                                                            # RETRIEVAL_WL_MIN=3.0, RETRIEVAL_VISITS=1,2 for the variants
prt:  stage_b_atmospheres/real_retrieval_report.py
python stage_b_atmospheres/retrieval_report.py
```

## 6. Stage C — priority ranking

```bash
python stage_c_priority_fusion/validate_shoreline.py     # cosmic shoreline vs 18 observed rocky planets + Solar System
python stage_c_priority_fusion/score_priority_targets.py --with-errors --tag-suffix _v2 --runs ...   # optional vetting columns
python stage_c_priority_fusion/build_priority_table.py
```

## 7. Figures and tests (minutes, no network)

```bash
python figures/make_figures.py      # figures/01-12 from committed results
python -m pytest tests -q           # 45 tests
```

## Where each result is written up

| Result | Write-up |
|---|---|
| Split v2, final Stage A, published-vetter comparison, failure analysis, injection test | `eda/STAGE_A_V2_FINAL_RESULTS.md` |
| K2 blind test | `eda/K2_BLIND_TEST_PROTOCOL.md` (registered before the build) and the results in `eda/K2_BLIND_TEST_RESULTS.md` |
| DR25 simulated-data benchmark | `eda/DR25_SIMULATED_PROTOCOL.md` and `eda/DR25_SIMULATED_RESULTS.md` |
| Prospective predictions | `eda/PROSPECTIVE_PREDICTIONS.md` |
| CH₄ classifier, rounds 1–3, contamination, label cross-check | `eda/METHANE_EXTENSION_RESULTS.md` |
| Gas forecast, contamination regimes, retrievals | `eda/FORECAST_RESULTS.md` |
| Priority ranking and cosmic shoreline | `eda/STAGE_C_RESULTS.md` |
| Chronological log of every run, bug and correction | `run_log.txt` |
