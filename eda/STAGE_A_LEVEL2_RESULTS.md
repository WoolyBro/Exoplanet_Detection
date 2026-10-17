# Stage A, Level 2 — CNN+LSTM transit vetting on Kepler light curves

**Date:** 2026-09-25
**Data:** 9,394 Astronet-style view pairs built from Kepler long-cadence photometry
(train 6,574 / val 1,414 / test 1,406), split by **star** so no star appears in two splits.
**Code:** `stage_a_transit_model/{cnn_lstm,baselines,sweep,failure_analysis,final_eval,make_figures}.py`
**Figure:** `eda/kepler_cnn_lstm_results.png`

---

> **Superseded for the headline result** by `STAGE_A_V2_FINAL_RESULTS.md` (2026-10-18). The numbers below use split v1, which leaked through cross-star signal twins. They are kept as the project's history and for the analyses not repeated on v2.

## 1. Headline

| | val (1,414) | **test (1,406)** |
|---|---|---|
| binary ROC-AUC (planet-like) | 0.9142 | **0.9158** |
| binary PR-AUC | 0.9108 | 0.9060 |
| 3-class accuracy | 0.7207 | 0.7304 |
| precision @ recall 90 % | 0.7705 | **0.7653** |
| precision @ recall 95 % | 0.7055 | 0.7267 |
| precision @ recall 99 % | 0.6134 | 0.6097 |

Chosen configuration `bn_aug_sched`: BatchNorm + flip augmentation + ReduceLROnPlateau,
40,371 parameters, selected epoch 38.

**Test came out marginally above val (+0.0016 ROC-AUC).** Across 4 seeds this
configuration scores 0.9121 ± 0.0013 on val, so the val→test difference is 1.2× the
run-to-run scatter — i.e. the model transfers, and best-of-17 selection on val did not
meaningfully inflate the reported number.

No threshold is hardcoded anywhere. Operating points are chosen by **recall**, because
planet hunting cares about completeness; 0.5 never appears as a decision rule.

---

## 2. Does the deep model earn its complexity?

The mentor's checklist requires baselines before deep models. All on the **same input**
(the global+local views) and the same star-disjoint val split:

| model | ROC-AUC | PR-AUC | acc (3-class) | P@R90 |
|---|---|---|---|---|
| majority class | 0.5000 | 0.4958 | 0.5042 | 0.496 |
| logistic regression on raw views | 0.7405 | 0.6866 | 0.5955 | 0.518 |
| 12 hand-built shape features + GBM | 0.8800 | 0.8669 | 0.7284 | 0.723 |
| CNN+LSTM, initial | 0.8919 | 0.8901 | 0.7020 | 0.723 |
| **CNN+LSTM, tuned** | **0.9142** | **0.9108** | 0.7207 | **0.7705** |

**Verdict: yes, but the margin is modest.** Twelve interpretable numbers get to 0.880.
The tuned network adds **+0.034 ROC-AUC** and, more usefully, **+0.047 precision at 90 %
recall** — at 90 % completeness it returns ~7 fewer false positives per 100 flagged
candidates. That is a real gain but not a transformation, and the shape-feature model
remains the better choice whenever interpretability matters more than the last few points.

A separate, earlier baseline on the KOI **catalog table** (`baseline_results/kepler_all.json`)
reaches ROC-AUC 0.971. That is **not** a fair competitor: it reads transit parameters the
Kepler pipeline already fitted (period, depth, duration, SNR, stellar properties), i.e. the
output of the vetting process this model is trying to reproduce from raw shape alone. The
leaky control with `koi_fpflag_*`/`koi_score` retained reaches 0.998 and is meaningless,
exactly as the guide warns.

---

## 3. What the tuning actually found

17 runs, all scored on validation only. Test was untouched until the single final evaluation
(`--eval-test` is off by default in `cnn_lstm.py` for this reason).

| ingredient | effect on val ROC-AUC |
|---|---|
| BatchNorm alone | **+0.015** (0.892 → 0.907) |
| flip augmentation alone | −0.002 (no help) |
| LR schedule alone (with BN) | −0.0005 (no help) |
| **all three together** | **+0.022** (0.914) |
| width ×1.5 (90k params) | −0.003 |
| width ×2.0 (158k params) | −0.005 |
| dropout 0.2 / 0.45 | −0.006 / −0.001 |

Two things worth keeping:

1. **Augmentation only pays once BatchNorm is present.** On its own it is slightly harmful;
   stacked on BatchNorm it adds +0.008. Left-right flipping is physically legitimate here —
   a phase-folded transit is symmetric about mid-transit, so a mirrored view is a real view
   of the same event, not a distortion.
2. **The model is data-limited, not capacity-limited.** Quadrupling the parameters to 158k
   made it *worse*. More KOIs would help; a bigger network will not.

**Seed scatter is ±0.0013.** The top seven configurations span 0.0063, so most of that
ranking is noise: `bn_aug_sched` (0.9121±0.0013) and `drop45` (0.9115±0.0015) are tied, and
`bn_aug_sched` was taken as the simpler of the two. Only the gaps to BatchNorm-only (~4σ)
and to the initial model (~15σ) are real.

---

## 4. Failure analysis — which false positives fool it?

At the 90 %-recall operating point, 188 of 713 true false positives are scored planet-like.
Flag prevalence, fooled vs correctly rejected:

| Kepler flag | fooled | rejected | difference | rejection rate |
|---|---|---|---|---|
| `koi_fpflag_nt` not transit-like | 44.7 % | 27.8 % | **+16.9 pp** | 63.5 % |
| `koi_fpflag_co` centroid offset | 45.2 % | 33.1 % | **+12.1 pp** | 67.2 % |
| `koi_fpflag_ec` ephemeris match | 23.9 % | 22.3 % | +1.7 pp | 72.2 % |
| `koi_fpflag_ss` stellar eclipse (EB) | 12.2 % | 55.6 % | **−43.4 pp** | **92.7 %** |

- **Eclipsing binaries are the model's strength, not its weakness** — 92.7 % rejected. But
  the reason is not what I first assumed (see §5): EBs are separated overwhelmingly by
  depth/SNR, not by a detected secondary eclipse.
- **Centroid offset is the expected structural blind spot.** Which star the light came from
  is simply not present in a folded light curve — no model reading only these views can see
  it. That 67 % are still rejected is incidental, via correlated shape properties.
- **Not-transit-like is the largest remaining gap** (+16.9 pp).
- Failures are low-SNR: median `local_snr` 5.9 among fooled vs 23.3 among rejected.
- 19 false positives carry **no** Kepler flag at all; 11 of them fool the model. Even the
  Kepler pipeline recorded no specific reason to reject these.

### Against the published disposition (reference, not truth)
Spearman ρ = 0.746 between the model score and `koi_score` (1,201 val rows). Disagreements:
- model planet-like, `koi_score` < 0.5: **149** rows (124 truly FALSE POSITIVE — model errs)
- model rejects, `koi_score` ≥ 0.5: **44** rows (14 CONFIRMED, 17 CANDIDATE — real planets missed)

`koi_score` is treated as a reference point, never as ground truth; both are estimates.

---

## 5. A correction worth recording

My first pass at the interpretable shape features contained two geometry errors, both found
by checking the features against the data rather than trusting them:

1. **`secondary_depth` was measured at the wrong phase.** `build_global_view` bins phase from
   −P/2 to +P/2 with the transit at the *centre*, so a secondary eclipse at phase 0.5 sits at
   the view **edges**, not at the quarter points where I measured it.
2. **More seriously, the statistic itself was wrong.** Using `−min(...)` over a region
   measures the deepest *noise* excursion, not a coherent dip. Measured against
   `koi_fpflag_ss`, both the old and the relocated version scored **AUC < 0.5** — they ranked
   noisy shallow candidates *above* real eclipsing binaries.

Replaced with a noise-normalised significance (median dip over the region divided by the
out-of-transit MAD). The corrected `secondary_sig` now points the right way (AUC 0.559), and
the noise term it had been hiding is exposed as its own feature, `oot_mad`.

**Why this mattered:** in the uncorrected run, the mislabelled `secondary_depth` ranked as the
**second most important feature**. Reported as-is, it would have supported the claim "the
model detects secondary eclipses to identify eclipsing binaries." That claim would have been
false. The corrected ranking puts `oot_mad` (+0.0495) in that slot and `secondary_sig` down
at +0.0136, and the true EB discriminator turns out to be `local_snr` (AUC 0.863 on its own).
Overall performance was unaffected (0.8797 → 0.8800); only the explanation changed.

A genuine **odd/even depth test is not computable from these views** and is not claimed: it
needs alternate transits separated (a fold at 2×period, or the unfolded series). What the
feature set offers instead is `transit_asymmetry`, named for what it measures.

---

## 6. Honest limitations

- **CANDIDATE is the weak class** (precision 0.445 on test): 207 of 711 false positives are
  called CANDIDATE. The class sits between the other two by construction — an unconfirmed
  signal — so this is partly label semantics, not purely model error.
- **Selection bias**: the val figure is best-of-17. The test figure is not, and is the one to
  quote.
- **Test is now spent.** Any further tuning must go back to val, and a second test evaluation
  would no longer be a clean held-out measurement.
- The 3-class accuracy (0.730) is much less flattering than the binary AUC (0.916). Both are
  reported; the binary planet-like task is the easier one.
- 28 test KOIs were excluded at build time for unreliable ephemerides, and 64 across all
  splits have unknown ephemeris reliability.

---

## 6b. Architecture ablation (guide stages 5.1 → 6.1 → 6.2)

The guide sequences this deliberately: plain CNN on the global view first, then the local
branch, then the LSTM. Run as a real ablation with everything else held at the winning
configuration (BatchNorm + augmentation + schedule), two seeds each:

| architecture | val ROC-AUC | params | gain | cost |
|---|---|---|---|---|
| `global_cnn` — plain CNN, global view only | 0.8945 | 12,603 | — | — |
| `dual_cnn` — + local view, no LSTM | 0.9073 | 16,051 | **+0.0128** | +3,448 params |
| `dual_cnn_lstm` — + BiLSTM (full) | 0.9121 ±0.0013 | 40,371 | +0.0048 | +24,320 params |

**The local view is worth 2.7× what the LSTM is worth, at one-seventh the parameter cost.**
Both gains exceed the ±0.002 within-architecture seed spread, so both are real — but the LSTM
buys the smaller half of the improvement for 2.5× the whole model. If this had to run on
limited hardware, `dual_cnn` is the configuration to ship.

Note also that even `global_cnn` (0.8945) beats the 12 hand-built shape features (0.8800), so
the CNN earns its place before any of the extras are added.

## 6c. Learned representation (guide 7.4)

t-SNE of the 64-unit head embedding, **plus a measurement**, because a t-SNE map will invent
clusters from noise at low perplexity and cannot be used as evidence on its own:

| measure | value | null (labels shuffled) |
|---|---|---|
| kNN accuracy, star-grouped 5-fold | **0.7808** | 0.4462 |
| silhouette | **0.1767** | −0.0073 |
| majority-class rate | 0.5042 | — |

The structure is real: kNN in the embedding space recovers the label at 0.781 against a
majority-class floor of 0.504 and a shuffled-label null of 0.446. Only 1 of 64 units is dead.

Worth noting: **the embedding's kNN accuracy (0.781) exceeds the model's own 3-class accuracy
(0.721)**. The representation carries more class information than the classifier head extracts
from it, which suggests the head, not the features, is the weaker part.

Figure: `eda/kepler_val_embedding_tsne.png`

## 6d. Where the false negatives come from (guide 7.3)

70 of 701 planet-like rows are missed at the 90 %-recall operating point. Bucketed by whether
the signal was there to find — using Kepler's own `koi_model_snr` threshold of 7.1 and the
out-of-transit scatter measured from the view:

| bucket | share of misses | share of all planets | verdict |
|---|---|---|---|
| below detection (SNR < 7.1) | 14.3 % | 2.6 % | **5.5× over-represented** |
| noise-dominated (worst-quartile scatter) | 21.4 % | 21.5 % | **exactly at base rate — not a driver** |
| genuine miss | 55.7 % | 74.8 % | under-represented |

Comparing each bucket against its base rate matters: "21 % of misses are noise-dominated"
sounds like a finding until you see that 21 % of *everything* is noise-dominated.

**The genuine misses have a median period of 1.54 d against 13.22 d for all planets.** The
worst-scored ones are at 0.49–0.71 d. Ultra-short-period planets are a systematic blind spot —
and this is the **same failure mode** found independently in §1b of the Stage C report, where
the model rejected the canonical hot Jupiters. Two different analyses, one root cause: short-
period, high-amplitude signals are rare or labelled false-positive in Kepler's training
distribution.

Two "genuine misses" should not be counted as model errors: K07865.01 (R_p 61 R⊕) and
K00971.01 (R_p 134 R⊕). Jupiter is 11 R⊕, so these radii are physically impossible; the model
scoring them low is more likely correct than the catalogue entry.

## 6e. Injection-recovery, and what the null control exposed (guide 7.1)

Synthetic limb-darkened transits (u1=0.4, u2=0.25, b=0.3) injected into **real Kepler
photometry** for 25 test-split stars at 8 depths — 200 injections. The signal goes in before
detrending, with its ephemeris handed to the Savitzky-Golay filter so it is masked exactly as
a real transit is; otherwise the filter would eat the injection and the model would take the
blame. The star's own KOIs are masked out, so a recovery cannot be a real planet showing
through. Code: `stage_a_transit_model/injection_recovery.py`. Figure:
`eda/injection_recovery.png`.

**This is not the DR25 INJ1 set.** INJ1's injected light curves span ~160,000 targets, and its
recovery table records what the *Kepler pipeline* recovered, not what this model recovers.
Injecting directly measures this model, at the cost of not being comparable with published
INJ1 completeness numbers.

### The null control is the result

| injected depth | median score | "recovered" | **paired excess over that star's own null** | stars responding |
|---|---|---|---|---|
| **0 — nothing injected** | **0.764** | **84 %** | — | — |
| 5 ppm | 0.759 | 84 % | +0.0000 | 13/25 |
| 10 ppm | 0.759 | 84 % | −0.0004 | 12/25 |
| 20 ppm | 0.757 | 80 % | +0.0003 | 14/25 |
| 40 ppm | 0.785 | 84 % | −0.0002 | 12/25 |
| 80 ppm | 0.857 | 84 % | **+0.0448** | 17/25 |
| 160 ppm | 0.924 | 92 % | **+0.1146** | **22/25** |
| 320 ppm | 0.987 | 92 % | **+0.1827** | **24/25** |

**With nothing injected at all, the model scores 84 % of these folds above its own
90 %-recall threshold, median 0.764.** An earlier version of this experiment, run without the
0 ppm row, reported "100 % recovery at 5 ppm" — which was entirely baseline. The null is the
only reason that is visible, and it is why the recovery-fraction column is reported here but
not plotted as a completeness curve.

### What it means

**The model is a candidate vetter, not a blind transit search.** Every training row — including
the FALSE POSITIVEs — is something the Kepler pipeline already flagged as transit-like. An
arbitrary fold at a random period is outside that distribution, so the model has no basis for
scoring it low. Nothing in the 0.9158 test ROC-AUC reveals this, because that number is
measured on curated KOIs: the vetting task, which is the task it was trained for.

Practical consequence: this model should be run **on candidates from a transit search**, never
as the search itself. Pointed at arbitrary ephemerides it will return high scores for noise.

### The signal that survives

Pairing each injection against its own star's null removes the per-star offset and leaves a
real measurement: the model starts responding at **80 ppm** and responds clearly at 160 ppm
(22/25 stars) and 320 ppm (24/25). Below 40 ppm it is a coin flip — indistinguishable from
folded noise.

For scale, an Earth across a Sun-like star is ~84 ppm, so the response threshold sits right at
the Earth-analogue boundary. That is a statement about this model on four years of Kepler
photometry, not about Kepler's own pipeline.

### Why depth is the wrong axis, strictly speaking

The views are depth-normalised — the deepest bin is set to −1 — so absolute depth is erased
before the model sees anything. What actually varies with injected depth is whether the
injection *dominates the fold*. Two earlier runs of this experiment returned flat recovery
across 100–6400 ppm for exactly that reason, and the depth grid was pushed down until the
transition appeared.

## 6f. Probability calibration

The planet-like score ranks well but is not a probability as it comes out of the network:
KOIs scored 0.5–0.7 on Kepler test are planet-like only ~40 % of the time. `calibrate.py` fixes
the probability scale without touching the model:

- **Fitted and chosen on validation only.** Three options — no calibration, Platt (logistic
  regression on logit(score), 2 parameters) and isotonic — compared by star-grouped 5-fold
  cross-validation on validation. Out-of-fold Brier: none 0.1468, **Platt 0.1187**, isotonic
  0.1189; Platt is chosen (isotonic has slightly lower ECE, 0.019 vs 0.029, but a worse log loss
  and a less stable map at n = 1,414).
- **Applied once to the already-saved test predictions** — no model is re-run on test. The map
  is p = sigmoid(0.835 · logit(score) − 1.021).

| Kepler test (1,406) | Brier | log loss | ECE | ROC-AUC |
|---|---|---|---|---|
| raw score | 0.1304 | 0.4127 | 0.0961 | 0.9158 |
| Platt, fitted on validation | **0.1170** | **0.3682** | **0.0314** | 0.9158 |

The ranking is untouched by construction (a monotone map), so every precision-at-recall number
in this report stands; only what the score *means* changes. After calibration, mid-range test
scores are slightly under-confident (observed rate a little above the predicted probability),
a residual val→test shift that is reported rather than tuned away. Figure:
`figures/05_probability_calibration.png`.

The same procedure on the TESS model, whose inverse-frequency class weights push raw scores far
below its 86 % planet-like base rate, chose Platt as well (p = sigmoid(1.035 · logit(score) +
1.658)): TESS test Brier 0.1596 → 0.1025, ECE 0.2231 → 0.0175, ROC-AUC 0.7843 unchanged.

This is a second look at each test set, but only at a calibration map that was fitted and
selected on validation; nothing about the model or the map was tuned on test.

---

## 6g. Does feeding the model the per-bin flux error help? (uncertainty-aware photometry)

Until now the views held binned depth only, so per-cadence flux errors never reached the model.
All 15,185 views were rebuilt (into `detection_views_err/`, leaving the originals untouched) with
a per-bin 1σ — the standard error of the bin median, √(π/2)·√Σσ²/n, in the same normalised units
as the depth — and `cnn_lstm.py --use-errors` stacks log₁₀ of it as a second input channel
(40,451 vs 40,371 parameters). The depth channel of the rebuilt views matches the originals to
3×10⁻⁸ (checked on 11 stars of each mission).

**Design.** The winning configuration (BatchNorm + flip + scheduler), trained *with* and *without*
the channel on the **same rebuilt views**, seeds 42/1/2/7, both missions: 16 models, **validation
only** (both test sets are already spent; the test views were rebuilt but never read).

| validation, mean ± std over 4 seeds | without | with | difference |
|---|---|---|---|
| Kepler ROC-AUC | 0.9111 ± 0.0014 | **0.9145 ± 0.0017** | **+0.0034** |
| Kepler PR-AUC | 0.9080 | 0.9107 | +0.0027 |
| Kepler precision @ 90 % recall | 0.7494 | 0.7496 | +0.0002 |
| TESS ROC-AUC | 0.7771 ± 0.0096 | **0.7884 ± 0.0049** | **+0.0113** |
| TESS PR-AUC (base rate 0.870) | 0.9541 | 0.9586 | +0.0045 |
| TESS precision @ 90 % recall | 0.9076 | 0.9106 | +0.0030 |

Per-seed ROC-AUC gain (seeds 42, 1, 2, 7): Kepler +0.0041, +0.0001, +0.0036, +0.0056; TESS
+0.0076, +0.0087, +0.0071, +0.0218. **With the channel is at or above without in all 8 pairs.**

**How strong is that?** Modest, and it should be stated as modest. With four seeds per arm the
gain is 2.2× (Kepler) and 1.5× (TESS) the seed-to-seed spread; a paired t-test over seeds gives
p = 0.062 (Kepler) and 0.049 (TESS), Welch p = 0.022 and 0.097. That is consistent evidence of a
small gain in the same direction in both missions — not a decisive one. It is comparable in size
to the LSTM (+0.0048) and a fifth of BatchNorm (+0.015). It sharpens the ranking rather than the
operating point: precision at 90 % recall is unchanged on Kepler.

**Caveats.**
- The "without" arm, retrained from scratch on this machine (macOS) on the rebuilt views, scored
  0.9094 at seed 42 against 0.9142 for the original run of the same configuration (Windows). The
  data are identical (6,574 / 1,414 rows), so this is cross-platform training variation; it is
  why the comparison is made against the new "without" arm, never against the old numbers.
- 20 TESS views (18 train, 2 val; none in Kepler) have a local window with no data, so their
  error channel is undefined; it is fed as "maximum uncertainty". Before that was handled, every
  TESS with-errors run stopped on epoch 1 (NaN guard).
- Validation only, no test numbers, and no claim that a with-errors model should replace the
  reported ones: those results stand as measured. Selection on validation across 16 runs is
  itself slightly optimistic.

Code: `tools/rebuild_with_errors.py`, `sweep.py --errors`. Table: `runs/error_channel_comparison.csv`.

---

## 7. Checklist status (mentor's evaluation & reporting list)

- [x] Splits by star/system, no leakage across folds — verified star-disjoint in code, asserted at load
- [x] Seeds reported — 42 primary, repeats at 1/2/7, scatter quantified
- [x] Baselines first (majority, logistic) before deep models
- [x] Precision/recall at fixed recall (90/95/99 %), no 0.5 threshold anywhere
- [x] Compared against published dispositions (`koi_score`) as reference, not truth
- [x] Failure analysis against `koi_fpflag_nt/ss/co/ec` and a variability proxy
- [x] Staged architecture ablation (5.1 plain CNN → 6.1 local view → 6.2 LSTM)
- [x] t-SNE on embeddings, with a measured separability check rather than the picture alone (7.4)
- [x] False negatives bucketed as undetectable / noise-dominated / genuine, against base rates (7.3)
- [x] Cross-mission evaluation (7.2) — see Stage C §1b and `TESS_AND_CROSS_MISSION_RESULTS.md`
- [x] Injection-recovery (7.1) — §6e. Synthetic transits into real photometry rather than the
      DR25 INJ1 set; the 0 ppm null control is the headline finding
- [x] TESS views: all three splits built — train 4,061 / val 874 / test 856 (5,791 views, 71 % yield).
      The TESS model and the four-way transfer matrix are in `TESS_AND_CROSS_MISSION_RESULTS.md`.
- [x] Uncertainty-aware — **done for spectra and now for photometry (§6g).** Stage B carries an explicit uncertainty channel and propagates it in quadrature
      into every band index and into the anomaly normalisation. Every result reported for Stage A
      above comes from views that hold binned depth only, so per-cadence flux errors never
      reached those models. Added 2026-10-13: `preprocessing_pipeline.py` now bins `flux_err` into
      a per-bin 1σ (standard error of the bin median, √(π/2)·√Σσ²/n, divided by the same scale
      as the depth so both are in units of the transit depth; empty bins take the largest
      filled-bin error), saved as `global_err` / `local_err`. The depth channel is unchanged
      (rebuilt Kepler and TESS views match the originals to 3×10⁻⁸). `cnn_lstm.py --use-errors`
      feeds log₁₀ of it as a second input channel (+80 parameters). The views were rebuilt into
      `detection_views_err/`; with vs without the channel, four seeds each, validation only:
      Kepler ROC-AUC +0.0034, TESS +0.0113, a small consistent gain (§6g).
- [x] Probability calibration — §6f: Platt scaling chosen and fitted on validation; Kepler test
      Brier 0.130 → 0.117, ECE 0.096 → 0.031, ranking unchanged.

---

## 8. Reproducing

```bash
python stage_a_transit_model/baselines.py                                   # ladder floor
python stage_a_transit_model/sweep.py                                       # 17 runs, val only
python stage_a_transit_model/failure_analysis.py --run stage_a_transit_model/runs/bn_aug_sched
python stage_a_transit_model/final_eval.py      --run stage_a_transit_model/runs/bn_aug_sched   # test, once
python stage_a_transit_model/make_figures.py    --run stage_a_transit_model/runs/bn_aug_sched
```
