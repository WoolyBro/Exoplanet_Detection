# Stage A, Level 2 — CNN+LSTM transit vetting on Kepler light curves

**Date:** 2026-09-25
**Data:** 9,394 Astronet-style view pairs built from Kepler long-cadence photometry
(train 6,574 / val 1,414 / test 1,406), split by **star** so no star appears in two splits.
**Code:** `stage_a_transit_model/{cnn_lstm,baselines,sweep,failure_analysis,final_eval,make_figures}.py`
**Figure:** `eda/kepler_cnn_lstm_results.png`

---

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
| 11 hand-built shape features + GBM | 0.8800 | 0.8669 | 0.7284 | 0.723 |
| CNN+LSTM, initial | 0.8919 | 0.8901 | 0.7020 | 0.723 |
| **CNN+LSTM, tuned** | **0.9142** | **0.9108** | 0.7207 | **0.7705** |

**Verdict: yes, but the margin is modest.** Eleven interpretable numbers get to 0.880.
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

Note also that even `global_cnn` (0.8945) beats the 11 hand-built shape features (0.8800), so
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
- [x] TESS views: all three splits built — train 4,061 / val 874 / test 856 (5,791 views, 71 % yield).
      The TESS model and the four-way transfer matrix are in `TESS_AND_CROSS_MISSION_RESULTS.md`.
- [ ] Uncertainty-aware: flux errors are **not** propagated into the views. The views carry
      depth only, so the per-cadence uncertainties never reach the model. Closing this means
      adding an error channel in `preprocessing_pipeline.py` and rebuilding all 15,185 views.
- [ ] Injection-recovery (INJ1) through the trained model (7.1) — the INJ1 set was never
      downloaded, so this is deferred rather than attempted and unfinished.

---

## 8. Reproducing

```bash
python stage_a_transit_model/baselines.py                                   # ladder floor
python stage_a_transit_model/sweep.py                                       # 17 runs, val only
python stage_a_transit_model/failure_analysis.py --run stage_a_transit_model/runs/bn_aug_sched
python stage_a_transit_model/final_eval.py      --run stage_a_transit_model/runs/bn_aug_sched   # test, once
python stage_a_transit_model/make_figures.py    --run stage_a_transit_model/runs/bn_aug_sched
```
