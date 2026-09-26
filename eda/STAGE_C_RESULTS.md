# Stage C — priority fusion over the 54-planet table

**Date:** 2026-09-25
**Code:** `stage_c_priority_fusion/build_priority_table.py` → `outputs/stage_c/`
**Figure:** `eda/stage_c_priority_overview.png`
**Inputs:** Stage A run `bn_aug_sched`, Stage B LOPO predictions, Stage B Level 2 scale heights,
pscomppars. The template in `05_final_ML_dataset_DO_NOT_USE/` was read only.

---

## 1. The headline is a coverage problem, not a modelling one

The template ships `transit_ML_probability = 0.99` for **all 54 planets**. That is not a
prediction — it encodes "this planet is confirmed". Carrying it forward would have been the
single most misleading thing this table could do, so it was not carried forward.

What the models can actually fill:

| column | model output | why the rest are empty |
|---|---|---|
| `transit_ML_probability` | **2 / 54** | Kepler stared at one ~115 deg² field; the other 52 are WASP/HD/HAT-P/GJ targets it never observed |
| `H2O_score`, `CO2_score`, `CH4_score` | **13 / 54** | no JWST spectrum in the pack |
| `O3_score` | **0 / 54** | zero positive examples anywhere — not modellable, by construction |

And of the two transit scores, only **one is a fair measurement**:

| planet | KOI | p(planet-like) | provenance |
|---|---|---|---|
| Kepler-62 f | K00701.04 | **0.8829** | test split → genuinely held out |
| Kepler-296 f | K01422.04 | 0.8942 | train split → **in-sample, not a fair score** |

**Every other cell is left empty with a reason recorded in `notes`.** No number appears in this
table that a model did not produce.

This is the honest state of the project: Stages A and B both work, but they were trained on the
populations that have data (Kepler KOIs; 13 JWST-observed planets), and the Stage C target list
is a different population — bright, ground-discovered planets chosen because JWST *can* observe
them. The TESS build now running is the route that closes most of this gap, because TESS is
all-sky and does cover these targets.

---

## 1a. FINAL STATE (2026-09-26): 41 of 54 now carry a real transit probability

Once the TESS splits finished and a TESS model was trained, the 54 targets were re-scored with
it. These are TESS light curves, so the TESS-trained model is the **same-mission** model and is
the primary column.

| column | filled | what it is |
|---|---|---|
| `transit_ML_probability_TESS` | **41 / 54** | **primary** — TESS-trained model on TESS light curves |
| `transit_ML_probability_kepler_model` | 41 / 54 | the Kepler model on the same views, kept because its bias is *opposite* (§1c) |
| `transit_ML_probability` | 2 / 54 | the strict column: only KOIs scored inside their own official split |
| `H2O/CO2/CH4_score` | 13 / 54 | Stage B, out-of-fold |
| `O3_score` | 0 / 54 | zero positive examples — not modellable |

13 of the 54 remain unscored: 10 lack a complete ephemeris in pscomppars, 2 were refused by the
ephemeris-reliability check, 1 has no SPOC light curve. Those cells stay empty.

## 1b. The Kepler model was wrong for these targets — and the TESS model fixes it

None of the 54 are in the TOI split being downloaded (they are WASP/HD/HAT-P/GJ planets, not
TOI candidates), so the big build will not score them. But all 54 TIC ids resolve from
pscomppars locally, so they were fetched directly: **41 of 54 scored** (44 had complete
ephemerides; 2 excluded for ephemeris drift, 1 had no SPOC light curve). Eight minutes on the
AWS mirror. Code: `stage_c_priority_fusion/score_priority_targets.py`.

These live in their **own column**, `transit_ML_probability_TESS_crossmission`, and were
deliberately *not* merged into `transit_ML_probability`. Here is why.

Every one of these 54 is a **confirmed planet**, so a model that transferred cleanly should
score them all high. Only **71 %** scored ≥ 0.5 (median 0.763). The 29 % it rejects are not
random:

| | rejected (n=12) | accepted (n=29) |
|---|---|---|
| median transit depth | **14,678 ppm** | 6,445 ppm |
| median period | **1.2 d** | 4.2 d |
| median radius | **16.6 R⊕** | 2.0 R⊕ |
| median T_eq | **2,036 K** | 586 K |

Correlations with the score: T_eq **−0.660**, radius −0.570, period +0.560, depth −0.410.
The rejected list is KELT-9 b (0.060), HD 189733 b (0.124), 55 Cnc e (0.158), WASP-19 b,
WASP-12 b, WASP-18 b, WASP-33 b, WASP-43 b, WASP-121 b, WASP-17 b, WASP-31 b, HD 209458 b —
i.e. **the canonical hot Jupiters**.

**This is not a bug; it is correct generalisation from the wrong population.** In the Kepler
training split, **93 % of KOIs deeper than 10,000 ppm are FALSE POSITIVES** (against 42 % for
shallower ones), because in Kepler's field a very deep transit is overwhelmingly an eclipsing
binary. The model learned that rule properly — and it is exactly wrong for targets that
ground-based surveys (WASP, HAT, KELT) were *built* to find, since those surveys select for
deep, short-period giants.

Consequences, stated plainly:
- The Kepler model **cannot be used** to score this priority table. Its errors are
  systematic and correlated with precisely the property that defines the target list.
- A TESS-trained model may not fix this either, if the TOI training split is similarly short of
  hot Jupiters. That should be **checked** on the TOI depth distribution before assuming.
- 55 Cnc e is the one rejection that this explanation does not cover (332 ppm, shallow). Its
  0.74 d period is the likely cause, but that is a guess and is flagged as one.

This is the most useful negative result in the project so far, and it only became visible
because the scores were checked against a population whose labels were known in advance.

**The fix, confirmed.** The TESS model was predicted (from the training distributions, before
it was trained) not to carry this prior, because in the TOI split only 11.5 % of transits
deeper than 10,000 ppm are false positives — *below* its 17.2 % base rate. Re-scoring the same
41 targets:

| planet | Kepler model | TESS model |
|---|---|---|
| KELT-9 b | 0.060 | **0.629** |
| HD 189733 b | 0.124 | **0.662** |
| 55 Cnc e | 0.158 | **0.723** |
| WASP-19 b | 0.174 | **0.714** |
| WASP-12 b | 0.199 | **0.770** |
| WASP-18 b | 0.244 | **0.782** |

Fraction of these confirmed planets scored ≥ 0.5 rises from **71 % to 93 %**.

## 1c. But the two models have OPPOSITE biases — neither is neutral

The TESS model did not simply turn out better. Measured across all 41 targets:

| Spearman vs score | Kepler model | TESS model |
|---|---|---|
| planet radius | **−0.570** | **+0.302** |
| transit depth | −0.410 | +0.221 |
| orbital period | +0.560 | −0.192 |

Every correlation flips sign. By planet size:

| bucket | Kepler model | TESS model | change |
|---|---|---|---|
| small, R_p < 4 R⊕ (n=17) | **0.786** | 0.633 | −0.153 |
| mid, 4–8 R⊕ (n=2) | 0.958 | 0.830 | −0.128 |
| giant, R_p ≥ 8 R⊕ (n=22) | 0.511 | **0.741** | +0.230 |

Each model is more confident on whatever its training population was rich in — Kepler's median
transit is 410 ppm, TESS's is 4,851 ppm. **This directly affects the top of this table:** the
TRAPPIST-1 planets rank first on the priority score, and they are exactly where the TESS model
is *least* confident (0.48–0.58, against the Kepler model's 0.73–0.88). The primary column is
the methodologically correct one, but for the small rocky targets that dominate the High band
it is the more pessimistic of the two, and that should be read as model uncertainty rather than
evidence about those planets.

Taking the maximum of the two scores recovers 100 % of the confirmed planets at ≥ 0.5. That is
**not adopted**: on a set containing no negatives, a max over two scores can only go up, so it
demonstrates nothing about precision. It is recorded here as an observation, not a method.

## 2. Priority from explicit weights

Replacing the template's heuristic bands. Every component is a measured quantity scaled to 0–1:

| component | weight | definition |
|---|---|---|
| `small` | 0.25 | log-scaled radius, 1 at Earth size → 0 at Jupiter size |
| `temperate` | 0.25 | Gaussian peaked at 300 K (a prioritisation preference, **not** a habitability claim) |
| `observable` | 0.30 | log-scaled `2·R_p·H/R_s²` — one scale height of absorption, in ppm |
| `evidence` | 0.20 | strongest out-of-fold molecular probability from Stage B |

Weights are **renormalised over the components a planet actually has**. This matters: 41 of 54
planets have no spectral evidence term, and scoring them as evidence = 0 would push every
un-observed planet to the bottom purely because nobody has pointed JWST at it yet. That would
be a statement about the observing log, not about the planet.

### Top targets

| rank | planet | score | R_p (R⊕) | T_eq (K) | 1 H (ppm) | components |
|---|---|---|---|---|---|---|
| 1 | TRAPPIST-1 d | 0.886 | 0.79 | 286 | 246 | small+temperate+observable |
| 2 | TRAPPIST-1 c | 0.880 | 1.10 | 340 | 234 | + evidence |
| 3 | TRAPPIST-1 b | 0.871 | 1.12 | 398 | 274 | small+temperate+observable |
| 4 | TRAPPIST-1 e | 0.863 | 0.92 | 250 | 192 | small+temperate+observable |
| 5 | TRAPPIST-1 f | 0.840 | 1.04 | 218 | 163 | small+temperate+observable |
| 6 | GJ 1002 b | 0.819 | 1.03 | 231 | 121 | small+temperate+observable |
| 11 | TOI-270 d | 0.721 | 2.00 | 383 | 49 | + evidence |

The ranking is dominated by the TRAPPIST-1 system, and the reason is physical rather than
sentimental: the host is 0.119 R☉, and `A_H = 2·R_p·H/R_s²` scales as 1/R_s². A small rocky
planet around a very small star produces a *larger* signal than the same planet around a
Sun-like star. That is precisely why TRAPPIST-1 is the most-observed rocky system in the sky,
and the score recovers it from first principles without being told.

Bands are terciles of this sample of 54 (18 High / 18 Medium / 18 Low) — relative to this list,
not absolute.

---

## 3. Ablation

Each contributor dropped in turn, re-scored, compared to the full ranking:

| dropped | weight | Spearman vs full | planets changing band |
|---|---|---|---|
| **observable** | 0.30 | **0.907** | 10 |
| small | 0.25 | 0.923 | 10 |
| temperate | 0.25 | 0.929 | 12 |
| evidence | 0.20 | 0.981 | 10 |

The ranking depends most on **observability** — the one term derived from physics rather than
preference. `evidence` moves it least (ρ = 0.981), which is a direct consequence of §1: it is
only present for 13 of 54 planets, so it cannot move a ranking it barely participates in.

Useful honesty: no contributor changes the ranking dramatically (all ρ > 0.90). The ordering is
fairly robust to the exact weights — which also means the weights are not doing heroic work, and
the result should not be over-interpreted as a finely-tuned optimum.

---

## 4. What would change this table

1. **Finish the TESS build** (running, ~20 h left). TESS covers most of these 54 targets, so it
   is the one change that could move `transit_ML_probability` from 2/54 towards most of them.
   That requires a TESS-trained model, not the Kepler one — cross-mission transfer is an
   assumption, and should be tested rather than assumed.
2. **More JWST spectra** would extend `evidence` beyond 13 planets. The pack's
   `download_spectra.py` skips existing files, so re-running it is cheap.
3. **O₃ will not be modellable** until positive examples exist. It stays empty, deliberately.

---

## 5. Limitations

- `temperate` peaks at 300 K by choice. It is a prioritisation preference and carries no
  habitability claim.
- `evidence` uses the maximum molecular probability, so a planet with one confident detection
  scores the same as one with three. Defensible for "is there anything to look at", crude as
  a chemistry summary.
- Scale-height amplitudes assume µ = 2.3 (H₂/He) for every planet, inherited from Stage B
  Level 2. Most doubtful for the rocky TRAPPIST-1 planets — which are also the top of the
  ranking, so this assumption is load-bearing and is flagged rather than buried.
- Bands are terciles, so "High" means "top third of these 54", not an absolute standard.
- Kepler-296 f's transit score is in-sample and is labelled as such; it should not be quoted.

---

## 6. Reproducing

```bash
python stage_c_priority_fusion/build_priority_table.py
```
Writes `outputs/stage_c/{final_priority_table.csv, ablation.csv, stage_c_summary.json}`
and `eda/stage_c_priority_overview.png`. Reads the template read-only.
