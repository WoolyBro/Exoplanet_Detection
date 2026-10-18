# Vetter v3: real false alarms and pipeline diagnostics (critique 1, Pillar A)

**Protocol, committed before any view was built or model trained:** `eda/VETTER_V3_PROTOCOL.md` (commit `ae91f4e`).
**Code:**
- `tools/build_v3_dataset.py` (samples, families, diagnostics);
- `tools/build_dr25_simulated_views.py` (views);
- `stage_a_transit_model/vetter_v3.py` (training);
- `stage_a_transit_model/evaluate_v3.py` (the one read).

**Output:** `stage_a_transit_model/runs/v3_final/v3_eval.json`, plus `predictions_*.csv`.
**Guard:** `runs/v3_final/TEST_READ.json`. The read was made once, 2026-10-19 00:38.
**Figure:** `figures/17_vetter_v3.png`.

> **Critique 1 was "it is not a complete vetter".** v2 passed 30 % of fresh systematics false alarms and caught light from neighbouring stars only 49 % of the time. "Why use it instead of ExoMiner?"
>
> **Answer.** Two additions remove most of that gap on held-out data:
> - **real false alarms in training:** 4,200 DR25 non-KOI TCEs ("NTP");
> - **a branch reading 27 raw pipeline measurements:** the Robovetter's inputs, with none of its decisions.
>
> **Results:**
> - **The primary model C matches ExoMiner** on the same 934 Kepler test KOIs: 0.9961 vs 0.9963, Δ = −0.0002 [−0.0025, +0.0021].
> - **It rejects 99.1 % of fresh false alarms** (v2 70.4 %, Robovetter 98.4 %) and 99.9 % of real held-out ones.
> - **It catches 80 % of neighbouring-star false positives** (v2 49 %).
>
> **What it does not fix:** subtle off-target transits and eclipsing-binary mimics in the simulations. On those the rule-based Robovetter is still far better (0.53 and 0.63 vs 0.20 and 0.23).

## 1. What was trained (validation only)

**Training data.**
- The KOIs of split v2, plus 4,200 / 900 / 900 NTP TCEs (train / validation / test).
- The NTP TCEs were split by star and signal family together with the KOIs. 507 NTPs whose families bridged KOI splits were dropped.
- Result: 0 shared stars and 0 cross-split twins.

**Diagnostics.** 27 raw DR25 measurements from the Robovetter input tables:
- centroid score, ghost/halo;
- odd/even and secondary-eclipse significances;
- LPP;
- the "chases" and SES/MES statistics;
- period, duration, depth.

They are signed-log transformed, standardised on training data and clipped at ±5, plus a missing flag. 1,373 KOIs have no DR25 TCE and get zeros with the flag set. No Robovetter *decision* (flags, dispositions or score) is an input.

| Config | Training data | Branches | Params | Validation ROC-AUC, KOI + NTP (5-seed ensemble) |
|---|---|---|---|---|
| v2 | KOIs | views | 40,451 | 0.932 |
| A | KOIs + NTP | views | 40,451 | 0.954 |
| B | KOIs | views + diagnostics | 44,483 | 0.972 |
| **C (primary)** | KOIs + NTP | views + diagnostics | 44,483 | **0.983** |

**Thresholds.** Each threshold is the recall-90 threshold on KOI + NTP validation: v2 0.593, A 0.607, B 0.612, **C 0.815**.

## 2. Results (one read; 95 % Wilson or star-bootstrap intervals; McNemar exact tests)

### T1: fresh DR25 simulations (1,000 TCEs each, on stars never used)

| Set (metric) | v2 | A | B | **C** | Robovetter | C vs v2 (Holm p) | C vs Robovetter (p) |
|---|---|---|---|---|---|---|---|
| INJ1, planets kept | 0.935 | 0.919 | 0.906 | **0.842** [0.818, 0.863] | 0.859 | lower, 1 × 10⁻¹⁵ | 0.14 (equal) |
| INV, false alarms rejected (n = 997) | 0.704 | 0.837 | 0.948 | **0.991** [0.983, 0.995] | 0.984 | higher, 8 × 10⁻⁸⁶ | 0.17 (equal) |
| INJ2, off-target rejected | 0.047 | 0.043 | 0.205 | **0.199** [0.175, 0.225] | 0.528 | higher, 4 × 10⁻³³ | lower, 3 × 10⁻⁶⁵ |
| INJ3, EB mimics rejected | 0.142 | 0.115 | 0.195 | **0.228** [0.203, 0.255] | 0.629 | higher, 2 × 10⁻⁹ | lower, 1 × 10⁻⁹⁶ |

**As a ranker** (ROC-AUC, INJ1 vs INV): v2 0.940, A 0.955, B 0.975, **C 0.989**.

### T2: real NTP false alarms, held-out test (900)

v2 rejects 0.762, A 0.942, B 0.949, **C 0.999** [0.994, 1.000]. C vs v2: Holm p = 6 × 10⁻⁶⁴.

### T3: Kepler v2 test KOIs (1,404), a second read declared in the protocol before training

| | v2 | A | B | **C** |
|---|---|---|---|---|
| ROC-AUC, planet-like vs FP | 0.924 | 0.918 | 0.969 | **0.962** [0.953, 0.971] |
| ROC-AUC, confirmed vs FP | 0.971 | 0.972 | 0.996 | **0.994** [0.990, 0.997] |
| Same, on the 934 KOIs ExoMiner also scored (ExoMiner 0.9963) | 0.977 | 0.977 | 0.997 | **0.996** |

**Gains of C.**
- Over v2, confirmed vs FP: +0.023 [0.015, 0.031].
- Over ExoMiner: −0.0002 [−0.0025, +0.0021].

**FPs rejected, by their only flag:**

| Only flag | n | v2 | A | B | C |
|---|---|---|---|---|---|
| ss (eclipsing binary) | 191 | 0.95 | 0.91 | 0.95 | 0.95 |
| ec (ephemeris match) | 22 | 0.91 | 0.86 | 0.86 | 0.77 |
| nt (not transit-like) | 150 | 0.55 | 0.61 | 0.73 | 0.73 |
| **co (centroid offset)** | 80 | 0.49 | 0.36 | **0.85** | **0.80** |

## 3. The registered expectations

| ID | Expectation | Outcome |
|---|---|---|
| H1 | C rejects ≥ 0.95 of INV | **held**: 0.991 |
| H2 | C keeps ≥ 0.90 of INJ1 | **failed**: 0.842 |
| H3 | C rejects INJ2 ≥ v2 + 0.15 **and** ≥ the Robovetter | **failed**: the first part held (+0.152); the second failed (0.20 vs 0.53) |
| H4 | C rejects INJ3 ≥ v2 | **held**: 0.228 vs 0.142 |
| H5 | C rejects ≥ 0.95 of the NTP test | **held**: 0.999 |
| H6 | C's confirmed-vs-FP ROC-AUC within 0.01 of ExoMiner's on the common KOIs | **held**: 0.9961 vs 0.9963 |
| H7 | C rejects ≥ 0.80 of centroid-only FPs | **held**: 0.800 (64 of 80; interval 0.70–0.87) |
| H8a | NTP training drives the INV gain (A, C > B) | **failed**: B (diagnostics only) 0.948 > A (NTP only) 0.837 |
| H8b | The diagnostics drive the INJ2 gain (B, C > A) | **held**: 0.205, 0.199 vs 0.043 |

**6 of 9 held.** Every C-vs-v2 difference survives Holm correction (largest adjusted p = 2 × 10⁻⁹).

## 4. Reading it

**1. The headline.** Training data and pipeline measurements, not architecture, close the gap to the state of the art.
- The network and recipe are v2's, with a 32-unit branch added.
- On the Kepler test it now equals ExoMiner, a larger network built around the same kind of diagnostics. On the fresh false alarms it equals the Robovetter, whose thresholds were set with DR25 simulated data (Thompson et al. 2018).
- The paper's data-centred thesis is confirmed by a fix, not only by a diagnosis.

**2. Which addition does what (the ablation, H8).**
- **The diagnostics do most of the work.** They alone take false-alarm rejection from 0.70 to 0.95, and neighbouring-star rejection from 0.49 to 0.85.
- **NTP training alone helps on real NTPs** (0.76 → 0.94) but much less on simulated INV (0.70 → 0.84). It also *lowers* centroid rejection (0.36): the view branch learns "transit-like = keep", and nothing in the views separates a neighbour's transit.
- **Together (C) they reach 0.991 and 0.999.** The two are complementary, but the attribution we registered (NTP drives INV) was wrong.

**3. The completeness trade (H2 failed).** C keeps 84 % of injected planets, against 93.5 % for v2. This is an operating-point effect, not a loss of skill:
- **As a ranker C dominates:** INJ1-vs-INV ROC-AUC 0.989 vs 0.940.
- **Why the threshold moved.** It was fixed as recall 90 on validation KOIs + NTPs, where planets are high-S/N. Injected planets are weak (median MES 9.8), and NTP training taught the model to distrust weak signals.
- **Where it ends up.** At that threshold C sits exactly at the Robovetter's operating point: 0.842 vs 0.859 kept (p = 0.14) and 0.991 vs 0.984 rejected (p = 0.17).
- **For a survey that values completeness,** a lower threshold trades back along a better curve.

**4. What remains (H3 failed): subtle off-target and eclipsing-binary signals.** The pipeline measurements of the simulated cases are much weaker than those of the KOI false positives C learned from (post-hoc diagnosis):

| Measurement (median) | KOI training FPs | Fresh simulations |
|---|---|---|
| centroid score | **0.001** (centroid-only KOI FPs; confirmed planets 1.0) | **0.62** (INJ2) |
| secondary-eclipse significance | **14** (KOI EBs) | **6.6** (INJ3) |
| MES | **706** (KOI EBs) | **24** (INJ3) |

- Half of the off-target injections look on-target to the centroid test. The EB mimics are faint eclipsing binaries with marginal secondaries, a regime the KOI false positives barely sample.
- **The Robovetter** applies fixed thresholds to the same measurements and catches 53 % and 63 %.
- **The learned model** puts its boundary where its training examples are, far out in the extreme tail.
- **This is the same lesson as the survey priors:** a learned vetter inherits the *distribution* of its training false positives. The fix this points to is simulated off-target and EB training examples (INJ2/INJ3 pools on other stars). It is untested here, because it would turn these sets from tests into training data.

**5. Circularity (stated, not removed).**
- The diagnostics are the measurements from which the Robovetter, and then human vetters, derived the KOI dispositions. So part of C's gain on *KOI* labels (T3) is in-sample.
- ExoMiner uses the same kind of DV diagnostics, so the ExoMiner comparison is like for like.
- The simulated sets (T1) and the NTP labels (T2) are free of this circularity. Their truth is known by construction, or is "not a KOI".

**6. Scope.** The diagnostics exist only for Kepler DR25, so v3 is a Kepler vetter. **v2 remains the model for TESS, K2 and the prospective register**, and the cross-mission results are unchanged.

## 5. Status of critique 1

| The critique's specifics | Before (v2) | After (v3-C) |
|---|---|---|
| Systematics false alarms passing | 30 % (fresh INV) | **0.9 %** |
| Real non-KOI false alarms passing | 24 % | **0.1 %** |
| Neighbouring-star FPs caught | 49 % | **80 %** |
| Gap to ExoMiner (confirmed vs FP, common KOIs) | −0.019 | **−0.0002** (n.s.) |
| Subtle simulated off-target / EB mimics caught | 5 % / 14 % | 20 % / 23 % (the Robovetter: 53 % / 63 %) |

The "not a complete vetter" critique is answered for real KOIs and real false alarms. The remaining weakness is specific, measured and explained.
