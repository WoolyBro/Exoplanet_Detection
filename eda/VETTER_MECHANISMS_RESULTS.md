# What the vetter learned — uncertainty, attention and the causal survey prior (critique 2)

**Code:** `stage_a_transit_model/critique2_analyses.py`
**Output:** `outputs/stage_a_mechanisms/mechanisms.json`
**Figure:** `figures/18_vetter_mechanisms.png`

**Data.** The frozen v2 Kepler ensemble, on the Kepler v2 **validation** split. Two sets already read once are used descriptively: K2, and the DR25 benchmark. No test split is read.

> **Critique 2 was "the method is not new."** CNN + BiLSTM over phase-folded views is standard. What this project adds is evidence about *how* such a vetter decides and where it fails, established by experiment rather than asserted. Three results:
> 1. **Its errors are predictable.** Seed disagreement flags them, and abstaining on the most uncertain half cuts the Kepler validation error rate from 16.8 % to 5.1 %.
> 2. **It looks at the right physics.** Occluding the transit core drops planet scores by up to 0.25. For eclipsing binaries, the most influential global window is phase 0.5, where the secondary eclipse falls.
> 3. **Its survey prior is causal, and acts through signal-to-noise.** Deep K2 planets that it under-scores (median 0.72) **rise to 0.95 when 5 % of their depth in noise is added**. Lowering the information makes them more planet-like: the network learned "too clean and deep = eclipsing binary" from Kepler's training data (94 % of transits deeper than 10,000 ppm there are false positives).

## 1. Uncertainty and selective prediction

The uncertainty of each signal is the standard deviation of the 5 seeds' planet-like scores.

| Set | n | Error rate at the threshold | AUROC: disagreement predicts an error | Error rate keeping the 50 % most certain | ROC-AUC, all → 50 % most certain |
|---|---|---|---|---|---|
| Kepler v2 validation | 1,415 | 0.168 | **0.76** | **0.051** | 0.932 → 0.978 |
| K2 (never trained on) | 1,594 | 0.307 | 0.68 | 0.194 | 0.854 → 0.909 |
| DR25 injections + inverted | 2,000 | 0.152 | 0.58 | 0.125 | 0.952 → 0.972 |

**Reading it.**
- In-distribution, the ensemble knows when it is unsure, so a vetter can route its uncertain half to a human and be right 95 % of the time on the rest.
- Out of distribution the signal weakens (0.68 on K2, 0.58 on simulated false alarms). Seed disagreement is not enough to catch errors the model makes *confidently*. The DR25 false alarms are such errors, and they are fixed by data (vetter v3), not by abstention.

## 2. Occlusion: where the decision lives

Windows of the view are replaced by the out-of-transit level, and the change in score is averaged over 250 confirmed planets and 250 eclipsing-binary false positives (validation).
- **Planets.** Blanking the in-transit core (phase −0.5 to +0.5 durations, local view) lowers the score by **0.18–0.25**. Ingress, egress and the baseline matter little.
- **Eclipsing binaries.** They stay rejected under almost any single occlusion; the largest local change is −0.04. In the global view, the window that raises their score most (+0.04) is the one centred on **phase 0.5**: hiding the secondary eclipse makes them look more like planets.

## 3. The survey prior is causal, and acts through S/N

Views are depth-normalised (the deepest bin is −1), so the network cannot see depth directly; it sees signal-to-noise.
- **The intervention.** Add Gaussian noise of σ (in units of the transit depth) to both views, and raise the error channel to match. If the Kepler prior "deep = eclipsing binary" acts through S/N, a deep, clean planet should *gain* score as information is removed.

| Group | n | σ = 0 | 0.01 | 0.02 | 0.05 | 0.10 | 0.20 |
|---|---|---|---|---|---|---|---|
| **K2 confirmed, deeper than 10,000 ppm** | 27 | **0.72** | 0.82 | 0.89 | **0.95** | 0.96 | 0.83 |
| K2 confirmed, shallower than 3,000 ppm | 200 | 0.89 | 0.91 | 0.91 | 0.91 | 0.92 | 0.81 |
| TESS light curves of the Stage C planets | 41 | 0.70 | 0.71 | 0.76 | 0.84 | 0.85 | 0.73 |
| Kepler confirmed, S/N > 100 (validation) | 57 | 0.98 | 0.98 | 0.98 | 0.99 | 0.99 | 0.86 |
| Kepler eclipsing binaries, S/N > 100 (validation) | 239 | 0.02 | 0.12 | 0.13 | 0.13 | 0.33 | 0.64 |

**Reading it.**
- **The depth prior is caused by S/N.** Deep K2 planets gain 0.23 from a few per cent of noise, and shallow ones barely move. The same happens to the hot Jupiters in TESS light curves.
- **Inside its own survey the prior is invisible.** High-S/N Kepler confirmed planets already score 0.98. The prior only bites on deep transits from *other* surveys, which is exactly the transfer failure of `figures/08` and of `K2_BLIND_TEST_RESULTS.md` §4.
- **Eclipsing-binary rejection depends on high S/N.** At 10–20 % noise, eclipsing binaries drift to 0.33–0.64: the fine detail that exposes them (secondary eclipse, V shape) is the first casualty of noise.
- **What it implies.**
  - Mission-specific priors are learned features of the label distribution, not of the light curves.
  - A cross-survey vetter needs either the target survey's labels or S/N-balanced training (e.g., noise augmentation of high-S/N training examples).

## 3b. Closing the loop: removing the S/N–label dependence from training (pre-registered: `eda/SNR_BALANCE_PROTOCOL.md`, commit `47a6930`)

**Outputs:** `outputs/stage_a_mechanisms/snr_balance.json`, `snr_balance_k2_scores.csv`, `figures/19_snr_balance.png`.
**Code:** `vetter_v3.py --config N`, `evaluate_snr_balance.py`.

**The test.** If the prior is learned from the training labels, a model trained so that S/N carries no label information should lose most of it.
- **Model N** is the v2 recipe with one change: per-example weights w = P(c) / P(c | S/N decile), clipped to [0.2, 10].
- It was trained with 5 seeds.
- **K2 was read a second time**, pre-declared and only for N (guard `runs/snr_balance/K2_READ.json`). The deep-planet subset was chosen *because* v2 failed on it, so this read tests the mechanism, not general skill.

| ID | Expectation | v2 | N | Outcome |
|---|---|---|---|---|
| N1 | Validation ROC-AUC ≥ v2 − 0.01 | 0.932 | 0.925 | **held** |
| N2 | K2 confirmed deeper than 10,000 ppm: median ≥ 0.85 | 0.72 | **0.84** (Wilcoxon p = 7 × 10⁻⁷ for the rise) | **failed**, narrowly |
| N3 | Deep-vs-shallow gap shrinks by at least half | 0.22 | **0.077**, change [−0.28, −0.08] | **held** (−65 %) |
| N4 | K2 P1 ROC-AUC ≥ v2 − 0.01 | 0.863 | **0.834** | **failed** |
| N5 | High-S/N validation FPs rejected at recall 90: ≥ v2 − 0.05 | 0.985 | 0.940 | **held**, narrowly |

Descriptive: the TESS light curves of the Stage C planets (mostly hot Jupiters) rise from a median of 0.70 to 0.83.

**What it shows.**
1. **The prior is learned from the training labels and is mostly removable.**
   - The deep-planet penalty on K2 falls by about two-thirds: the gap goes from 0.22 to 0.08, and the deep-vs-shallow test goes from p = 2 × 10⁻⁶ to p = 0.05.
   - This was done by changing nothing but the weight of training examples. The explanation of §3 thus predicted a fix, and the fix worked in the predicted direction.
2. **The prior is not pure error: it carried real base-rate information** (post-hoc breakdown of the same read).
   - N raises deep false positives too: their median goes from 0.05 to 0.12.
   - Within deep K2 transits, discrimination is unchanged: ROC-AUC 0.689 → 0.681.
   - The overall K2 loss comes mostly from shallow signals (0.836 → 0.796), where N is simply a slightly weaker network, as on validation (−0.007).
   - In short, the prior acted as a **level shift** on deep signals, not as a shape feature. Removing it moves deep planets and deep eclipsing binaries up together.
3. **The pre-registered reading applies:** N4 failed, so the fix is reported as a **trade-off, not a fix**. The remedy this points to, **untested here** because it would need a third read of K2, is per-survey recalibration *conditional on depth or S/N*, fitted on the target survey's own labels. That is a small correction, not retraining. This agrees with E7 of the K2 test: the frozen Platt map does not transfer.

## 4. Where this leaves the method's contribution

Taken together with the rest of Stage A:
- **Data hazards found and measured:** signal-twin leakage; survey priors that act causally through S/N; the absence of false-alarm examples in KOI-based training (DR25 INV: 24 % pass).
- **A protocol:** pre-registration, single guarded reads, and external tests on a new mission (K2) and on the mission's own simulations (DR25).
- **Fixes tested on fresh data:**
  - vetter v3 (`VETTER_V3_PROTOCOL.md`);
  - S/N-balanced training (§3b). It removes about two-thirds of the survey prior but costs 0.03 ROC-AUC on K2, because the prior was partly correct base-rate information.
- **Operating guidance:** abstention on seed disagreement; per-survey recalibration.

The architecture is deliberately standard, so that every effect above can be attributed to data and evaluation rather than to a new network.
