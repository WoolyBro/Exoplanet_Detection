# Stage B, Level 2 — physics-informed and unsupervised analysis of JWST spectra

**Date:** 2026-09-25
**Data:** the 53 spectra / 13 planets already assembled by Level 1 (1–5 µm, 100 bins,
channels = depth, uncertainty, coverage mask). Level 1's outputs and
`verified_gas_labels.csv` were read only and are unchanged.
**Code:** `stage_b_atmospheres/stage_b_level2.py` → `outputs/stage_b_level2/`
**Figure:** `eda/stage_b_level2_overview.png`

> Scope, stated up front: 53 spectra of 13 planets. Everything here is **descriptive**.
> The clusters are illustrations, and an "anomaly" is a flag for a human to look at — not
> a detection, and not a population result.

---

## 1. Scale-height normalisation

A transmission feature is roughly `2·R_p·H / R_s²` deep, where `H = kT / (µ·m_u·g)`. Dividing
each spectrum by that quantity converts ppm into **number of scale heights**, which is the only
way a hot Jupiter and a sub-Neptune can be compared on one axis. Assumed µ = 2.3 (H₂/He).

| planet | T_eq (K) | R_p (R⊕) | g (m/s²) | H (km) | 1 H depth (ppm) |
|---|---|---|---|---|---|
| WASP-107 b | 736 | 10.48 | 2.95 | 901 | **576.0** |
| WASP-39 b | 1166 | 14.34 | 4.27 | 988 | 422.8 |
| WASP-52 b | 1315 | 14.24 | 7.08 | 671 | 403.3 |
| HAT-P-18 b | 841 | 10.70 | 4.76 | 639 | 350.2 |
| WASP-17 b | 1755 | 20.96 | 5.54 | 1145 | 284.6 |
| TRAPPIST-1 c | 340 | 1.10 | 10.67 | 115 | 233.8 |
| HD 209458 b | 1459 | 15.58 | 9.39 | 562 | 162.8 |
| HD 189733 b | 1209 | 12.67 | 21.98 | 199 | 117.9 |
| TOI-270 d | 383 | 2.00 | 10.31 | 134 | 49.0 |
| K2-18 b | 284 | 2.37 | 15.60 | 66 | **24.3** |

The ordering is a sanity check that passes: WASP-107 b tops it (a genuinely low-gravity
"super-puff"), K2-18 b sits at the bottom with a 24 ppm feature scale, and the two differ by a
factor of **24**. After normalisation the median spectrum varies by **3.2 scale heights**, which
is the physically expected range for a cloud-free atmosphere.

---

## 2. Band indices — which molecules are actually measurable

Band minus a local continuum, in scale heights, uncertainty propagated. Positive = absorption.

| molecule | band (µm) | measurable | median (H) | > 3σ |
|---|---|---|---|---|
| **CO₂** | 4.20–4.42 | 28/53 | **+1.17** | **20** |
| **H₂O** | 1.35–1.50 | 15/53 | **+0.98** | **11** |
| CH₄ | 3.20–3.45 | 32/53 | −0.13 | 9 |
| SO₂ | 4.00–4.12 | 26/53 | −0.34 | 2 |
| CO | 4.55–4.75 | 28/53 | −0.19 | **0** |

This reproduces the known JWST picture without being told the labels: **CO₂ at 4.3 µm is the
strongest and most consistently measurable feature**, H₂O second. CH₄ and SO₂ have a median
near zero (present in some planets, absent in most), and **CO is never recovered at 3σ** — it
sits at the edge of the 1–5 µm grid and overlaps the CO₂ wing, so this method cannot see it.
That is a limitation of the band-index approach, not evidence CO is absent.

Note CH₄ is *measurable* in more spectra than CO₂ (32 vs 28) yet is detected far less often —
coverage is not detection.

---

## 3. Clustering

PCA on the normalised spectra explains 25.3 % / 23.7 % / 13.3 % / 11.3 % — no single dominant
axis, i.e. the spectra are genuinely diverse rather than one family with noise.

2-means on the first three PCs agrees with the physical giant / sub-Neptune split
(R_p ≥ 8 R⊕) **73 %** of the time. It was *checked* against that split, not assumed to
reproduce it. With 12 planets in the clustering set this is an illustration; the honest
reading is "spectral shape carries some size information, but does not cleanly separate the
two families at this sample size."

---

## 4. Anomaly detection — and a correction that changed the answer

Reconstruction error under **leave-one-planet-out** PCA (a linear autoencoder; with 53 samples
a deep one would mostly memorise), plus a small MLP autoencoder as a nonlinear check. The
planet is never in the fit that reconstructs it, so the score means "unlike the other planets"
rather than "badly memorised".

**The raw ranking was wrong and I nearly reported it.** Raw error put K2-18 b first (2.11) and
TOI-270 d second — which is a seductive result, since those are exactly the two contested
sub-Neptunes in the literature. But normalising divides by `A_H`, so a planet with a small
`A_H` has its **noise** inflated most. Checking:

- Spearman(raw error, normalised uncertainty) = **+0.690**
- K2-18 b: error 2.11 vs its own noise 1.93 → ratio **1.09**, i.e. nothing beyond noise
- K2-18 b has the smallest `A_H` in the sample (24.3 ppm)

So K2-18 b topped the list *because it is the noisiest in normalised units*, not because its
spectrum is unusual. Ranking by **error ÷ that spectrum's own noise** instead:

| planet | LOPO error (H) | own noise (H) | excess |
|---|---|---|---|
| WASP-17 b | 0.82 | 0.21 | **3.99** |
| HAT-P-18 b | 0.48 | 0.14 | **3.34** |
| WASP-39 b | 0.67 | 0.14 | **2.93** |
| WASP-107 b | 0.22 | 0.08 | 2.88 |
| HD 189733 b | 0.52 | 0.19 | 2.69 |
| … | | | |
| TOI-270 d | 0.90 | 0.71 | 1.27 |
| K2-18 b | 2.11 | 1.93 | 1.09 |

The corrected list is physically sensible: bright, well-measured hot Jupiters with strong
features. **No claim is made about K2-18 b.** Its spectrum is consistent with its own noise
under this method — which is neither support for nor evidence against anything reported about
that planet elsewhere.

This is the same failure mode as the `secondary_depth` bug in Stage A Level 2: an unnormalised
amplitude that silently measured noise. Both were caught by testing the feature against
something external rather than trusting it.

---

## 5. Limitations

- **µ = 2.3 is assumed for every planet.** It is the standard H₂/He value and is what makes the
  comparison possible at all, but it is wrong for a high-metallicity or secondary atmosphere.
  It is most doubtful for **TRAPPIST-1 c** (rocky; may have little or no H/He) and **K2-18 b**
  / **TOI-270 d** (metallicity debated). Their normalised amplitudes should be read as
  "assuming an H₂/He envelope", not as measurements.
- One spectrum spans 20.3 scale heights (K2-18 b) — far beyond physical. That is the noise
  amplification above, and it is why the anomaly ranking had to be noise-normalised.
- 13 planets. No population inference is possible, and none is made.
- Band indices are a blunt instrument: overlapping bands (CO/CO₂) and grid edges limit them.
  They complement Level 1's classifier, they do not replace it.
- The MLP autoencoder is reported alongside PCA but adds nothing at this sample size; its
  scores track the PCA ones closely.

---

## 6. What this adds to the guide's checklist

- [x] Scale-height normalisation by (T_eq, gravity, stellar radius)
- [x] Band-index features with uncertainty propagated
- [x] Clustering of hot Jupiters vs sub-Neptunes — done and *checked*, not assumed
- [x] Autoencoder anomaly detection — done under leave-one-planet-out, and noise-normalised
- [x] Uncertainty-aware: the uncertainty channel is propagated into every band index and into
      the anomaly normalisation

Still open from the guide: O₃ remains a forward-look/anomaly target with **zero** positive
examples and is not treated as supervised learning anywhere.

---

## 7. Reproducing

```bash
python stage_b_atmospheres/stage_b_level2.py
```
Reads `outputs/stage_b/stage_b_dataset.npz` and the pscomppars catalogue; writes only to
`outputs/stage_b_level2/` and `eda/stage_b_level2_overview.png`.
