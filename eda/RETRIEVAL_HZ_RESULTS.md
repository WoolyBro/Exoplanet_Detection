# Contamination-aware retrievals of two more habitable-zone planets (critique 5)

**Protocol, committed before any retrieval:** `stage_b_atmospheres/RETRIEVAL_HZ_PROTOCOL.md` (commit `4b6b107`).
**Code:** `stage_b_atmospheres/retrieval_real_general.py` (`TARGET=...`; `--report` applies the credit rule).
**Outputs:**
- `outputs/retrieval_trappist1f/` and `outputs/retrieval_lhs1140b/` (`results.json`, `summary.json`, `posteriors/`);
- figures `eda/retrieval_trappist1f.png` and `eda/retrieval_lhs1140b.png`.

> **Critique 5 was "there is no new science result".**
>
> TRAPPIST-1 e was the only planet given the visit-by-visit contamination method (`FORECAST_RESULTS.md` §4d–4e), and the method only re-checked a published conclusion. It is now applied, unchanged and under a pre-registered credit rule, to the two other habitable-zone rocky planets in the ranking that have public JWST spectra:
> - **TRAPPIST-1 f:** five NIRISS visits that were released in August 2026, after this project's data pack;
> - **LHS 1140 b:** the planet whose N₂ atmosphere claim is debated.

## Method (identical for both planets)

- **Forward model.** petitRADTRANS, isothermal (T 150–400 K), with CH₄, CO₂ and H₂O (log VMR −10 to −0.5) and a surface pressure of 10⁻³–10 bar.
- **Three hypotheses:**
  - an opaque disc (bare rock);
  - a secondary atmosphere with N₂ background;
  - an H₂/He envelope.
- **Contamination, per spectrum.** Each spectrum gets its own linear contamination: an unocculted-spot term (0.86 T_eff) and a facula term (T_eff + 150 K), from PHOENIX, plus a depth offset. Priors are N(0, 0.5), N(0, 0.5) and N(0, 500 ppm), marginalised analytically.
- **Evidence.** dynesty, 400 live points, Δln Z = 0.3.
- **Credit rule R1 (fixed before the runs).** An atmosphere is credited only if ln B(atmosphere vs bare) ≥ 3 **and** the χ² improves in at least ⌈2/3⌉ of the spectra.

## 1. TRAPPIST-1 f: no atmosphere credited (R4 held)

**Data:** NIRISS SOSS visits 1–5 (Lim et al. 2026, arXiv:2608.17207), 0.73–2.80 µm, 249 points. The combined spectrum (file 6) is excluded to avoid double counting.

| Model | ln Z | ln B vs bare rock | Best log-likelihood |
|---|---|---|---|
| bare rock | −1706.66 ± 0.09 | — | −1705.13 |
| N₂ background | −1707.36 ± 0.14 | **−0.70 ± 0.16** | −1701.07 |
| H₂/He envelope | −1709.75 ± 0.15 | **−3.09 ± 0.17** | −1705.09 |

**Reading it.**
- **No atmosphere is credited.** The best atmosphere (N₂) fits better by 4 in log-likelihood using 5 more parameters, and the evidence does not reward it (ln B = −0.70, inconclusive and leaning bare). R1 fails on its first condition.
- **The per-spectrum condition is met but uninformative.** The χ² improves in all 5 visits, by 1.9, 3.1, 0.5, 2.0 and 0.5. The atmosphere model nests a flat spectrum, so small gains everywhere are expected.

  > **Lesson for the rule.** The ⌈2/3⌉ condition guards against a preference carried by a few visits, as for TRAPPIST-1 e. It is not a detection test by itself. ln B carries the weight.
- **A clear H₂-rich envelope is disfavoured** against both bare rock (ln B = −3.1, "moderate" on the Trotta 2008 scale, 2.5–5) and an N₂ atmosphere (+2.4). A low-mean-molecular-weight atmosphere would show large features in these data.
- **Upper limits** (95 %, N₂ background): log VMR(CH₄) < −3.37, log VMR(CO₂) < −1.76, log VMR(H₂O) < −1.19.
  - **CH₄** is excluded above ≈ 400 ppm even in an N₂ atmosphere. That agrees with Lim et al.'s rejection of a CH₄-dominated atmosphere, and with the CH₄ classifier (P = 0.03; `METHANE_EXTENSION_RESULTS.md` §10).
  - **CO₂ and H₂O** are not constrained below the percent level. The NIRISS range misses the 4.3 µm CO₂ band.
- **The visit-to-visit depth differences need no planet.** The fits are good with contamination alone (bare rock χ²/N = 1.09, 0.92, 0.58, 0.78, 0.49). The ~400 ppm spread between visits is absorbed by the per-visit spot, facula and offset terms, as R4 predicted.

**What is new.**
- Lim et al. analysed these visits with their own contamination treatment. This is an independent re-analysis by a method calibrated on a different planet of the same star (TRAPPIST-1 e, 8 visits).
- It gives the same answer: TRAPPIST-1 f's NIRISS spectrum is consistent with bare rock or a high-mean-molecular-weight atmosphere, and not with an H₂ envelope.
- **For the habitability factor,** the 0.33 fitted retention probability for f (`STAGE_C_RESULTS.md` §2c) is not contradicted by the data. Transmission spectroscopy cannot yet separate N₂ atmosphere from bare rock at this precision.

## 2. LHS 1140 b

LHS1140B_PLACEHOLDER
