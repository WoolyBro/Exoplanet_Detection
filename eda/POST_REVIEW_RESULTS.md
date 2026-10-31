# Post-review results

Rules: `eda/POST_REVIEW_PROTOCOL.md`, written before each analysis. Everything here is **post hoc** and is not counted among the 66 expectations.

| Analysis | Code | Result | Reading under the fixed rule |
|---|---|---|---|
| A. Fair architecture comparison | `stage_a_transit_model/nested_architectures.py` | Kepler (7,990): CNN-BiLSTM 0.917, ResNet 0.914, Transformer 0.915. TESS (4,942): 0.761, 0.753, 0.758 | No material gain (limit 0.01). The TESS interval allows about +0.011 |
| B. Label-shift baseline | `stage_a_transit_model/label_shift_baseline.py` | ECE, depth correction vs Platt refit: 0.090 vs 0.052, 0.072 vs 0.048, 0.071 vs 0.077 | Not label shift by depth; call it recalibration |
| C. Noise vs shape | `stage_a_transit_model/noise_vs_shape.py` | Median 0.716 to 0.948 everywhere, 0.838 out of transit, 0.557 in transit, 0.612 near phase 0.5 | Out-of-transit share 53%, below two-thirds: not separated, S/N favored |
| D. CH4 masking | `stage_b_atmospheres/ch4_band_mask.py` | GBM drop 0.49 vs −0.002, p = 6 × 10⁻⁷; ROC-AUC 1.00 to 0.80 | Specific to the bands |
| E. Retention priors | `stage_c_priority_fusion/retention_prior_sensitivity.py` | P(β<0) 0.986 registered, 0.85–0.89 neutral, intervals include 0 | Direction depends on the prior |
| F. Design baseline | `stage_c_priority_fusion/design_baseline.py` | Median rank 29.5 (information) vs 28.5 (S/N per visit) | "Independently recovers" dropped |
| H. Statistics | `tools/post_review_stats.py` | Holm m = 15, largest adjusted p 0.009; Newcombe intervals; CO2 labels vs exoplanet.eu κ 0.58 | Corrected in the paper |
| I. Known-truth injection | `stage_b_atmospheres/detector_injection.py` | Above 3 × noise the GBM recovered 100% at every abundance, 0 false calls before injection | Sensitive on real noise; precision sets the limit |
| J. Molecule baselines | `stage_b_atmospheres/molecule_baselines.py` | CH4: temperature 0.68, radius 0.71. CO2: 0.82, 0.84 | Planet properties do not explain the ranking |
| K. Retention comparison | `stage_c_priority_fusion/retention_model_comparison.py` | ln B vs intercept: magma +1.26, slope −1.22, both +0.97 | The magma term is needed; the slope is not yet |
| L. Priority decomposition | `stage_c_priority_fusion/priority_decomposition.py` | Without observability, 9 of the top 10 stay (Spearman 0.84); TRAPPIST-1 d falls to 11th. Without HZ 0.26 (5 of 10); without the rocky term 0.88 (9 of 10). Script corrected on 2026-10-31 to apply the 0.40 and 0.30 weights; the ranks and the Spearman value did not change | Priority and habitability terms can be read apart |
| M. Label evidence | `tools/label_evidence.py`, `tools/label_audit.py` | No clear contradiction; 12 of 81 flagged; removing them leaves ROC-AUC almost unchanged | Not a substitute for a human check |
| G. Retrieval injection | `stage_b_atmospheres/injection_recovery_hz.py` | Hydrogen envelope credited on all three (ln B +320 to +2,330). Nitrogen credited on TRAPPIST-1 e only (+11.1 in 3 of 4 visits at 400 ppm CO2, +22.0 in 4 of 4 at 1% CO2), not on TRAPPIST-1 f or LHS 1140 b (−0.53 to −0.17). Real TRAPPIST-1 e with the hydrogen model: +3.62 ± 0.20 against bare rock, nitrogen preferred by +3.7 ± 0.3 | Nulls constrain hydrogen, not 1-bar nitrogen, on f and LHS 1140 b; Earth-like credit on e rests on contaminated visits |
| N. Figure support | `stage_a_transit_model/embeddings_v2.py`, `stage_b_atmospheres/trappist1e_fit_curves.py` | Vetter v2 embedding on Kepler validation: nearest-neighbor accuracy 0.755, majority class 0.506, shuffled labels 0.472. Best-fit TRAPPIST-1 e curves saved without a new retrieval | Validation data only; no test set read |
| O. Learning curve | `stage_a_transit_model/learning_curve.py` | Doubling the training set from 50% to 100%: Kepler +0.005 [+0.002, +0.007], TESS −0.002 [−0.013, +0.009]. Out-of-fold ROC-AUC at 12.5, 25, 50, 100%: Kepler 0.898, 0.904, 0.912, 0.917; TESS 0.695, 0.757, 0.763, 0.761 | Both below the 0.01 limit: more examples of this kind are no longer the limit |
| P. Rocky Worlds forecast | `stage_c_priority_fusion/rocky_worlds_predictions.py` | Six targets, P(retain) 0.24 to 0.54; 2.2 expected atmospheres [1.0, 3.5]; P(none) 0.10; P(five or six) 0.05; SHA-256 db14e8ab7b13a4e90f4ea90bf33f4647260b44eb6acb6663a90fb928cbaa3c13 | Locked 2026-10-31; open until at least three verdicts exist |
