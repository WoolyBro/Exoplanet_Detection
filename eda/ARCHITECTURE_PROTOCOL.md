# Is the architecture the limit? A controlled comparison (pre-registered)

**Written and committed before any model below is trained.**
**Code:** `stage_a_transit_model/vetter_v3.py --config R|T`; evaluation `stage_a_transit_model/evaluate_architectures.py`.

## Why

The paper's thesis is that data, not architecture, limits a transit vetter. So far that rests on two facts:
- v2's CNN + BiLSTM is a standard design (the 2018 AstroNet family plus an LSTM);
- every large gain came from data: removing signal twins (split v2), adding real false alarms, and adding pipeline measurements.

A reviewer can still say that a modern architecture might have done better. This protocol tests that directly.

## Design (fixed now)

**Fixed for every model:**
- data: Kepler v2 train/validation views with the flux-error channel, KOIs only, no diagnostics;
- training recipe (`vetter_v3.HP`), augmentation (mirror flip), class weights and seeds (42, 1, 2, 7, 11);
- model selection: the best validation ROC-AUC epoch.

**Two alternatives to v2:**

| Config | Architecture | Size |
|---|---|---|
| v2 (baseline, existing runs) | dual CNN + BiLSTM over the global and local views | 40,451 parameters |
| **R** | 1-D ResNet per view: a stem conv, then 3 stages of 2 residual blocks (32/64/128 channels, stride 2, batch norm), global average pooling, then the v2 head | ≈ 0.5 M parameters |
| **T** | Transformer per view: non-overlapping patch embedding (global 25 bins, local 9 bins → 80 and 22 tokens), learned positions, 3 encoder layers (d = 64, 4 heads, feed-forward 128, dropout 0.1), mean pooling, then the v2 head | ≈ 0.2 M parameters |

Both alternatives are larger than v2, so a "more capacity" advantage, if any, is given to them.

## The test (validation only; every test set is spent)

The test is the 5-seed ensemble ROC-AUC (planet-like vs false positive) on the Kepler v2 validation split, with a star-bootstrap CI and a paired difference from v2. It is set against the data effect already measured on the same split: config B (v2 + pipeline diagnostics) scored 0.970–0.973 per seed against v2's 0.928.

| ID | Expectation |
|---|---|
| X1 | Neither R nor T beats v2's ensemble by more than 0.01 ROC-AUC |
| X2 | The best architecture gain over v2 is less than a quarter of the data gain (B's ensemble minus v2's) |

**Interpretation, fixed now.**
- **X1 and X2 hold:** within this data, the architecture is not the limit, and the thesis is supported by a controlled experiment, not asserted.
- **X1 fails:** the gain is reported, and the paper says a modern architecture helps. The v2/v3 results stand, because every comparison in the paper was like for like.
