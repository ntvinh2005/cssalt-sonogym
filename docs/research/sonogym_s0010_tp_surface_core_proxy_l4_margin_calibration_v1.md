# s0010 TP surface-core proxy L4 margin calibration

This report records the calibration-only s0010_tp_surface_core_proxy_l4_margin_calibration_v1
experiment.  It keeps the fitted checkpoint, preprocessing, 0.50 class thresholds and L5 decision
unchanged; no development image is opened.

## Result

**The L4 margin experiment fails its predeclared calibration gate because baseline calibration has zero clean-negative false positives to reduce; no margin is frozen and no new development acquisition should run for this rule.**

| step | what was measured | criterion | measured | verdict |
|---|---|---|---|---|
| 1 | whole-image L4 margin on calibration | reduce clean-negative FP and retain at least 90% of baseline TP | baseline clean-negative FP `0`; selected margin `null` | fail |
| 2 | scope integrity | no fit, no development image, no L5 change | all three conditions preserved | pass |

The margin can remove false positives whose accepted proposals overlap uncertainty, but that is not the user-approved target and does
not justify a new independent-development capture.

## 1. Whole-image calibration — can one L4 margin reduce clean-negative errors?

`workflows/two_stage_navigation/calibrate_tp_surface_core_l4_margin.py`, run by Slurm job
`51953988`, scores every calibration proposal with the frozen checkpoint.

**Criterion.** A candidate margin must reduce false-positive images whose accepted L4 proposals
contain no projected core or uncertainty while retaining at least 90% of baseline L4 true-positive
images.  This criterion was approved by the researcher on 2026-11-18 and sealed before execution.

### 1.0 Quantities

| symbol | definition | source |
|---|---|---|
| L4 margin | [repository definition](../notation.md#l4-margin) | `docs/notation.md` |

### 1.1 Method

- The unchanged proposal gate first requires L4 argmax and L4 probability at least 0.50.
- L4 margin candidates are `0.000, 0.025, ..., 0.500`.
- A whole image is L4-positive when any actor-visible proposal passes both gates.
- The required retained count is `ceil(0.90 × 9) = 9` baseline true-positive images.
- Selection minimizes clean-negative false positives, then maximizes true positives, minimizes total
  false positives and finally chooses the smaller margin.
- Uncertainty and competing-L5 activations are reported separately and do not become relabelled
  positives or negatives.

### 1.2 Setup

| what | value | spec |
|---|---|---|
| run / checkpoint | job `51953988`; checkpoint SHA-256 `338351fd...2f8b` | §1 |
| execution | commit `a841e25cedc4144417fa23929e96a16b2df2ccde`; 4 CPU; 16 GB; 0 GPU | §1 |
| protocol | SHA-256 `e561901e5dd6cd39336532dfc60196e5182e6e6357885e27f794c86e0804ee7a` | §1 |
| input audit / selected manifest | SHA-256 `67dfdfe4...3175` / `72f74f70...c871` | §1 |
| allowed data | calibration only | §1 |

```bash
python workflows/two_stage_navigation/calibrate_tp_surface_core_l4_margin.py \
  --training-dir "$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_simulator_proxy_exploratory_v1/51836756/training" \
  --protocol experiments/unity_two_stage_transfer/s0010_tp_surface_core_proxy_l4_margin_calibration_v1.yaml \
  --output-dir "$RUN_DIR/calibration"
```

| sample accounting | value |
|---|---:|
| calibration images attempted | 110 |
| calibration proposals attempted | 1,062 |
| dropped for invalid image, feature or hash | 0 |
| **final calibration images / proposals** | **110 / 1,062** |

### 1.3 Results

| margin | TP | FP | FN | precision | recall | clean-negative FP | uncertainty FP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 baseline | 9 | 3 | 3 | 0.7500 | 0.7500 | 0 | 3 |
| 0.175 | 9 | 2 | 3 | 0.8182 | 0.7500 | 0 | 2 |
| 0.200 | 8 | 2 | 4 | 0.8000 | 0.6667 | 0 | 2 |
| 0.400 | 5 | 0 | 7 | 1.0000 | 0.4167 | 0 | 0 |

No grid point can reduce clean-negative FP below the baseline value of zero.  Margin 0.175 retains
all nine baseline TP and removes one FP, but the removed case has an accepted proposal overlapping uncertainty.

**Verdict: fail, because no candidate satisfies the required clean-negative FP reduction even
though margin 0.175 preserves the baseline TP count.**

### 1.4 Reading it

- Calibration does not contain the clean-negative failure mode that dominates four opened-development errors.
- Selecting 0.175 would optimize against uncertainty labels rather than the requested model error.
- Margins at least 0.200 violate the nine-TP retention gate.
- The result provides no basis for acquiring a new independent development set for this margin.

### 1.5 Files

| file | contents |
|---|---|
| `calibration/result.json` | fail-closed result and scope flags |
| `calibration/calibration_l4_margin_grid.csv` | all 21 predeclared grid points |
| `calibration/calibration_proposal_scores.csv` | 1,062 proposal probabilities and geometry strata |
| `calibration/selected_margin_image_decisions.csv` | baseline image decisions because no margin passed |

## 2. Scope integrity — did calibration avoid retraining, development and L5 changes?

`hpg/jobs/s0010_tp_surface_core_proxy_l4_margin_calibration_v1.sbatch` records the sealed inputs,
tests and resource usage.

**Criterion.** `fit_called=false`, `development_images_opened=false`, L5 decisions unchanged, and
all focused and existing proxy regression tests pass.

### 2.1 Method

- The script checks all three frozen input hashes before loading the checkpoint.
- Only rows whose split is exactly `calibration` reach image loading and feature extraction.
- L5 uses the original argmax and 0.50 threshold; the L4 margin is not passed to its function.
- A frozen-margin file is emitted only after a passing calibration result.

### 2.2 Setup

| what | value | spec |
|---|---|---|
| job | `51953988`; `COMPLETED 0:0`; elapsed 39 s | §2 |
| calibration process | 29.07 s wall time; peak RSS 135,168 KiB | §2 |
| sessions opened | `s0010tpscv6_cal_00`, `s0010tpscv8_cal_l4_00`, `s0010tpscv10_cal_l5_00` | §2 |

```bash
python tests/test_tp_surface_core_l4_margin_calibration.py
python tests/test_tp_surface_core_proxy_training.py
```

Nothing was dropped.

### 2.3 Results

| check | result |
|---|---|
| focused margin tests | 9 / 9 pass |
| existing proxy-training regression | 4 / 4 pass |
| fit called | false |
| development images opened | false |
| frozen-margin file | absent |
| L5 modified | false |
| L5 calibration TP/FP/FN/TN | 28 / 4 / 1 / 77 |

**Verdict: pass, because every information-boundary and regression condition is satisfied.**

### 2.4 Reading it

- `EXPLORATORY_TRAINING_PASS` remains unchanged.
- `ACTIVE_ACQUISITION_INTEGRATION_HOLD` remains unchanged.
- No statement about independent development performance follows from calibration failure.

### 2.5 Files

| file | contents |
|---|---|
| `tests-margin.log` | focused test output |
| `tests-proxy-regression.log` | frozen training-path regression output |
| `context/frozen-inputs.sha256` | protocol, checkpoint and manifest identities |
| `metrics/time-calibration.txt` | CPU and memory accounting |

Runtime directory:
`/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_tp_surface_core_proxy_l4_margin_calibration_v1/51953988/`.
