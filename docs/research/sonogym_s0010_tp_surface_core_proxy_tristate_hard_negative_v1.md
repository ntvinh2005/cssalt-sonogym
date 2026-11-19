# s0010 TP surface-core L4 proxy tri-state and hard-negative preparation

This report records the CPU-only `s0010_tp_surface_core_proxy_tristate_hard_negative_v1`
experiment. It preserves the fitted proxy-v1 checkpoint and applies a three-state L4 proxy
evaluation before mining clear hard negatives from existing training images.

## Result

**The three-state evaluation is reproducible, but the existing training captures contain only 10
independent clear hard-negative images that false-trigger the frozen L4 decision, below the frozen
minimum of 40; L4 v2 fitting is not justified and no retraining protocol was emitted.**

| step | criterion | measured | verdict |
|---|---|---:|---|
| tri-state evaluation | unknown or uncertainty-overlapping evidence must not become negative | 790 ambiguous proposals; 73 ambiguous images | pass |
| opened-development diagnosis | reclassify 14 historical L4 false positives without training on them | 7 ambiguous; 7 negative; 0 training rows | pass |
| hard-negative sufficiency | at least 40 independent training images that satisfy the frozen L4 false-trigger and clear-negative contracts | 10 | fail |
| scope integrity | no fit, capture, development-array loading, historical rewrite or L5 modification | all preserved | pass |

`EXPLORATORY_TRAINING_PASS` remains the proxy-v1 baseline,
`ACTIVE_ACQUISITION_INTEGRATION_HOLD` remains unchanged, and the L4 margin experiment remains
closed as `FAIL`.

## 1. Tri-state proxy evaluation — are uncertainty cases separated from negatives?

**Criterion.** Every proposal and image must be assigned one of
[`PROXY_POSITIVE`](../notation.md#proxy_positive),
[`PROXY_NEGATIVE`](../notation.md#proxy_negative) or
[`PROXY_AMBIGUOUS`](../notation.md#proxy_ambiguous), with positive-before-ambiguous-before-negative
image precedence. Ambiguous rows must remain outside the primary positive/negative metric and
training manifests.

### 1.0 Quantities

| symbol | definition | source |
|---|---|---|
| three proxy states | repository definitions | `docs/notation.md` |
| hard-negative preparation | [`s0010_tp_surface_core_proxy_tristate_hard_negative_v1`](../notation.md#s0010_tp_surface_core_proxy_tristate_hard_negative_v1) | `docs/notation.md` |

### 1.1 Method

- A proposal is positive only when it satisfies the complete existing L4 proxy contract.
- It is negative only when projection is verified and the patch contains zero L4 core and zero L4
  uncertainty intersections.
- It is ambiguous when projection is unverified, L4 uncertainty intersects, or L4 core evidence
  exists without the complete positive contract.
- Whole-image state uses positive, then ambiguous, then negative precedence across every proposal.
- The 14 already-opened development false positives are reclassified descriptively; their image
  arrays are not loaded and none can enter training.

### 1.2 Setup

| what | value |
|---|---|
| job / execution revision | `51956725` / `0a5ebe54d32fdb26d2364c0b9c187c00cc83374c` |
| protocol SHA-256 | `cce6506e49b9bd46ac350aaa0a5097070b69bffed8a06c7627e538cb31fcefd` |
| checkpoint SHA-256 | `338351fdee5a8444dee5a6ec250d828b8f1eedbc3f76e79495aa1aafbdeb2f8b` |
| execution | 4 CPU; 16 GB requested; 0 GPU; 70 s; peak RSS 193,564 KiB |
| input rows | 5,469 proposals in 560 fixed-anatomy development images |

```bash
python workflows/two_stage_navigation/prepare_tp_surface_core_proxy_tristate_hard_negatives.py \
  --training-dir "$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_simulator_proxy_exploratory_v1/51836756/training" \
  --failure-audit-csv "$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_proxy_l4_failure_audit_v1/51936904/audit/l4_false_positive_audit.csv" \
  --protocol experiments/unity_two_stage_transfer/s0010_tp_surface_core_proxy_tristate_hard_negative_v1.yaml \
  --output-dir "$RUN_DIR/preparation"
```

### 1.3 Results

| split | positive images | negative images | ambiguous images |
|---|---:|---:|---:|
| train | 55 | 242 | 50 |
| calibration | 12 | 88 | 10 |
| development evaluation | 17 | 73 | 13 |

Proposal counts are positive `183`, negative `4,496` and ambiguous `790`. The 14 historical L4
development false positives divide exactly into seven ambiguous and seven negative images. This
does not change their historical false-positive status; it explains which failures should remain
outside a clean-negative metric.

**Verdict: pass, because ambiguity is explicit and no ambiguous or development row enters the
hard-negative manifest.**

### 1.4 Reading it

- Half of the old 14 false positives overlap proxy uncertainty and should not be forced into the
  negative class.
- The other seven remain clean proxy-negative errors and are evidence of genuine morphology or
  competing-structure confusion under the simulator-proxy contract.
- These development images remain opened diagnostic data, not confirmation data.

### 1.5 Files

| file | contents |
|---|---|
| `preparation/l4_proxy_tristate_proposals.csv` | state and reason for all 5,469 proposals |
| `preparation/l4_proxy_tristate_images.csv` | state summary for all 560 images |
| `preparation/opened_development_fp_reclassification.csv` | descriptive reclassification of the 14 old false positives |

## 2. Training-only hard-negative mining — is there enough clear error evidence for L4 v2?

**Criterion.** The frozen checkpoint must predict L4 by its unchanged argmax and 0.50 threshold on
at least 40 distinct training images whose selected actor-visible patch is proxy-negative, has no
L4/L5 uncertainty intersection and passes exact image/feature hash checks. At most one selected
patch may come from one source image.

### 2.1 Method

- The original 240-tree checkpoint and production feature extractor are loaded unchanged.
- Candidate patches come only from the `train` split and the actor-visible bright-interface
  proposal generator.
- Geometry eligibility is local to the patch; whole-image state is retained as a diagnostic field.
- Among qualifying false-triggering patches in one image, the highest L4-probability patch is kept,
  with sample ID as the deterministic tie-break.
- Competing L5 core and other bone-interface confusers are counted separately.

### 2.2 Setup

Every loaded training image and re-extracted feature must match the sealed source and feature hash.
The minimum of 40 independent source images is inherited from the existing proxy pilot's minimum
training-image requirement and was sealed before mining.

### 2.3 Results

| quantity | measured |
|---|---:|
| eligible false-trigger proposal rows | 16 |
| independent selected source images | 10 |
| required source images | 40 |
| deficit | 30 |
| selected competing-L5-core confusers | 0 |
| selected other-bone-interface confusers | 10 |
| selected L4 probability range / mean | 0.5125–0.6875 / 0.5950 |

The ten images span all four existing training sessions: `3/2/3/2` selected images from
`train_00/train_01/train_02/train_03`. Every selected source image is itself `PROXY_NEGATIVE`;
none depends on distant image-level ambiguity.

**Verdict: fail, because only 25% of the predeclared independent-image minimum is available.**

### 2.4 Reading it

- Lowering the 0.50 error threshold or padding the set with easy negatives would answer a different
  question and was not done.
- Keeping multiple correlated proposals from the same image would inflate the count and was not
  done.
- The current data contain no high-L4-probability clear competing-L5-core hard negative under this
  contract, so L4-versus-L5 repair is not supported by the mined set.
- The exact blocker for the proposed L4 v2 pilot is 30 additional independent clear hard-negative
  training images, including supported competing-L5 examples if that failure mode is to be repaired.

### 2.5 Files

| file | contents |
|---|---|
| `preparation/l4_hard_negative_candidates.csv` | all 16 eligible false-trigger proposal rows |
| `preparation/l4_hard_negative_selected.csv` | ten deterministic one-per-image rows; SHA-256 `c67f8f52...e97b` |
| `preparation/result.json` | machine-readable sufficiency and scope result |

## 3. Scope and regression integrity — was the historical experiment preserved?

**Criterion.** No model fitting, new capture, threshold change, development training use,
historical-result rewrite or L5 checkpoint modification may occur; focused and existing proxy
regression tests must pass.

### 3.1 Method

The Slurm wrapper checks sealed inputs, runs the new unit tests and the existing proxy-training
regression suite, then verifies that every selected row is train-only and source-image unique.

### 3.2 Setup

Job `51956725` completed with exit code `0:0`. The preparation process used 62.17 seconds wall time
and 190,492 KiB peak RSS; the complete batch used 70 seconds and 193,564 KiB peak RSS.

### 3.3 Results

| check | result |
|---|---|
| focused tri-state/mining tests | 11 / 11 pass |
| existing proxy-training regression | 4 / 4 pass |
| fit called | false |
| new capture | false |
| development rows used for training | 0 |
| L5 role | frozen control |
| L4 v2 protocol emitted | no |

**Verdict: pass, because the diagnostic is deterministic and preserves every frozen boundary.**

### 3.4 Reading it

The useful outcome is a fail-closed data diagnosis, not another trained checkpoint. The next
scientifically coherent action is to design a bounded hard-negative acquisition or an explicitly
smaller ablation protocol; either is a new experiment and requires researcher approval.

### 3.5 Files

Runtime directory:
`/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_tp_surface_core_proxy_tristate_hard_negative_v1/51956725/`.

The repository result is
`experiments/unity_two_stage_transfer/s0010_tp_surface_core_proxy_tristate_hard_negative_v1_result.json`.
