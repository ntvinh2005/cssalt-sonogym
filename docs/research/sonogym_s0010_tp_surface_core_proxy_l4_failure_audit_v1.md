# s0010 TP surface-core proxy L4 full-image failure audit

This report audits the frozen `SIMULATOR_GEOMETRY_PROXY` checkpoint on the already-opened
`s0010` development set.  It does not fit a model, change the 0.50 operating thresholds, collect
images, or establish anatomical recognition.

## Result

**The reported L4 precision loss is not one homogeneous model failure: 7/14 false-positive images
contain projected L4 core evidence that the strict proxy contract excludes because accepted
proposals also touch the declared uncertainty region; only 4/14 are clean proxy-negative
morphology confusions.**

| outcome | measured result | decision |
|---|---:|---|
| frozen L4 replay | TP 12, FP 14, FN 5, TN 72 | exact reproduction |
| frozen L5 replay | TP 16, FP 4, FN 3, TN 80 | exact reproduction |
| L4 FP primary cause | proxy coverage/uncertainty 7; clean morphology 4; competing L5 2; proposal multiplicity 1 | heterogeneous |
| model fitting | `fit_called=false` | no retraining |
| project decision | `EXPLORATORY_TRAINING_PASS`; `ACTIVE_ACQUISITION_INTEGRATION_HOLD` | unchanged |

## 1. Frozen replay — whether the audit reproduces the original decision

**Criterion.** The audit must reproduce the exact frozen L4 and L5 full-image confusion counts,
use checkpoint SHA-256 `338351fdee5a8444dee5a6ec250d828b8f1eedbc3f76e79495aa1aafbdeb2f8b`,
retain both 0.50 thresholds, and never call `fit()`.

### 1.1 Method

Every actor-visible proposal in all 103 development images was re-extracted through the production
feature path and scored by the saved 240-tree ExtraTrees checkpoint.  The image decision remains
positive when any proposal passes the frozen class decision.  Only after that decision was fixed
did the audit join evaluator-only L4/L5 core and uncertainty projections.

### 1.2 Setup

| item | value |
|---|---|
| successful job | `51936904`; 4 CPU; 16 GB requested; 0 GPU; 29 s |
| execution commit | `6688af7854ced99d4f29577520b6e75e24819147` |
| dataset | `s0010_tp_surface_core_dataset_v10/51831979` |
| frozen training run | `s0010_tp_surface_core_simulator_proxy_exploratory_v1/51836756` |
| audit peak memory | 454,932 KiB |
| failed predecessor | `51936414`; checkpoint module alias missing; preserved, then fixed |

```bash
python workflows/two_stage_navigation/audit_tp_surface_core_proxy_failures.py \
  --dataset-dir "$SOURCE_RUN/dataset" \
  --training-dir "$TRAINING_RUN" \
  --source-zip "$SOURCE_RUN/s0010_tp_surface_core_review_v3_51831979.zip" \
  --output-dir "$RUN_DIR/audit" \
  --surface-slab-tolerance-mm 1.5
```

### 1.3 Results

The replay reproduced L4 `12/14/5/72` and L5 `16/4/3/80` for
TP/FP/FN/TN.  The audit recorded `fit_called=false`.  Seven new focused tests and four existing
proxy-training regression tests passed.

### 1.4 Reading it

Exact reproduction makes the case audit traceable to the original checkpoint.  It does not turn
opened development data into independent confirmation.

### 1.5 Files

`audit_result.json`, `l4_failure_proposal_scores.csv`, the test logs, and the timing records are in
the job directory listed in Section 5.

## 2. Fourteen L4 false positives — evidence and corresponding repair direction

**Criterion.** Every one of the 14 frozen false-positive images must have a clean image, all
actor-visible proposal boxes, every L4-triggering crop, all three class probabilities, a separate
privileged proxy overlay, and an evidence-backed primary classification.

### 2.1 Method

Primary classifications use the following precedence: a triggering proposal overlapping the L5
core; otherwise projected L4 evidence excluded by the uncertainty/support contract; otherwise
proposal multiplicity; otherwise clean proxy-negative morphology confusion.  Supporting causes
remain recorded, so the primary counts are not presented as mutually exclusive mechanisms.

### 2.2 Results

| case | session:frame | accepted L4 proposals | max L4 probability | evaluator-only evidence | primary cause | corresponding repair direction |
|---|---|---:|---:|---|---|---|
| FP01 | `v9_dev_l4:19` | 1 | 0.5375 | no projected core/uncertainty in trigger | morphology confusion | calibration-only L4-vs-rest margin |
| FP02 | `v6_dev:6` | 2 | 0.5500 | 12 L4 core pixels; both triggers touch uncertainty | proxy coverage/uncertainty | review this exact proxy exclusion; do not tune model to reject it |
| FP03 | `v6_dev:5` | 1 | 0.6083 | 38 L4 core pixels; trigger touches uncertainty | proxy coverage/uncertainty | review this exact proxy exclusion |
| FP04 | `v9_dev_l4:22` | 1 | 0.5500 | no projected core/uncertainty in trigger | morphology confusion | calibration-only L4-vs-rest margin |
| FP05 | `v6_dev:3` | 1 | 0.5417 | trigger has no core/uncertainty overlap; L5 exists elsewhere | morphology confusion | calibration-only L4-vs-rest margin |
| FP06 | `v10_dev_l4l5:13` | 1 | 0.5667 | 31 L4 core pixels; trigger touches uncertainty | proxy coverage/uncertainty | review this exact proxy exclusion |
| FP07 | `v6_dev:10` | 1 | 0.5208 | 5 L4 core pixels; trigger touches uncertainty | proxy coverage/uncertainty | review this exact proxy exclusion |
| FP08 | `v6_dev:39` | 1 | 0.5083 | trigger overlaps competing L5 core; 26 L5 core pixels | cross-level confusion | calibration-only L4-vs-L5 margin |
| FP09 | `v9_dev_l4:2` | 1 | 0.5208 | 23 L4 core pixels; trigger touches uncertainty | proxy coverage/uncertainty | review this exact proxy exclusion |
| FP10 | `v10_dev_l4l5:23` | 1 | 0.5000 | trigger overlaps competing L5 core; 56 L5 core pixels | cross-level confusion | calibration-only L4-vs-L5 margin |
| FP11 | `v10_dev_l4l5:17` | 3 | 0.6708 | 42 L4 core pixels; all three triggers touch uncertainty | proxy coverage/uncertainty | review exclusion; retain multiplicity as secondary evidence |
| FP12 | `v9_dev_l4:17` | 2 | 0.6083 | neither trigger overlaps projected core/uncertainty | proposal multiplicity | calibration-only image aggregation test |
| FP13 | `v9_dev_l4:9` | 1 | 0.6292 | 25 L4 core pixels; trigger touches uncertainty | proxy coverage/uncertainty | review this exact proxy exclusion |
| FP14 | `v9_dev_l4:21` | 1 | 0.5208 | no projected core/uncertainty in trigger | morphology confusion | calibration-only L4-vs-rest margin |

The 14 images contain 18 L4-triggering proposals.  Proposal multiplicity is nonexclusive evidence
in FP02, FP11 and FP12, but it is the primary explanation only for FP12.  The false-positive
maximum L4 probabilities span 0.5000--0.6708, with mean 0.5595.

### 2.3 Reading it

- The seven coverage/uncertainty cases are false positives under the frozen proxy contract, but
  they are not clean evidence that the image model fired on unrelated anatomy.
- The four clean morphology cases are the strongest demonstrated model-specific failure group.
- The two competing-L5 cases show level confusion rather than generic background confusion.
- Increasing the decision threshold on this opened development set would be post-hoc tuning and
  is not a repair.

### 2.4 Files

`l4_false_positive_audit.csv` is the machine-readable 14-row table.  Exact per-proposal background,
L4 and L5 probabilities are in `l4_failure_proposal_scores.csv`.  `assets/*_clean.png`,
`assets/*_context.png`, `assets/*_proxy_overlay.png`, and `assets/proposal_crops/` keep actor-visible
and privileged evidence separate.

## 3. Five L4 false negatives and session/L5 comparison

**Criterion.** All five false-negative images must be exported and the L4-specific pattern must be
compared with L5 without changing either class.

### 3.1 Results

| case | session:frame | max L4 probability | projected L4 core pixels | evidence |
|---|---|---:|---:|---|
| FN01 | `v10_dev_l4l5:2` | 0.4792 | 21 | no proposal reaches 0.50 |
| FN02 | `v9_dev_l4:7` | 0.3333 | 7 | no proposal reaches 0.50 |
| FN03 | `v9_dev_l4:4` | 0.4792 | 43 | no proposal reaches 0.50 despite the largest projected support among FNs |
| FN04 | `v6_dev:13` | 0.3458 | 5 | no proposal reaches 0.50 |
| FN05 | `v10_dev_l4l5:6` | 0.4583 | 24 | no proposal reaches 0.50 |

False-negative maximum L4 probabilities span 0.3333--0.4792, with mean 0.4192.

| session | L4 TP/FP/FN | L4 precision/recall | L5 TP/FP/FN | L5 precision/recall |
|---|---|---|---|---|
| `s0010tpscv6_dev_00` | 4/5/1 | 0.444/0.800 | 7/4/1 | 0.636/0.875 |
| `s0010tpscv9_dev_l4_00` | 2/6/2 | 0.250/0.500 | 0/0/0 | undefined/undefined |
| `s0010tpscv10_dev_l4l5_00` | 6/3/2 | 0.667/0.750 | 9/0/2 | 1.000/0.818 |

### 3.2 Reading it

L4 errors occur in all three sessions and are worst in the L4-focused V9 session, where three of
six false positives are clean morphology confusions.  L5 false positives occur only in V6.  This
does not support changing both classes together: L4 has a distinct proxy-coverage and morphology
problem, while L5 retains materially better pooled precision.

### 3.3 Files

See `l4_false_negative_audit.csv`, `session_l4_l5_comparison.csv`, and
`l4_failure_contact_sheet.png`.

## 4. One bounded repair test to run next

**Criterion.** Keep the checkpoint and original 0.50 class gate frozen; select any additional rule
using calibration only; evaluate once on a newly designated, session-disjoint development set; and
retain at least 90% of the original paired true positives while reducing clean-negative L4 false
accepts.  Cases whose accepted proposals overlap proxy uncertainty must be reported separately, not silently converted into
negatives or positives.

### 4.1 Proposed test

Add one L4 margin after the existing gate:

1. A proposal must still satisfy L4 probability at least 0.50.
2. On calibration only, choose the smallest margin between L4 probability and the larger of
   background/L5 probability that meets the existing 0.90 calibration-precision preference while
   maximizing recall.
3. Freeze that margin before opening a new development session.
4. On new development, report paired original-gate and margin-gate results, clean negatives,
   competing-L5 negatives, and cases whose accepted proposals overlap proxy uncertainty separately.

This directly tests the four clean morphology cases and two cross-level cases.  It does not claim
to repair the seven proxy-coverage cases; those remain an evaluator-label question and are kept as
a separate stratum.

### 4.2 Not executed

No model fitting, checkpoint change, threshold search, new capture, or active-acquisition
integration was performed in this audit.  The next test requires a separately frozen protocol and
new development evidence; the current opened cases may be used for diagnosis only.

## 5. Artifacts and reproducibility

| artifact | location |
|---|---|
| runtime directory | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_tp_surface_core_proxy_l4_failure_audit_v1/51936904/` |
| review ZIP | `s0010_tp_surface_core_proxy_l4_failure_audit_v1_51936904.zip` |
| ZIP SHA-256 | `6278ff84d1cb7e582fef5beb61f6be8baa1c0da39aa25ce00ee1d2302b3964cd` |
| result record | `experiments/unity_two_stage_transfer/s0010_tp_surface_core_proxy_l4_failure_audit_v1_result.json` |

The ZIP contains the complete 14-FP/5-FN evidence bundle; all 86 listed artifact checksums pass.
It is a diagnostic review package, not an independent validation set.
