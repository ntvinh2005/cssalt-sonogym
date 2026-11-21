# s0010 RIGHT L4/L5 SAP ultrasound validation and conditional training

This experiment validates simulator-proxy SAP-related cortical-interface labels and fits the established image-only ExtraTrees baseline on fixed anatomy `s0010`. It does not establish clinician-verified SAP anatomy, patient generalization, an MBB target, or intervention readiness.

## Result

**Both levels remain scientifically inconclusive: the frozen RIGHT L5 classifier made no error on five clear held-out positives but the positive count and correspondence gates failed, while RIGHT L4 remains untrained.**

| step | what was measured | criterion | measured | verdict |
|---|---|---|---|---|
| 1 | session-separated correspondence | matched at least 0.50; mismatch at most 0.10; top-1 at least 0.70 | L4: 0.40 / 0.1148 / 0.3778; L5: 0.6552 / 0.0747 / 0.7241 | L4 fail; L5 pass |
| 2 | binary image eligibility | no retained negative image contains a non-negative actor proposal | 138 previously selected negative images excluded; 0 added | pass |
| 3 | RIGHT L5 calibration | precision and recall at least 0.80; false-positive rate at most 0.20 | precision 13 / 14 (92.86%); recall 13 / 16 (81.25%); false positives 1 / 206 (0.49%) | pass |
| 4 | session-disjoint held-out evaluation | at least two new reset sessions and at least 10 positive and 10 negative images per evaluated level | L5: 5 positive, 37 negative; L4: 2 positive, 35 negative | inconclusive |
| 5 | review package | ZIP integrity and complete evidence accounting | 59 / 59 checksums; 1,889,607 bytes | pass |
| 6 | correspondence alternatives on newer calibration | matched at least 0.50; mismatch at most 0.10; top-1 at least 0.70 | L4: 0.40 / 0.15 / 0.4667; L5: 0.5333 / 0.1667 / 0.2333 | fail |
| 7 | focused data-completion preflight | split-disjoint poses, executable transitions, fail-closed integrity and actor/evaluator separation | 220 / 220 distinct planned poses; 58 / 58 tests | pass, capture not run |

The existing checkpoint remains useful only as a simulator-development baseline. The next capture is frozen around supported poses and clear controls, but it requires a new GPU authorization because the previously approved one-job allowance was consumed by job `52182100`.

## 1. Correspondence validation — do image interfaces identify the matched pseudo-SAP projection?

`workflows/two_stage_navigation/validate_s0010_sap_correspondence.py`, run by Slurm job `52138127`. It fits the evaluator-only correspondence model on training sessions and applies the frozen threshold to opened calibration sessions.

**Criterion.** Matched acceptance at least 0.50, mismatched acceptance at most 0.10, distinct-pose acceptance at most 0.10, and top-1 ranking at least 0.70; thresholds were frozen before job `52138127`.

### 1.1 Method

- The model receives stored interface-alignment measurements only; it is not an actor input.
- Training sessions fit the model and select its threshold.
- Calibration sessions evaluate the frozen rule.
- Raw projection statistics remain reported separately and are not replaced retroactively.

### 1.2 Setup

| what | value | spec |
|---|---|---|
| run | job `52138127` | §1 |
| code | `59c765be4347238e9950f67e28e854504fb002c9` | §1 |
| inputs | `s0010_sap_negative_image_gate_development_v1/52138127/dataset_archive_v3` | §1 |
| protocol | `s0010_sap_existing_capture_development_protocol_v3.yaml`, SHA-256 `018df8584a181f255b8d301e088d0b5a44d38e94746332d5173f334b754eeb22` | §1 |
| resources | 4 CPUs, 16 GB requested, no GPU | §1 |

```bash
export SONOGYM_CONFIG_FILE="$PWD/.sonogym.env"
sbatch --export=ALL,SONOGYM_CONFIG_FILE="$SONOGYM_CONFIG_FILE" \
  hpg/jobs/s0010_sap_negative_image_gate_and_train.sbatch
```

| | value |
|---|---:|
| L4 calibration matched comparisons | 45 |
| L4 calibration mismatched comparisons | 270 |
| L5 calibration matched comparisons | 29 |
| L5 calibration mismatched comparisons | 174 |
| **final comparisons** | **518** |

### 1.3 Results

| level | threshold | matched | mismatch | distinct pose | top-1 | verdict |
|---|---:|---:|---:|---:|---:|---|
| RIGHT L4 | 0.66 | 18 / 45 (40.00%) | 31 / 270 (11.48%) | 31 / 225 (13.78%) | 17 / 45 (37.78%) | fail |
| RIGHT L5 | 0.72 | 19 / 29 (65.52%) | 13 / 174 (7.47%) | 13 / 145 (8.97%) | 21 / 29 (72.41%) | pass |

**Verdict: RIGHT L4 failed and RIGHT L5 passed the frozen development correspondence criterion.**

### 1.4 Reading it

- L4 remains insufficiently specific and poorly ranked.
- L5 passed on previously opened calibration data, not an independent holdout.
- The L5 external validation does not alter the failed historical raw correspondence statistics.

### 1.5 Files

| file | contents |
|---|---|
| `52138127/correspondence_validation/correspondence_validation.json` | thresholds, calibration metrics and comparison-specific acceptance |
| `52138127/correspondence_validation/s0010_l5_sap_correspondence.joblib` | evaluator-only L5 correspondence checkpoint |
| `52138127/dataset_archive_v3/correspondence_controls.csv` | session-separated matched and mismatched rows |

## 2. Negative-image eligibility — does the dataset exclude unresolved whole-image evidence?

`workflows/two_stage_navigation/prepare_s0010_sap_dataset.py`, run by job `52138127`. It retains an image as negative only when every actor-visible proposal for that conditioned level is negative.

**Criterion.** A retained negative image must have no `BOUNDARY_OR_IDENTITY_AMBIGUOUS` or positive actor-visible proposal; this is the user-approved binary label contract.

### 2.1 Method

- Candidate proposals are generated from clean ultrasound only.
- Privileged geometry assigns offline positive, negative or ambiguous states.
- The gate is opt-in and recorded in three checksum-sealed protocols, preserving historical datasets.
- Positive selection and correspondence calculations are unchanged.

### 2.2 Setup

| what | value | spec |
|---|---|---|
| run | job `52138127` | §2 |
| initial protocol | `s0010_sap_validation_development_protocol_v2.yaml`, SHA-256 `7b1e93cc8f8c3e69fcc21673bdc534c9ade1b870009a89b9ebbb25521ebfbd3e` | §2 |
| follow-up protocol | `s0010_sap_validation_followup_protocol_v2.yaml`, SHA-256 `8e062375c7e145f07760eda1c49a26155e1b856d5baebecada6b6a11a1b1313d` | §2 |
| archive protocol | `s0010_sap_existing_capture_development_protocol_v3.yaml`, SHA-256 `018df8584a181f255b8d301e088d0b5a44d38e94746332d5173f334b754eeb22` | §2 |

```bash
export SONOGYM_CONFIG_FILE="$PWD/.sonogym.env"
sbatch --export=ALL,SONOGYM_CONFIG_FILE="$SONOGYM_CONFIG_FILE" \
  hpg/jobs/s0010_sap_negative_image_gate_and_train.sbatch
```

| | value |
|---|---:|
| previous selected rows | 1,356 |
| dropped, negative image contained a non-negative proposal | 138 |
| **final selected rows** | **1,218** |

### 2.3 Results

| level / split | prior negatives | final negatives | dropped |
|---|---:|---:|---:|
| L4 train | 321 | 315 | 6 |
| L4 calibration | 106 | 104 | 2 |
| L5 train | 529 | 453 | 76 |
| L5 calibration | 260 | 206 | 54 |

**Verdict: pass, because every retained negative satisfies the complete-image label contract and no positive was removed.**

### 2.4 Reading it

- The change fixes label admission; it is not a paired model improvement on an unchanged negative set.
- Removed images remain ambiguous evidence and must be handled by abstention during future evaluation.
- Exact train/calibration frame-hash overlap is 0.

### 2.5 Files

| file | contents |
|---|---|
| `52138127/dataset_merged_v2/sample_manifest.csv` | 1,218 admitted image labels |
| `52138127/dataset_merged_v2/proposal_manifest.csv` | all 23,232 actor-visible proposals, including excluded ambiguity |
| `52138127/dataset_merged_v2/readiness_report.json` | counts, sources, gates and deficits |

## 3. Conditional classifier fit — does RIGHT L5 meet the frozen calibration gate?

`workflows/two_stage_navigation/train_s0010_sap_classifier.py`, run by job `52138127`. It fits 240 ExtraTrees on training rows only and selects one whole-image threshold on calibration rows only.

**Criterion.** Full-image precision at least 0.80, recall at least 0.80 and false-positive rate at most 0.20; thresholds were frozen before fitting.

### 3.1 Method

- The classifier uses the established 774-value clean-ultrasound patch feature vector.
- Whole-image score is the maximum positive probability across actor-visible proposals.
- L4 is not fit because its annotation and calibration gates fail.
- The fitted checkpoint is saved and reload-verified regardless of calibration outcome.

### 3.2 Setup

| what | value | spec |
|---|---|---|
| run | job `52138127` | §3 |
| input | `52138127/dataset_merged_v2` | §3 |
| model | 240-tree ExtraTrees, seed `20261121`, balanced class weights | §3 |
| threshold grid | 0.05 through 0.95 in increments of 0.05 | §3 |
| resources | 4 CPUs, 16 GB requested, no GPU; elapsed 4 minutes 14 seconds for the complete pipeline | §3 |

```bash
python workflows/two_stage_navigation/train_s0010_sap_classifier.py \
  --dataset-dir "$RUN/dataset_merged_v2" \
  --output-dir "$RUN/training_v2" --fit --level L5
```

| | train | calibration |
|---|---:|---:|
| positive unique images | 80 | 16 |
| negative unique images | 453 | 206 |
| **final unique images** | **533** | **222** |

### 3.3 Results

| setting | TP | FP | FN | TN | precision | recall | false-positive rate | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full image, threshold 0.55 | 13 | 1 | 3 | 205 | 92.86% | 81.25% | 0.49% | 86.67% |
| selected patch, threshold 0.55 | 13 | 0 | 3 | 206 | 100.00% | 81.25% | 0.00% | 89.66% |

**Verdict: pass on opened development calibration; the checkpoint status is `TRAINED_CALIBRATED_HOLDOUT_UNOPENED`.**

### 3.4 Reading it

- The checkpoint was reloaded and reproduced every calibration probability exactly.
- Only training IDs entered `fit()`; only calibration IDs selected the threshold.
- The result applies to clear positive and clear negative simulator-proxy images, not excluded ambiguous images.
- Geometry is absent from model features.

### 3.5 Files

| file | contents |
|---|---|
| `52138127/training_v2/s0010_l5_sap_classifier_v1.joblib` | reload-verified candidate checkpoint, SHA-256 `c173aa9812a2df25114f99604b7330ed06b435c3a77edc7e2901f2b5dd3896e5` |
| `52138127/training_v2/training_result.json` | threshold grid, confusion metrics and split IDs |
| `52138127/training_v2/preflight_result.json` | no-fit schema and readiness validation |

## 4. Held-out evaluation — does the calibrated checkpoint generalize to new resets?

`workflows/two_stage_navigation/evaluate_s0010_sap_classifier.py`, run after the frozen resume capture. It evaluated the unchanged L5 checkpoint and threshold once on session-disjoint evaluation observations.

**Criterion.** At least two independently reset evaluation sessions, at least 10 positive and 10 negative unique images per evaluated level, and the same precision, recall and false-positive gates as §3.

### 4.1 Method

- `s0010sapval_evaluation_02` and `s0010sapval_evaluation_03` are preassigned holdout sessions.
- No existing session qualifies as unopened SAP evaluation data.
- The checkpoint, preprocessing and threshold were frozen before acquisition.
- Insufficient holdout positives force an inconclusive verdict even when descriptive confusion metrics pass.
- Images without an actor-visible proposal are counted as abstentions, not silently dropped.

### 4.2 Setup

| what | value | spec |
|---|---|---|
| protocol | `s0010_sap_validation_resume_protocol_v1.yaml`, SHA-256 `3460dd70dda759e3e68692ba1e253d8dd9113e27b971d291a54f941215f35425` | §4 |
| checkpoint | L5 SHA-256 `c173aa9812a2df25114f99604b7330ed06b435c3a77edc7e2901f2b5dd3896e5` | §4 |
| first authorized capture | job `52181462`; failed before acquisition | §4 |
| replacement capture | job `52182100`, execution commit `9446859b1002db29c23498aae6ab2c34ea8af028` | §4 |
| resources | 1 NVIDIA L4, 4 CPUs, 30 GB, 30-minute limit | §4 |
| CPU analysis | jobs `52182756` failed on merge input, `52183365` completed after commit `1a2c5e3` | §4 |
| frozen evaluation | job `52184029`, code `6a30797`; no retraining or threshold selection | §4 |
| evaluation sessions | 2 / 2 acquired | §4 |

```bash
sbatch --export=ALL,SONOGYM_CONFIG_FILE="$PWD/.sonogym.env",SAP_VALIDATION_PHASE=resume \
  hpg/jobs/s0010_sap_validation_capture.sbatch

sbatch --export=ALL,SONOGYM_CONFIG_FILE="$PWD/.sonogym.env",SAP_RESUME_JOB=52182100 \
  hpg/jobs/s0010_sap_resume_analysis.sbatch
```

| | value |
|---|---:|
| planned calibration/evaluation images | 60 / 80 |
| acquired fresh images | 60 / 80 |
| exact-pose replay and coordinate-hash checks | 140 / 140 |
| exact train/calibration/evaluation hash overlap | 0 |
| exact session overlap between splits | 0 |

### 4.3 Results

| level / split | positive | negative | result |
|---|---:|---:|---|
| L4 merged train | 40 | 315 | existing fit pool only |
| L4 merged calibration | 6 | 133 | insufficient; no L4 fit |
| L5 merged train | 80 | 453 | frozen L5 checkpoint preserved |
| L5 merged calibration | 30 | 235 | frozen correspondence gate failed |
| L4 evaluation | 2 | 35 | insufficient |
| L5 evaluation | 5 | 37 | insufficient |

The frozen L5 classifier produced TP=5, FP=0, FN=0 and TN=37 on the 42 clearly labelled evaluation images: precision 1.0, recall 1.0, false-positive rate 0.0 and F1 1.0. Per session, evaluation_02 had TP=2/TN=19 and evaluation_03 had TP=3/TN=18. These are descriptive metrics, not a held-out pass, because the frozen requirement was at least ten positive images and the correspondence gate also failed.

Across all 80 evaluation frames, 42 received clear L5 labels, 37 had ambiguous proposal evidence and one produced no actor-visible proposal. The resulting abstention count is 38/80 (47.5%); two ambiguous images exceeded the frozen classifier threshold but remain excluded from the confusion matrix.

**Verdict: RIGHT L4 `SAP_TRAINING_INCONCLUSIVE`; RIGHT L5 `SAP_TRAINING_INCONCLUSIVE`. `SAP_TRAINED_HELDOUT_PASS` is not supported.**

### 4.4 Reading it

- All three replacement sessions passed acquisition freshness, array-hash integrity, native-map replay, synchronized full-pose replay and viewpoint diversity.
- RIGHT L4 was not trained: calibration had six positives after merging, below ten, and frozen correspondence failed.
- RIGHT L5 was neither retrained nor recalibrated. Its checkpoint hash remained `c173aa9812a2df25114f99604b7330ed06b435c3a77edc7e2901f2b5dd3896e5` and threshold remained 0.55.
- L5 calibration correspondence had matched acceptance 0.5333, mismatched acceptance 0.1556 and top-one ranking 0.2333; the last two violate the frozen gate. Evaluation correspondence also failed with 0.225 matched acceptance and 0.375 top-one ranking.
- The perfect 5-positive descriptive confusion is too small to establish recognition performance and does not justify Stage 2 integration.

### 4.5 Files

| file | contents |
|---|---|
| `experiments/unity_two_stage_transfer/s0010_sap_validation_resume_protocol_v1.yaml` | frozen calibration/evaluation waypoint bank |
| `hpg/jobs/s0010_sap_validation_capture.sbatch` | opt-in resume capture path |
| `s0010_sap_validation_resume_v1/52182100/sessions/` | synchronized capture artifacts and replay reports |
| `s0010_sap_validation_resume_v1/52182100/analysis_v2/heldout_evaluation_final.json` | frozen L5 descriptive held-out metrics and inconclusive verdicts |
| `workflows/two_stage_navigation/evaluate_s0010_sap_classifier.py` | single-use held-out evaluator with fail-closed sample sufficiency |

## 5. Review package — is the current evidence independently inspectable?

`workflows/two_stage_navigation/build_s0010_sap_validation_training_review.py`, finalized by CPU job `52184578`. It packages clean images, privileged overlays, manifests, capture verification, the frozen L5 checkpoint and the inconclusive held-out result.

**Criterion.** ZIP integrity, all internal checksums valid, complete split/coverage accounting and no promotion of insufficient metrics to PASS.

### 5.1 Method

- Clean ultrasound and privileged overlays are separate files.
- The package includes the development and evaluation readiness reports and both frozen correspondence reports.
- The checkpoint is included for reload and hash verification.
- Exact split/session overlap and all-80-frame abstention accounting are machine-readable.

### 5.2 Setup

| what | value | spec |
|---|---|---|
| run | job `52184578` | §5 |
| code | `cb08a302476933b60e971c4f1dbe877d86c68f43` | §5 |
| inputs | development job `52138127`; capture job `52182100`; analysis job `52183365` | §5 |
| resources | 4 CPUs, 8 GB requested, no GPU | §5 |

```bash
sbatch --export=ALL,SONOGYM_CONFIG_FILE="$PWD/.sonogym.env",SAP_RESUME_JOB=52182100 \
  hpg/jobs/s0010_sap_resume_review.sbatch
```

Nothing was dropped from the declared compact package.

### 5.3 Results

| check | result |
|---|---|
| ZIP test | pass |
| internal SHA-256 manifest | pass |
| ZIP size | 1,889,607 bytes |
| internal SHA-256 entries | 59 / 59 |
| ZIP SHA-256 | `3495754a369348c740f417f6990eff97872dc5490bf7bed3bb0a9e973c1b9b68` |

**Verdict: package pass; scientific result remains inconclusive for both levels.**

### 5.4 Reading it

- The package supports review of fresh holdout evidence and the frozen L5 model response.
- It does not support a held-out recognition PASS, SAP ground truth or Stage 2 integration.

### 5.5 Files

| file | contents |
|---|---|
| `52182100/final_review/s0010_right_l4l5_sap_validation_training_review.zip` | compact final review artifact |
| `52182100/final_review/s0010_right_l4l5_sap_validation_training_review/checksums.sha256` | internal integrity manifest |

## 6. Correspondence alternatives — do small model or contour changes transfer to newer calibration?

`workflows/two_stage_navigation/validate_s0010_sap_correspondence.py`, run by CPU jobs `52189713` and `52191101`. It compared the existing logistic and ExtraTrees estimators with the original and contour-extended alignment measurements, selected on previously opened archive calibration, then evaluated the frozen per-level choices once on `s0010sapval_calibration_02`.

**Criterion.** Matched acceptance at least 0.50, mismatched acceptance at most 0.10, distinct-pose acceptance at most 0.10 and top-1 ranking at least 0.70; the criterion is unchanged from §1.

### 6.1 Method

- Training rows fit each candidate; archive calibration rows select one candidate per level.
- `s0010sapval_calibration_02` is opened development evidence and is not a holdout.
- No evaluation session, classifier checkpoint or actor feature was changed.
- Failure on the newer calibration stops further post-hoc model and threshold changes.

### 6.2 Setup

| what | value | spec |
|---|---|---|
| candidate run | job `52189713`, code `c98815f` | §6 |
| frozen newer-calibration evaluation | job `52191101`, code `a668174` | §6 |
| archive input | `s0010_sap_correspondence_cpu/52189713/dataset_archive` | §6 |
| newer calibration input | `s0010_sap_validation_resume_v1/52182100/analysis_v2/dataset_calibration` | §6 |
| resources | 4 CPUs, 16 GB requested, no GPU for each job | §6 |

```bash
sbatch --export=ALL,SONOGYM_CONFIG_FILE="$PWD/.sonogym.env" \
  hpg/jobs/s0010_sap_correspondence_improvement.sbatch

sbatch --export=ALL,SONOGYM_CONFIG_FILE="$PWD/.sonogym.env",SAP_CORRESPONDENCE_SOURCE_JOB=52189713 \
  hpg/jobs/s0010_sap_correspondence_opened_calibration.sbatch
```

| | value |
|---|---:|
| archive L4 calibration matched / mismatched | 45 / 270 |
| archive L5 calibration matched / mismatched | 29 / 174 |
| newer L4 calibration matched / mismatched | 30 / 180 |
| newer L5 calibration matched / mismatched | 30 / 180 |
| **final newer-calibration comparisons** | **420** |

### 6.3 Results

| level | archive-selected estimator and inputs | threshold | newer matched | newer mismatch | newer distinct pose | newer top-1 | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| RIGHT L4 | ExtraTrees, contour-extended | 0.57 | 12 / 30 (40.00%) | 27 / 180 (15.00%) | 21 / 150 (14.00%) | 14 / 30 (46.67%) | fail |
| RIGHT L5 | logistic, original | 0.705 | 16 / 30 (53.33%) | 30 / 180 (16.67%) | 30 / 150 (20.00%) | 7 / 30 (23.33%) | fail |

**Verdict: fail for both levels; the archive gains did not transfer to the newer calibration.**

### 6.4 Reading it

- Local contour measurements improved archive L4 but did not control physically distinct projections in the newer session.
- L5 retained matched acceptance but failed mismatch and ranking.
- The failure supports changing pose and control coverage rather than another threshold or estimator adjustment on opened data.

### 6.5 Files

| file | contents |
|---|---|
| `52189713/selected/selection.json` | archive-only per-level candidate selection |
| `52189713/opened_calibration_shape/evaluation/correspondence_validation.json` | frozen newer-calibration results |
| `52189713/opened_calibration_shape/evaluation/correspondence_predictions.csv` | per-comparison scores and decisions |

## 7. Data-completion preflight — is the next bounded capture executable and properly separated?

`workflows/two_stage_navigation/generate_s0010_sap_validation_resume_protocol_v2.py`, run locally from Blue storage at commit `5d6bd2e`. It ranks supported archived frames and clear controls, expands only the positive neighborhoods, assigns disjoint planned poses to train, calibration and evaluation sessions, and writes a checksum-sealed protocol.

**Criterion.** Every planned pose must be unique across splits; each adjacent transition must exceed the position or angle tracking margin; planned coverage must exceed the declared per-level image minimums before actual post-capture labeling; array and coordinate hashes must fail closed.

### 7.1 Method

- Supported archived frames are ranked by their frozen projection and clean-image alignment measurements.
- Clear controls require zero candidate, uncertainty and other-level surface pixels in the selected patch.
- Positive neighborhoods use fixed position or angle offsets; the planned label remains coverage intent, not a training label.
- Exact labels are rebuilt only from synchronized native coordinates after capture.
- Evaluation sessions remain unopened until training and calibration are frozen.

### 7.2 Setup

| what | value | spec |
|---|---|---|
| implementation | commit `5d6bd2e3ee67e2ac3fe142b8eb5664986a79fb88` | §7 |
| source manifest | `52189713/dataset_archive/sample_manifest.csv`, SHA-256 `277e6768ca6612b006a741c7b5a14dd8c72f83eddd1ff7c21c44cb10c0e6b965` | §7 |
| protocol | `s0010_sap_validation_resume_protocol_v2.yaml`, SHA-256 `7d3a6c46f50f7123351a14daca69745bc44dfd13364d333eec7db391e1410a7a` | §7 |
| resources used | bounded local manifest processing, no GPU | §7 |

```bash
PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$PWD/workflows/two_stage_navigation" \
  "$SONOGYM_ENV_PREFIX/bin/python" \
  workflows/two_stage_navigation/generate_s0010_sap_validation_resume_protocol_v2.py \
  --source-existing-protocol experiments/unity_two_stage_transfer/s0010_sap_existing_capture_development_protocol_v3.yaml \
  --source-dataset-dir "$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_correspondence_cpu/52189713/dataset_archive" \
  --source-resume-protocol experiments/unity_two_stage_transfer/s0010_sap_validation_resume_protocol_v1.yaml \
  --implementation-commit 5d6bd2e3ee67e2ac3fe142b8eb5664986a79fb88 \
  --output experiments/unity_two_stage_transfer/s0010_sap_validation_resume_protocol_v2.yaml
```

| | value |
|---|---:|
| planned train images | 80 |
| planned calibration images | 60 |
| planned evaluation images | 80 |
| duplicate planned poses across splits | 0 |
| **final planned images** | **220** |

### 7.3 Results

| split / level | planned supported-pose neighborhoods | planned clear controls |
|---|---:|---:|
| train / L4 | 24 | 16 |
| train / L5 | 24 | 16 |
| calibration / L4 | 18 | 12 |
| calibration / L5 | 18 | 12 |
| evaluation / L4 | 24 | 16 |
| evaluation / L5 | 24 | 16 |

All 58 focused regression tests passed. The protocol seal, Python compilation, shell syntax and Git whitespace checks passed. Slurm `--test-only` accepted the 1-GPU, 4-CPU, 30-GB, 30-minute request and no job remained queued. The capture itself was not run.

**Verdict: engineering preflight pass; scientific readiness remains pending actual capture and post-capture gates.**

### 7.4 Reading it

- Planned coverage is deliberately larger than the minimum because nearby poses can become ambiguous after actual acquisition.
- The verifier now checks saved ultrasound, raw-label and coordinate-array hashes for SAP validation protocols rather than defaulting their coordinate hash status to true.
- No GPU job was submitted because job `52182100` consumed the exact one-job authorization.

### 7.5 Files

| file | contents |
|---|---|
| `experiments/unity_two_stage_transfer/s0010_sap_validation_resume_protocol_v2.yaml` | sealed five-session, 220-frame plan |
| `workflows/two_stage_navigation/generate_s0010_sap_validation_resume_protocol_v2.py` | deterministic plan generator |
| `tests/test_s0010_sap_validation_resume_protocol_v2.py` | split, pose, tracking-margin and actor/evaluator boundary checks |
| `hpg/jobs/s0010_sap_validation_capture.sbatch` | opt-in capture entry point; not executed |
