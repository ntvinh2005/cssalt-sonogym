# SonoGym s0010 supervised anatomical-discovery pilot

This report tests whether clean SonoGym ultrasound pixels add anatomical development evidence beyond executed Home-relative motion, and whether that evidence supports offline L4 localization in the [Unity Stage 1](../notation.md#unity-stage-1) architecture. It covers one previously inspected anatomy and does not establish independent confirmation, patient generalization, or Phase 2 readiness.

## Result

**The bounded development pilot passed: ultrasound morphology added measurable information beyond position/history, and image-derived bidirectional spatial evidence localized L4 in all four development sessions. Independent confirmation and generalization remain untested.**

| step | what was measured | criterion | measured | verdict |
|---|---|---|---|---|
| 1 | capture, label, motion, and inference-boundary integrity | 16 expected sessions; unique valid hashes; exact odometry reconstruction; no privileged actor fields | 16 sessions, 3,376 unique frames; 0.0 mm and 0.0 rad maximum reconstruction error; boundary checks passed | PASS |
| 2 | image evidence and controls | shared precision/recall at least 0.90; image-only macro recall at least 0.70; image-minus-shuffled at least 0.20 | shared 0.990/0.954; image-only macro recall 0.844; image-minus-shuffled 0.697 | PASS |
| 3 | offline L4 spatial inference | precision at least 0.90; recall at least 0.80; zero wrong stops; median error and forward/reverse difference at most 10 mm | 4/4 correct; precision/recall 1.00/1.00; zero wrong stops; median error 6.36 mm; median direction difference 3.5 mm | PASS, development only |

The canonical CPU job was `51192890`. Job `51192265` is retained as a numerically equivalent predecessor; it motivated recording separate fit/inference timing and omitting control models from the development checkpoint. Model, data, splits, thresholds, and scientific results were unchanged.

## 1. Data, label, motion, and leakage audit

### 1.1 Criterion

Every expected capture session must have valid public and privileged manifests, matching frame hashes, reconstructable executed motion, correct forward/reverse ordering, and no privileged anatomical fields in the actor interface. Raw Group A identities may be joined only for offline training labels and evaluator measurements.

### 1.2 Method

- The runner audits all 16 sessions from capture job `50908807` before fitting.
- Actor inputs are clean ultrasound, executed Home-relative motion, relative angle, direction history, and calibrated pixel geometry.
- Offline labels use the dominant raw Group A identity overlapping each tracked-interface-centred 48 x 80 pixel patch, with minimum overlap fraction 0.01.
- Raw labels, segmentation, patient identity, mesh identity, reference coordinates, goal pose, target distance, and evaluator score are forbidden from actor inputs and checkpoint features.
- Training and development are separated by complete acquisition session; adjacent frames are never randomly split.

### 1.3 Setup

| item | value |
|---|---|
| patient | `s0010` only |
| renderer capture | job `50908807`; `US_slicer.us_img_tensor` |
| training sessions | `td_00`--`td_07` |
| development sessions | `cd_00`--`cd_03` |
| integrity-only unused sessions | `ua_00`--`ua_03` |
| raw annotation | `s0010_anatomical_annotations_v2.json`; SHA-256 `25df75aadb83b6d1faf39cfd8b376fb8976a08dbd834e302bad8f38e9de023c3` |
| protocol | `sonogym-stage1-supervised-anatomical-discovery-v1`; SHA-256 `4f0b7df56d58b791b938b426144ae74c99c0d49e1ca9ee7f2a544898749474a7` |

### 1.4 Results

| measure | result |
|---|---:|
| audited sessions / frames | 16 / 3,376 |
| unique frame hashes | 3,376 |
| modeled sessions | 12: 8 training, 4 development |
| unused sessions | 4, integrity audit only |
| modeled candidate patches | 33,945 |
| training / development anatomy patches | 16,645 / 8,366 |
| ambiguous positive patches | 2 |
| maximum position reconstruction error | 0.0 mm |
| maximum angle reconstruction error | 0.0 rad |

### 1.5 Reading it

The images are genuine SonoGym renderer outputs with consistent acquisition provenance and executed-motion histories. The patch label is nevertheless an evaluator-derived geometric overlap label. It demonstrates a valid supervised target, not that every raw vertebral identity has a uniquely recognizable acoustic appearance. All 16 sessions and this fixed anatomy were previously inspected; these results are development evidence, not a locked test.

### 1.6 Files

| file | contents |
|---|---|
| [frozen protocol](../../experiments/unity_two_stage_transfer/stage1_supervised_pilot_protocol.yaml) | splits, inputs, models, gates, and resource bound |
| [protocol digest](../../experiments/unity_two_stage_transfer/stage1_supervised_pilot_protocol.sha256) | immutable protocol SHA-256 |
| [runner](../../workflows/two_stage_navigation/run_stage1_supervised_pilot.py) | integrity audit, features, controls, and spatial inference |

## 2. Supervised morphology recognition and controls

### 2.1 Criterion

The [Image evidence gate](../notation.md#image-evidence-gate) requires shared-structure precision and recall at least 0.90, image-only seven-level macro recall at least 0.70, and an image-only minus shuffled-image macro-recall difference at least 0.20. A separate diagnostic requires image-plus-position macro recall to exceed the [Position-only control](../notation.md#position-only-control) by at least 0.05 before calling the pixel contribution meaningful.

### 2.2 Method

- Each tracked patch is normalized by its 0.05 and 0.95 intensity quantiles and resized to 16 x 16.
- Flattened normalized intensity and gradient features feed CPU `ExtraTreesClassifier` models.
- The primary model is a [Shared anatomical-structure detector](../notation.md#shared-anatomical-structure-detector); a seven-level classifier is a diagnostic alternative, not seven independent binary detectors.
- Controls use executed Home-relative x/z, relative angle, direction, and normalized frame sequence; or deterministically permute image features within each development session.
- The common candidate rows are found by the image-derived interface tracker. Therefore the position-only control excludes morphology features but is not an image-independent candidate generator.

### 2.3 Setup

| item | value |
|---|---|
| model family | ExtraTrees; 240 trees; depth 18; minimum leaf 3; balanced classes |
| random seed | `20261113` |
| shuffled-image seed | `20261114` |
| software | Python 3.10.22; NumPy 1.26.4; SciPy 1.15.2; scikit-learn 1.5.2; joblib 1.4.2 |
| code commit | `91886a49f3099f8c23874507041a34c40b22fe4d` |

### 2.4 Results

| shared-structure setting | accuracy | precision | recall | FP | FN |
|---|---:|---:|---:|---:|---:|
| image only | 0.959 | 0.990 | 0.954 | 81 | 387 |
| position/history only | 0.694 | 0.918 | 0.642 | 481 | 2,997 |
| shuffled image | 0.630 | 0.758 | 0.730 | 1,954 | 2,260 |
| image plus position/history | 0.961 | 0.991 | 0.956 | not primary | not primary |

| seven-level diagnostic | accuracy | macro precision | macro recall |
|---|---:|---:|---:|
| image only | 0.876 | 0.855 | 0.844 |
| position/history only | 0.729 | 0.767 | 0.810 |
| image plus position/history | 0.918 | 0.910 | 0.902 |
| shuffled image | 0.173 | 0.145 | 0.147 |
| shuffled image plus position/history | 0.233 | 0.206 | 0.210 |

Image-only macro recall exceeded shuffled image by 0.697. Image plus position/history exceeded position/history alone by 0.092, above the frozen 0.05 meaningful-increment threshold. The image evidence gate therefore passed.

For L4 patches, image-only precision was 0.866 and recall was 0.840 over 808 development patches. Difficult neighboring negatives included 63 / 1,163 L3 patches and 27 / 1,190 L5 patches predicted as L4. L4 false negatives were most often predicted as L5 (62) or L3 (39).

### 2.5 Reading it

Ultrasound morphology contributes substantial session-separated development signal: destroying the image-label association collapses seven-level macro recall from 0.844 to 0.147, and adding images improves over the strong fixed-anatomy position prior by 0.092. This does not prove patient-independent anatomy recognition. Position remains predictive because all data come from one anatomy and a constrained scan corridor.

### 2.6 Files

| file | contents |
|---|---|
| [result manifest](../../experiments/unity_two_stage_transfer/stage1_supervised_pilot_result.json) | concise data, control, timing, spatial, and artifact evidence |
| [result digest](../../experiments/unity_two_stage_transfer/stage1_supervised_pilot_result.sha256) | result-manifest SHA-256 |
| [tests](../../tests/test_stage1_supervised_pilot.py) | control alignment, deterministic shuffle, metrics, and spatial gate tests |

## 3. Bidirectional spatial evidence and L4 inference

### 3.1 Criterion

After the image evidence gate passes, the offline actor must reach L4 precision at least 0.90 and recall at least 0.80, with zero wrong stops, median [L4 localization error](../notation.md#l4-localization-error) at most 10 mm, and median forward/reverse peak difference at most 10 mm. The actor may abstain when evidence is insufficient.

### 3.2 Method

Image-only shared-structure and seven-level probabilities are projected onto a 0.5 mm Home-relative grid using executed motion and calibrated image geometry. Candidate events require forward and reverse evidence. A frozen threshold grid from 0.30 to 0.90 is evaluated on the four development sessions; no coordinate offset is calibrated. The selected coordinate is compared to the evaluator-only L4 reference only after actor inference.

### 3.3 Setup

~~~bash
sbatch \
  --account=anest-dept \
  --qos=anest-dept \
  --partition=hpg-default \
  --cpus-per-task=4 \
  --mem=16gb \
  --time=00:30:00 \
  --job-name=sg-stage1-supervised \
  --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env \
  hpg/jobs/two_stage_stage1_supervised_pilot.sbatch
# Canonical job 51192890: 2026-11-13T11:59:46-05:00 to 12:02:50-05:00.
~~~

| resource | measured |
|---|---:|
| Slurm state / exit | COMPLETED / 0:0 |
| allocation | `anest-dept`; 4 CPU; 16 GB requested; 0 GPU |
| Slurm / application elapsed | 184 / 175.659 seconds |
| CPU time | 248.011 seconds |
| Slurm batch MaxRSS | 351,576 KiB |
| `/usr/bin/time` maximum RSS | 1,117,384 KiB |
| output growth | 72,034,539 bytes |

### 3.4 Results

| session | inferred L4 coordinate (mm) | error (mm) |
|---|---:|---:|
| `cd_00` | -77.0 | 6.36 |
| `cd_01` | -77.0 | 6.36 |
| `cd_02` | -76.5 | 6.86 |
| `cd_03` | -77.5 | 5.86 |

The selected threshold was 0.50. All four sessions localized within 10 mm: precision 1.00, recall 1.00, zero wrong stops, zero abstentions, median error 6.36 mm, p90 error 6.71 mm, and median forward/reverse peak difference 3.5 mm. Per-session forward/reverse differences were 3, 2, 4, and 5 mm.

| timed stage | seconds |
|---|---:|
| data audit and feature extraction | 137.881 |
| shared-model fit / development inference | 12.109 / 0.918 |
| anatomy-model fit / development inference | 9.828 / 2.155 |
| spatial development inference | 8.629 |
| checkpoint serialization | 3.710 |

The 71,695,094-byte development checkpoint contains only the image-only shared and anatomy models, class order, patch specification, tracker configuration, and protocol digest. It is explicitly marked `deployment: false` and structurally loaded successfully after the job.

### 3.5 Reading it

The final scientific answer is **yes for this bounded s0010 development setting**: clean SonoGym ultrasound plus executed relative motion supported meaningful L4 localization through supervised morphology and bidirectional spatial aggregation. It is not yet an independently confirmed Phase 1 capability. The threshold was selected and reported on the same four development sessions, and the four unused sessions are prior-inspected rather than a locked test.

This L4-specific result does not retroactively pass the earlier seven-structure Group A reconstruction. Job `51116088` remains a valid development NO-GO for complete sacrum/S1/L1--L5 mapping. The current pilot establishes an L4 target estimate, not a complete ordered landmark map, patient generalization, closed-loop acquisition, Phase 2 performance, or clinical MBB validity.

The next smallest experiment is a separately approved, preregistered confirmation capture on `s0010` with new session-held-out starts and the current protocol, model, threshold, and inference code frozen before capture. No tuning should use those confirmation sessions.

### 3.6 Files

| file | contents |
|---|---|
| [Slurm launcher](../../hpg/jobs/two_stage_stage1_supervised_pilot.sbatch) | bounded CPU execution and provenance capture |
| external raw report | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage1_supervised_pilot/51192890/results/pilot_report.json` |
| external development checkpoint | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage1_supervised_pilot/51192890/results/stage1_l4_development.joblib` |
| external run root | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage1_supervised_pilot/51192890/` |
