# SonoGym s0010 at a fixed 2,000-image training budget

The study compares reuse of the original L5 training examples and targeted
sampling on recorded acquisitions. Every primary replay uses the original
L4 checkpoint, rules for extracting representative points, and point mapper.

## Result

**The 2,000-image experiments did not meet the recorded-development readiness gate.**

| Step | Measured | Criterion | Result | Verdict |
| --- | --- | --- | --- | --- |
| Diagnosis | Visible interface versus representative | ≤3 px | Two localizations outside gate, one absence | Failures traced |
| Training | Per-arm distinct image count | 2,000 images | 2,000 in each completed arm | Pass |
| Primary replay | Readiness gate | ≥14/16 correct, ≥3/4 complete, zero wrong and historical/TP checks | T4 TP boundary: 13/16 correct, 0 wrong, 1/4 complete | FAIL |
| Artifact verification | Recorded replay MP4 | All 102 camera/ultrasound joins and complete decode | 102/102 joins, 246 decoded frames | Pass |

## 1. Failure diagnosis — What produced the incorrect L5 TP map and historical SAP regressions?

`workflows/perception_mixed_s0010/diagnose.py` extracts the actual ultrasound,
semantic probabilities, accepted components, and simulator-proxy labels at
the recorded frame and pose.

**Criterion.** An accepted L5 observation satisfies the existing
visible-interface criterion when the distance from its representative pixel
to the nearest visible target pixel is at most 3 pixels. The recorded
frames retain the original evaluator labels and tolerance.

### 1.0 Quantities

| Quantity | Definition | Source |
|---|---|---|
| Correct-interface precision | Frames with at least one accepted representative within 3 px divided by frames with any accepted representative | `perception_metrics.statistics` |
| Correct-interface recall | Same numerator divided by frames with a labeled visible interface | `perception_metrics.statistics` |
| Landmark correctness | A resolved landmark whose selected support has a strict majority within 3 px | `run_simple_landmark_mapping.score_map` |
| Complete map | All four anatomical landmarks correct with required acquisition and projection integrity | `replay.replay` |
| Hard positive | Frame with a visible TP or SAP that either checkpoint missed under its original rules | `composition.verified_training_candidates` |
| Hard negative | Frame with a predicted region on labeled other bone, with no visible or projected target, or displaced at least 5 px onto a labeled pixel | `composition.verified_training_candidates` |

### 1.1 Method

The nearest pixel of the predicted component and the chosen representative
were evaluated independently. Native structure IDs provide coarse provenance.
A representative at ignored proxy label 255 cannot identify a precise
competing bone structure.

### 1.2 Setup

| Input | Value |
|---|---|
| Runs | `52729667` failed before output because the model name was passed as the anatomical level. `52729998` completed the corrected diagnosis |
| Reference model | L5 from `52498821`, SHA-256 `ca13c8fb6bf99bd363dd87c5e30dc1eeb5b22da5cf3f87e991893fad96b41acf` |
| Previous pilot | L5 from `52723139` T3, SHA-256 `26b648777b88873a5b7a22aa6727d56f06e305151eba42e40ee36b52f0c5ed63` |
| Observations | All five supports of the pilot's wrong L5 TP map in medium seed `2026121700`. Regressed SAP in 54-waypoint seed `2026121703` and v6r19 sessions `00` and `03` |
| Reproduction | `bash reproduce.sh diagnosis /blue/.../diagnosis-new` from the experimental checkout on a CPU compute job |

### 1.3 Results

| Medium waypoint | Visible TP pixels | Pilot region pixels | Mean confidence | Representative distance | Exact TP overlap | Finding |
|---|---:|---:|---:|---:|---:|---|
| 6 | 0 | 11 | 0.915 | — | 0 | False positive with no visible TP |
| 33 | 21 | 101 | 0.963 | 1 px | 21 | Correct |
| 71 | 18 | 118 | 0.969 | 1 px | 18 | Correct |
| 85 | 8 | 59 | 0.948 | 6 px | 8 | Enlarged mask moved the representative away from the visible interface |
| 94 | 7 | 81 | 0.959 | 8 px | 7 | Enlarged mask moved the representative away from the visible interface |

**Verdict.** Only 2/5 supports met the criterion. Point selection accounted
for two of the errors on this locked set of components. The third frame
supplied no visible TP for an alternate point to recover.

| Historical SAP case | Original winning / next cluster views | Pilot winning / next cluster views | Evidence |
|---|---:|---:|---|
| 54 waypoints, `2026121703` | 2 / 2 | 2 / 2 | SAP became ambiguous. New high-confidence false-localization components on frames 21 and 31 reached 3.16 and 3.61 px from the reference |
| v6r19 `00` | 5 / 3 | 4 / 4 | A good component on frame 126 moved outside the selected hypothesis and each leading cluster had four views |
| v6r19 `03` | 16 / 12 | 16 / 16 | The pilot missed small visible interfaces at frames 129, 138, 150, and 174 and admitted competing components |

### 1.4 Reading it

- The medium false positive at waypoint 6 has an L5 source vertebral
  label, with ignored interface pixels at the representative. The exact
  substructure is undetermined.
- Point distances and observed cluster changes explain the recorded SAP
  failures. The comparison cannot distinguish distribution shift from
  forgetting by itself.
- White symbols in the historical grids identify a representative in either
  of the two inspected hypotheses. The JSON separates selected and
  competing clusters.

### 1.5 Files

| File | Contents |
|---|---|
| `diagnosis-52729998/diagnosis.json` | Frame IDs, representatives, 3D coordinates, confidence, distances, and memberships |
| `diagnosis-52729998/*.png` and `*.npz` | Ultrasound, full probability maps, masks, and visible interfaces |
| `diagnosis-52729998/evaluation_image_hashes.json` | Exact raw-image exclusion set |

## 2. Training composition — Do original examples and hard cases help at the same image budget?

`workflows/perception_mixed_s0010/train.py` starts each of the three matched
runs from the original L5 checkpoint. T0 is the original model. T1 is the
saved pilot. A matched rerun on the exact pilot data controls for its
different sampling and checkpoint-selection schedule.

**Criterion.** Each arm has exactly 2,000 distinct original-resolution
images. Compare the validation-selected checkpoints at unchanged inference
thresholds, then apply the development readiness criteria in §3.

### 2.1 Method

- A matched uses the same 2,000 rendered images as the earlier pilot.
- T2 and T3 use an identical, SAP-prioritized 400-image subset of the
  original training partition and the same eligible rendered pool.
- Both mixed manifests exclude the five inspected medium source groups
  and their 3 mm neighborhoods. A matched retains its archived composition.
  This is an additional data difference in comparisons against A.
- Hard mining uses the original and pilot L5 models, with a 0.60 and four
  pixel candidate floor for mining only. Primary L5 inference uses TP
  0.80 / 8 pixels and SAP 0.90 / 2 pixels. Hard-positive and hard-negative
  manifest categories are disjoint.
- All matched runs use one shuffled pass over the manifest per epoch,
  seed 394406, batch size 12, AdamW with fixed (1.2\times 10^{-4})
  learning rate and (10^{-4}) weight decay. Weights computed from
  Dataset A are shared across all three runs. The objective combines
  weighted cross-entropy, Tversky with false-positive weight 0.35, and
  negative-presence weight 0.25.
- The best epoch maximizes mean grouped validation correct-interface F1
  equally across classes and splits. Earliest wins ties. Max 22 epochs,
  with five-patience early stopping starting at epoch 14.

### 2.2 Setup

| Input | Value |
|---|---|
| Training run | Slurm `52730372`, one L4 GPU, four allocated CPU cores |
| Execution revision | `f3a35be44fcec291c7fc839d2871a26a191ce4aa` |
| Rendered data | Archived `dataset-52720837/render`, 3,925 candidate training images |
| Original data | Dataset job `52497655`, L5 included training split from `52498821`. Input hashes in the frozen profile |
| Validation | 340 unseen-speckle images on familiar geometry and 84 unseen-geometry images. Seven source groups totaling 26 images excluded from the original 110 to enforce a 3 mm legacy training buffer. Each class has 13 positives in the retained geometry set |
| Opened comparison | Exact original 99 images with separated geometry and fields. Used for downstream readiness and for reporting, outside epoch selection and training |
| Validation and acquisition exclusions | 1,426 distinct saved playback images checked for exact raw image reuse. Zero overlaps. Re-rendered training data derive from the same anatomy and some of the development trajectory neighborhoods |
| Reproduction | `bash reproduce.sh training /blue/.../fit-new /blue/.../diagnosis-new` on an allocated L4 GPU |

### 2.3 Results

| Arm | Representative rendered | Hard positive | Hard negative | Original | Total |
|---|---:|---:|---:|---:|---:|
| Pilot and A matched | 2,000 | 0 | 0 | 0 | 2,000 |
| T2 legacy replay | 1,600 | 0 | 0 | 400 | 2,000 |
| T3 targeted mixture | 1,079 | 300 | 221 | 400 | 2,000 |

The eligible rendered pool held 2,975 images. All 221 qualifying negatives
were kept. The remaining 79 slots used representative images. The original
400 included 216 TP-positive and 169 SAP-positive images, with possible
co-occurrence.

| Arm | Epochs | Selected epoch | Grouped interface F1 | L5 checkpoint SHA-256 |
| --- | --- | --- | --- | --- |
| A matched | 15 | 10 | 0.7366 | e360c1c1822d8f90c8565cfc289b4188735dff9cb2e4020b35c5d24ea3f93833 |
| T2 legacy replay | 20 | 15 | 0.7581 | 65a67b583002a154540a427f8a22dda4140721919faadbc8f1359be1cae5ad16 |
| T3 targeted mixture | 22 | 18 | 0.7387 | 8af2b2026aa69159072331bea0bd70cc23f20cf9763dc5fb8fc7535c1942cfae |
| T4 TP boundary | 20 | 15 | 0.7423 | 1efc323e9c379bcf657eb59fb793cfa30b98e618033a3abc1966a4f3cc69bf47 |

**Verdict.** The cap and split audits passed for all three runs.

### 2.4 Reading it

- T2 and T3 each use 595 rendered physical poses covering 90 source groups
  and five fields. A uses 785 physical poses across 101 source groups.
- The comparison between T2 and T3 measures the joint replacement of 521
  representative samples. It cannot attribute an effect to negative or
  positive examples in isolation.
- No simulation or new rendering was required. Each split is from
  `s0010`.

### 2.5 Files

| File | Contents |
|---|---|
| `training-52730372/fit/*/training_manifest.json` | Exact source images, identities, poses, checksums and assignments |
| `training-52730372/fit/validation_manifest.json` | Matched epoch selection and opened 99-image evaluation split |
| `training-52730372/fit/mining.json` | Per-image records and selected hard cases |
| `training-52730372/fit/leakage_audit.json` | Raw hash checks, buffers and residual geometric provenance |
| `training-52730372/fit/*/history.json` and `best.pt` | Learning curves, complete metrics and checkpoints |

## 3. Replay — Did perception or point selection improve the frozen maps?

`workflows/perception_mixed_s0010/evaluate.py` evaluates the same 99 opened
images, four 102-waypoint sweeps, four 54-waypoint sweeps and 816
historical frames. Separate offline point selection changes only a
component's representative, keeping the model and mask fixed.

**Criterion.** Readiness requires 14/16 or more correct landmarks, 3/4 or
more complete maps, zero wrong landmarks on the 102-waypoint sweep, 4/4
correct historical L5 SAP on both replays, and on the opened 99 images
L5 TP correct-interface recall within one true-positive frame and
precision within five percentage points of the pilot.

### 3.1 Method

All primary arms use the 102 eligible TP waypoints for the medium
evaluation, the archived 50-waypoint TP subset on the 54-waypoint
historical sweeps, and original frame-center routing for v6r19.
L4 checkpoint SHA-256 is
`c6b7ecc9bc245a88ccc0a5db31dbfd919b57407988dde5ea8211dd7eea2c1b2a`.
The original calibration, thresholds, class size floors, viewpoint
requirements, and point mapper are preserved. The primary comparison
uses no threshold calibration. The offline point sensitivity uses a
confidence-weighted medoid and a maximum-probability pixel of the same
accepted component. Both choices use only the predicted component and
its probability map.

The optional T4 arm changes the training loss on the exact A matched
manifest. In training frames with a visible TP, the auxiliary term
adds 0.20 times the mean squared-probability-weighted distance cost
over pixels. The cost is zero to 3 px, grows linearly to 12 px,
and is capped thereafter. This geometric term uses the distance to
the nearest labeled TP pixel, including predicted probability on
ignored pixels, while the segmentation targets and inference remain
fixed. A CPU smoke test checked that distant predictions incur a
larger loss, that no-visible-TP frames contribute zero, and that
backpropagation yields finite gradients.

### 3.2 Setup

| Input | Value |
|---|---|
| Recorded 102-waypoint development | Seeds `2026121700`, `2026121702`, `2026121900`, `2026121901` |
| Recorded 54-waypoint history | Seeds `2026121700` through `2026121703` from `52699995` |
| Recorded 816-frame history | v6r19 `52626449` archive with evaluator labels from `52699992` |
| Reserved Stage 2 seeds | `2026121800` through `2026121803` remain unused |
| Optional arm | Slurm `52731568`, execution revision `a93bbf07b61e2ec596b79121f7214d58a526dee3` |
| Optional reproduction | `bash reproduce.sh boundary /blue/.../boundary-new /blue/.../selection.json /blue/.../fit-new` on an L4 |
| Primary and optional evaluation | Slurm `52730514` and `52731587` with identical replays |
| Reproduction | `bash reproduce.sh evaluation /blue/.../evaluation-new /blue/.../fit-new` on a CPU compute job |

### 3.3 Results

**Opened 99-image comparison.**

| Arm | Class | Precision | Recall | Dice | IoU | Accepts on absent | Accepts on clear negative | Correct on original hard positives | Accepts on original hard negatives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0 original | TP | 6/12 | 6/15 | 0.596 | 0.425 | 6/84 | 2/60 | 0/9 | 2/2 |
| T0 original | SAP | 6/11 | 6/17 | 0.403 | 0.253 | 4/82 | 4/78 | 0/11 | 0/0 |
| T1 previous pilot | TP | 11/14 | 11/15 | 0.861 | 0.756 | 2/84 | 0/60 | 5/9 | 0/2 |
| T1 previous pilot | SAP | 12/13 | 12/17 | 0.632 | 0.462 | 1/82 | 0/78 | 6/11 | 0/0 |
| A matched | TP | 11/16 | 11/15 | 0.857 | 0.750 | 4/84 | 0/60 | 5/9 | 0/2 |
| A matched | SAP | 13/14 | 13/17 | 0.615 | 0.444 | 1/82 | 0/78 | 7/11 | 0/0 |
| T2 legacy replay | TP | 12/19 | 12/15 | 0.882 | 0.789 | 6/84 | 0/60 | 6/9 | 0/2 |
| T2 legacy replay | SAP | 9/11 | 9/17 | 0.600 | 0.429 | 1/82 | 0/78 | 3/11 | 0/0 |
| T3 targeted mixture | TP | 12/21 | 12/15 | 0.856 | 0.748 | 7/84 | 0/60 | 6/9 | 0/2 |
| T3 targeted mixture | SAP | 9/9 | 9/17 | 0.562 | 0.391 | 0/82 | 0/78 | 3/11 | 0/0 |
| T4 TP boundary | TP | 11/14 | 11/15 | 0.886 | 0.795 | 2/84 | 0/60 | 6/9 | 0/2 |
| T4 TP boundary | SAP | 13/14 | 13/17 | 0.622 | 0.451 | 1/82 | 0/78 | 7/11 | 0/0 |

**Frozen 102-waypoint primary replay.**

| Arm | Correct | Wrong | Unresolved | Complete | L4 TP | L4 SAP | L5 TP | L5 SAP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0 original | 10 | 0 | 6 | 0/4 | 2/4 | 3/4 | 3/4 | 2/4 |
| T1 previous pilot | 12 | 1 | 3 | 1/4 | 2/4 | 3/4 | 3/4 | 4/4 |
| A matched | 11 | 1 | 4 | 1/4 | 2/4 | 3/4 | 2/4 | 4/4 |
| T2 legacy replay | 10 | 2 | 4 | 0/4 | 2/4 | 3/4 | 1/4 | 4/4 |
| T3 targeted mixture | 10 | 2 | 4 | 0/4 | 2/4 | 3/4 | 1/4 | 4/4 |
| T4 TP boundary | 13 | 0 | 3 | 1/4 | 2/4 | 3/4 | 4/4 | 4/4 |
| Oracle | 15 | 0 | 1 | 3/4 | 4/4 | 4/4 | 4/4 | 3/4 |
| Oracle with actor routing | 15 | 0 | 1 | 3/4 | 4/4 | 4/4 | 4/4 | 3/4 |

**102-waypoint replay by seed.**

| Development seed | Arm | L4 TP / views | L4 SAP / views | L5 TP / views | L5 SAP / views |
| --- | --- | --- | --- | --- | --- |
| 2026121700 | T0 original | unresolved / 5 | correct / 4 | unresolved / 4 | unresolved / 3 |
| 2026121700 | T1 previous pilot | unresolved / 5 | correct / 4 | wrong / 5 | correct / 5 |
| 2026121700 | A matched | unresolved / 5 | correct / 4 | wrong / 5 | correct / 5 |
| 2026121700 | T2 legacy replay | unresolved / 5 | correct / 4 | wrong / 6 | correct / 5 |
| 2026121700 | T3 targeted mixture | unresolved / 5 | correct / 4 | wrong / 6 | correct / 5 |
| 2026121700 | T4 TP boundary | unresolved / 5 | correct / 4 | correct / 6 | correct / 5 |
| 2026121700 | Oracle | correct / 11 | correct / 4 | correct / 6 | correct / 6 |
| 2026121702 | T0 original | correct / 6 | correct / 4 | correct / 4 | unresolved / 3 |
| 2026121702 | T1 previous pilot | correct / 6 | correct / 4 | correct / 6 | correct / 5 |
| 2026121702 | A matched | correct / 6 | correct / 4 | correct / 6 | correct / 5 |
| 2026121702 | T2 legacy replay | correct / 6 | correct / 4 | wrong / 6 | correct / 5 |
| 2026121702 | T3 targeted mixture | correct / 6 | correct / 4 | unresolved / 5 | correct / 5 |
| 2026121702 | T4 TP boundary | correct / 6 | correct / 4 | correct / 4 | correct / 5 |
| 2026121702 | Oracle | correct / 8 | correct / 4 | correct / 7 | correct / 5 |
| 2026121900 | T0 original | correct / 6 | unresolved / 4 | correct / 5 | correct / 3 |
| 2026121900 | T1 previous pilot | correct / 6 | unresolved / 4 | correct / 6 | correct / 6 |
| 2026121900 | A matched | correct / 6 | unresolved / 4 | correct / 6 | correct / 6 |
| 2026121900 | T2 legacy replay | correct / 6 | unresolved / 4 | correct / 6 | correct / 6 |
| 2026121900 | T3 targeted mixture | correct / 6 | unresolved / 4 | correct / 6 | correct / 5 |
| 2026121900 | T4 TP boundary | correct / 6 | unresolved / 4 | correct / 6 | correct / 6 |
| 2026121900 | Oracle | correct / 8 | correct / 4 | correct / 7 | unresolved / 6 |
| 2026121901 | T0 original | unresolved / 5 | correct / 4 | correct / 5 | correct / 5 |
| 2026121901 | T1 previous pilot | unresolved / 5 | correct / 4 | correct / 6 | correct / 5 |
| 2026121901 | A matched | unresolved / 5 | correct / 4 | unresolved / 6 | correct / 5 |
| 2026121901 | T2 legacy replay | unresolved / 5 | correct / 4 | unresolved / 6 | correct / 4 |
| 2026121901 | T3 targeted mixture | unresolved / 5 | correct / 4 | wrong / 6 | correct / 5 |
| 2026121901 | T4 TP boundary | unresolved / 5 | correct / 4 | correct / 6 | correct / 5 |
| 2026121901 | Oracle | correct / 10 | correct / 4 | correct / 8 | correct / 6 |

**Historical regression.**

| Arm | Historical replay | Correct | Wrong | Unresolved | Complete | L4 TP | L4 SAP | L5 TP | L5 SAP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0 original | 54 waypoints | 10 | 2 | 4 | 0/4 | 4/4 | 1/4 | 1/4 | 4/4 |
| T0 original | 816 frames | 15 | 0 | 1 | 3/4 | 4/4 | 4/4 | 3/4 | 4/4 |
| T1 previous pilot | 54 waypoints | 11 | 0 | 5 | 1/4 | 4/4 | 1/4 | 3/4 | 3/4 |
| T1 previous pilot | 816 frames | 14 | 0 | 2 | 2/4 | 4/4 | 4/4 | 4/4 | 2/4 |
| A matched | 54 waypoints | 9 | 1 | 6 | 0/4 | 4/4 | 1/4 | 2/4 | 2/4 |
| A matched | 816 frames | 16 | 0 | 0 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| T2 legacy replay | 54 waypoints | 11 | 2 | 3 | 1/4 | 4/4 | 1/4 | 2/4 | 4/4 |
| T2 legacy replay | 816 frames | 16 | 0 | 0 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| T3 targeted mixture | 54 waypoints | 9 | 2 | 5 | 1/4 | 4/4 | 1/4 | 1/4 | 3/4 |
| T3 targeted mixture | 816 frames | 15 | 1 | 0 | 3/4 | 4/4 | 4/4 | 3/4 | 4/4 |
| T4 TP boundary | 54 waypoints | 10 | 2 | 4 | 1/4 | 4/4 | 1/4 | 3/4 | 2/4 |
| T4 TP boundary | 816 frames | 14 | 1 | 1 | 3/4 | 4/4 | 4/4 | 3/4 | 3/4 |

**Offline representative diagnostic.** Distances condition on a visible interface for the eligible observation.

| Model | Representative | Class | Correct / eligible | Visible beyond 3 px | No visible interface | Median px | P95 px |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1 previous pilot | weighted_center | TP | 47/64 | 10 | 7 | 1.00 | 7.20 |
| T1 previous pilot | weighted_center | SAP | 60/82 | 13 | 9 | 2.00 | 9.12 |
| T4 TP boundary | weighted_center | TP | 46/58 | 6 | 6 | 1.00 | 4.71 |
| T4 TP boundary | weighted_center | SAP | 60/80 | 12 | 8 | 2.00 | 8.06 |
| T1 previous pilot | weighted_medoid | TP | 47/64 | 10 | 7 | 1.00 | 8.00 |
| T1 previous pilot | weighted_medoid | SAP | 61/82 | 12 | 9 | 2.00 | 9.12 |
| T4 TP boundary | weighted_medoid | TP | 46/58 | 6 | 6 | 1.00 | 4.28 |
| T4 TP boundary | weighted_medoid | SAP | 61/80 | 11 | 8 | 2.00 | 8.06 |
| T1 previous pilot | maximum_probability | TP | 51/64 | 6 | 7 | 0.00 | 6.71 |
| T1 previous pilot | maximum_probability | SAP | 59/82 | 14 | 9 | 1.00 | 9.12 |
| T4 TP boundary | maximum_probability | TP | 47/58 | 5 | 6 | 0.00 | 3.89 |
| T4 TP boundary | maximum_probability | SAP | 58/80 | 14 | 8 | 1.00 | 7.52 |

| Model | Representative | Correct | Wrong | Unresolved | Complete out of 4 |
| --- | --- | --- | --- | --- | --- |
| T1 previous pilot | weighted_center | 12 | 1 | 3 | 1 |
| T4 TP boundary | weighted_center | 13 | 0 | 3 | 1 |
| T1 previous pilot | weighted_medoid | 12 | 0 | 4 | 1 |
| T4 TP boundary | weighted_medoid | 12 | 1 | 3 | 1 |
| T1 previous pilot | maximum_probability | 11 | 1 | 4 | 0 |
| T4 TP boundary | maximum_probability | 12 | 0 | 4 | 1 |

**Matched optional loss comparison.**

| Arm | Correct | Wrong | Unresolved | Complete |
| --- | --- | --- | --- | --- |
| A matched | 11 | 1 | 4 | 1/4 |
| T4 TP boundary | 13 | 0 | 3 | 1/4 |

T4 achieved TP precision 11/14 and recall 11/15 on the opened images.

On seed 2026121700, the unchanged point picker moved from 6 to 0 px at waypoint 85 and from 8 to 4.12 px at waypoint 94. A false TP component remained at waypoint 6. The winning map cluster switched to six different supports, five of which met the existing 3 px criterion.

**Verdict.** T4 ranked first and failed the frozen readiness gate.

### 3.4 Reading it

- The frozen original and previous pilot medium results reproduced exactly, including the previous pilot's wrong seed-1700 L5 TP. Both mixed arms retained the wrong seed-1700 cluster. T2 additionally accepted wrong L5 TP on seed 1702, while T3 accepted wrong L5 TP on seed 1901.
- T2 recovered historical L5 SAP to 4/4 on both replays. The pilot recovered 3/4 and 2/4. T3 recovered 3/4 and 4/4, and did not improve the primary map totals over T2. The A matched run also recovered 4/4 SAP on v6r19 using no legacy data, so historical recovery cannot be assigned to legacy data alone.
- On the opened 99 images, T2 improved TP recall from 11/15 to 12/15 relative to the pilot while precision fell from 11/14 to 12/19. T3 matched 12/15 recall with 12/21 precision. Both mixed arms made zero TP accepts on the 60 clear-negative frames.
- On the previous pilot masks, switching to the confidence-weighted medoid moved the wrong seed-1700 L5 TP to unresolved. The primary map changed from 12 correct, one wrong, three unresolved, and one complete to 12 correct, zero wrong, four unresolved, and one complete. On the selected retrained arm, see the paired results table.
- On the previous pilot masks, using the maximum-probability pixel put the two shifted representatives at 0 px distance in the locked failure images. The same substitution across all four sessions changed the map totals to 11 correct, one wrong, and zero complete. On T4 the original representative gave 13 correct and zero wrong, the medoid gave 12 correct and one wrong, and the peak gave 12 correct and zero wrong. The primary point extractor is retained.
- T4 reached 8/8 correct L5 landmarks and fixed the medium trace with the same 2,000 images. The original L4 checkpoint gave 5/8 correct L4 landmarks and had both L4 landmarks correct in only one session. The L5-only ceiling on this set is 13/16 correct and 1/4 complete maps. The full readiness gate stays frozen at 14/16 and 3/4.
- On the 54-waypoint replay, T4 accepted wrong L5 SAP in seeds 2026121701 and 2026121703. In both sessions, two of three selected observations were on visible SAP frames but lay outside the frozen 3 px gate. The four errors were 3.61, 3.61, 3.16, and 3.61 px. Correct medium L5 performance did not transfer to that SAP history. On v6r19, session 01 had a wrong L5 SAP cluster after a 3/6 split under the strict-majority evaluator and also had unresolved L5 TP.
- Keep the 2,000-image cap. The matched loss experiment improved map correctness without increasing coverage or data size. Diagnose the two T4 SAP errors and evaluate L4 separately before asking to run reserved confirmation.
- The recorded demo shows seed 2026121900. Its final evaluator outcomes were L4 TP correct, L4 SAP unresolved, L5 TP correct, L5 SAP correct. The reserved confirmation seeds remain unused.

### 3.5 Files

| File | Contents |
|---|---|
| `evaluation-52731587/opened_99_images.json` | Exact image-level metrics, hard-case breakdown, confidence reliability and curve points |
| `evaluation-52731587/downstream.json` | Per-session and per-landmark scored results, view counts, spreads, distances and supporting IDs |
| `evaluation-52731587/v6r19_regression.json` | Reproduced historical scoring and per-landmark regression |
| `evaluation-52731587/representatives.json` | Paired point comparison on identical masks |
| `evaluation-52731587/learning_curves.png`, `precision_recall_99.png`, `paired_segmentation.png` | Review figures |
| `evaluation-52731587/landmark_sweep.mp4`, `video_verification.json` | Actual synchronized medium recording, with the new perception replay and offline evaluator panel |
| `reproduce.sh`, `slurm_accounting.txt`, `SHA256SUMS` | Resolved reproduction commands, resource accounting and package checksums |
