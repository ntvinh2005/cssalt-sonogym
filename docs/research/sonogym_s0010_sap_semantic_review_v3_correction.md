# s0010 SAP semantic review V3 targeted correction

This records the CPU-only correction of the V3 anatomical-consistency metric and V2/model comparison semantics. It reuses the frozen raw model prediction and does not approve SAP ground truth, alter masks or repeat inference.

## Result

**The correction passes: the invalid `segmentation > 0` metric is replaced by explicit source-label checks, and 19 distance/class conflicts are exported for review without changing the `MODEL_SAP_PROPOSAL_READY_FOR_REVIEW` proposal verdict.**

| step | what was measured | criterion | measured | verdict |
|---|---|---|---|---|
| 1 | anatomical consistency | use explicit vertebral IDs and renderer bone mapping; never use nonzero source label as bone | `109,139 / 110,736 (98.56%)` foreground voxels lie in acoustic-bone IDs; `1,597 / 110,736 (1.44%)` do not | pass; original 100% claim invalidated |
| 2 | V2/model comparison semantics | report 1.5 mm proximity independently from sampled model class | L4: 4 distance outliers and 7 sampled-IAP conflicts; L5: 7 and 1 | pass |
| 3 | compact correction package | every flagged point has coordinates, class, distance and MPR callout; checksum/ZIP pass | 19 callouts; `8 / 8` checksums; ZIP pass | pass |

The model-generated SAP masks remain research proposals. The correction narrows what the package demonstrates; it does not establish ultrasound SAP visibility or MBB readiness.

## 1. Anatomical consistency — where model foreground lies in source anatomy

`anatomical_consistency_report` in `workflows/two_stage_navigation/build_s0010_sap_semantic_review_v3.py`, run by CPU job `52085026`. It compares the frozen model foreground against explicit raw source labels.

**Criterion.** Vertebral consistency must use source IDs 25–31; renderer-bone consistency must use only IDs mapped to acoustic class 13 by the sealed `label_conversion.yaml`; `segmentation > 0` is prohibited as a bone definition.

### 1.1 Method

- Preserve raw prediction SHA-256 `a78daeae9159f0df28ba6fffcb8e8d7ef6c2e87f490c9598acfcb82efad1677f`.
- Count prediction foreground by raw source label and by model class.
- Report explicit vertebral labels 25–31 separately from all labels mapped to acoustic bone class 13.
- Report L3/L4/L5 class intersections separately.

### 1.2 Setup

| what | value | spec |
|---|---|---|
| run / checkpoint | correction job `52085026`, commit `405e7b5f71827a3f631842fb0c09b3a8634c8a2c`; original inference job `52078067` | §1 |
| inputs | frozen CT, segmentation and raw prediction from V3 | §1 |
| label mapping | `source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml`, SHA-256 `4ce73c412ef26f69fefe0caf003a531f4fd0928ac24f2248af0a019c4d76ff9a` | §1 |
| filters | source labels 25–31; acoustic class exactly 13 | §1 |

```bash
sbatch hpg/jobs/s0010_sap_semantic_review_v3_correction.sbatch
# Resolved builder command is recorded in runtime metrics/time-build.txt.
```

Nothing was dropped: all `110,736` raw model foreground voxels were counted.

### 1.3 Results

| mask definition | inside | outside | inside fraction |
|---|---:|---:|---:|
| explicit vertebral IDs 25–31 | 71,411 | 39,325 | 64.49% |
| all raw IDs mapped to acoustic bone class 13 | 109,139 | 1,597 | 98.56% |

| non-acoustic-bone source label | model foreground voxels |
|---|---:|
| 120 | 1,576 |
| 86 | 13 |
| 87 | 8 |

**Verdict: pass**, because the correction uses documented label sets and exposes all `1,597` foreground voxels outside acoustic bone. The previous `110,736 / 110,736` result from `segmentation > 0` is invalid and superseded.

### 1.4 Reading it

- The 64.49% figure is not a model failure: labels 25–31 cover only the declared spinal levels while the full prediction spans additional labelled vertebrae.
- The acoustic-bone comparison is the relevant full-volume consistency check.
- The correction does not independently validate SAP semantic accuracy.

### 1.5 Files

| file | contents |
|---|---|
| package `03_ANATOMICAL_CONSISTENCY.json` | full-volume, per-class and per-level counts |
| package `05_PROVENANCE.json` | frozen input hashes and invariants |

## 2. Point semantics — which V2 points require targeted review

The corrected comparison retains nearest model-SAP surface distance and sampled model class as independent fields.

**Criterion.** Every V2 point farther than 1.5 mm must be flagged as a distance outlier; every point sampling model IAP must be flagged as a class conflict regardless of distance. Background samples remain unresolved rather than automatic SAP errors.

### 2.1 Method

- Compute nearest model-SAP boundary distance in physical millimetres.
- Sample the raw model class at each rounded V2 index independently.
- Assign explicit categories for proximal SAP, proximal background uncertainty, sampled-IAP conflict and distant background.
- Export enlarged axial, sagittal and coronal MPRs around every distance outlier or IAP conflict.

### 2.2 Setup

| what | value | spec |
|---|---|---|
| V2 inputs | frozen 55-point L4 and 77-point L5 cores | §2 |
| distance tolerance | 1.5 mm, one source CT voxel | §2 |
| review inclusion | distance greater than 1.5 mm or sampled model label 7 | §2 |

```bash
correct_s0010_sap_semantic_review_v3.py \
  --prediction <frozen V3 raw prediction> --v2-package <frozen V2 package> \
  --label-conversion <sealed label conversion> --output-root <job output>
```

| | L4 | L5 |
|---|---:|---:|
| attempted V2 points | 55 | 77 |
| distance outliers | 4 | 7 |
| sampled-IAP conflicts | 7 | 1 |
| **targeted callouts** | **11** | **8** |

### 2.3 Results

| structure | sampled background | sampled SAP | sampled IAP | maximum distance |
|---|---:|---:|---:|---:|
| RIGHT L4 SAP | 44 | 4 | 7 | 2.121 mm |
| RIGHT L5 SAP | 71 | 5 | 1 | 22.198 mm |

**Verdict: pass**, because all 19 targeted points are explicitly represented without conflating geometric proximity with sampled semantics.

### 2.4 Reading it

- L4 sampled-IAP conflicts cluster at the SAP/IAP interface and may include voxel discretization or true proposal conflict; they are not automatically rejected.
- L5 callout 12 is the 22.198 mm midline outlier at world `(-3.120, 67.880, 319.300)` mm.
- L5 callouts 13–17 are additional background-sampled distance outliers from 3.000 to 7.348 mm.
- No V2 point or model voxel was relabelled.

### 2.5 Files

| file | contents |
|---|---|
| package `01_ALL_V2_POINT_COMPARISON.csv` | all 132 V2 points with independent distance/class fields |
| package `02_TARGETED_REVIEW_POINTS.csv` | 19 numbered conflicts/outliers |
| package `figures/RIGHT_L4_SAP_targeted_correction.png` | 11 L4 MPR callouts |
| package `figures/RIGHT_L5_SAP_targeted_correction.png` | 8 L5 MPR callouts |

## 3. Package validation — is the correction independently inspectable

The CPU job writes a small package that references, but does not duplicate, the frozen CT and raw prediction.

**Criterion.** Both focused test suites, all package checksums and ZIP integrity must pass; the original model mask and V3 ZIP hashes must remain unchanged.

### 3.1 Method

- Run V3 and correction tests before building.
- Hash all package files and test the ZIP.
- Record source and original artifact hashes in provenance.

### 3.2 Setup

| what | value | spec |
|---|---|---|
| job | `52085026`, one CPU, 8 GB requested, zero GPUs | §3 |
| original V3 ZIP | SHA-256 `ce55b17345507a5baa2ee6552daeb63c924e162be285507f3bfb9c8750f3a442` | §3 |
| correction ZIP | SHA-256 `06e7719b228382dcba04dc4dbb5e9b3ca7ec45d8a4204f86ff696c5bacc4405c` | §3 |

```bash
cd <correction package>
sha256sum -c checksums.sha256
cd ..
unzip -t s0010_right_l4l5_sap_semantic_review_v3_correction.zip
```

Nothing was dropped.

### 3.3 Results

| check | measured |
|---|---:|
| V3 focused tests | `9 / 9` pass |
| correction tests | `2 / 2` pass |
| checksums | `8 / 8` pass |
| ZIP integrity | pass |
| elapsed / maximum process RSS | 20 seconds / 2,089,392 KB |
| ZIP size | 1,093,756 bytes |

**Verdict: pass**, because every test and integrity gate succeeded and the frozen input hashes match.

### 3.4 Reading it

- This correction supersedes only the invalid validation metric and ambiguous comparison categories.
- The original V3 inference and proposal verdict remain preserved.
- `SAP_GT_NOT_VALIDATED`, `ULTRASOUND_SAP_NOT_CONFIRMED` and `MBB_END_TO_END_BLOCKED` remain the scientific status.

### 3.5 Files

| file | contents |
|---|---|
| `experiments/unity_two_stage_transfer/s0010_sap_semantic_review_v3_correction_result.json` | concise correction result |
| runtime `review_packages/correction_result.json` | full measured output |
| correction ZIP | compact independent-review artifact |
