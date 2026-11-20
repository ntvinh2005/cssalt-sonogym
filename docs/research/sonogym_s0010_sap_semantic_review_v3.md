# s0010 RIGHT L4/L5 SAP semantic review V3

This records one pretrained CT subregion-inference pass and the resulting static review package for provisional RIGHT L4/L5 SAP annotations. It does not approve semantic ground truth, define an MBB target, change actor inputs or validate ultrasound visibility.

## Result

**The V3 verdict is `MODEL_SAP_PROPOSAL_READY_FOR_REVIEW` for both RIGHT L4 SAP and RIGHT L5 SAP: inference and alignment passed, while semantic acceptance remains pending independent review.**

| step | what was measured | criterion | measured | verdict |
|---|---|---|---|---|
| 1 | original CT integrity and alignment | every frozen CT preflight gate passes | real HU CT, `259 × 259 × 283`, 1.5 mm isotropic RAS; CT/segmentation affine difference `0.0` | pass |
| 2 | official checkpoint inference | prediction passes every frozen inference/alignment gate | labels `0–7`; output affine difference `0.0`; correction job `52084265` finds `109,139 / 110,736 (98.56%)` foreground voxels in explicitly mapped acoustic-bone IDs | pass; original nonzero-label metric superseded |
| 3 | level/laterality mapping and V2 comparison | both SAP and paired IAP proposals are nonempty and inspectable; no semantic-accuracy threshold | L4 `51 / 55 (92.7%)`; L5 `70 / 77 (90.9%)` V2 core locations within 1.5 mm of model SAP | ready for review, not accepted anatomy |
| 4 | static-package integrity | all references, checksums and ZIP members pass | `49 / 49` references; `60 / 60` checksums; ZIP test pass | pass |

V3 adds independent pretrained-model evidence that V2 lacked. V2/model proximity compares imperfect proposals and is not accuracy against verified SAP anatomy.

## 1. CT preflight — is the original s0010 input valid for inference

`workflows/two_stage_navigation/build_s0010_sap_semantic_review_v3.py`, run by the V3 Slurm entry point. It validates the original CT intensity volume and its registered whole-vertebra segmentation before model loading.

**Criterion.** The CT and segmentation must match the sealed hashes, shape, 1.5 mm spacing and RAS orientation; their affines must match exactly; L3/L4/L5 labels must be present; CT intensities must be plausible HU values.

### 1.1 Method

- Read the original NIfTI intensity and label volumes without resampling.
- Compare hashes, shapes, affines, orientation and declared L3/L4/L5 labels against the sealed protocol.
- Reject a binary mask, STL rasterization or synthetic intensity input.

### 1.2 Setup

| what | value | spec |
|---|---|---|
| run / checkpoint | job `52078067`, commit `0bb79e0c82c3096c6e57332f8c3d0e92d245e264` | §1 |
| CT | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/assets/data/HumanModels/selected_dataset/s0010/ct.nii.gz`, SHA-256 `eff5e5f5829cf8cc3e413b3ac09ce455e45bc0f91f5ba37fab9b37b1fe683546` | §1 |
| segmentation | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/assets/data/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz`, SHA-256 `0d435584ed7851736f044c0a9c906231ef3f139b7cda16770d3066773992a1ed` | §1 |
| protocol | `experiments/unity_two_stage_transfer/s0010_sap_semantic_review_v3_protocol.yaml` | §1 |

```bash
sbatch hpg/jobs/s0010_sap_semantic_review_v3.sbatch
# The job runs: build_s0010_sap_semantic_review_v3.py preflight --ct <sealed CT> --segmentation <sealed segmentation> --model-archive <official archive> --expected-model-sha256 384b6a... --output-dir <run>/preflight
```

Nothing was dropped: one sealed CT and one registered segmentation were attempted and accepted.

### 1.3 Results

| quantity | measured |
|---|---:|
| shape | `259 × 259 × 283` |
| spacing | `1.5 × 1.5 × 1.5` mm |
| orientation | RAS |
| affine maximum absolute difference | `0.0` |
| CT range | `-1047` to `1682` HU |
| L3 / L4 / L5 voxels | `15,471` / `15,489` / `14,926` |

**Verdict: pass**, because every sealed CT and registration gate passed on the original intensity data.

### 1.4 Reading it

- The input is the original CT, not a label-derived surrogate.
- Exact CT/segmentation alignment supports level assignment but not SAP semantics.

### 1.5 Files

| file | contents |
|---|---|
| runtime `preflight/ct_preflight.json` | measured CT and affine checks |
| package `01_CT_PREFLIGHT.md` | reviewer-facing summary |
| runtime `preflight/ct_preflight.png` | representative raw CT views |

## 2. Pretrained inference — does the official model produce aligned subregions

The official legacy nnU-Net checkpoint is run once, without training, fine-tuning or parameter search.

**Criterion.** The official archive hash must match; prediction shape and affine must match the CT; labels must be a subset of `0–7` with every foreground class present; no unsupported label may be emitted.

### 2.1 Method

- Use release `V1`, `Task005`, `nnUNetTrainerV2`, `nnUNetPlansv2.1`, `3d_fullres`, fold `0`, `model_final_checkpoint`.
- Keep official test-time mirroring enabled.
- Preserve the raw prediction before level/laterality mapping; no postprocessing file was supplied by the release, so none was invented.

### 2.2 Setup

| what | value | spec |
|---|---|---|
| model archive | `nnUNetTrainerV2__nnUNetPlansv2.1.zip`, SHA-256 `384b6ead3dd4df2b34d3fc6dc1d72c845dab76807b237a488c91d56758c06938` | §2 |
| runtime | nnU-Net `1.7.1`, PyTorch `2.5.1+cu121` | §2 |
| resources | one NVIDIA L4, four CPUs, 30 GB requested; walltime 30 minutes | §2 |
| raw prediction | runtime `raw_prediction/s0010.nii.gz`, SHA-256 `a78daeae9159f0df28ba6fffcb8e8d7ef6c2e87f490c9598acfcb82efad1677f` | §2 |

```bash
nnUNet_predict -i <run>/input -o <run>/raw_prediction -t 005 -m 3d_fullres \
  -tr nnUNetTrainerV2 -p nnUNetPlansv2.1 -f 0 \
  --num_threads_preprocessing 2 --num_threads_nifti_save 1 \
  --overwrite_existing -chk model_final_checkpoint
```

Nothing was dropped: one CT case was attempted and one prediction was exported.

### 2.3 Results

| check | measured |
|---|---:|
| prediction labels | `[0, 1, 2, 3, 4, 5, 6, 7]` |
| input/output affine maximum difference | `0.0` |
| foreground inside acoustic-bone source IDs | `109,139 / 110,736 (98.56%)` |
| foreground outside acoustic-bone source IDs | `1,597 / 110,736 (1.44%)` |
| inference wall time | `160.43` seconds |
| inference process maximum RSS | `8,082,652` KB |

**Verdict: pass**, because the raw prediction met every sealed inference and alignment gate.

### 2.4 Reading it

- The release archive is a legacy nnU-Net v1 checkpoint despite current repository wording that mentions nnU-Net v2.
- Successful inference is not proof that the predicted SAP class is correct on s0010.
- The original `segmentation > 0` consistency metric was invalid because the label map contains non-bone tissues; correction job `52084265` supersedes that metric without changing the raw prediction.
- The academic non-commercial checkpoint is stored outside Git and is not redistributed in the ZIP.

### 2.5 Files

| file | contents |
|---|---|
| runtime `raw_prediction/s0010.nii.gz` | untouched full-volume model labels |
| runtime `logs/inference.log` | complete inference log |
| package `02_MODEL_INFERENCE.md` | provenance and resolved command |

## 3. Anatomical comparison — is there enough semantic evidence for external review

The V3 builder assigns global model classes to vertebral level by intersection with frozen whole-vertebra labels and assigns RIGHT using physical RAS x above each level's median x coordinate.

**Criterion.** Both RIGHT SAP and paired adjacent IAP proposals must be nonempty, aligned and anatomically inspectable. No absolute V2-proximity threshold exists; V2 is an imperfect sparse proposal and is used only for comparison.

### 3.1 Method

- Pair RIGHT L4 SAP with RIGHT L3 IAP and RIGHT L5 SAP with RIGHT L4 IAP.
- Measure V2 surface-point support within one source CT voxel, 1.5 mm.
- Preserve model voxels near/not-near the V2 core separately; report V2 sampled class independently from distance proximity.
- Project model proposals into archived ultrasound only as privileged geometry; visibility and semantic identity remain `NOT_REVIEWED`.

### 3.2 Setup

| what | value | spec |
|---|---|---|
| V2 input | `s0010_right_l4l5_sap_review_v2`, frozen hashes `46b92e...` (L4) and `9a62fa...` (L5) | §3 |
| level labels | L3 `29`, L4 `28`, L5 `27` | §3 |
| laterality | physical RAS x above per-level median is RIGHT | §3 |
| comparison tolerance | 1.5 mm | §3 |

```bash
build_s0010_sap_semantic_review_v3.py build \
  --ct <sealed CT> --segmentation <sealed segmentation> \
  --prediction <run>/raw_prediction/s0010.nii.gz \
  --v2-package <frozen V2 package> --archived-ultrasound-root <V3 archived frames> \
  --output-root <run>/review_packages --job-id 52078067
```

| | L4 | L5 |
|---|---:|---:|
| attempted V2 core locations | 55 | 77 |
| outside 1.5 mm support | 4 | 7 |
| **supported** | **51** | **70** |

### 3.3 Results

| quantity | RIGHT L4 SAP | RIGHT L5 SAP |
|---|---:|---:|
| model voxels | 259 | 338 |
| model volume | 874.125 mm³ | 1,140.750 mm³ |
| connected components | 1 | 1 |
| V2 supported locations | `51 / 55 (92.7%)` | `70 / 77 (90.9%)` |
| V2-to-model median / p95 / maximum | 1.500 / 2.121 / 2.121 mm | 1.500 / 4.790 / 22.198 mm |
| minimum paired-IAP distance | 1.500 mm | 1.500 mm |
| curated ultrasound examples | 2 | 2 |
| projected SAP pixels per ultrasound example | 2–3 | 3–5 |

**Verdict: `MODEL_SAP_PROPOSAL_READY_FOR_REVIEW` for both structures**, because the level-specific model proposals and paired IAP relations are aligned and inspectable. The L5 22.198 mm outlier and sparse ultrasound projections remain explicit review issues; neither structure is accepted semantic ground truth.

### 3.4 Reading it

- Model/V2 distance proximity is strong enough to make external review more decisive, but is not semantic agreement.
- Correction job `52084265` separately flags seven L4 and one L5 sampled-IAP conflicts, including conflicts that lie within 1.5 mm.
- Most model SAP volume is model-only: 685.125 mm³ at L4 and 894.375 mm³ at L5.
- The L5 maximum-distance outlier requires reviewer attention; it is not hidden by the median.
- Archived ultrasound projections are too sparse to establish SAP cortical visibility or semantic identity.

### 3.5 Files

| file | contents |
|---|---|
| package `03_COMPARISON_RESULTS.csv` and `.json` | per-structure measurements |
| package `figures/disagreement/*` | CT/V2/model disagreement sheets |
| package `figures/3d/*paired_facet.png` | SAP/IAP/TP context |
| package `figures/ultrasound/*comparison.png` | clean image versus privileged projection |
| package `masks/*` | compact physical-coordinate review masks |

## 4. Package validation — is the review artifact self-contained

The build validates references, hashes every package member, excludes model weights and tests the ZIP.

**Criterion.** Every required/reference file and checksum must pass; the ZIP must open without error; no checkpoint or executable may be packaged.

### 4.1 Method

- Resolve package-relative file references.
- Recompute SHA-256 for every listed member.
- Test every ZIP entry and scan for prohibited weight files.

### 4.2 Setup

| what | value | spec |
|---|---|---|
| package | runtime `review_packages/s0010_right_l4l5_sap_semantic_review_v3` | §4 |
| ZIP | runtime `review_packages/s0010_right_l4l5_sap_semantic_review_v3.zip` | §4 |
| ZIP SHA-256 | `ce55b17345507a5baa2ee6552daeb63c924e162be285507f3bfb9c8750f3a442` | §4 |

```bash
cd <package>
sha256sum -c checksums.sha256
cd ..
unzip -t s0010_right_l4l5_sap_semantic_review_v3.zip
```

Nothing was dropped.

### 4.3 Results

| check | measured |
|---|---:|
| referenced files | `49 / 49` |
| checksum entries | `60 / 60` |
| prohibited weight files | 0 |
| ZIP size | 9,714,988 bytes |
| V3 focused tests | `7 / 7` pass in the inference environment |
| V2 static-review regressions | `7 / 7` pass in the project environment |
| geometry-audit regressions | `3 / 3` pass in the project environment |

**Verdict: pass**, because every integrity, exclusion and regression check succeeded.

### 4.4 Reading it

- Job `52078067` completed `0:0` in 5 minutes 29 seconds; Slurm batch MaxRSS was 5,123,032 KB.
- The 9.7 MB ZIP is self-contained for review but intentionally excludes the CT source and third-party weights.
- V1/V2 artifacts and verdicts remain unchanged.

### 4.5 Files

| file | contents |
|---|---|
| `experiments/unity_two_stage_transfer/s0010_sap_semantic_review_v3_result.json` | concise result record |
| runtime `review_packages/build_result.json` | package paths, verdict and hash |
| runtime `metrics/` | GPU telemetry and timing |
