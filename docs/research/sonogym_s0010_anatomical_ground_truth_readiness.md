# SonoGym s0010 anatomical ground-truth readiness

This report validates a privileged provisional annotation layer for SonoGym patient s0010. It covers the anatomy needed to begin the verified two-stage Unity-method transfer and audits, without claiming clinical validity, bony geometry relevant to a later simulated lumbar medial branch block extension.

## Result

**The source anatomy and major ordered lumbar references are ready for a bounded supervised Unity Stage 1 observability experiment; discriminative ultrasound recognition is unproven and nerve-specific medial branch block work is blocked.**

| step | what was measured | criterion | measured | verdict |
|---|---|---|---|---|
| 1 | source-anatomy and transform integrity | [A0](../notation.md#a0) | identical CT/segmentation shape and affine; maximum bidirectional surface p95 1.5 mm | PASS |
| 2 | major vertebral and sacral coverage | [A1](../notation.md#a1) | 7 / 7 structures, 3.90--7.01 mm uncertainty | PASS |
| 3 | fine bony geometry audit | [A2](../notation.md#a2) | 14 / 34 PROVISIONAL; 20 / 34 UNRESOLVED | PASS |
| 4 | actor-visible anatomical evidence | [A3](../notation.md#a3) | all labels present in forward/reverse captures; no disjoint-trajectory recognition test | INCONCLUSIVE |
| 5 | task-specific readiness | [A4](../notation.md#a4) | Stage 1 READY; L4 view acquisition PROVISIONAL; procedure-specific work BLOCKED | MIXED |

The evidence supports the smallest image-recognition and spatial-reconstruction experiment. It does not establish image-only level discrimination, learned L4 view acquisition, or a known medial branch nerve location.

## 1. Source anatomy integrity — do the CT, segmentation, meshes, and simulator transforms agree?

workflows/two_stage_navigation/annotate_s0010_anatomy.py reconstructs source geometry, converts among recorded frames, and measures bidirectional mesh/segmentation surface agreement.

**Criterion.** [A0](../notation.md#a0), specified by the researcher on 2026-11-12. The implemented surface criterion is p95 no greater than one CT-voxel diagonal, 2.598 mm.

### 1.0 Quantities

| symbol | definition | source |
|---|---|---|
| A0 | source-anatomy integrity gate | [notation](../notation.md#a0) |
| A1 | major-structure coverage gate | [notation](../notation.md#a1) |
| A2 | fine-bony-geometry feasibility gate | [notation](../notation.md#a2) |
| A3 | ultrasound observability gate | [notation](../notation.md#a3) |
| A4 | task-specific readiness classification | [notation](../notation.md#a4) |

### 1.1 Method

- NIfTI headers determine shape, spacing, affine, and RAS orientation.
- Mesh vertices are interpreted in voxel-index coordinates because the running task applies 0.0015 m per index and the patient root transform without NIfTI affine translation.
- Nearest-surface distances are measured in both mesh-to-segmentation and segmentation-to-mesh directions.
- The patient-local-to-world transform is checked by numerical round trip.

### 1.2 Setup

| what | value | spec |
|---|---|---|
| generator commit | 7a99c2feecbdd2130bebad51ab311a39729948d5 | annotation implementation |
| evidence commit | 798581e | generated manifest and figures |
| Unity reference | f7e182b5e2ac71cc73ac41f8b2a928ce10e220c0 | verified merged implementation |
| CT | /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/assets/data/HumanModels/selected_dataset/s0010/ct.nii.gz | SHA-256 eff5e5f5829cf8cc3e413b3ac09ce455e45bc0f91f5ba37fab9b37b1fe683546 |
| segmentation | /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/assets/data/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz | SHA-256 0d435584ed7851736f044c0a9c906231ef3f139b7cda16770d3066773992a1ed |
| runtime | Python 3.10.22; NumPy 1.26.4; SciPy 1.15.2; nibabel 5.3.2; trimesh 5.2.0 | CPU only |

~~~bash
source .sonogym.env
export PYTHONNOUSERSITE=1
export PYTHONPATH="$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/evaluation"
taskset -c 0 "$SONOGYM_ENV_PREFIX/bin/python" workflows/two_stage_navigation/annotate_s0010_anatomy.py \
  --asset-root "$SONOGYM_RUNTIME_DIR/assets/data/HumanModels" \
  --capture-root "$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/g0_g1_v2/50908807" \
  --output-json experiments/unity_two_stage_transfer/s0010_anatomical_annotations_v1.json \
  --figure-dir docs/research/figures/s0010_anatomical_ground_truth \
  --unity-commit f7e182b5e2ac71cc73ac41f8b2a928ce10e220c0
# Resolved: one CPU; no Slurm job; no GPU.
~~~

Nothing was dropped.

### 1.3 Results

| setting | measured value |
|---|---|
| CT and segmentation shape | both (259, 259, 283) |
| spacing | both (1.5, 1.5, 1.5) mm |
| affine maximum absolute difference | 0.0 |
| orientation | RAS; +x patient right, +z cranial before root transform |
| maximum bidirectional surface p95 | 1.5 mm |
| maximum observed surface distance | 3.354 mm |
| transform round-trip maximum absolute error | 2.78e-17 m |

**Verdict: PASS**, because the frame checks are exact and 1.5 mm is below the 2.598 mm surface criterion.

### 1.4 Reading it

- CT and segmentation share header geometry.
- STL surfaces show the expected approximately one-voxel offset from raw-label boundaries.
- Centroid disagreement is retained in annotation uncertainty because mesh vertices are not volumetrically uniform.
- This validates simulator coordinate reconstruction, not clinical registration.

### 1.5 Files

| file | contents |
|---|---|
| [annotation manifest](../../experiments/unity_two_stage_transfer/s0010_anatomical_annotations_v1.json) | transforms, hashes, geometry, validation, and annotations |
| [manifest digest](../../experiments/unity_two_stage_transfer/s0010_anatomical_annotations_v1.sha256) | immutable JSON digest |
| [annotation module](../../source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/anatomical_ground_truth.py) | schema, transforms, comparison, and leakage guard |

## 2. Major anatomy coverage — are sacrum, S1, and L1--L5 usable as ordered references?

The same workflow creates one privileged [Group A annotation](../notation.md#group-a-annotation) per raw anatomical identity and a candidate posterior surface region.

**Criterion.** [A1](../notation.md#a1), specified by the researcher on 2026-11-12: all seven structures require source identity, geometry, uncertainty, and supporting evidence.

### 2.1 Method

- Raw-label identities are resolved from the current task label-conversion source and checked against meshes.
- Reference centroids use all voxels of each raw label.
- Candidate ultrasound-facing regions use the posterior 20% of mesh vertices by level-specific y-coordinate; these are regions, not point targets.
- Uncertainty is the maximum of half the voxel diagonal, surface p95, and mesh/segmentation centroid disagreement.
- Longitudinal coordinates are measured relative to frozen Home index 195 at 1.5 mm per index.

### 2.2 Setup

| what | value | spec |
|---|---|---|
| run identity | manifest SHA-256 8b58eaa09cd20ac3bd92f667c397440c1f08550ef649edea304890a99ce7b20a | §1 invocation |
| raw labels | 25--31 | current label_conversion.yaml, SHA-256 4ce73c412ef26f69fefe0caf003a531f4fd0928ac24f2248af0a019c4d76ff9a |
| Home reference | z index 195 | frozen capture protocol |
| inputs | CT, segmentation, and seven meshes from §1 | no additional assets |

~~~bash
# Exact invocation: §1.2; Group A is emitted in the same manifest.
~~~

| | value |
|---|---|
| attempted | 7 |
| dropped | 0 |
| **final** | **7** |

### 2.3 Results

| anatomy | raw label | z centroid, index | Home-relative coordinate, mm | uncertainty, mm | status |
|---|---:|---:|---:|---:|---|
| sacrum | 25 | 87.652 | -161.021 | 3.901 | PROVISIONAL |
| S1 | 26 | 103.914 | -136.629 | 4.462 | PROVISIONAL |
| L5 | 27 | 120.164 | -112.254 | 5.456 | PROVISIONAL |
| L4 | 28 | 139.425 | -83.362 | 6.004 | PROVISIONAL |
| L3 | 29 | 160.523 | -51.716 | 6.833 | PROVISIONAL |
| L2 | 30 | 181.319 | -20.522 | 7.007 | PROVISIONAL |
| L1 | 31 | 202.620 | +11.430 | 6.195 | PROVISIONAL |

**Verdict: PASS**, because 7 / 7 requested major structures have geometry-supported annotations and explicit uncertainty.

### 2.4 Reading it

- The prior raw-label mapping and centroids were independently reproduced from current files.
- Sacrum and S1 remain distinct privileged identities; they need not produce two separable ultrasound events.
- All seven identities map to acoustic class 13, so renderer class does not encode level identity.
- The ordered references can supervise anatomical discovery but do not prove image-only recognition.

### 2.5 Files

| file | contents |
|---|---|
| [L3 CT views](figures/s0010_anatomical_ground_truth/ct_mpr_L3.png) | axial, sagittal, and coronal review |
| [L4 CT views](figures/s0010_anatomical_ground_truth/ct_mpr_L4.png) | axial, sagittal, and coronal review |
| [L5 CT views](figures/s0010_anatomical_ground_truth/ct_mpr_L5.png) | axial, sagittal, and coronal review |
| [3D review](figures/s0010_anatomical_ground_truth/lumbar_mesh_regions.png) | oriented meshes and investigated regions |

## 3. Fine bony geometry — which later medial branch block references are supportable?

The same workflow investigates [Group B annotations](../notation.md#group-b-annotation) and preserves unresolved records when geometry alone is insufficient.

**Criterion.** [A2](../notation.md#a2), specified by the researcher on 2026-11-12: every requested structure is inspected, and unsupported anatomy is UNRESOLVED rather than fabricated.

### 3.1 Method

- Bilateral L3--L5 transverse-process, L3--S1 facet, and sacral-ala candidates remain low-confidence PROVISIONAL regions.
- Superior and inferior articular, mammillary, and accessory processes remain UNRESOLVED because mesh quantiles do not establish their identity.
- Lumbar medial branches relate to the transverse-process/superior-articular-process and mammillo-accessory regions; L5 has a distinct S1/sacral-ala relationship ([CT-anatomic correlation](https://pmc.ncbi.nlm.nih.gov/articles/PMC7977091/), [3D mapping](https://pmc.ncbi.nlm.nih.gov/articles/PMC11372964/), [consensus guidelines](https://pmc.ncbi.nlm.nih.gov/articles/PMC7362874/)).
- No nerve is segmented; pedicle-screw trajectories are not used as targets.

### 3.2 Setup

| what | value | spec |
|---|---|---|
| run identity | manifest SHA-256 8b58eaa09cd20ac3bd92f667c397440c1f08550ef649edea304890a99ce7b20a | §1 invocation |
| anatomy | s0010 L3--L5, S1, sacrum | one fixed patient |
| evidence | CT multiplanar and 3D mesh review | evaluator-only |

~~~bash
# Exact invocation: §1.2; Group B is emitted in the same manifest.
~~~

| | value |
|---|---|
| attempted | 34 |
| retained PROVISIONAL | 14 |
| retained UNRESOLVED | 20 |
| dropped | 0 |
| **final** | **34** |

### 3.3 Results

| structure | levels | result | uncertainty |
|---|---|---|---|
| transverse-process region | L3--L5, bilateral | 6 / 6 PROVISIONAL, low confidence | 6.0 mm |
| facet-joint region | L3--L4, L4--L5, L5--S1, bilateral | 6 / 6 PROVISIONAL, low confidence | 6.0 mm |
| sacral-ala region | bilateral | 2 / 2 PROVISIONAL, low confidence | 8.0 mm |
| superior/inferior articular processes | L3--L5, bilateral | 12 / 12 UNRESOLVED | not assigned |
| mammillary/accessory processes | L3--L5 | 6 / 6 UNRESOLVED | not assigned |
| medial branch block target | L4--L5, bilateral | 2 / 2 MBB_TARGET_UNRESOLVED | not assigned |

**Verdict: PASS**, because every requested structure was inspected and ambiguous structures remain unresolved; this is not a pass for procedure-target readiness.

### 3.4 Reading it

- Resolved regions are simulator engineering candidates, not clinician-validated landmarks.
- Visual 3D review kept articular-process candidates UNRESOLVED rather than promoting quantile regions to named points.
- No exact nerve centre or supported procedure target can be inferred.
- Missing nerve data do not block Group A work.

### 3.5 Files

| file | contents |
|---|---|
| [3D region review](figures/s0010_anatomical_ground_truth/lumbar_mesh_regions.png) | evidence for provisional and unresolved candidates |
| [annotation manifest](../../experiments/unity_two_stage_transfer/s0010_anatomical_annotations_v1.json) | all 34 records with evidence and limitations |

## 4. Ultrasound correspondence — are the annotated structures present and discriminative?

The same workflow audits existing job 50908807 captures and produces separate clean-image and evaluator-overlay examples.

**Criterion.** [A3](../notation.md#a3), specified by the researcher on 2026-11-12: geometric presence is insufficient; repeatable actor-visible discrimination must be shown on disjoint trajectories.

### 4.1 Method

- All 3,376 frames from 16 sessions are scanned for raw-label presence.
- Outbound caudal and return cranial observations are counted separately.
- Review examples include L4 forward/reverse positives and L3/L5 difficult negatives.
- No classifier is trained and no recognition score is estimated.

### 4.2 Setup

| what | value | spec |
|---|---|---|
| capture job | 50908807 | pre-existing GPU capture; no new GPU job |
| protocol digest | a08fefa87e8f4b0cfb54d1ab0194ebe4e78bcbb05bda68d5071c33f27cba4322 | frozen manifest |
| capture root | /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/g0_g1_v2/50908807 | 16 sessions |
| actor view | clean ultrasound pixels | raw labels excluded |
| evaluator view | raw-label contours | privileged visualization only |

~~~bash
# Exact invocation: §1.2; no new capture was performed.
~~~

| | value |
|---|---|
| attempted | 3,376 frames |
| invalid or missing | 0 |
| **final** | **3,376 frames** |

### 4.3 Results

| anatomy | visible frames | forward | reverse | sessions |
|---|---:|---:|---:|---:|
| sacrum | 1,282 | 625 | 657 | 16 / 16 (100%) |
| S1 | 1,142 | 569 | 573 | 16 / 16 (100%) |
| L5 | 915 | 475 | 440 | 16 / 16 (100%) |
| L4 | 778 | 394 | 384 | 16 / 16 (100%) |
| L3 | 930 | 458 | 472 | 16 / 16 (100%) |
| L2 | 1,040 | 517 | 523 | 16 / 16 (100%) |
| L1 | 586 | 317 | 269 | 16 / 16 (100%) |

**Verdict: INCONCLUSIVE**, because presence is bidirectional and complete by session, but no image-only held-out test establishes anatomical-level discrimination.

### 4.4 Reading it

- Existing data include positives and adjacent-level difficult negatives for a small supervised experiment.
- All identities render as acoustic class 13.
- Evaluator overlays prove geometry in the field of view only and are forbidden actor inputs.
- A3 requires a lightweight model evaluated on held-out acquisition sessions.

### 4.5 Files

| file | contents |
|---|---|
| [ultrasound review](figures/s0010_anatomical_ground_truth/ultrasound_correspondence.png) | clean positives/negatives beside evaluator-only overlays |
| [annotation manifest](../../experiments/unity_two_stage_transfer/s0010_anatomical_annotations_v1.json) | counts, frame hashes, directions, and session provenance |

## 5. Readiness and verification — what experiment may proceed now?

The annotation tests and existing two-stage CPU suites validate records, transforms, protocol integrity, and ground-truth leakage rejection.

**Criterion.** [A4](../notation.md#a4), specified by the researcher on 2026-11-12: readiness is classified separately for immediate navigation and later procedure capabilities.

### 5.1 Method

- Schema validation rejects incomplete status, provenance, geometry, confidence, or uncertainty fields.
- Actor-payload validation rejects raw labels, segmentation, mesh identity, target coordinates, evaluator scores, and medial branch block regions.
- Annotation, contract, protocol, and Stage 1 tests run together.
- Each capability requires its own evidence; later capabilities cannot inherit Group A readiness.

### 5.2 Setup

| what | value | spec |
|---|---|---|
| test runtime | Python 3.10.8; pytest 7.4.3 | UF module, one CPU |
| test files | test_anatomical_ground_truth.py plus three two-stage suites | 42 tests |
| artifact validation | 41 records; 5 visualizations | all paths and hashes checked |
| resources | one CPU; no GPU; no Slurm job | bounded local execution |

~~~bash
. ./.sonogym.env
module load python/3.10
export PYTHONNOUSERSITE=1
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/evaluation"
taskset -c 0 python -m pytest -q \
  tests/test_anatomical_ground_truth.py \
  tests/test_two_stage_contracts.py \
  tests/test_two_stage_protocol.py \
  tests/test_two_stage_stage1.py
# Resolved result: 42 passed in 0.43 s.
~~~

Nothing was dropped.

### 5.3 Results

| capability | classification | evidence boundary |
|---|---|---|
| begin Unity Stage 1 experiment | READY | A0/A1 pass; positives and negatives exist; leakage tests pass |
| define L4 best-valid-view labels | PROVISIONAL | L4 geometry and images exist; acceptable-view semantics remain to freeze |
| provisional bony medial branch block target | BLOCKED | named fine structures and nerve relationship unresolved |
| simulated instrument positioning | BLOCKED | no supported target, trajectory, or safety model |
| nerve-specific validation | BLOCKED | no nerve data or clinician validation |
| CPU verification | PASS | 42 / 42 passed in 0.43 s |

**Verdict: PASS for starting the bounded Unity Stage 1 experiment and PROVISIONAL for defining Stage 2; procedure-specific capabilities remain BLOCKED.**

### 5.4 Reading it

- The next experiment is not PPO and not another annotation pass.
- The smallest useful experiment trains the verified patch-feature ExtraTrees baseline on session-disjoint data to distinguish L4-relevant morphology from adjacent L3/L5 negatives.
- It then aggregates image-derived scores with executed motion and tests L4-region reconstruction without privileged target coordinates.
- An A3 failure must stop Stage 2 rather than be bypassed using the raw L4 coordinate.
- Expected compute: existing-data preparation and training on at most four CPU cores; after review, one bounded L4 GPU closed-loop evaluation. No new GPU capture is needed for the first offline test.

### 5.5 Files

| file | contents |
|---|---|
| [annotation tests](../../tests/test_anatomical_ground_truth.py) | schema, transforms, completion, and leakage checks |
| [annotation workflow](../../workflows/two_stage_navigation/annotate_s0010_anatomy.py) | reproducible CPU generation |
| [machine-readable result](../../experiments/unity_two_stage_transfer/s0010_anatomical_annotations_v1.json) | annotation deliverable |
| [notation registry](../notation.md) | definitions used by this report |
