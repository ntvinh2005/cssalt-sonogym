# s0010 SAP observability V1 projection correction

This records a CPU-only correction of the V1/V1R1 geometry-to-image rasterization and a first-bone-surface audit of the same 12 saved frames. It does not repeat capture or inference, approve SAP anatomy, train a classifier, define an MBB target or authorize Stage 3.

## Result

**The axis-ordering defect is reproduced and corrected, but both RIGHT L4 and RIGHT L5 remain `INCONCLUSIVE` under the fail-closed two-session surface-evidence criterion.**

| step | what was measured | criterion | measured | verdict |
|---|---|---|---|---|
| 1 | native flat-index pixel position | corrected raster must match renderer width/depth ordering at off-center and asymmetric pixels | positional parity tests pass | pass |
| 2 | offline replay of job `52092083` | all saved raw labels must reproduce exactly from native coordinates | 12 / 12 (100%) | pass |
| 3 | first-bone pseudo-SAP surface evidence | at least two projection-gated aligned frames from two sessions per level | L4: 1 frame, 1 session; L5: 0 frames, 0 sessions | fail closed; both `INCONCLUSIVE` |
| 4 | review artifact | all checksums, tests and ZIP integrity pass | 39 / 39 tests; 59 / 59 checksums | pass |

The correction invalidates only V1/V1R1 spatial projection evidence. Freshness, synchronized pose/native-coordinate capture, frame hashes and raw-label replay remain valid.

## 1. Native raster correction — are flattened coordinates placed at the renderer's pixels

`workflows/two_stage_navigation/run_s0010_sap_observability.py:project_mask`, tested by `tests.test_s0010_sap_observability`. It converts one flattened native coordinate map into the displayed 200-row by 150-column mask.

**Criterion.** The raster must use the renderer's native width-major flattening and display transpose; an off-center pixel and an asymmetric pattern must match `raw.reshape(width, depth).T`. This criterion comes from `stage2_sim.raw_label_slice()` and the native slicer.

### 1.1 Method

- Invalid V1/V1R1 path: direct reshape to 200 rows by 150 columns.
- Corrected path: reshape to 150 width positions by 200 depth positions, then transpose.
- Preserve `project_mask_legacy_v1()` only to reproduce before/after evidence.
- Compare pixel positions, not only projected pixel counts.

### 1.2 Setup

| what | value | spec |
|---|---|---|
| source commit | `02ccbbf11bf828765a58ca4770125d164e024373` | §1 |
| image dimensions | width 150, depth 200, one elevation sample | §1 |
| positional fixtures | `(row=100, column=40)` and three asymmetric pixels | §1 |

```bash
PYTHONPATH=<runtime>/envs/stage1-python:<repo>/source/spinal_surgery:<repo>/workflows/two_stage_navigation \
  <runtime>/envs/sonogym/bin/python -m unittest -v \
  tests.test_s0010_sap_observability \
  tests.test_s0010_sap_observability_correction \
  tests.test_s0010_sap_observability_contract \
  tests.test_ultrasound_plane_geometry \
  tests.test_acquisition_integrity
```

Nothing was dropped.

### 1.3 Results

| fixture | invalid V1/V1R1 position | corrected position |
|---|---|---|
| flat index 8100 | row 54, column 0 | row 100, column 40 |
| asymmetric multi-pixel mask | displaced | exact native raw-label raster |

**Verdict: pass**, because all positional fixtures reproduce the renderer-compatible row/column image.

### 1.4 Reading it

- V1/V1R1 pixel counts remain numerically unchanged while their locations move.
- Historical centroids, margins, gates and alignment results are invalid.
- Historical acquisition and saved-array integrity are unaffected.

### 1.5 Files

| file | contents |
|---|---|
| `run_s0010_sap_observability.py` | corrected and legacy rasterization helpers |
| `test_s0010_sap_observability.py` | off-center, asymmetric and raw-label parity tests |

## 2. Saved-frame correction — what changes on the 12 acquired observations

`workflows/two_stage_navigation/correct_s0010_sap_observability_v1.py`, run by `hpg/jobs/s0010_sap_observability_v1_correction.sbatch`. It reloads job `52092083`, reproduces raw labels, and emits legacy/corrected positions without changing ultrasound or CT masks.

**Criterion.** All 12 raw-label images must exactly equal source-volume sampling through the saved native coordinates before any corrected projection is interpreted.

### 2.1 Method

- Reload raw ultrasound, raw labels and 30,000 native coordinates for every frame.
- Resample the unchanged source segmentation and require exact array equality.
- Apply the legacy and corrected rasterizations to the same pseudo-SAP, core and uncertainty masks.
- Recompute centroids, margins and the unchanged 20-pixel/four-pixel-margin projection gate.

### 2.2 Setup

| what | value | spec |
|---|---|---|
| correction run | job `52115861`, commit `02ccbbf11bf828765a58ca4770125d164e024373` | §2 |
| source run | job `52092083`, 12 frames, two sessions | §2 |
| prediction SHA-256 | `a78daeae9159f0df28ba6fffcb8e8d7ef6c2e87f490c9598acfcb82efad1677f` | §2 |
| segmentation SHA-256 | `0d435584ed7851736f044c0a9c906231ef3f139b7cda16770d3066773992a1ed` | §2 |
| resources | one CPU, 8 GB requested, zero GPUs; 31 seconds | §2 |

```bash
sbatch hpg/jobs/s0010_sap_observability_v1_correction.sbatch
# Resolved builder invocation is recorded in the job script and package provenance.
```

| sample accounting | value |
|---|---:|
| attempted saved frames | 12 |
| dropped | 0 |
| **final** | **12** |

### 2.3 Results

| measurement | historical | corrected |
|---|---:|---:|
| target frames passing projection gate | 7 / 8 (87.5%) | 6 / 8 (75.0%) |
| frames preserving pseudo/core pixel counts | — | 12 / 12 (100%) |
| exact raw-label replay | — | 12 / 12 (100%) |
| mean target-core centroid displacement | — | 65.267 pixels |

**Verdict: pass**, because 12 / 12 raw-label rasters reproduce exactly and corrected positions are exported without altering inputs.

### 2.4 Reading it

- The L4 target gate changes from 3 / 4 to 2 / 4; L5 remains 4 / 4.
- One L4 session-00 frame changes from gate pass to fail after its corrected core lies within four pixels of the right boundary.
- The 65.267-pixel mean shift demonstrates that the old spatial conclusions cannot be retained.

### 2.5 Files

| file | contents |
|---|---|
| package `01_CORRECTED_PROJECTION_METRICS.csv` | per-frame old/corrected positions, gates and surface evidence |
| package `03_CORRECTED_BINARY_MASKS.npz` | corrected pixel masks |
| package `verification_arrays/` | ultrasound, raw labels, native coordinates, poses and frame records |

## 3. First-bone surface audit — do corrected target views support a repeatable cortical interface

`correct_s0010_sap_observability_v1.py:first_bone_surface` and `summarize_structure`. They compare clean-image cortical polylines with the first source-label bone hit along each image column, then identify whether that hit belongs to the pseudo-SAP proposal.

**Criterion.** `SAP_SURFACE_VISIBLE_PROVISIONAL` requires at least two projection-gated `SAP_SURFACE_CANDIDATE_VISIBLE` frames from two reset sessions. The four-pixel image tolerance is frozen before this correction and no result-dependent threshold search is performed.

### 3.1 Method

- Lock 20 clean-image cortical polylines before corrected overlays are opened; SHA-256 `b965a384887e8c20cf3bfe211756c05d7f19b11e3aa4141bf9ec5f4707370ee2`.
- For each image column, locate the shallowest raw source label mapped to acoustic bone class 13.
- Record whether that first hit belongs to the pseudo-SAP proposal or another bone region.
- Measure the minimum row/column distance from the clean polyline to first-hit pseudo-SAP pixels.
- Apply the projection gate before aggregating repeatable evidence across sessions.

### 3.2 Setup

| what | value | spec |
|---|---|---|
| clean review | single-agent opened-development notes; not independent semantic truth | §3 |
| alignment tolerance | four pixels | §3 |
| target frames | four L4 and four L5 across two sessions | §3 |
| controls | two L4 and two L5 context frames | §3 |

Preliminary CPU job `52114287` completed but is superseded: its structure
aggregator counted aligned edge frames before applying the corrected projection
gate. Job `52115861` adds that fail-closed requirement and is the only reported
correction result.

```bash
<runtime>/envs/sonogym/bin/python \
  workflows/two_stage_navigation/correct_s0010_sap_observability_v1.py \
  --run-root <runtime>/outputs/unity_two_stage_transfer/s0010_sap_observability_v1r1/52092083 \
  --prediction <runtime>/outputs/unity_two_stage_transfer/s0010_sap_semantic_review_v3/52078067/raw_prediction/s0010.nii.gz \
  --segmentation <runtime>/assets/data/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz \
  --label-conversion source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml \
  --polylines experiments/unity_two_stage_transfer/s0010_sap_observability_cortical_polylines_v1.csv \
  --output-root <job-output>/review_packages/s0010_right_l4l5_sap_observability_v1_correction \
  --zip <job-output>/review_packages/s0010_right_l4l5_sap_observability_v1_correction.zip
```

Nothing was dropped.

### 3.3 Results

| level | target frames | projection-gated | per-frame surface candidate visible | qualifying frames | qualifying sessions | verdict |
|---|---:|---:|---:|---:|---:|---|
| RIGHT L4 | 4 | 2 | 3 | 1 | 1 / 2 | `INCONCLUSIVE` |
| RIGHT L5 | 4 | 4 | 0 | 0 | 0 / 2 | `INCONCLUSIVE` |

L4 polyline-to-first-hit-pseudo-SAP distances are 3.000, 1.000, 2.236 and 5.385 pixels. Only the 2.236-pixel frame also passes the corrected projection gate. L5 distances are 8.062, 16.125, 9.220 and 5.831 pixels.

**Verdict: fail closed**, because neither level has qualifying aligned evidence in both sessions; no SAP training label is released.

### 3.4 Reading it

- Corrected L4 geometry is substantially closer to visible right-edge interfaces than V1R1 reported.
- Two aligned L4 session-00 frames remain excluded because the proposed core is too close to the image boundary.
- No L5 frame aligns within four pixels despite all four passing the corrected projection gate.
- No frame supports a structure-level `OCCLUDED_BY_OTHER_BONE` conclusion under the implemented rule.
- The clean polylines are development annotations by the same agent and cannot establish semantic SAP identity.

### 3.5 Files

| file | contents |
|---|---|
| package `02_CORTICAL_POLYLINES.csv` | locked clean-image polylines |
| package `figures/frame_comparisons/` | clean, invalid, corrected and first-bone views |
| package decision sheets | per-level review evidence and limits |

## 4. Artifact validation — can the correction be reproduced independently

The job packages source arrays, code, tests, figures and machine-readable masks under one checksum manifest.

**Criterion.** All focused tests, all listed checksums and ZIP integrity must pass; the ZIP must contain the native coordinates and raw labels required to recompute the correction.

### 4.1 Method

- Run 39 positional, surface, acquisition and contract tests.
- Verify every package member listed in `checksums.sha256`.
- Test the compressed archive and record its SHA-256.

### 4.2 Setup

| what | value | spec |
|---|---|---|
| package job | `52115861` | §4 |
| ZIP size | 7,531,349 bytes | §4 |
| ZIP SHA-256 | `9dc92294b354ad56bb6071678cbe0a4ef4aad83b25b06c9c98dea1b462613fbd` | §4 |

```bash
cd <package-root> && sha256sum -c checksums.sha256
unzip -t s0010_right_l4l5_sap_observability_v1_correction.zip
```

Nothing was dropped.

### 4.3 Results

| validation | measured |
|---|---:|
| focused tests | 39 / 39 (100%) |
| package checksums | 59 / 59 (100%) |
| ZIP integrity | pass |

**Verdict: pass**, because all validation checks succeed and both sessions' original verification arrays are included.

### 4.4 Reading it

- Engineering correction is complete.
- Scientific status remains `INCONCLUSIVE` for both structures.
- `SAP_ULTRASOUND_GROUND_TRUTH`, MBB target validation and MBB end-to-end remain blocked.

### 4.5 Files

| file | contents |
|---|---|
| `experiments/unity_two_stage_transfer/s0010_sap_observability_v1_correction_result.json` | concise measured result |
| correction ZIP | self-contained external-review package |
