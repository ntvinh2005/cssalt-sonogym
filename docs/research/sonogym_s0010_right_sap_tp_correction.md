# s0010 RIGHT SAP/TP correspondence and boundary correction

## Result

| workstream | result |
|---|---|
| Native coordinate reference and inverse-projection tests | PASS |
| Existing-frame correspondence recovery | **PROJECTION_REPAIR_BLOCKED (0 / 3,826)** |
| L4 RIGHT TP boundary | PROPOSED_REVISION_PENDING_ANATOMICAL_REVIEW |
| L5 RIGHT TP boundary | PROPOSED_REVISION_PENDING_ANATOMICAL_REVIEW |
| L4/L5 RIGHT SAP boundaries | UNRESOLVED_BOUNDARY |
| Detector dataset | TRAINING_BLOCKED_WITH_SPECIFIC_REASON (0 eligible rows) |
| Training | NOT_RUN |

The historical v2 result remains unchanged: 3,826 audited frames, zero verified
candidate projections, and zero independently approved semantic boundaries.  The native-axis
repair is real and materially improves replay agreement, but the reduced three-value pose record
does not reconstruct any stored frame at the unchanged dual gate.  Training on these projected
regions would therefore turn registration error and unresolved anatomy into labels.

## 1. Native coordinate repair

### Criterion

Image-local coordinates must use `(width, elevation, depth)`, array rows must advance along
depth, candidate plane distance must use elevation, and the full-pose CPU reference must match the
origin-sign behavior of `LabelImgSlicer`.  The historical X-Y/X-Z mismatch must remain detectable.

### Method

`ultrasound_plane_geometry.py` transcribes the native slicer coordinate construction without
importing Isaac Sim.  It separates `EXACT_POSE_REPLAY` from
`REDUCED_POSE_RECONSTRUCTION`, validates proper rotation matrices, mirrors the native image-origin
normal sign rule, and provides the inverse pixel projection.  Tests use independently specified
coordinates for identity and nonidentity transforms rather than generating expectations with the
helper under test.

### Setup

```bash
sbatch \
  --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env \
  hpg/jobs/two_stage_s0010_right_landmark_correction.sbatch
```

Final job: `51779234`; source commit used by the job:
`6fbe38df115d2483c334874ce5027c9f3c2d6b83`.

### Results

All 30 focused and historical regression tests passed.  These include seven native-coordinate
tests, seven correction-contract tests, and the 16 pre-existing RIGHT landmark-review tests.
The old helper's median exact-label agreement was 0.3002 and median bone IoU was 0.0000.  The
corrected helper reached median 0.8157 and 0.5638 respectively.  Exact agreement improved on
3,825 / 3,826 frames and bone IoU improved on 3,808 / 3,826 frames.

### Reading it

The reviewed axis mismatch and image-origin discrepancy were reproduced and corrected.  This
establishes mathematical parity with the inspected native source, not runtime exact-pose parity:
the captures did not persist the full `human_to_ee_pos` and `human_to_ee_quat` that the renderer
uses.

### Files

- `source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/ultrasound_plane_geometry.py`
- `tests/test_ultrasound_plane_geometry.py`
- `tests/test_s0010_right_landmark_correction.py`

## 2. Existing-frame replay

### Criterion

Do not weaken the frozen requirements: exact raw-label agreement must be at least 0.985 and bone
mask IoU must be at least 0.90 on the same frame.  Unverified projections remain unknown and
cannot become negative labels.

### Method

The replay joined all public frames to privileged labels and recorded three-value realized poses,
then reconstructed each plane from the surface map.  The code also evaluated the historical
helper on the same frames.  Local bone agreement is reported only as a diagnostic and does not
replace the frozen gate.

### Setup

The replay used the opened development sources from jobs `50908807` and `51377943`.  Search of
their public, annotation, home-action, scan-action, and Stage 2 action records found only reduced
poses; no full end-effector transform or per-pixel sampled volume coordinates were persisted.

### Results

| quantity | historical | corrected |
|---|---:|---:|
| frames | 3,826 | 3,826 |
| source-wise unique hashes | 3,573 | 3,573 |
| median exact agreement | 0.3002 | 0.8157 |
| maximum exact agreement | 0.6256 | 0.9732 |
| median bone IoU | 0.0000 | 0.5638 |
| maximum bone IoU | 0.3747 | 0.9439 |
| frames passing exact gate | 0 | 0 |
| frames passing bone-IoU gate | 0 | 27 |
| frames passing both gates | 0 | **0** |

Stage 1 median exact agreement improved from 0.3094 to 0.8134 and median bone IoU from 0.0000
to 0.5519.  Stage 2 improved from 0.2657 to 0.8403 and from 0.0795 to 0.6372.  The best joint
candidate was Stage 2 session `s2v13tr_03`, frame 38: exact agreement 0.9709 and bone IoU 0.9439;
it still failed the exact gate.  No sampling coordinate was clipped out of the volume.

### Reading it

The axis repair explains a large fraction of the historical mismatch but not the remaining edge
and interface displacement.  The task has full actual end-effector position/quaternion at runtime,
while `realized_pose_index` retains only surface x, surface z, and one angle.  Missing full pose is
therefore a demonstrated metadata limitation.  The remaining error may contain actual tracking,
image-origin calibration, roll/tilt, or timing effects; existing records cannot distinguish these,
and no constants were fitted post hoc.

### Files

- Runtime output: `projection_repair/per_frame_metrics.csv`
- Session summary: `projection_repair/replay_summary.json`
- Four representative failures: `projection_repair/failure_examples/`

## 3. Anatomical boundary review

### Criterion

An automated revision must preserve source mesh indices, label uncertainty explicitly, and remain
separate from qualified anatomical approval.  A region that cannot be separated from adjacent
structures must remain unresolved.

### Method

For TP, the review identifies the largest lateral CT cross-section narrowing inside the historical
candidate and exports a conservative distal surface core, a two-voxel boundary uncertainty zone,
and removed vertices.  This is a surface-region proposal, not a whole-process volumetric
segmentation.  For SAP, CT and mesh do not provide an independent separation from lamina and
adjacent posterior elements, so all historical candidate vertices remain in the uncertainty set
and no replacement mask is proposed.

### Setup

The original meshes and selected vertex indices were reused without remeshing.  Before/after mesh
views and axial, sagittal, and coronal CT slabs use the same source coordinates.  Slab overlays are
labeled `+/-1.5 mm`; they are not presented as exact in-plane contours.

### Results

| candidate | original | proposed | uncertain | removed | result |
|---|---:|---:|---:|---:|---|
| L4 RIGHT SAP | 2,198 | 0 | 2,198 | 0 | UNRESOLVED_BOUNDARY |
| L4 RIGHT TP | 4,833 | 1,681 | 988 | 2,164 | PROPOSED_REVISION_PENDING_ANATOMICAL_REVIEW |
| L5 RIGHT SAP | 2,422 | 0 | 2,422 | 0 | UNRESOLVED_BOUNDARY |
| L5 RIGHT TP | 4,343 | 2,589 | 1,305 | 449 | PROPOSED_REVISION_PENDING_ANATOMICAL_REVIEW |

The L4 TP transition is at x index 145; its proposed core begins at x 147 and shifts the centroid
11.62 mm laterally relative to the historical quantile candidate.  The L5 transition is also at x
145; its proposed core begins at x 147 and shifts the centroid 4.54 mm.  Self-review found the TP
cores cleaner and removed the conspicuous central/anterior extensions, but also confirmed that
the base cut remains an anatomical-review question.  It did not justify promotion.  The SAP views
confirmed the original coordinate-cut ambiguity, so retaining the full regions as uncertainty is
the fail-closed outcome.

### Reading it

Track B advanced from unchanged quantile selections to two editable TP proposals and two explicit
SAP unresolved records.  These are useful review artifacts but are not semantic labels.  Agent
self-review is engineering QA and cannot supply reviewer identity or qualification.

### Files

- Editable masks and source indices: `boundary_revisions/editable_geometry/`
- Before/after CT and mesh sheets: `boundary_revisions/figures/`
- Reviewer checklist: `ANATOMICAL_REVIEW_CHECKLIST.md`

## 4. Ultrasound integration and detector dataset

### Criterion

Candidate overlays require verified correspondence.  Geometry intersection, acoustic visibility,
semantic identity, and training label are distinct states.  Unknown or unreviewed cases must not
become negatives.

### Method

The candidate manifest uses explicit states: `VERIFIED`, `UNVERIFIED`, or `MISSING_METADATA` for
projection; `PRESENT`, `ABSENT`, or `UNKNOWN` for candidate intersection; and separate acoustic,
semantic, and training-label fields.  The intended supervision is a 48x80 image patch, reduced to
16x16 intensity and gradient features.  At inference, patch proposals must come from actor-visible
bone-interface morphology, not privileged candidate coordinates.

### Setup

All source sessions, including `ua_00` through `ua_03`, are declared previously opened
development data.  The split manifest no longer describes them as untouched audit data.

### Results

Because 0 / 3,826 frames passed the correspondence gate, there are zero candidate intersections
that can be asserted and zero candidate-specific clean/overlay pairs.  The concrete sample
manifest consequently has zero rows and zero training-eligible examples.  Acoustic visibility and
semantic identity remain `NOT_REVIEWED`; no unknown value was converted to a negative.

### Reading it

The detector dataset is transparently empty, not cosmetically populated.  User authorization to
train does not remove the data gate: running ExtraTrees now would train on registration error and
unapproved anatomy.  No model fitting or MBB target construction was performed.

### Files

- `training_preparation/candidate_sample_manifest.csv`
- `training_preparation/label_target.json`
- `training_preparation/session_split_manifest.yaml`

## 5. Reproducibility and next gate

### Criterion

The run must use at most four CPU cores, 16 GB RAM, zero GPUs, preserve the historical package,
and produce a checksum-verified compact bundle.

### Method

Job `51779234` ran on account `anest-dept` with linked-library thread pools limited to one thread.
The historical ZIP checksum was verified before export.

### Setup

Python 3.10.22 in the pinned SonoGym environment; 4 allocated CPUs, 16 GB requested RAM, 0 GPUs.

### Results

The job completed in 1 minute 29 seconds with exit code 0.  Slurm batch peak RSS was 655,084 KiB;
the timed exporter peak was 694,272 KiB.  The run directory occupies 16 MB and the final ZIP 3.4
MB.  The historical ZIP SHA-256 remains
`9cfc8be656e219043010eea78b2787cd7f4de0c1e7f442c3d5205c0afb5bfc88`; the correction ZIP
SHA-256 is `1dcf71bd463bb53abd72877e831961bb5c70060b2e66cf7229c3ea1ca1f20ebd`.

### Reading it

The smallest data repair is a bounded development capture that records, for every ultrasound
frame, the exact `human_to_ee_pos`, `human_to_ee_quat`, producer sequence, and preferably the exact
sampled pixel-to-volume coordinates used by `LabelImgSlicer`.  It must include a synchronized raw
label array and hash.  Before any positive SAP/TP label is promoted, a qualified anatomical
reviewer must also accept or correct the exact geometry version.  Only then is a TP-only CPU
detector pilot defensible; SAP and any procedure target remain blocked until their boundaries and
ultrasound identity are supported.

### Files

- Run: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_right_sap_tp_correction/51779234/`
- ZIP: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/reviewer/s0010_right_sap_tp_correction_v1_51779234.zip`
