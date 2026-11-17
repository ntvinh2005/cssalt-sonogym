# SonoGym s0010 RIGHT SAP/TP review v2 with ultrasound evidence

## Result

**Job `51764274` produced the requested review package, but semantic detector training remains
`BLOCKED_INDEPENDENT_REVIEW_AND_CORRESPONDENCE`.** The package contains four high-resolution
anatomical sheets, four real `conv` ultrasound sheets, editable geometry, a contact sheet and a
prepared-but-not-run CPU training contract. No annotation was promoted.

| step | criterion | measured | verdict |
|---|---|---|---|
| package integrity | tests and ZIP exit `0`; all payload files checksummed | 16 / 16 tests; 46 / 46 payload checksums; ZIP SHA-256 `9cfc8be...afb5bfc88` | PASS |
| anatomy review evidence | four candidates have mesh, CT, adjacent slices and editable masks | 4 / 4 exported; 0 / 4 independently accepted | REVIEW_READY |
| ultrasound evidence | candidate overlay only after raw-label round trip | 3,826 frames audited; 0 passed projection gate | NOT_OBSERVED_IN_EXISTING_DATA |
| semantic training | accepted boundary plus reviewed positive/negative correspondence | 0 / 4 eligible | BLOCKED |

## 1. Anatomical evidence — are the four candidate boundaries supportable?

**Criterion.** A candidate needs a documented qualified review confirming level, laterality,
semantic isolation, uncertainty and corrected geometry when required. Visual plausibility alone is
not acceptance.

### 1.1 Method

- Reconstructed the four annotation-v2 candidates exactly from the source STL vertex masks.
- Rendered a consistent posterior-oblique whole-vertebra view and close-up.
- Rendered axial, sagittal and coronal CT at five adjacent slices each (-6, -3, 0, +3, +6 mm).
- Distinguished the complete vertebral segmentation boundary (green) from the candidate points
  (blue SAP; orange TP) and included 10 mm scale bars and orientation labels.
- Preserved CSV, NPZ Boolean vertex masks, PLY point clouds, source STL and non-resampled CT/label
  crops for reviewer correction.

### 1.2 Setup

| item | value |
|---|---|
| source patient / side | `s0010` / `RIGHT` |
| task scope | L4/L5 SAP and TP only |
| source annotation | `s0010_anatomical_annotations_v2.json`; development-only provisional reference |
| execution | Slurm `51764274`; commit `9abed5f7b286e359306d93aa34c97144acac4de2` |
| resources | `anest-dept`; 4 CPU; 16 GB requested; 0 GPU; peak RSS 743.50 MiB |
| runtime | 1:45; `COMPLETED 0:0` |

```bash
sbatch --parsable --account=anest-dept --qos=anest-dept --partition=hpg-default \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:30:00 \
  --job-name=sg-sap-tp-review-v2 \
  --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env \
  hpg/jobs/two_stage_s0010_right_landmark_review_v2_ultrasound.sbatch
```

Nothing was dropped.

### 1.3 Results

| candidate | source status | observed boundary concern | advisory recommendation |
|---|---|---|---|
| L4 RIGHT SAP | UNRESOLVED | axis-aligned selection does not isolate SAP from lamina/posterior elements | REVISE_BOUNDARY |
| L4 RIGHT TP | PROVISIONAL | selection may include anterior/pedicle- or body-facing adjacent surface | REVISE_BOUNDARY |
| L5 RIGHT SAP | UNRESOLVED | same semantic-isolation problem as L4 SAP | REVISE_BOUNDARY |
| L5 RIGHT TP | PROVISIONAL | same anterior-adjacent-surface concern as L4 TP | REVISE_BOUNDARY |

The TP geometry is anatomically plausible enough to merit correction rather than rejection, but it
is not clean enough to serve as truth. SAP remains unresolved. These are agent recommendations,
not qualified anatomical decisions.

**Verdict: 4 / 4 review-ready; 0 / 4 accepted.**

### 1.4 Reading it

The enhanced views make the previous concerns easier to inspect but do not resolve them. The
editable masks let a reviewer correct the exact vertices rather than describe a boundary in prose.
No CT/mesh field is actor-visible.

### 1.5 Files

- ZIP: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/reviewer/s0010_right_sap_tp_review_v2_51764274.zip`
- Run: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_right_sap_tp_review_v2_ultrasound/51764274/`
- Machine result: `experiments/unity_two_stage_transfer/s0010_right_sap_tp_review_v2_ultrasound_result.json`

## 2. Ultrasound correspondence — can SAP/TP be identified in existing captures?

**Criterion.** Candidate-specific correspondence requires the proposed geometry to intersect a
verified ultrasound plane. A vertebral raw-label contour or bright interface alone does not prove
SAP/TP identity or recognizability.

### 2.1 Method

- Audited the actual model-based `conv` capture jobs `50908807` and `51377943`.
- Joined public image hashes with evaluator-only raw label frames and recorded reduced probe pose.
- Attempted to reconstruct each image plane using the checked SonoGym image geometry.
- Required >=0.985 exact raw-label agreement and >=0.90 bone-mask IoU before drawing a candidate
  overlay. Failed reconstructions show an explicit missing-data panel instead of an inferred mask.
- Included clean actor-visible images plus separately marked whole-vertebra evaluator contours.

### 2.2 Setup

| source | sessions | frames | role |
|---|---:|---:|---|
| Stage 1 G0/G1 job `50908807` | 16 | 3,376 | opened development capture |
| Stage 2 dataset job `51377943` | 6 | 450 | opened development capture |
| total | 22 | 3,826 | not an untouched test set |

### 2.3 Results

- 3,826 / 3,826 frames joined to image hash, raw label record and reduced pose.
- 0 / 3,826 passed the image-plane round-trip gate.
- The archives store `(surface_x, surface_z, in_plane_angle)` but not the full physical
  end-effector transform or exact per-pixel `human_img_coords` used by the renderer.
- Selected clean images visibly contain lumbar bone interfaces, and raw labels confirm L4/L5
  vertebral presence. This does not identify the interface as SAP or TP.
- Candidate-specific overlays and corresponding CT planes were deliberately withheld.

**Verdict: all four semantic structures are `NOT_OBSERVED_IN_EXISTING_DATA`.** This means the
required correspondence cannot be established from these records; it does not prove the anatomy
is intrinsically invisible to ultrasound.

### 2.4 Reading it

The most important result is the information deficit, not a negative detector result. The current
captures are useful for bone-morphology review, but they cannot produce defensible SAP/TP positive
or negative labels. A new capture must preserve the exact image-plane mapping.

### 2.5 Files

The ZIP contains `ultrasound_sheets/`, `ultrasound_correspondence_records.json` and
`contact_sheet.png`. All reviewed rows remain `training_label_eligible=false`.

## 3. Training preparation — what remains before the first fit?

**Criterion.** Fit only after an accepted/corrected boundary, verified ultrasound correspondence,
independent negative review, session-disjoint manifests and leakage checks exist.

### 3.1 Method

Prepared a semantic label schema, correspondence CSV, session-disjoint development split,
ambiguity/ignore rules, checksums and a four-CPU ExtraTrees configuration using normalized 16x16
intensity and gradient features. No model was fit.

### 3.2 Setup

Review-selected sessions are `review_only_excluded`; they cannot later be described as unseen
confirmation. The future model-selection split remains development-only on one patient.

### 3.3 Results

| capability | readiness |
|---|---|
| CT/mesh candidate correction | READY_FOR_QUALIFIED_REVIEW |
| L4/L5 whole-bone ultrasound context | AVAILABLE |
| SAP/TP-specific ultrasound labels | BLOCKED_CORRESPONDENCE |
| CPU ExtraTrees fit | PREPARED_NOT_RUN |
| MBB surrogate target region | UNRESOLVED |

If a reviewer accepts/corrects only TP, a TP-only feasibility detector is the largest defensible
claim. The complete unilateral L4-L5 two-site task remains blocked until L4/L5 SAP is supported.

**Verdict: training is not authorized or scientifically ready.**

### 3.4 Reading it

The shortest path is: qualified boundary correction, then one bounded targeted capture that stores
the full transform or exact per-pixel image coordinates for positives, adjacent ambiguous views and
independent negatives. After correspondence review, the prepared CPU baseline can run. A later new
sealed session is still required for independent confirmation.

### 3.5 Files

Inside the ZIP, see `TRAINING_READINESS_REPORT.md` and `training_preparation/`.
