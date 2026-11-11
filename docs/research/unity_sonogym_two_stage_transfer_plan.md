# Phase 5C: verified Unity two-stage method transfer to SonoGym

Date: 2026-11-11
Status: source audit and implementation design only
Execution: no simulator code change, GPU job, dataset capture, or model training was performed

## 1. Decision summary

The Unity `main` merge does preserve the complete successful two-stage method. The audited
Unity HEAD is `f7e182b5e2ac71cc73ac41f8b2a928ce10e220c0`, which is both `origin/main` and the
merge of `codex/phase2-best-valid-t7-acquisition`. The source, frozen checkpoints, final demo
controller, tests, actor artifacts, and independent evaluator artifacts are present at that
revision.

The method to transfer is specifically:

1. **Stage 1 anatomical discovery:** physically return to an anatomically unlabeled Home,
   perform forward and reverse longitudinal sweeps, detect and classify image patches around
   bright interfaces, project patch evidence into a Home-relative coordinate map using the
   calibrated image footprint and executed motion, require bidirectional support, infer an
   ordered landmark map, and abstain if the map is incomplete or ambiguous.
2. **Stage 2 best-valid-view acquisition:** physically move to the Stage 1 estimate, rotate
   into the acquisition plane, execute a bounded deterministic spatial/angular search, use an
   image-only validity classifier and a separate image-only quality regressor, physically
   return to the best valid recorded pose, acquire fresh images, confirm them independently,
   and then run a privileged evaluator after actor output is frozen.

The SonoGym transfer is **not ready for model fitting yet**. SonoGym clearly renders structured
bone interfaces and shadows, and its source assets contain separately identifiable L1--L5/S1
meshes. However, the running `conv` ultrasound renderer collapses all vertebral source labels
to the same acoustic bone class. Existing evidence does not establish that L4 identity, or a
complete ordered lumbar event sequence that uniquely determines L4, is observable from
ultrasound plus admissible action history. Gate G0 is therefore `NOT_ESTABLISHED`. The next
approved implementation must begin with a bounded anatomical-observability and label-integrity
pilot, not with Stage 2 and not with target-coordinate control.

## 2. Verified source revisions and runtime

| Component | Audited revision or version | Evidence |
| --- | --- | --- |
| Unity Thoracic US Coach | `f7e182b5e2ac71cc73ac41f8b2a928ce10e220c0` | Current `origin/main`; verified merge is an ancestor of HEAD and equals HEAD |
| CSSALT SonoGym destination | `5ed04d117bb0e425b7abbb1f2feca3bef26b726f` | Local `main == origin/main` before this report; prior untracked documentation existed |
| Upstream SonoGym | `e67be58334d1a5274f0913af36f56e4b0b7ffe5a` | Present in destination history and pinned manifests |
| Isaac Lab | `21f7136325136ca3f6ca4e0a8125edffe5c24f7e` | Clean detached runtime checkout |
| Isaac Sim / Python | 4.5.0.0 / 3.10.22 | Current installed package metadata |
| PyTorch / torchvision / skrl | 2.5.1+cu121 / 0.20.1+cu121 / 1.4.2 | Current installed package metadata |
| NumPy / SciPy | 1.26.4 / 1.15.3 | Current installed package metadata; SciPy is 1.15.3 although `pinned-versions.txt` records 1.15.2 |
| Anatomy asset revision | `b37b080a8673f856266a2306724e48d5e034521a` | `environment/assets-manifest.json` |

The current SonoGym working tree had no tracked modification before this document. The only
pre-existing worktree item was an untracked `docs/` draft that described an older, incorrect
Unity method. It is superseded by this report rather than treated as evidence.

## 3. Verified Unity Stage 1

### 3.1 Physical acquisition and Home-relative history

The final demo entry point in
`Assets/ThoracicUSCoach/Runtime/ThoracicUSCoachController.RandomStartLandmarkDemo.cs`
configures the ground-truth-free acquisition in
`ThoracicUSCoachController.G2R2ContinuousSweep.cs`.

The actual episode is:

- optional seeded 2-D random start;
- physical lateral and longitudinal return to a fixed scene-authored operator Home;
- fresh Home baseline;
- caudal outbound and physical reverse to Home;
- cranial outbound and physical reverse to Home;
- blind actor-side analysis after capture.

The physical step is 1.5 mm. The ground-truth-free setup traces the skin in both directions
within a 120 mm mechanical ceiling and keeps a 6 mm boundary reserve. Every observation is
captured through `CaptureDirectProducerFreshFrame`, receives a SHA-256 hash and monotonic pose/
producer identifiers, and is paired with requested action, accepted status, achieved
displacement, cumulative executed travel, contact/workspace status, and rollback state.
Rejected actions do not advance the actor-relative coordinate. Initial random-start placement
and return-to-Home frames are explicitly excluded from anatomical inference.

### 3.2 Interface candidates and supervised patch likelihood

The final inference path is
`Tools/ThoracicUSCoach/ground_truth_free_exploration.py`, using
`autonomous_t2t8_patch_likelihood.py` and the interface tracker in
`g2_r4_multi_hypothesis_tracker.py`.

The tracker creates up to three bright-interface hypotheses from intensity and vertical-gradient
ridges, enforces a smooth path, and keeps only spatially diverse alternatives. The supervised
patch classifier then evaluates patches at projected Home-relative coordinates.

Verified frozen primary feature/model details are:

- source patch: 48 pixels wide x 80 pixels deep;
- interface offset: 34 pixels from patch top;
- per-patch 5th/95th percentile normalization;
- resize to 16 x 16;
- flattened normalized intensity, x-gradient, y-gradient, plus six summary values;
- feature width: 774;
- `ExtraTreesClassifier`: 240 trees, maximum depth 18, minimum leaf size 3,
  balanced class weights, seed 20261103;
- actor-relative aggregation grid: 0.5 mm;
- minimum event separation: 10 mm;
- primary frozen acceptance threshold: 0.70;
- at least two forward and two reverse observations at an accepted event.

Training coordinates are privileged development labels, but the saved actor receives only
ultrasound patches, acquisition footprint, and executed Home-relative motion. Pixel-to-map
projection uses the rendered footprint and the explicit image-axis sign; it does not read the
Unity world pose at inference.

### 3.3 Final primary-plus-verifier bundle

The final Stage 1 checkpoint is:

`Artifacts/ThoracicUSCoach/sweep_count/G2_5_RANDOM_START_HOME_ROBUSTNESS_20261103_R01/actor_only_evidence/home_robust_primary_v2_verifier_bundle.joblib`

SHA-256: `0b9ccabe2e4d5b5e9546b13870cd6fa2ce02a5cb3046b632c223142d0df73236`.

It contains both `model` and `verifierModel`. The verifier is not optional in the final demo.
It only rescues a primary candidate that:

- failed solely because its primary score is below 0.70;
- still has primary score at least 0.65;
- has a verifier-supported coordinate within a 2 mm search margin;
- has bidirectional support; and
- reaches verifier score at least 0.90.

The bundle metadata identifies three primary development sessions and nine independent verifier
sessions. Both models use the same 16 x 16 intensity/gradient feature definition. These Unity
thresholds and spatial values are source facts, not proposed SonoGym hyperparameters.

### 3.4 Landmark acceptance and abstention

`build_relative_map` assigns T2--T8 only when exactly seven accepted events remain in order.
T7 is event six. Otherwise `predictedT7RelativeMm` is null and the map status is
`INCONCLUSIVE`. `ground_truth_free_exploration.py` additionally checks bidirectional support and
that the two edge events have observed margins; it may extend coverage and reanalyze, but it
does not fabricate an event.

This absolute-level inference depends on the thoracic task's complete seven-level corridor and
ordering prior. It must not be copied to SonoGym without a verified lumbar anchor and complete
observable sequence.

## 4. Verified Unity Stage 2

### 4.1 Frozen image-only models

The selected checkpoint is:

`Artifacts/ThoracicUSCoach/active2d/phase2_best_view_20261103/tilted_transverse_v5/models/phase2_best_view_adapted_global.joblib`

SHA-256: `f0b79e965023bdfca6ae15711d53c795aa3880fd16c99ed69d10ef05fad1beb1`.

Its schema is `phase2-best-view-extra-trees-adapted-v2`, selected feature mode is `global`, and
the frozen validity threshold is `0.8010418379849094`.

`phase2_best_view_recognition.py`, `phase2_best_view_adaptation.py`, and the checkpoint establish:

- grayscale resize to 128 x 128, then 1st/99th percentile normalization;
- normalized image resized to 32 x 32;
- global feature = 32 x 32 intensity + 16 x 16 absolute x-gradient + 16 x 16
  absolute y-gradient, width 1,536;
- Model A: `ExtraTreesClassifier`, 360 trees, depth 28, minimum leaf size 1,
  balanced weights, `max_features=0.5`, seed 20261103;
- Model B: `ExtraTreesRegressor`, 360 trees, depth 28, minimum leaf size 1,
  `max_features=0.5`, seed 20261104;
- classifier label: privileged `targetAcceptable` during offline fitting;
- regressor label: privileged `targetScore`, fitted on positives and near-target frames with
  score at least 0.50;
- actor input contract: only `cleanUltrasoundImage`;
- explicitly forbidden inputs: anatomy labels, anatomical coordinates, evaluator score,
  probe world pose, and T7 target position.

The threshold is calibrated just above the maximum development-negative probability. The
training code labels the adapted study `DEVELOPMENT_ONLY_ALL_CHALLENGES_PREVIOUSLY_CONSUMED`;
the later frozen full-pipeline episodes therefore matter because they execute fresh closed-loop
capture and independent scoring rather than merely repeating offline classifier metrics.

### 4.2 Deterministic physical search actually executed

`ThoracicUSCoachController.Phase2BestView.cs` starts only from a valid Stage 1 map. It physically
moves from Home to the inferred T7 coordinate, performs 30 counter-clockwise rotation actions,
adds approximately +3 degrees of acquisition tilt, and then executes:

- 70 coarse poses: longitudinal -12 to +6 mm in 3 mm rows, right-lateral 0 to 27 mm
  in 3 mm increments, serpentine order, yaw 0;
- 90 fixed boundary-refinement poses: longitudinal -3, -1.5, 0, +1.5, +3 mm;
  right-lateral 24, 25.5, 27 mm; yaw 0 to 15 degrees in 3-degree increments;
- 75 micro poses around the best accepted candidate or best unaccepted proposal:
  longitudinal offsets -0.75, -0.375, 0, +0.375, +0.75 mm; lateral offsets
  -1.5, -0.75, 0, +0.75, +1.5 mm; yaw offsets -3, 0, +3 degrees;
- physical return to the best accepted pose;
- two new confirmation frames.

This explains the observed 237 Stage 2 frames: 70 + 90 + 75 + 2. Candidate eligibility is
set by Model A; Model B ranks only accepted candidates. A high quality score cannot make an
invalid candidate eligible. At least one fresh confirmation must remain valid and no more than
0.08 below the remembered best quality.

Contact enforcement, workspace checks, surface raycasts, meaningful-progress checks, and
teleport-like displacement checks occur on every movement. A rejected movement or rollback
sets failure and aborts the pipeline. Freshness is checked using pose epoch and producer scan
serial. Privileged `ScoreCandidate`, target marker distance, pose, and anatomy labels are stored
only in evaluator artifacts and do not select the actor candidate.

One source discrepancy is important for faithful transfer: `BuildPhase2FineTargets` and
`Phase2FineSearchSeeds` exist and have Editor tests, but the final `RunPhase2BestViewCore`
does not call them. The verified 237-frame runs use the fixed 90-pose boundary refinement
described above, then micro refinement. The transfer should reproduce executed behavior, not
infer an adaptive fine search from unused helpers.

## 5. Verified Unity evidence and limits

The full development confirmation at
`RANDOM_START_LANDMARK_DEMO_20261104_110000` reports:

- 249 fresh Stage 1 observations, 247 accepted and zero rejected movements;
- seven accepted T2--T8 landmarks, TP 7, FP 0, FN 0;
- T7 error 0.079838 mm;
- Stage 2 `ACTOR_SUCCESS_FROZEN`, 237 observations, and 454 accepted movements;
- two of two fresh confirmation frames accepted;
- final image SHA-256
  `2c3516ecfcb00df73984a959610804c9894814897b53c8536109a042122cc1b4`;
- independent final score 0.605084 versus frozen floor 0.591231;
- independent evaluator overall PASS.

The separate bounded 2-D random-start smoke at
`RANDOM_START_LANDMARK_DEMO_20261104_115048` began at +12.000022 mm longitudinal and
-6.000000 mm lateral, physically returned to Home, captured 257 Stage 1 frames with zero
rejected movements, produced seven landmarks, completed the same 237-frame Stage 2 search,
and passed independent evaluation with score 0.591754 versus the same 0.591231 floor.

These results prove that the complete method executed on one fixed Unity simulator anatomy,
including physical return and fresh confirmation. They do not prove reliability over many
episodes, patient-held-out generalization, real-ultrasound transfer, clinical validity, or MBB
performance. The fixed Home, search corridor, seven-level ordering prior, target score, and
T7-right geometry are Unity-specific experimental assumptions.

## 6. Current SonoGym task audit

The running task is `Isaac-robot-US-guidance-v0` with:

- patient `s0010` and target mesh `vertebrae_L4`;
- model-based `conv` ultrasound;
- policy observation `float32 [N, 1, 200, 150]`, verified as
  `[1, 1, 200, 150]` in Phase 4;
- ultrasound source image size `[width=150, depth=200]` at 0.5 mm/pixel;
- physical image footprint 75 mm across-probe x 100 mm depth;
- privileged command state `[patient_x_index, patient_z_index, angle_rad]`;
- label resolution 0.0015 m/index = 1.5 mm/index;
- continuous actor space `[-2, 2]^3`, scaled by `[2, 2, 0.1]` and clamped to
  command increments `[4 indices, 4 indices, 0.2 rad]`;
- reset range `[100,70,1.5]` to `[180,200,3.5]`, followed by the native command clamp,
  whose angular maximum is 3.0 rad;
- target angle 3.14 rad;
- no native goal termination and disabled out-of-bounds termination.

The two translation actions are expressed in the current probe/image plane and transformed
into patient x-z before the surface planner and differential IK. The task exposes ultrasound
only to the policy, but stores `cur_cmd_pose`, `goal_cmd_pose`, reward, robot state, and target
geometry as privileged simulator/evaluation state.

Phase 5A job 50656802 established `METRIC_GO`: the privileged deterministic oracle reached
32/32 final 10 mm/10 degree successes, while zero and bounded random actions reached 0/32.
The Phase 4 PPO pilot checkpoint reached 0/32 final success and is diagnostic only. This proves
action-path feasibility and metric discrimination, not image-only localization.

### 6.1 Available anatomy and label semantics

The installed asset package contains 12 patient directories with an L4 mesh:
`s0004`, `s0006`, `s0010`, `s0012`, `s0014`, `s0015`, `s0024`, `s0028`, `s0029`,
`s0030`, `s0034`, and `s0038`. Only `s0010` is configured and runtime-validated in the current
baseline.

For `s0010`, individual S1 and L1--L5 STL files exist. Read-only NIfTI/STL inspection found:

| Structure | Approximate mesh centroid in label indices (x, y, z) |
| --- | --- |
| L1 | (125.93, 78.63, 201.60) |
| L2 | (126.32, 83.32, 180.17) |
| L3 | (126.96, 88.69, 159.53) |
| L4 | (127.36, 91.98, 138.95) |
| L5 | (128.10, 90.96, 120.85) |

Raw label 28 has centroid (127.50, 95.95, 139.43) and matches the L4 mesh. This is a useful
offline annotation source. It is not visible as label 28 to the actor: `label_conversion.yaml`
maps labels 25--50, including L4, to ultrasound acoustic class 13 (`bone`). Thus the renderer
does not encode vertebral level as a unique intensity class. Any L4 recognition must come from
morphology, spatial context, repeated scan evidence, and a declared initialization prior.

The current z command range ends at index 200 while the L1 centroid is near 201.6. A complete
S1/L5/L4/L3/L2/L1 sequence may therefore be edge-limited even though the 100 mm ultrasound
footprint can see beyond the probe center. Coverage and visibility must be measured, not assumed.

Four preserved Phase 4 raw frames show repeatable-looking bright interfaces and acoustic
shadows and vary with motion. They establish structured observation generation only. They are
not a bidirectional labeled lumbar sweep and cannot establish that L4 is identifiable.

### 6.2 Correct anatomical equivalent of Unity Stage 1

The closest defensible equivalent is a **Home-relative ordered lumbar posterior-bone interface
map**, initially on `s0010`:

- a fixed, declared, anatomy-unlabeled mechanical Home;
- a physical longitudinal patient-z sweep with forward/reverse acquisition;
- supervised detection of posterior bony-interface patches;
- projection into a Home-relative patient x-z map using executed relative motion;
- consolidation of repeated evidence into vertebral-level events;
- L4 inference only if a complete, independently annotated lumbar sequence and its anchor are
  observable, or if an independently validated L4-specific verifier distinguishes L4 from
  adjacent-level hard negatives;
- explicit `ABSTAIN` if coverage, ordering, identity, or directional support is ambiguous.

Two Stage 1 label hypotheses should be tested at G0, without choosing post hoc:

1. **Generic lumbar-event primary:** a patch intersects a true posterior osseous interface from
   the predeclared lumbar/sacral set. The map supplies identity through a complete verified
   order and declared Home/corridor prior.
2. **L4 verifier:** among primary event patches, distinguish raw-label-28/L4 support from
   adjacent L2/L3/L5/S1 interfaces. This is only admissible if disjoint-trajectory evidence
   shows that local/contextual morphology carries the information. If it fails, it must not be
   replaced by the true L4 coordinate at inference.

Unity's 0.5 mm grid, 10 mm event separation, 0.70 threshold, seven-event count, and T2--T8
order are not SonoGym settings. They become candidate implementation defaults only after G0/G1
measure the SonoGym image footprint, lumbar spacing, motion error, and support distribution.

## 7. Information boundaries

### 7.1 Actor-visible contract

The actor may receive only:

- current clean `float32 [1,200,150]` ultrasound pixels and their hash/sequence number;
- prior ultrasound pixels if the frozen model requires history;
- requested action and actually executed Home-relative translation/rotation;
- accepted/completed/clamped/rejected status without absolute patient pose;
- contact/workspace/image-valid booleans;
- its own evidence map and exploration history;
- calibrated 75 x 100 mm image footprint and declared experimental Home prior.

### 7.2 Low-level motion/safety channel

The adapter may read robot joints, end-effector pose, patient surface normal/map, command clamps,
IK convergence, collision/contact state, and workspace bounds to move reliably. It may expose
only executed relative deltas and status booleans to the actor. It must not expose absolute
patient pose, L4 direction, target distance, or goal state.

### 7.3 Privileged annotation/evaluation channel

This channel may contain patient ID, original unconverted label volume, raw L4 label 28,
individual vertebra meshes, `cur_cmd_pose`, `goal_cmd_pose`, geometric errors, reward,
segmentation slices, and evaluator scores. It is used only to create offline labels and to score
frozen actor output. Public actor artifacts and privileged artifacts must be separate files with
independent schema checks.

Tests must fail if actor inference receives keys such as `patient_id`, `label_image`,
`vertebra_mesh_id`, `cur_cmd_pose`, `goal_cmd_pose`, `target_position`, `target_distance`,
`reward`, or evaluator score.

## 8. Coordinate and image mapping

### 8.1 Proposed fixed-`s0010` convention

SonoGym command x/z values are label-volume indices; multiply by 1.5 mm to obtain patient-frame
distances. The `s0010` vertebral centroids show patient +z progressing L5 -> L4 -> L3 -> L2 ->
L1, so +z is the proposed cranial longitudinal direction for this pilot. The L4 right-trajectory
mesh lies at larger x than the L4 center, so +x is the proposed right-lateral direction. Both
signs must be verified with known simulator poses and rendered slices before freezing the
protocol and must not be assumed for another patient.

Define a Home-relative actor coordinate from accepted execution feedback:

`r_H(t) = [e_right, e_cranial]^T (p_patient(t) - p_patient(Home))`, in millimetres,

and orientation

`yaw_H(t) = wrap(angle(t) - angle(Home))`, in radians.

Absolute `p_patient(t)` may be used inside the motion adapter to compute the delta, but only
`r_H(t)-r_H(t-1)`, `yaw_H(t)-yaw_H(t-1)`, and status flags cross into the actor interface.

### 8.2 Pixel projection

`LabelImgSlicer` creates a 150-column centered across-probe grid and a 200-row depth grid at
0.5 mm spacing, then `USSlicer` returns H x W = 200 x 150. Before sign calibration, the nominal
local coordinates are:

- column offset: `(column - 74.5) * 0.5 mm` along the probe image-x axis;
- row depth: `row * 0.5 mm` from the image-plane origin along the acquisition-depth axis.

The current probe quaternion maps these local axes to patient x-z. The Stage 1 projector must
use this calibrated transform plus actor-owned relative odometry. It must not replace the
projection with the L4 mesh center.

### 8.3 Required coordinate tests

- patient-index <-> millimetre round trip;
- patient x/z <-> Home longitudinal/lateral round trip;
- pixel-to-patient and patient-to-pixel round trip within one pixel;
- known positive/negative translation direction;
- rotation sign and half-open angle wrapping;
- reconstruction from actually executed actions;
- clamped/rejected actions do not advance the actor map;
- physical return-to-best error;
- contact/workspace failure handling;
- fresh-frame hash uniqueness and monotonic acquisition sequence.

## 9. SonoGym Stage 2 adaptation

Stage 2 begins only after Stage 1 returns an L4-relative region with uncertainty and accepted
evidence. It uses ordinary SonoGym actions and a deterministic search; it does not use PPO.

### 9.1 Validity label

Before Stage 2 model fitting, freeze a simulator-defined engineering image target:

`image_valid = technical_image_valid AND Phase5A_10mm_10deg AND L4_interface_visible`.

`L4_interface_visible` is computed offline from the unconverted label-28 slice projected into a
predeclared image ROI. Its numerical coverage/support threshold must be selected from G0/G1
development data and frozen before Stage 2 training. It is not a clinical view definition.

The classifier predicts this image validity from clean ultrasound pixels only.
`physical_transition_valid` is not a learned image label: it is a separate hard execution gate
and a frame-eligibility flag. Position/orientation and label-28 projection are offline
label/evaluator fields; physical metadata are execution/provenance fields. None are inference
inputs to the perception model.

### 9.2 Quality label

The quality regressor ranks only technically valid or near-valid candidates. A proposed frozen
target is the conjunction-preserving product of:

- normalized L4-interface visibility/support; and
- a monotonic geometric proximity term computed from the unchanged Phase 5A position and
  wrapped orientation errors.

The exact normalization constants and ROI threshold must be preregistered after G0 but before
fitting. The 3.0-rad command ceiling versus 3.14-rad target remains unchanged and is reported;
the quality definition must not silently substitute a reachable target angle.

### 9.3 Deterministic controller

The SonoGym controller will:

1. move to the Stage 1 estimate using relative executed-motion feedback;
2. define a bounded search box in Home-relative patient x/z and probe angle;
3. execute a coarse serpentine search;
4. retain candidates only when the classifier accepts them;
5. rank accepted candidates with the quality regressor;
6. perform fine then micro refinement within the frozen search box;
7. inverse-replay/correct relative motions to return physically;
8. acquire at least two new frames with new hashes;
9. require frozen classifier confirmation and an allowed quality drop;
10. freeze actor output, then run Phase 5A geometric and privileged image-label evaluation.

The output is exactly `SUCCESS`, `FAILURE`, or `ABSTAIN`. A high quality prediction never
overrides an invalid frame, rejected/clamped/incomplete move, missing contact, or failed return.

The Unity numerical tube and target score are not transferred. SonoGym bounds must be derived
from G0/G1 feasibility, the native action lattice, and the 300-step locked evaluation horizon.

## 10. Source-to-destination mapping

| Unity source | Actual role | Proposed SonoGym component | Adaptation and validation |
| --- | --- | --- | --- |
| `G2R2ContinuousSweep.cs` | Home return, bidirectional physical sweep, fresh capture, executed motion | `workflows/two_stage_navigation/collect_observability.py` plus motion adapter | Redesign around `env.step`; verify completion, clamps, freshness, surface/IK feedback |
| `RandomStartLandmarkDemo.cs` | End-to-end Stage 1 orchestration and abstention | `workflows/two_stage_navigation/run_stage1.py` | Adapt state machine; no target mesh access |
| `g2_r4_multi_hypothesis_tracker.py` | Bright-interface candidates and diverse contour hypotheses | `.../two_stage_navigation/stage1_interfaces.py` | Port algorithms, not constants; validate on 200 x 150 conv images |
| `autonomous_t2t8_patch_likelihood.py` | 48 x 80 patches, 16 x 16 intensity/gradient features, ExtraTrees likelihood | `.../stage1_features.py`, `train_stage1.py` | Reuse method; retrain SonoGym labels; freeze patch geometry only after G0 |
| Primary-plus-verifier bundle | Main event classifier plus independent borderline verifier | `.../stage1_models.py` | Preserve two-model logic if G0 supports L4/adjacent discrimination; never load Unity weights as valid |
| Home-relative 0.5 mm evidence map | Multi-frame spatial aggregation | `.../stage1_map.py` | Adapt to patient x-z and calibrated footprint; explicit uncertainty and abstention |
| T2--T8 seven-event prior | Absolute T7 identity | Lumbar sequence/anchor protocol | Cannot reuse; requires verified S1/L1--L5 coverage or an L4 verifier |
| `phase2_best_view_recognition.py` and adaptation | Global image features, validity classifier, quality regressor | `.../stage2_features.py`, `train_stage2.py` | Reuse lightweight architecture; retrain on SonoGym labels |
| `run_phase2_best_view_server.py` | Pixel-only scorer contract | `.../stage2_models.py` | Direct Python inference is sufficient; preserve forbidden-key checks |
| `Phase2BestView.cs` | Coarse/boundary/micro search, return, fresh confirm | `.../controller.py` and `evaluate_closed_loop.py` | Redesign action lattice and bounds; preserve validity-before-quality logic |
| Direct fresh capture token/hash | Reject stale/reused images | dataset and controller contracts | Require post-step hash, sequence, simulator step, and action association |
| Unity contact/workspace/rollback | Physical feasibility | motion adapter using IK/joints/surface/no-collide | Expose only relative deltas/status to actor; fail closed |
| Unity privileged labels/`ScoreCandidate` | Offline training labels and independent final evaluation | `build_annotations.py` and separate evaluator manifest | Redefine for L4; never expose to actor |
| Unity independent evaluator | Score frozen actor result | Phase 5A metrics plus frozen L4-image label evaluator | Reuse unchanged 10/10 metric; keep image and geometry conclusions separate |

## 11. Dataset protocol

### 11.1 Required records

Every frame record must include clean float32 ultrasound, SHA-256, patient/anatomy ID in the
privileged manifest, session/trajectory ID, seed, frame sequence, acquisition configuration,
requested action, actual executed relative delta, completion/clamp/rejection state, contact/
workspace/image validity, Home-relative reconstruction, and history ID.

The separate privileged annotation contains original label-28 projection, vertebral identity,
`cur_cmd_pose`, `goal_cmd_pose`, Phase 5A errors, Stage 1 patch/event labels, Stage 2 validity,
and quality. The public actor manifest contains no patient target or geometry fields.

### 11.2 Minimum G0/G1 pilot

The smallest useful first capture is a bounded **signal-detection pilot**, not a final dataset:

- one validated anatomy: `s0010`;
- 16 independently reset bidirectional longitudinal sessions, capped rather than assumed
  sufficient;
- preassigned 8 train-development, 4 calibration-development, and 4 untouched audit sessions;
- multiple safe patient-x lateral offsets and probe angles;
- approximately 3,000--6,000 fresh frames total;
- positive L4/adjacent-interface regions, near-target views, difficult bone negatives, and
  non-bone negatives;
- no Phase 5A locked evaluation start used for training or threshold selection.

All adjacent frames and patches from one session stay in one split. Report patient count,
trajectory count, frame count, patch count, and positive support separately. Continue to model
fitting only if positives and adjacent-level negatives occur in multiple independent sessions
and both scan directions. If the capped pilot is insufficient, report G0 `INCONCLUSIVE` and
request a new bounded protocol; do not reinterpret scan index as L4.

### 11.3 Later Stage 2 data

Only after Stage 1 passes G3, collect separate local search trajectories around actor-inferred
regions. Include valid views, position-compatible/orientation-wrong frames, orientation-
compatible/position-wrong frames, near-boundary positives, hard L3/L5 negatives, clamped moves,
and failed/abstained returns. Split by complete trajectory and keep the locked 32 Phase 5A
starts untouched until final evaluation.

### 11.4 Patient-held-out extension

The asset package suggests future multi-patient feasibility, but every additional patient must
first pass runtime and G0 checks. Patient IDs, not frames, define train/development/test splits.
No number of frames from `s0010` establishes G5, and no fixed number of available asset patients
guarantees generalization.

## 12. Evaluation gates and stop conditions

### G0 -- anatomical observability

Require labeled bidirectional evidence that real lumbar interfaces are visible, repeat across
independent sessions, and can be distinguished from renderer artifacts and adjacent levels.
Require a feasible complete sequence/anchor or a disjoint-trajectory L4 verifier. Otherwise
`BLOCKED` or `INCONCLUSIVE`; do not train around the failure.

### G1 -- data and coordinate integrity

Require zero hash/provenance/split mismatches, zero forbidden actor fields, pixel round-trip
error at most one pixel, Home-relative transform round-trip error within numerical tolerance,
and executed-motion reconstruction within one 1.5 mm command index and one degree over accepted
moves. Rejected/clamped actions must not create fictitious map motion. Any stale image or label
association fails G1.

### G2 -- supervised recognition

On disjoint development trajectories, report patch/event and frame-level precision, recall,
F1, PR-AUC, calibration, false positives by adjacent level, and failure images. A proposed
promotion target is at least 0.90 event-level precision and recall for Stage 1 and, later, at
least 0.95 validity precision with at least 0.80 recall for Stage 2 at a frozen threshold.
Quality ranking must have positive held-out rank correlation and beat a position-agnostic
constant baseline. These targets require researcher approval before fitting.

### G3 -- relative landmark localization

Require the Stage 1 map to identify L4 without target coordinates in at least 80% of eligible
fixed-anatomy development sessions, with median L4 localization error no greater than 10 mm,
zero false L4 acceptance on predeclared incomplete/wrong-anchor negatives, and explicit
abstention otherwise. Report coverage failures separately from perception failures.

### G4 -- full closed loop

On the unchanged 32-start Phase 5A protocol, execute Scan -> Map -> Infer L4 -> Navigate ->
Coarse/Fine/Micro -> Return -> Fresh Confirm. A proposed transfer gate is at least 26/32 final
geometric 10/10 successes, at least a 25-percentage-point advantage over zero and bounded
random controls, no observed wrong-target STOP, complete fresh-image confirmation, and zero
unhandled contact/workspace/nonfinite failures. Report abstention separately; do not count
timeout as success. The PPO pilot remains diagnostic and the privileged oracle remains an
upper bound.

### G5 -- anatomy-held-out generalization

Cannot pass with `s0010`. Freeze a patient-level protocol before using other assets, train on
one set of patient IDs, calibrate on disjoint patients, and open a held-out patient set once.
Report per-patient results and uncertainty. Do not claim clinical or MBB generalization.

## 13. Minimal implementation after approval

Keep the upstream task, PPO rewards, observations, actions, dynamics, checkpoints, and frozen
Phase 5A files unchanged. Add an isolated experimental package:

| Proposed path | Purpose |
| --- | --- |
| `experiments/unity_two_stage_transfer/protocol.yaml` | Frozen G0/G1 capture, splits, labels, resources, and stop conditions |
| `source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/contracts.py` | Public/privileged schemas and forbidden-field checks |
| `.../coordinates.py` | Patient/Home/probe/pixel transforms and executed odometry |
| `.../stage1_interfaces.py` | Bright-interface extraction and candidate tracks |
| `.../stage1_features.py` | SonoGym patch normalization and intensity/gradient features |
| `.../stage1_map.py` | Spatial aggregation, peak consolidation, uncertainty, abstention |
| `.../stage1_models.py` | Primary/verifier ExtraTrees interfaces and checkpoint metadata |
| `.../stage2_features.py` and `stage2_models.py` | Validity classifier and quality regressor |
| `.../controller.py` | Deterministic scan/search/return/fresh-confirm state machine |
| `workflows/two_stage_navigation/collect_observability.py` | Bounded G0/G1 capture through ordinary environment stepping |
| `workflows/two_stage_navigation/build_annotations.py` | Separate privileged labels from original label volume/meshes |
| `workflows/two_stage_navigation/train_stage1.py` | CPU ExtraTrees training after G0/G1 approval |
| `workflows/two_stage_navigation/train_stage2.py` | Deferred until Stage 1 passes G3 |
| `workflows/two_stage_navigation/evaluate_closed_loop.py` | Deferred Phase 5A-compatible G4 evaluation |
| `tests/test_two_stage_*.py` | Leakage, transforms, freshness, splits, map, return, and abstention tests |

The first approved implementation increment should add only the protocol, contracts,
coordinates, annotation builder, Stage 1 feature/map code, and CPU tests. It should not alter
`robotic_US_guidance.py` unless a reviewed evaluation-only hook is proven necessary.

After CPU tests and protocol review, request separate authorization for one bounded G0/G1
capture job. Train only the Stage 1 primary/verifier ExtraTrees baseline if G0/G1 pass. Stage 2
implementation remains blocked until Stage 1 produces defensible anatomical information.

## 14. Compute and storage budget

Measured Phase 5A throughput was 38,400 transitions in 638 seconds, approximately 60.2
transitions/s aggregated across four environments and repeated process startup. This is a
reference, not a measured capture-with-hashing rate.

| Increment | Resources | Bound | Estimated output/time |
| --- | --- | --- | --- |
| CPU contracts/features/tests | 1--2 CPUs, <=8 GB, no GPU | <=30 min | Source/tests only |
| G0/G1 pilot capture | `anest-dept`, 1 L4, 4 CPUs, 30 GB | <=30 min, one job | 3k--6k frames; estimated 10--20 min including startup/I/O |
| Stage 1 ExtraTrees | 4 CPUs, <=16 GB, no GPU preferred | <=30 min | Small checkpoint/report; estimated 5--20 min |
| Deferred Stage 2 capture | 1 L4, 4 CPUs, 30 GB | separate approval, <=45 min | Local search frames only |
| Deferred G4 evaluation | 1 L4, 4 CPUs, 30 GB | separate approval, <=2 h | Same 32-start protocol |

One float32 1 x 200 x 150 frame is 120,000 bytes. The G0/G1 raw image payload is therefore
approximately 0.36--0.72 GB before metadata/compression. Reserve at most 1.5 GB for raw frames,
privileged annotations, previews, hashes, and logs. Store large data under
`/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/`,
not in Git. These runtimes are estimates until measured.

The proposed jobs use one L4 and at most four CPUs, stay within the total 4-CPU/2-GPU agreement,
and do not use the `thai` allocation. No job is authorized by this document.

## 15. Scientific limitations

- The verified Unity result is a fixed-anatomy simulator engineering demonstration.
- The SonoGym renderer, anatomy, image size, speckle process, target semantics, and action frame
  differ from Unity; checkpoint weights are not portable.
- SonoGym's runtime acoustic label erases vertebral-level identity, so L4 may not be locally
  identifiable even though bone is visible.
- A fixed Home and corridor can encode a strong per-anatomy prior and enable memorization.
- Thousands of frames or patches from `s0010` are still one patient anatomy.
- Stage 2's proposed label is a simulator engineering definition, not a clinician-validated
  L4 or MBB view.
- The current boundary termination is disabled and is not equivalent to Unity's fail-closed
  contact/workspace controller.
- The target angle remains partly outside the command range; the 5 mm/5 degree sensitivity is
  currently unattainable even for the oracle.
- Offline classifier success does not establish physical navigation, fresh return, or
  closed-loop success.
- No result here supports real-ultrasound transfer, MBB accuracy, clinical safety, or patient
  generalization.

## 16. Approval decision required

Approve or reject the first implementation increment:

1. add the isolated contracts, coordinate transforms, Stage 1 patch/interface/map modules,
   annotation builder, tests, and frozen G0/G1 protocol;
2. review those CPU-only artifacts;
3. only then authorize one bounded `s0010` G0/G1 capture on one L4;
4. fit a 240-tree Stage 1 ExtraTrees primary/verifier baseline only if observability,
   provenance, and split gates pass.

Do not implement Stage 2, collect its dataset, or run a closed-loop evaluation until G3 is
passed and the researcher gives a new explicit approval.

## 17. Implementation status (2026-11-11)

The approved first CPU-only increment is implemented without changing the SonoGym task, PPO
baseline, or Phase 5A protocol. It adds public/privileged contracts, frame freshness checks,
patient/Home/pixel transforms, fail-closed executed-motion odometry, configurable Stage 1
interface and patch features, bidirectional spatial evidence aggregation with abstention, a
separate annotation-manifest validator, and the pre-GPU G0/G1 protocol.

Frozen protocol SHA-256 after adding the verified `s0010` raw-label mapping and explicit
tracking/image-validity semantics:
`dc240b5bb333312f6bf85a9ff509f1beafd81e18f1ed458e75b498b5a017017a`.

Verification from the project worktree:

- 29 new CPU unit tests passed;
- 11 unchanged Phase 5A metric tests passed in the same run;
- Python syntax compilation passed;
- the annotation-builder CLI import and help path passed without initializing Isaac Sim;
- Ruff was not installed and was therefore not run;
- the bounded one-session-per-process Isaac collector, capture summarizer, and Slurm launch
  path passed CPU syntax, protocol-seal, and launch dry-run checks;
- GPU capture, model fitting, and Stage 2 remain `NOT_RUN` at this revision.

The existing `spinal_surgery` package imports Isaac modules at package import time. CPU tests
and the annotation workflow therefore import the isolated `two_stage_navigation` evaluation
package directly instead of changing the upstream package initializer. The installed runtime
also lacks scikit-learn; no dependency was installed because ExtraTrees fitting is outside this
increment and remains gated on G0/G1 evidence.
