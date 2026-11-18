# s0010 RIGHT L4/L5 TP surface-core dataset readiness

| Result | Value |
|---|---|
| Final readiness | `DATASET_INCOMPLETE` |
| Verified real acquisition | job `51811811`; v6 smoke 8/8 plus development 300/300 |
| Repair executions | v4 timeout and v5 verifier mismatch preserved; v6 completed `0:0` |
| Synchronized smoke | **PASS**, 8/8 reloaded frames pass all five gates |
| Development dataset | **COLLECTED**, 300/300 distinct frames across six preassigned sessions |
| Review candidates | 414 geometric-intersection patches; 168 split-stratified review assets |
| Training eligibility | 0 samples; anatomy and image-semantic review remain pending |
| Training implementation | train-only fit, calibration-only threshold selection, development-only evaluation, background class, save/reload |
| CPU finalization | job `51813176`; 26/26 tests; 4 CPUs; 1:16; exit `0:0` |
| Final package | SHA-256 `259746e83b127c573f47eda31dbecd3d6e8edcb4e62f4953eda442420191402d` |

The first bounded capture exposed a float64-to-float32 hash mismatch, V4 reached the fixed episode
horizon, and V5 exposed a reset-boundary verifier error.  Those failures remain preserved.  V6
passed strict smoke and automatically collected all six 50-frame development sessions.  Post-run
auditing then found that 414 candidate patches represented fewer independent L4 source images than
the preregistered planning gate; the corrected summary therefore remains `DATASET_INCOMPLETE`
instead of treating correlated patches as independent evidence.

## 1. Acquisition synchronization and physical viewpoints

### Criterion

The pose, native sampling map, raw label and clean ultrasound must share one acquisition identity.
Every intended smoke viewpoint must have advancing acquisition/producer sequences, executed motion,
a technically valid image, and successful replay after the files are reloaded.

### Method

The opt-in snapshot is now created inside the native slicing call from immutable copies of the
actual world-to-patient and world-to-probe transforms.  The clean image is attached to that same
snapshot after rendering.  The collector joins all saved records by acquisition ID and rejects
waypoints swallowed by tolerance before capture and stale/repeated observations at runtime.

### Setup

```bash
sbatch --account=anest-dept --qos=anest-dept --partition=hpg-turin \
  --constraint=l4 --gres=gpu:l4:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=30G --time=00:30:00 \
  --job-name=sg-tp-core-v1 \
  --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env \
  hpg/jobs/s0010_tp_surface_core_dataset_v1.sbatch
```

Protocol SHA-256: `cb912fca43294650e5c1a93ad5a2c9a34f2bbcddfd3332e98b561ad889f21a99`.
The execution checkout was `0ea74a60a7666a4cf0d14e86693717315d467e2c`; the acquisition
implementation named by the sealed protocol was `7039c7ee559ddd40631b8ab25ec81659fe143d01`.

### Results and verdict

The smoke produced 8 frames, 8 exact image hashes and 8 acquisition IDs.  Producer sequences were
`122, 175, 215, 266, 359, 407, 453, 520`; action counts were
`121, 53, 40, 51, 93, 48, 46, 67`.  Consecutive realized translations were 17.10--59.32 mm and
rotations were 6.73--8.20 degrees.  Reloaded native-map and full-pose replay both achieved raw-label
agreement 1.0 and bone IoU 1.0 for all frames, with no clipping and maximum coordinate disagreement
`2.8149133442e-05` index.  `NATIVE_MAP_REPLAY`, `SYNCHRONIZED_FULL_POSE_REPLAY`,
`ACQUISITION_FRESHNESS`, and `VIEWPOINT_DIVERSITY` all pass.

**Verdict: synchronized smoke PASS; repeated-waypoint failure repaired.**

### Reading it

- The real-frame result verifies synchronized simulator logging, not semantic TP visibility.
- Eight saved coordinate hashes differ from their recorded hashes only because job `51793371`
  hashed float64 arrays and then saved float32 arrays; source now hashes the serialized float32.
- The scientific replay uses the reloaded float32 map and remains exact at the declared gates.

### Files

- Runtime: `s0010_tp_surface_core_dataset_v1/51793371/sessions/s0010tpsc_smoke/`
- Replay: `verification_report.json`
- Protocol: `experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_protocol_v1.yaml`

## 2. Dataset and candidate labels

### Criterion

The pilot requires 300 accepted frames across four train, one calibration and one development-
evaluation session.  Geometry intersection, acoustic visibility, semantic identity and anatomical
approval remain separate gates; unknown rows cannot become negatives.

### Method

The dataset builder projects the exact versioned L4/L5 RIGHT proposed TP surface-core vertices
through the saved native-consistent full pose.  It uses a 1.5 mm surface slab and retains the
uncertain-base projection separately.  Proposed patches are 48x80 pixels and use the existing
16x16 normalized intensity/gradient feature path.  SAP, whole-TP, MBB targets and injection claims
are excluded.

### Results and verdict

All six dataset sessions contain 50 distinct synchronized observations, for 300 development images
plus eight smoke images.  The actor-visible proposal path produced 2,940 candidate patches: 414
geometric positive-review targets, 2,038 hard-negative review targets and 488 ambiguous rows.  The
independent source-image counts are:

| class | train | calibration | development evaluation |
|---|---:|---:|---:|
| background / other structure | 199 | 50 | 49 |
| L4 RIGHT TP surface core | 33 | 7 | 8 |
| L5 RIGHT TP surface core | 53 | 13 | 16 |

L4 is below the declared 40/10/10 planning target in every split.  All rows remain `NOT_REVIEWED`
for acoustic visibility and semantic identity, `PENDING` for positive anatomy, and
`training_eligible=false`.

**Verdict: `DATASET_INCOMPLETE`.**

### Reading it

- The final package contains 168 unique-image review assets: background 40/10/10, all available
  L4 candidates 33/7/8, and L5 40/10/10.
- A geometric surface intersection does not prove a visible or distinguishable TP interface.
- Review cannot manufacture the missing 7/3/2 independent L4 candidate images.

### Files

- `dataset/candidate_sample_manifest.csv`
- `dataset/dataset_summary.json`
- `dataset/review/`

## 3. End-to-end no-fit training path

### Criterion

No-fit must load real image arrays, verify hashes/shapes, resolve real crops, extract finite features,
check splits/duplicates/eligibility, instantiate the intended ExtraTrees configuration, and never
call `fit()`.

### Method

```bash
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery"
"$SONOGYM_ENV_PREFIX/bin/python" workflows/two_stage_navigation/train_tp_surface_core.py \
  --dataset-dir "$DATASET_DIR" --output-dir "$PREFLIGHT_DIR" --no-fit
```

### Results and verdict

The real path resolved all 2,940 candidate crops and produced finite 774-value features.  It
instantiated a deterministic 240-tree `ExtraTreesClassifier`, found no cross-split exact duplicates,
and recorded `fit_called=false`.  It returned code 3 with `ENGINEERING_PREFLIGHT_ONLY` because L4
candidate coverage is incomplete, geometry approval is pending, and all semantic labels remain
excluded.

**Verdict: implementation path works; fitting is correctly blocked.**

### Reading it

- The future bounded training command is the same command without `--no-fit`.
- It has no bypass flag and will refuse to fit until the manifest contains approved, reviewed labels
  with at least 40/10/10 positive source images per class and reviewed hard negatives in each split.
- No model was trained in this task.

### Files

- `preflight/preflight_result.json`
- `workflows/two_stage_navigation/train_tp_surface_core.py`

## 4. Training and capture repair — whether the reviewed defects are closed in code

### Criterion

Only training rows assigned to `train` may enter `fit()`.  Calibration chooses operating thresholds,
development evaluation remains read-only, every active split contains reviewed background examples,
and a fitted checkpoint must reload without changing development predictions.  Capture must reject
serialized coordinate-hash mismatches and accept only a protocol explicitly registered in both the
collector and verifier.

### Method

The dataset builder now proposes patches from actor-visible bright interfaces and uses privileged
geometry only to prepare review targets.  The trainer partitions eligible rows before constructing
the estimator, fits only the training partition, selects class thresholds only from calibration,
evaluates only on development, and saves then reloads the model bundle.  Synthetic regression data
cover all three classes and all three splits; no real research model is fitted.

### Setup

| what | value | spec |
|---|---|---|
| capture job | `51811811` | one L4, 4 CPUs, 30 GB; 6:58; exit `0:0` |
| finalization job | `51813176` | 4 CPUs, 16 GB; 1:16; exit `0:0` |
| source | `6ed6259` capture; `0e5b3c7` finalization | clean worktree at capture |

```bash
python tests/test_acquisition_integrity.py
python tests/test_ultrasound_plane_geometry.py
python tests/test_two_stage_contracts.py
python tests/test_tp_surface_core_training.py
python tests/test_tp_surface_core_capture_contract.py
```

### Results and verdict

The pinned run passed `14 + 11 + 23 + 4 + 20 = 72` tests.  The synthetic end-to-end test resolved
180 feature rows, fitted only training IDs, calibrated only on calibration IDs, evaluated only on
development IDs, and verified checkpoint reload identity.  Job `51798167` independently passed all
five strict smoke gates before the first bulk session exposed float32 rotation validation; job
`51798995` stopped before its first frame because the new protocol ID was absent from the explicit
allow-list.  Both source defects have focused regression coverage.  V4 subsequently passed strict
smoke, then the first dataset session reached the fixed episode horizon at action 599 after 11
completed waypoints.  Protocol v5 splits the unchanged waypoint sequence into at-most-three-frame
reset segments; its worst-case declared segment budget is 550 actions against the 600-step horizon.
V5 acquired all 8 smoke frames across three resets, but its verifier treated segment-start frames 3
and 6 as stale.  CPU job `51810728` applied the corrected reset rule to the unchanged stored files
and returned all five gates true.

**Verdict: v6 acquisition and replay PASS; 300-frame dataset collected; L4 coverage and review gates keep the dataset incomplete.**

### Reading it

- The code no longer permits calibration or development rows to enter model fitting.
- Background proposals have a complete review and feature-extraction path; unreviewed rows remain
  excluded instead of becoming negatives.
- Passing synthetic fitting tests does not create a usable real dataset or approve TP anatomy.
- Jobs `51797926`, `51798167` and `51798995` remain technical failures and are not rewritten.

### Files

- `workflows/two_stage_navigation/train_tp_surface_core.py`
- `source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/tp_surface_core_dataset.py`
- `experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_protocol_v6.yaml`
- `hpg/jobs/s0010_tp_surface_core_dataset_v6.sbatch`

## 5. Completed coverage supplement and exact remaining gate

### Criterion

Training can start only after review imports produce at least 40/10/10 unique positive images per
active class plus reviewed background examples in each split.  The combined V6/V7/V8/V9 dataset
must first close the prior 7/3/2 L4 image deficit without using failed partial sessions.

### Method

V7 supplied a verified 24-frame training supplement but stopped before calibration.  V8 preserved
that session and added verified 12-frame training and 24-frame calibration sessions; its
development session failed because a 2.985-rad waypoint was outside the demonstrated realized
angular envelope.  V9 replaced only that missing split with a sealed 24-frame bank bounded at
2.94 rad, then rebuilt the dataset from verified sessions only.

### Setup

GPU completion job `51818619` used one L4, four CPUs and 30 GB and completed `0:0` in 2:08.  CPU
jobs `51818555` and `51819343` passed 87 and 89 focused tests respectively.  The review-ready ZIP
contains exact geometry assets, 180 stratified overlays, a decision template, synchronized smoke
arrays and fail-closed preflight.

### Results and verdict

The combined dataset contains 384 development observations across ten sessions, plus the separate
eight-frame opened smoke set.  Unique source-image coverage is background `234/74/73`, L4 RIGHT TP
surface core `49/14/14`, and L5 RIGHT TP surface core `55/15/16` for
train/calibration/development.  All exceed `40/10/10`; no session is missing and no exact image hash
crosses splits.  All 3,732 actor-visible patches resolve to finite 774-value features.  Training
eligibility remains zero solely because exact geometry and image-semantic decisions have not been
approved; SAP remains excluded.

**Verdict: `ENGINEERING_READY_LABEL_REVIEW_PENDING`; acquisition and coverage are complete, while
real model fitting remains fail-closed on exact review decisions.**

### Reading it

- The reviewer must complete all 180 checksum-bound rows: exact L4/L5 surface-core geometry,
  acoustic visibility, semantic identity, reviewer identity and qualification.
- `apply_tp_surface_core_review.py` rejects missing, duplicate or identity-mismatched decisions and
  promotes only exact approved positives and visible reviewed negatives.
- After import, `train_tp_surface_core.py --no-fit` must report `TRAINING_READY_FOR_APPROVAL`; only
  then may the separately authorized fit command train on `train`, calibrate on `calibration` and
  evaluate once on `development_evaluation`.

### Files

- Result: `experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_v9_result.json`
- Review ZIP: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_tp_surface_core_dataset_v9/51818619/s0010_tp_surface_core_review_51818619_final.zip`
- Review ZIP SHA-256: `aec872965c310c8659a483a393f92efefb98ba7eca9e5af26f059b998876bb8b`

