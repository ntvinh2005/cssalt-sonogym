# s0010 RIGHT L4/L5 TP surface-core dataset readiness

| Result | Value |
|---|---|
| Final readiness | `DATASET_INCOMPLETE` |
| Verified real acquisition | job `51810449`; synchronized segmented v5 smoke 8/8; dataset 0/300 |
| Repair executions | v4 timeout preserved; v5 job `51810449` captured segmented smoke then verifier mismatch |
| Synchronized smoke | **PASS**, 8/8 reloaded frames pass all five gates |
| Development dataset | **NOT COLLECTED**, 0/300 frames across the six preassigned sessions |
| Review candidates | 3 verified geometric intersections: 1 L4 RIGHT TP core and 2 L5 RIGHT TP core |
| Training eligibility | 0 samples; anatomy and image-semantic review remain pending |
| Training implementation | train-only fit, calibration-only threshold selection, development-only evaluation, background class, save/reload |
| Current CPU regression | job `51810903`; 72/72 tests; 4 CPUs; 9 s; exit `0:0` |
| Next sealed protocol | v6, SHA-256 `55fc3d7a426e8475421e6c77b3c98acaa22289fa7667633447ea4e4a7cb9142d`; not run |

The first bounded capture exposed a float64-to-float32 hash mismatch, and later repairs preserved
that result while fixing serialized hashing.  V4 then passed smoke but reached the 600-step episode
horizon during its first development session.  Segmented V5 acquired 8/8 synchronized smoke frames
across three reset trajectories.  Its original verifier incorrectly rejected the first frames after
reset; CPU re-verification of the unchanged saved arrays passed all five gates after that rule was
corrected.  The V5 wrapper had already stopped, so the required six-session dataset remains absent.

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

Only the smoke session exists.  Its 16 candidate/frame rows contain 3 verified geometric
intersections: smoke frame 1 with L4 core (336 projected core vertices, 165 uncertain), and frames
6 and 7 with L5 core (312/0 and 469/180).  All three remain `NOT_REVIEWED` for acoustic visibility
and semantic identity, `PENDING` for anatomy, and `training_eligible=false`.  The six planned
dataset sessions and all train/calibration/development-evaluation counts are zero.

**Verdict: `DATASET_INCOMPLETE`.**

### Reading it

- The three overlays are actionable review examples, not positive training labels.
- A geometric surface intersection does not prove a visible or distinguishable TP interface.
- The exact missing data are the six named 50-frame sessions in the frozen protocol; no split can
  be activated from the smoke alone.

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

The real path resolved all 3 available candidate crops and produced finite 774-value features.  It
instantiated a deterministic 240-tree `ExtraTreesClassifier`, found no cross-split exact duplicates,
and recorded `fit_called=false`.  It returned code 3 with `ENGINEERING_PREFLIGHT_ONLY` because the
dataset is incomplete, geometry approval is pending, reviewed hard negatives are absent and every
required split/class count is zero.

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
| CPU job | `51810903` | 4 CPUs, 8 GB, pinned environment |
| source | `ce5e06c` | corrected segmented protocol-v6 revision |
| capture retry | protocol v6 | acquisition contract unchanged from v5; corrected reset-boundary verifier |

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

**Verdict: segmented v5 acquisition PASS after corrected replay; dataset NOT RUN; v6 CPU preflight PASS.**

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

## 5. Exact remaining gate

### Criterion

Training can start only after protocol v6 captures all 300 frames, every session passes strict
reloaded correspondence/hash/freshness checks, and review imports produce at least 40/10/10 unique
positive images per active class plus reviewed background examples in each split.

### Method

No additional method is run in this section.

### Setup

The next execution requires separate approval because the approved v5 GPU allocation has been
consumed.  The ready job is `hpg/jobs/s0010_tp_surface_core_dataset_v6.sbatch`.

### Results and verdict

Current counts remain 0/300 development frames and zero training-eligible samples.  L4 and L5 RIGHT
distal TP surface-core geometry and image-semantic decisions remain pending for the exact candidate
checksums; SAP remains excluded.

**Verdict: `DATASET_INCOMPLETE`; real model fitting remains blocked.**

### Reading it

- The next bounded action is one v6 smoke-gated segmented capture, not model training.
- After capture, the reviewer must classify exact overlays as supported, ambiguous, occluded or
  rejected and approve or reject the exact L4/L5 surface-core geometry.
- If imported review counts pass no-fit, the supplied fit command can run without another capture.

### Files

- Protocol: `experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_protocol_v6.yaml`
- Protocol seal: `experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_protocol_v6.sha256`

