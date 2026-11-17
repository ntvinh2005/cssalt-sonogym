# s0010 exact-pose replay capture

## Result

| gate | result |
|---|---|
| Focused CPU tests | PASS (33 / 33) |
| V1 capture | TECHNICAL_PRECAPTURE_FAILURE, job `51782795` |
| Native stored-coordinate replay | PASS (6 / 6 exact) |
| Full-pose historical dual gate | PASS (6 / 6) |
| Strict coordinate replay | FAIL (0 / 6) |
| Capture diversity | FAIL (2 / 6 distinct hashes; 4 required) |
| Overall | **PROJECTION_REPAIR_BLOCKED** |
| Semantic detector training | **NOT_RUN** |

The bounded capture did establish that persisting the renderer's native sampled coordinates gives
an exact raw-label reconstruction. It did not produce enough distinct observations and did not
meet the separately frozen coordinate-level replay tolerance. The old 3,826-frame result remains
unchanged, and no SAP/TP candidate was promoted or used for model fitting.

## 1. Pre-GPU integrity

### Criterion

The capture must use the pinned SonoGym environment, the `anest-dept` account, one L4, four CPU
cores, 30 GB requested memory, no concurrent project job, and a checksum-sealed protocol. Native
geometry and historical Stage 2 contract tests must pass before simulator startup.

### Method

The collector was extended only under the exact-pose protocol. It stores native
`human_img_coords`, full `human_to_ee` position/quaternion/rotation, runtime image constants,
producer sequence, image hash, clean ultrasound and the corresponding raw-label slice. Actor
records remain separate from privileged replay data.

### Setup

```bash
python tests/test_ultrasound_plane_geometry.py
PYTHONPATH=source/spinal_surgery python tests/test_two_stage_contracts.py
```

### Results

Native geometry tests passed 10 / 10 and Stage 2 contract tests passed 23 / 23. The frozen v2
protocol SHA-256 is
`57b5b213a5141499a70aedadb85bca5644fd6e95e82d59f8a5ed3d9aed73bc70`.

**Verdict: PASS.**

### Reading it

- The helper and information-boundary changes were tested before simulator execution.
- Synthetic/CPU parity tests do not establish successful replay of a real frame.

### Files

- `source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/ultrasound_plane_geometry.py`
- `workflows/two_stage_navigation/collect_stage2_dataset.py`
- `workflows/two_stage_navigation/verify_exact_pose_capture.py`
- `tests/test_ultrasound_plane_geometry.py`

## 2. V1 technical failure and bounded correction

### Criterion

The first failure may receive at most one technically justified retry without weakening the
scientific gates or changing the six requested poses.

### Method

V1 used the legacy independent-axis `move_to_pose` helper. It stopped before frame acquisition when
one command mapping error was 0.161271 index against the unchanged 0.15 limit. V2 replaced only
that helper with the coupled proportional controller already used by prior Stage 2 captures.

### Setup

```bash
sbatch --account=anest-dept --qos=anest-dept --partition=hpg-turin \
  --constraint=l4 --gres=gpu:l4:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=30G --time=00:20:00 \
  --job-name=sg-exact-pose2 \
  --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env \
  hpg/jobs/s0010_exact_pose_replay_v2.sbatch
```

### Results

Job `51782795` preserved the V1 technical failure with zero captured frames. The single retry,
job `51783444`, captured six frames and exited `3:0` after 43 seconds because the frozen gates did
not all pass.

**Verdict: V1 TECHNICAL_PRECAPTURE_FAILURE; V2 completed capture and returned NO_GO.**

### Reading it

- The V1 event says nothing about projection accuracy because it acquired no frame.
- V2 is the final bounded retry; no third GPU job was submitted.

### Files

- `experiments/unity_two_stage_transfer/s0010_exact_pose_replay_protocol_v1.yaml`
- `experiments/unity_two_stage_transfer/s0010_exact_pose_replay_protocol_v2.yaml`
- Runtime V1: `s0010_exact_pose_replay_v1/51782795/`
- Runtime V2: `s0010_exact_pose_replay_v2/51783444/`

## 3. Exact correspondence evidence

### Criterion

Every frame must reconstruct the stored raw-label slice exactly from the persisted native
pixel-to-volume coordinates. Full-pose replay must also meet exact raw-label agreement at least
0.985 and bone-mask IoU at least 0.90. The sealed v2 protocol additionally requires maximum
coordinate disagreement at most 0.001 volume index.

### Method

The post-capture CPU verifier independently resamples the original s0010 raw label volume using
both the saved native coordinates and coordinates reconstructed from saved full pose. It joins
frames by sequence and SHA-256 and uses the renderer's truncation and display transpose.

### Setup

```bash
python workflows/two_stage_navigation/verify_exact_pose_capture.py \
  --protocol experiments/unity_two_stage_transfer/s0010_exact_pose_replay_protocol_v2.yaml \
  --session-dir "$RUN_DIR/session" \
  --assets-data-dir "$SONOGYM_ASSETS_DATA_DIR" \
  --output "$RUN_DIR/results/exact_pose_replay_report.json"
```

### Results

| quantity | required | measured | verdict |
|---|---:|---:|---|
| stored-coordinate exact raw-label frames | 6 / 6 | 6 / 6 | PASS |
| full-pose agreement | >= 0.985 | 0.9879--0.9906 | PASS |
| full-pose bone IoU | >= 0.90 | 0.9514--0.9576 | PASS |
| clipped samples | 0 | 0 | PASS |
| maximum coordinate error | <= 0.001 index | 0.1918--0.2136 | FAIL |

**Verdict: PROJECTION_REPAIR_BLOCKED under the sealed protocol.**

### Reading it

- The saved native pixel-to-volume map is exact and is suitable for rechecking the raw-label
  slice without guessing the pose.
- Full-pose replay is already within the historical image-level gates, but not the much stricter
  coordinate equality gate frozen for this capture.
- The current evidence does not isolate quaternion precision, timing or another transform detail
  as the remaining cause; constants must not be fitted to these six opened frames.

### Files

- `session/privileged/pixel_to_volume_coordinates.npz`
- `session/privileged/full_pose_replay.jsonl`
- `results/exact_pose_replay_report.json`

## 4. Observation diversity and training gate

### Criterion

At least four of six ultrasound frames must have different exact hashes. A requested waypoint
that executes zero actions and does not advance the producer cannot count as a new observation.
No semantic training sample may be created without reviewed anatomy and verified correspondence.

### Method

Frame sequence, producer sequence, action count, requested waypoint, realized pose and exact image
hash were audited together after the actor record was frozen.

### Setup

The six requested poses used x offsets `[-2, 0, 2]` volume indices, one z offset and angles 2.90
and 2.95 radians. Physical position tolerance remained 5 mm.

### Results

Frames 0 and 1 executed 81 and 6 actions and advanced producer sequence to 82 and 88. Frames 2--5
executed zero actions, retained producer sequence 88 and repeated the same hash. Their 3.0--4.5 mm
requested translation differences remained within the unchanged 5 mm movement tolerance.

| quantity | required | measured | verdict |
|---|---:|---:|---|
| distinct hashes | >= 4 | 2 | FAIL |
| producer-advanced observations | >= 4 | 2 | FAIL |
| training-eligible semantic samples | > 0 | 0 | BLOCKED |

**Verdict: FAIL / training remains NOT_RUN.**

### Reading it

- This is a demonstrated waypoint-spacing error, not evidence that the renderer cannot produce
  distinct SAP/TP views.
- Exact correspondence alone does not approve TP/SAP anatomy. The two TP surface cores still need
  qualified review and both SAP boundaries remain unresolved.

### Files

- `session/public/frames.jsonl`
- `session/privileged/annotations.jsonl`
- `session/privileged/actions.jsonl`

## 5. Resources, artifacts and next gate

### Criterion

Report actual scheduler evidence, storage growth, software provenance and the smallest next action;
do not start training from a failed capture.

### Method

Slurm `sacct`, job context, `/usr/bin/time`, GPU telemetry and on-disk file sizes were inspected
after completion.

### Setup

Job `51783444` used source commit `0b8294e77008a5d296a3bc53c213e232a565e0c7`, Isaac Lab commit
`21f7136325136ca3f6ca4e0a8125edffe5c24f7e`, Python 3.10.22 and NVIDIA driver 580.178.04.

### Results

The job requested one L4, four CPUs and 30 GB; elapsed time was 43 seconds. Peak sampled GPU memory
was 5,009 MiB, sampled utilization peaked at 1%, `/usr/bin/time` maximum RSS was 3,607,956 KiB,
and output growth was 2.9 MB. Slurm recorded `FAILED (3:0)` because the fail-closed gates returned
NO_GO, not because the simulator crashed.

**Verdict: bounded execution complete; training blocked.**

### Reading it

- The next protocol must use waypoint separations larger than movement tolerance and require an
  action/producer advance before accepting each requested observation.
- Before another capture, the researcher must review whether coordinate equality at 0.001 index is
  scientifically necessary in addition to the already passed image-level dual gate.
- After a diverse exact-mapping capture, TP-only preparation can proceed only for independently
  accepted TP geometry. SAP training remains blocked until SAP boundaries are resolved.

### Files

- Result: `experiments/unity_two_stage_transfer/s0010_exact_pose_replay_v2_result.json`
- Runtime: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_exact_pose_replay_v2/51783444/`
- Reviewer ZIP: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/reviewer/s0010_exact_pose_replay_review_51783444.zip`
- ZIP SHA-256: `0263a99829125431d53c962a6ddcdd25c3f55f56cc08e9b3e2777ced6d79a7f8`
