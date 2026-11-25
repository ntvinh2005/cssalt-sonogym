# Independent anatomical mapping confirmation on s0010

This campaign evaluates the matched repair from job `52694480` on four unused simulator seeds. Stage 1 is supplied by the previously recorded actor output. Acquisition is fresh for all four mapping sessions.

## Result

**Independent confirmation fail with 0/4 complete maps and 10/16 correct landmarks.**

| Step | Measurement | Criterion | Result |
|---|---|---|---|
| §1 Replay | Historical mapping results and compatible revised SAP routing | Report changes on the same 816 valid frames | Exact replay reproduced. Revised SAP lost one correct landmark. |
| §2 Independent confirmation | Correct landmarks and complete maps on four distinct starts | Four complete maps, 16 correct landmarks, zero wrong and zero unresolved | FAIL, 0/4 maps and 10/16 correct |
| §3 Recorded demonstration | Source integrity and full MP4 decode | Verified video and map artifacts for all scored sessions | Pass, full decode |

The experiment started at 04:41:10 UTC on 2026-11-25. Its scope is provisional simulator-proxy anatomical identity on fixed anatomy `s0010`.

## 1. Replay — what part of the final acquisition pipeline is testable in the historical records?

`run_simple_landmark_mapping.py` reproduces the three original replay arms. `replay_independent_sap.py` replaces only the historical SAP observations with those from the frozen revised level assignment.

**Criterion.** Compare the historical simple mapper to the archived 15/16 correct, 0 wrong and 3/4 complete maps. Report the revised SAP result on the same valid frames and identify the limits of recreating the full acquisition and evidence schedule.

### 1.0 Quantities

| Quantity | Definition | Source |
|---|---|---|
| Map and medoid | [Simple anatomical landmark mapping](../notation.md#simple-anatomical-landmark-mapping) | `simple_map` |
| Acceptance and identity | [Anatomical mapping status](../notation.md#anatomical-mapping-status) | `simple_map`, `score_map` |
| Localization | [Distance to the visible interface](../notation.md#distance-to-the-visible-interface) | `score_map` |

### 1.1 Method

- Use the historical logged frame hashes, executed 6DoF poses, freshness/contact checks, Stage 1 actor references and provisional cortical truth. Run all three historical arms with the pinned checkpoints and confirm the same frame set.
- Apply both U-Nets to each valid frame in the revised SAP arm. Fit the same public cranial direction using frames tagged as transverse with an executed angle within 1° of 2.95 rad. Preserve the frozen fit residual bound, reference ambiguity bands, component acceptance and mapping settings.
- Preserve TP inputs from the historical simple-mapper arm. The actively selected v6r19 trajectories have no specified partition into the final 50 original and four supplementary SAP waypoints. The revised arm isolates the SAP routing effect on historical observations.

### 1.2 Setup

| Setting | Value | Specification |
|---|---|---|
| v6r19 source | Job `52626449`, seeds `2026120400`–`2026120403` | 821 acquired, 816 valid |
| Historical regression reference | Job `52640447` | Four source sessions |
| CPU regression job | `52699992`, four CPU cores, 12 GB requested, no GPU | `anest-dept`, `hpg-default` |
| Execution commit | `3dceed443fde53d218883a74a36cea7d9ece2d68` | Same scientific files as the reference commit |
| Checkpoints | Job `52498821`, L4 and L5 U-Nets | SHA-256 recorded in §2.2 |

```bash
source_run=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_mbb_single_anatomy_poc_v6r19/52626449
results=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_independent_20261125T044110Z
source_protocol="$source_run/context/s0010_mbb_single_anatomy_poc_protocol_v6r19.yaml"
"$SONOGYM_ENV_PREFIX/bin/python" workflows/two_stage_navigation/run_simple_landmark_mapping.py \
  --run-root "$source_run" --protocol "$source_protocol" \
  --label-conversion source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml \
  --output "$results/replay-52699992/results" --batch-size 8
"$SONOGYM_ENV_PREFIX/bin/python" workflows/two_stage_navigation/replay_independent_sap.py \
  --protocol "$source_protocol" --baseline "$results/replay-52699992/results" \
  --output "$results/replay-52699992/revised_sap.json"
```

| Source session | Attempted | Dropped by existing integrity flags | Valid |
|---|---:|---:|---:|
| 00 | 201 | 2 | 199 |
| 01 | 196 | 1 | 195 |
| 02 | 215 | 1 | 214 |
| 03 | 209 | 1 | 208 |
| Total | 821 | 5 | 816 |

### 1.3 Results

| Arm, 816 valid frames | Correct mapped landmarks | Wrong mapped landmarks | Unresolved | Complete maps |
|---|---:|---:|---:|---:|
| Oracle plus simple, reproduced | 14/16 | 0/16 | 2/16 | 2/4 |
| Historical U-Net plus simple, reproduced | 15/16 | 0/16 | 1/16 | 3/4 |
| Historical U-Net plus existing association, reproduced | 7/16 | 4/16 | 5/16 | 0/4 |
| Revised SAP and historical TP, compatible replay | 14/16 | 1/16 | 1/16 | 2/4 |

**Verdict. The historical replay reproduced its archived totals. The compatible revised SAP replay regressed by one wrong landmark and one complete map.** The four fit sets each contained 16 transverse acquisitions. The maximum fit residual was 0.2693 command index, within the frozen 1.5 limit.

### 1.4 Reading it

- The revised replay changed L4 SAP on historical session 02 from correct to wrong. The selected cluster went from 14/16 correct supporting views to 2/8.
- The changed L4 SAP point was 0.424 mm from the nearest visible interface in the session. The identity criterion still failed because six of eight supporting images lacked the corresponding visible interface.
- The exact TP subset is available only on the 54-waypoint systematic sweep. The compatible replay does not measure the entire frozen acquisition pipeline.
- The CPU job passed 45 focused tests and completed in 3:59. The GPU campaign used the sealed scientific settings.

### 1.5 Files

| File | Contents |
|---|---|
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_independent_20261125T044110Z/replay-52699992/results/comparison.json` | Exact replay and per-session evidence |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_independent_20261125T044110Z/replay-52699992/revised_sap.json` | Revised SAP routing and map outcomes on each recorded session |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_independent_20261125T044110Z/replay-52699992/tests.log` | Focused replay tests |

## 2. Independent confirmation — does the frozen map reproduce on four unused resets?

The launcher calls the unchanged sweep actor and evaluator sequentially. It checks the 54-waypoint plan and seals each actor JSON with SHA-256 before correctness evaluation.

**Criterion.** Require 4/4 complete maps, 16/16 correct landmarks, zero wrong and zero unresolved landmarks. A complete map requires all 54 valid synchronized acquisitions and passing coordinate integrity. Verify distinct realized starts from the recorded reset poses and image sequences.

### 2.1 Method

- Reuse the sealed Stage 1 actor record `cf7_00`. Use all four previously unused seeds with a fresh simulator reset. Search the repository and 668 archived runtime protocols, manifests and actor-result records for the candidate seeds before freezing.
- Use the same 14 longitudinal and 40 transverse viewpoints as the successful matched repair, including four additional SAP slices. Credit TP proposals only from the original 50 waypoint indices. Credit SAP from all 54 if projected level and checkpoint agree. Preserve uncertainty near Stage 1 level references and apply the existing U-Net acceptance thresholds.
- Require two distinct supporting views in a 4 mm cluster. Distinct views meet the established 2 mm translation or 1° rotation threshold. Retain the confidence-margin abstention rule. Use existing 0.5 mm ultrasound pixel spacing, 133 mm image-origin offset and executed 6DoF projection.
- Compare first-hit, image-supported provisional visible interfaces. An accepted landmark needs a strict majority of selected views at most 3 pixels from the corresponding visible interface. The projection check uses a maximum residual of 0.0015 mm. The evaluator runs after the actor JSON seal is written. Oracle diagnostics are offline.
- Check all six pairs of realized initial poses and complete image-hash sequences. Use the established physical viewpoint separations to characterize differences.
- Record the robot camera on every session, as in the successful development run. Use the first failed scored session for review when a failure exists and otherwise use the first scored session.

### 2.2 Setup

| Setting | Value | Specification |
|---|---|---|
| Development reference | Job `52694480`, seed `2026120500`, execution `a53cf7be668a7aa47918511eaefeae5fc8d3bfb3` | 54/54 frames, 4/4 correct |
| Scientific reference | `9009900be0d7c3238b45cff485237dbe42fbd988` | Scientific source matches the development executable |
| Confirmation execution | `3dceed443fde53d218883a74a36cea7d9ece2d68`, Slurm `52699995` | Launcher and sealed protocol added |
| Confirmation protocol | `s0010_simple_landmark_independent_v1.yaml` | SHA-256 `2c4e60945c38b6d501a1fa1e1134c7118fe915a2615e77fd0f32788110fdc669` |
| Seeds | `2026121700`–`2026121703`, fresh mapping on `s0010` | Never used in searched records |
| Stage 1 reused | `cf7_00`, SHA-256 `1fcfd37760034feaf5eccca586a687713a04181fcb943df293d5b1c213a1f4a8` | Fresh Stage 2 acquisition |
| U-Net L4 | Training job `52498821` | SHA-256 `c6b7ecc9bc245a88ccc0a5db31dbfd919b57407988dde5ea8211dd7eea2c1b2a` |
| U-Net L5 | Training job `52498821` | SHA-256 `ca13c8fb6bf99bd363dd87c5e30dc1eeb5b22da5cf3f87e991893fad96b41acf` |
| Probability / minimum component size | L4 TP 0.90 / 5, SAP 0.90 / 3, other bone 0.60 / 2. L5 TP 0.80 / 8, SAP 0.90 / 2, other bone 0.90 / 8 | Frozen checkpoint metadata |
| Resources | Four CPUs, one NVIDIA L4, 30 GB, 18-minute time limit | `anest-dept`, `hpg-turin` |

```bash
source_run=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_mbb_single_anatomy_poc_v6r19/52626449
results=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_independent_20261125T044110Z
source_protocol="$source_run/context/s0010_mbb_single_anatomy_poc_protocol_v6r19.yaml"
sbatch --dependency=afterok:52699992 --kill-on-invalid-dep=yes \
  --deadline=2026-11-25T00:29:00 --output="$results/confirm-%j.log" \
  --export=ALL,SONOGYM_CONFIG_FILE="$source_run/context/machine-config.env",SIMPLE_MAPPING_RUN_ROOT="$source_run",SIMPLE_MAPPING_PROTOCOL="$source_protocol",SIMPLE_MAPPING_OUTPUT_ROOT="$results",SIMPLE_MAPPING_REPLAY_JOB=52699992,SIMPLE_MAPPING_EXECUTION_COMMIT=3dceed443fde53d218883a74a36cea7d9ece2d68 \
  hpg/jobs/s0010_simple_landmark_independent.sbatch
# The wrapper creates a sealed session profile with the chosen seed and uses:
"$SONOGYM_ENV_PREFIX/bin/python" workflows/two_stage_navigation/capture_simple_landmark_sweep.py \
  --protocol "$source_protocol" --revision "$results/52699995/context/session_$index.yaml" \
  --stage1-session cf7_00 --stage1-actor "$source_run/stage1/results/actor_results.json" \
  --seed "$seed" --output "$results/52699995/session_$index" --headless --device cuda:0 \
  --record-robot --enable_cameras \
  --evaluator-segmentation "$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
"$SONOGYM_ENV_PREFIX/bin/python" workflows/two_stage_navigation/evaluate_simple_landmark_sweep.py \
  --session "$results/52699995/session_$index" --protocol "$source_protocol" \
  --label-conversion source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml \
  --oracle-diagnostic "$results/52699995/session_$index/oracle_diagnostic.json"
```

### 2.3 Results

The number in parentheses is the selected distinct-view count. Complete maps include acquisition and projection integrity.

| Session | Seed | Valid frames | L4 TP | L4 SAP | L5 TP | L5 SAP | Correct / Wrong / Unresolved | Complete map |
|---|---:|---:|---|---|---|---|---|---|
| 00 | 2026121700 | 54/54 | CORRECT (5) | UNRESOLVED (2) | UNRESOLVED (2) | CORRECT (2) | 2 / 0 / 2 | False |
| 01 | 2026121701 | 54/54 | CORRECT (7) | UNRESOLVED (2) | CORRECT (2) | CORRECT (3) | 3 / 0 / 1 | False |
| 02 | 2026121702 | 54/54 | CORRECT (4) | CORRECT (2) | WRONG (2) | CORRECT (2) | 3 / 1 / 0 | False |
| 03 | 2026121703 | 54/54 | CORRECT (7) | UNRESOLVED (2) | WRONG (2) | CORRECT (2) | 2 / 1 / 1 | False |

The support frame numbers are local to each session. Full acquisition IDs and frame hashes are in `confirmation.json`.

| Session | Landmark | Actor status | Support frames | RMS dispersion, mm | Competing clusters | Visible-interface error, mm | Oracle outcome |
|---|---|---|---|---:|---:|---:|---|
| 00 | RIGHT L4 TP | MAPPED | 12, 8, 9, 10, 11 | 1.428 | 5 | 0.550 | CORRECT |
| 00 | RIGHT L4 SAP | AMBIGUOUS | 52, 51 | 1.978 | 3 | 0.500 | UNRESOLVED |
| 00 | RIGHT L5 TP | AMBIGUOUS | 5, 41 | 2.453 | 2 | 1.000 | UNRESOLVED |
| 00 | RIGHT L5 SAP | MAPPED | 43, 31 | 1.838 | 1 | 1.000 | CORRECT |
| 01 | RIGHT L4 TP | MAPPED | 11, 9, 10, 12, 24, 13, 8 | 1.730 | 5 | 0.500 | CORRECT |
| 01 | RIGHT L4 SAP | AMBIGUOUS | 48, 49 | 1.315 | 3 | 1.000 | UNRESOLVED |
| 01 | RIGHT L5 TP | MAPPED | 33, 42 | 1.870 | 2 | 0.500 | CORRECT |
| 01 | RIGHT L5 SAP | MAPPED | 43, 31, 21 | 1.628 | 2 | 1.000 | UNRESOLVED |
| 02 | RIGHT L4 TP | MAPPED | 12, 11, 10, 9 | 1.671 | 6 | 0.783 | CORRECT |
| 02 | RIGHT L4 SAP | MAPPED | 48, 49 | 1.458 | 3 | 1.500 | UNRESOLVED |
| 02 | RIGHT L5 TP | MAPPED | 33, 5 | 1.882 | 2 | 0.500 | CORRECT |
| 02 | RIGHT L5 SAP | MAPPED | 43, 31 | 2.194 | 1 | 1.000 | UNRESOLVED |
| 03 | RIGHT L4 TP | MAPPED | 10, 27, 11, 12, 13, 9, 8 | 1.450 | 6 | 0.000 | CORRECT |
| 03 | RIGHT L4 SAP | AMBIGUOUS | 48, 49 | 1.221 | 2 | 1.500 | UNRESOLVED |
| 03 | RIGHT L5 TP | MAPPED | 33, 42 | 1.528 | 2 | 0.500 | CORRECT |
| 03 | RIGHT L5 SAP | MAPPED | 43, 31 | 1.967 | 1 | 1.000 | UNRESOLVED |

| Session | Initial executed pose index | Actor elapsed excluding mapping, s | Mapping, ms | Acquisition-to-evaluation wall time, s | Maximum projection residual, mm |
|---|---|---:|---:|---:|---:|
| 00 | 177.498, 103.835, 2.593 | 93.285 | 2.930 | 181.0 | 0.0001037 |
| 01 | 132.732, 76.585, 3.000 | 80.455 | 3.152 | 145.0 | 0.0000812 |
| 02 | 158.676, 170.644, 2.455 | 85.284 | 2.791 | 155.0 | 0.0001159 |
| 03 | 132.413, 155.683, 2.781 | 105.301 | 3.315 | 171.0 | 0.0001273 |

**Verdict. FAIL.** The campaign scored 4/4 sessions, with 0/4 complete maps and 10/16 correct landmarks. There were 2 wrong and 4 unresolved scored landmarks.

### 2.4 Reading it

- All four sessions acquired 54 / 54 valid frames and passed the projection gate. The GPU job completed in 10:59 with one NVIDIA L4 and four CPU cores. The actor's elapsed time excluding the mapper summed to 364.325 s. Mapper time totaled 12.188 ms.
- L4 SAP was ambiguous in three actor maps and all four Oracle maps. Each Oracle case had competing clusters with two views apiece. Session 00's two leading actor positions were 7.300 mm apart. Its Oracle positions were 6.020 mm apart.
- L5 TP was accepted incorrectly in sessions 02 and 03 with one of two views meeting the 3-pixel identity gate. Session 02 had one view with no corresponding visible interface. Session 03 had one view 3.162 pixels from the provisional interface. Oracle mapped L5 TP correctly in both sessions.
- L4 TP and L5 SAP each mapped correctly in all four actor sessions.
- Researcher review should examine the competing L4 SAP surface evidence and the two false L5 TP acceptances before selecting a subsequent experiment.

- Session 00, RIGHT L4 SAP. Competing spatial clusters had insufficient separation in support or confidence. Oracle scored UNRESOLVED. 1/2 selected views met the 3-pixel gate.
- Session 00, RIGHT L5 TP. Competing spatial clusters had insufficient separation in support or confidence. Oracle scored UNRESOLVED. 2/2 selected views met the 3-pixel gate.
- Session 01, RIGHT L4 SAP. Competing spatial clusters had insufficient separation in support or confidence. Oracle scored UNRESOLVED. 2/2 selected views met the 3-pixel gate.
- Session 02, RIGHT L5 TP. Accepted U-Net cluster failed the frozen majority visible-interface score. Oracle scored CORRECT. 1/2 selected views met the 3-pixel gate.
- Session 03, RIGHT L4 SAP. Competing spatial clusters had insufficient separation in support or confidence. Oracle scored UNRESOLVED. 2/2 selected views met the 3-pixel gate.
- Session 03, RIGHT L5 TP. Accepted U-Net cluster failed the frozen majority visible-interface score. Oracle scored CORRECT. 1/2 selected views met the 3-pixel gate.
- Minimum pairwise reset separation was 45.338 mm and minimum rotation difference was 7.891°. Distinct realized starts passed: `True`.

### 2.5 Files

| File | Contents |
|---|---|
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_independent_20261125T044110Z/review/confirmation.json` | All attempted sessions, exact evidence, timing, pairwise starts and verdict |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_independent_20261125T044110Z/review/session_<index>/` | Sealed actor result, separate evaluator score, Oracle diagnosis and map |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_independent_20261125T044110Z/52699995/context/` | Frozen profiles, checksums, scheduler and dependency provenance |

## 3. Demonstration — can the representative result be inspected?

The existing renderer consumes the preserved camera, ultrasound, U-Net masks and executed pose records.

**Criterion.** Decode every MP4 frame at positive duration, verify the ultrasound hashes and executed-pose joins, and identify the selected session's real map result.

### 3.1 Method

- Select the first failed scored session when one exists. Show all sessions separately in the result table and package their static actor maps.
- Render actor evidence in acquisition order. Place offline evaluator identity and viewpoint counts on the final card.

### 3.2 Setup

| Setting | Value | Specification |
|---|---|---|
| CPU review job | `52700804`, two CPU cores, no GPU | `anest-dept`, `hpg-default` |
| Source | Confirmation execution `52699995` | §2 |
| Renderer source | `review_source/review_independent_landmark_mapping.py` | SHA-256 in the review package |

```bash
"$SONOGYM_ENV_PREFIX/bin/python" "$results/review_source/review_independent_landmark_mapping.py" \
  --campaign "$results/52699995" \
  --protocol experiments/unity_two_stage_transfer/s0010_simple_landmark_independent_v1.yaml \
  --source-protocol "$source_protocol" --output "$results/review"
```

### 3.3 Results

| Seed | Decode | Duration | Actual ultrasound/pose joins | Synchronized robot pictures |
|---:|---:|---:|---:|---:|
| 2026121700 | 150/150 at 6 fps | 25.0 s | 54 | 54 |

**Verdict. Pass.** The footage decoded fully and shows the selected seed's observed outcome. The 39-second review job packaged separate static maps for all four sessions.

### 3.4 Reading it

- The scope is fixed simulator anatomy and reused Stage 1. Fresh physical return, clinical validity and MBB target localization remain outside the experiment.

### 3.5 Files

| File | Contents |
|---|---|
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_independent_20261125T044110Z/review/session_00/independent_systematic_sweep.mp4` | Representative independent result |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_independent_20261125T044110Z/simple-landmark-independent-review.tar.gz` | All scored maps, evidence, review video, replay and frozen protocol |
