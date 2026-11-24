# Simple anatomical landmark mapping on s0010

This experiment compares two spatial mappers on shared ultrasound observations and uses provisional visible-interface labels to diagnose observability. It also tests one fixed, two-plane development trajectory after the offline comparison.

## Result

**On v6r19 recordings, the simple mapper produced three complete anatomical maps across four starts, with zero wrong mapped landmarks.**

| Step | Measured | Criterion | Result |
|---|---|---|---|
| §1 Projection | Maximum 3D residual on 50,117 sampled pixels | At most 0.0015 mm | Pass, 0.000130 mm |
| §1 Association | Correct mapped landmarks and complete maps on shared U-Net evidence | Four correct mapped landmarks in a session | B 15/16 and 3/4. C 7/16 and 0/4 |
| §1 Oracle diagnostic | Complete maps from first-hit, image-supported interfaces | Four correct mapped landmarks in a session | A 14/16 and 2/4 |
| §2 Fresh acquisition | Fixed survey and provisional anatomical identity | Both planes acquired, projection passes, and all four landmarks correct | Partial, 50/50 valid frames and 2/4 correct landmarks |

The experiment started at 20:42:01 UTC on 2026-11-24. The v6r19 starts were opened for this new development comparison. The original integrated results retain their sealed verdict. Results concern provisional simulator anatomy on `s0010`.

## 1. Recorded-data comparison — can simple association recover four landmarks?

`workflows/two_stage_navigation/run_simple_landmark_mapping.py`, run by `hpg/jobs/s0010_simple_landmark_mapping.sbatch`, produces one map per session and arm.

**Criterion.** A mapped landmark passes identity when a strict majority of its selected distinct views fall within 3 pixels of the same provisional visible interface in the source image. The 3-pixel and majority rules come from `evaluate_stage2_tp_active.py:incremental_map_identity`. Both landmarks must pass to credit a level. A complete map requires all four landmarks.

### 1.0 Quantities

| Quantity | Definition | Source |
|---|---|---|
| Map and medoid | [Simple anatomical landmark mapping](../notation.md#simple-anatomical-landmark-mapping) | `simple_landmark_mapping.py` |
| Status and correctness | [Anatomical mapping status](../notation.md#anatomical-mapping-status) | `simple_map`, `score_map` |
| Localization error, mm | [Distance to the visible interface](../notation.md#distance-to-the-visible-interface) | `score_map` |

### 1.1 Method

- Admit published frame records with valid image, contact, advancing producer, and valid transition after verifying ultrasound content hash and executed 6DoF. Residuals compare the public projection with sampled native coordinates on a 25-pixel-spaced lattice and provisional target-interface pixels, excluding samples clamped to a volume boundary. Estimate the evaluator-only rigid conversion between recorded human and robot bases from the synchronized probe poses of the first valid frame. Use the same conversion for every frame in the session.
- A uses the existing whole-TP and whole-SAP provisional masks and the first bone hit along each acoustic ray. A hard interface label requires a bright-interface track within 3 pixels and excludes the uncertain boundary. Both vertebral levels are scored on each valid frame. A uses the same transform as B and C. The public transform uses 0.5 mm image pixels and a 133 mm origin offset with recorded probe position and rotation.
- B routes each pose's cranial-caudal command index using the midpoint between the Stage 1 L4 and L5 reference intervals. The ambiguous band covers the interval of possible midpoints plus 1.5 command units on each side, half the recorded survey spacing. Both U-Nets are applied in the band and their outputs are recorded as ambiguous. Those proposals are withheld from support in B and C.
- Form 8-connected TP and SAP components. Select the actual component pixel nearest its confidence-weighted image centroid. A uses unit confidence and retains component pixels with at least one visible-interface pixel. B and C use each checkpoint's existing class-specific probability and minimum-size thresholds. Both preserve component masks.
- A and B cluster within 4 mm complete-link distance. Count one observation per frame/hash among mutually distinct views with at least 2 mm translation or 1° relative rotation. Rank by distinct-view count, mean semantic confidence and dispersion. Require two supporting views and mean confidence of at least 0.5. Abstain if the top two clusters tie in support and differ in mean confidence by less than 0.01.
- C replays the existing `IncrementalAnatomicalLandmarkMap` selectors in acquisition order on exactly B's usable observations. C uses the v6r19 association configuration. C has no return links because those require the original frozen intermediate decisions and physical actions. All arms use a common semantic pair criterion. The original active decision path, return requirements, and TP-core evaluation are outside the comparison.

### 1.2 Setup

| Setting | Value | Specification |
|---|---|---|
| Branch starting revision | `ffe23a5b15c44fe64103672b8e2acd3dfe4a54b9` | Branch `experiment/simple-landmark-mapping` |
| Offline executable | `56c19fa` | Job `52640447`, clean checkout |
| Sources | `mbbpoc_v6r19_00` through `03` in job `52626449` | Source revision `ed1ab888f585e10b395329ec7990f2dd9898d1d6` |
| Anatomy and starts | `s0010`, seeds `2026120400`–`2026120403`, references `cf7_00`–`03` | Opened development evidence |
| Checkpoints | Job `52498821`, `s0010_l4_sonoanatomy_unet_v1.pt` and `s0010_l5_sonoanatomy_unet_v1.pt` | L4 SHA-256 `c6b7ecc9bc245a88ccc0a5db31dbfd919b57407988dde5ea8211dd7eea2c1b2a`. L5 SHA-256 `ca13c8fb6bf99bd363dd87c5e30dc1eeb5b22da5cf3f87e991893fad96b41acf` |
| Threshold / minimum component pixels | L4 TP 0.90 / 5, SAP 0.90 / 3, other bone 0.60 / 2. L5 TP 0.80 / 8, SAP 0.90 / 2, other bone 0.90 / 8 | Frozen checkpoint metadata |
| Shared inputs | All valid source images and poses. Identical calibrated public projection and anatomical scoring definitions | §1.1 |
| Perception contrast | Oracle visible interfaces in A and checkpoint masks in B/C. A has directly labeled level. B/C use Stage 1 routing | §1.1 |
| Replay contrast | C's incremental association and selection among input observations. A/B's fixed complete-link clusters after acquisition. Original policy actions held as recorded | §1.1 |
| Scoped tests | 26 offline job tests, 22 fresh job tests and 29 local visible-label and calibration tests passed | `tests.log` and bounded local runs |
| Software | Python 3.10.22, NumPy 1.26.4, PyTorch 2.5.1+cu121 on CPU, NiBabel 5.3.2 | Full `pip-freeze.txt` in job directory |
| Resources | 4 CPUs, 12 GB requested, 0 GPUs, `anest-dept` / `hpg-default` | Elapsed 2:38, total CPU 3:08.382, batch peak RSS 885184 KiB |

```bash
source_run=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_mbb_single_anatomy_poc_v6r19/52626449
experiment=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_mapping_20261124T204201Z
cd "$experiment/checkout"
sbatch --parsable --chdir="$experiment/checkout" --output="$experiment/offline-%j.log" \
  --export=ALL,SONOGYM_CONFIG_FILE="$source_run/context/machine-config.env",SIMPLE_MAPPING_RUN_ROOT="$source_run",SIMPLE_MAPPING_PROTOCOL="$source_run/context/s0010_mbb_single_anatomy_poc_protocol_v6r19.yaml",SIMPLE_MAPPING_OUTPUT_ROOT="$experiment" \
  hpg/jobs/s0010_simple_landmark_mapping.sbatch
# On the compute node the wrapper runs the pinned Python:
$SONOGYM_ENV_PREFIX/bin/python workflows/two_stage_navigation/run_simple_landmark_mapping.py \
  --run-root "$source_run" \
  --protocol "$source_run/context/s0010_mbb_single_anatomy_poc_protocol_v6r19.yaml" \
  --label-conversion source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml \
  --output "$experiment/${SLURM_JOB_ID}/results" --batch-size 8
```

For exact reproduction use `56c19fa` at the compute checkout. The sealed source protocol SHA-256 is `c26802d61f211c62acd16fe12208183e3442ebf776e9188d867a77126c6a114e`. Full input hashes, labels, and per-frame membership are recorded in the job output.

| Source session | Attempted frames | Invalid image, stale producer and invalid transition | Final valid frames | Valid frames with ambiguous Stage 1 level |
|---|---:|---:|---:|---:|
| 00 | 201 | 2 | 199 | 11 |
| 01 | 196 | 1 | 195 | 73 |
| 02 | 215 | 1 | 214 | 12 |
| 03 | 209 | 1 | 208 | 7 |
| Total | 821 | 5 | 816 | 103 |

### 1.3 Results

| Metric, four v6r19 starts | A: Oracle + simple | B: U-Net + simple | C: U-Net + existing |
|---|---:|---:|---:|
| RIGHT L4 TP correct | 4/4 | 4/4 | 1/4 |
| RIGHT L4 SAP correct | 4/4 | 4/4 | 1/4 |
| RIGHT L5 TP correct | 4/4 | 3/4 | 3/4 |
| RIGHT L5 SAP correct | 2/4 | 4/4 | 2/4 |
| Correct mapped landmarks | 14/16 | 15/16 | 7/16 |
| Wrong mapped landmarks | 0/16 | 0/16 | 4/16 |
| Unresolved landmarks | 2/16 | 1/16 | 5/16 |
| Correct L4 / L5 pairs | 4/4 / 2/4 | 4/4 / 3/4 | 0/4 / 2/4 |
| Complete anatomical map | 2/4 | 3/4 | 0/4 |
| Observation count | 2102 | 794 | 794 |
| Perception + mapping, seconds | 93.017 | 21.275 | 22.585 |
| Mapping only, seconds | 0.343 | 0.124 | 1.434 |

| Landmark | B/C observations correct / all usable | B distinct support across sessions | B localization error for correctly mapped landmarks, mm |
|---|---:|---:|---:|
| RIGHT L4 TP | 254/303 | 10–17 | 0.000–0.500 |
| RIGHT L4 SAP | 113/148 | 7–16 | 0.500–0.945 |
| RIGHT L5 TP | 164/213 | 8–16 | 0.000–0.594 |
| RIGHT L5 SAP | 92/130 | 5–16 | 0.000–0.500 |

**Verdict. B outperformed C on matched perceptions. It passed the complete-map criterion on sessions 00, 02 and 03.** Projection passed in all four sessions. The evaluator verified anatomical identity separately from TP surface-core target-view performance.

### 1.4 Reading it

- C accepted three wrong landmarks in session 00. For L5 SAP it selected 0/3 correct views while B selected 5/5.
- A observed all four landmarks in all four sessions. L5 SAP had tied clusters in sessions 00 and 02 with 7 and 11 distinct views per cluster, respectively. The top positions were 7.804 mm and 2.289 mm apart. The oracle's zero medoid-to-observed-surface distance follows from construction.
- B's unresolved RIGHT L5 TP in session 01 had 9 versus 9 views at a separation of 1.917 mm. Their mean confidences differed by 0.005. The top cluster had 5/9 correct views, and its competitor had 9/9.
- Of 293 L5 TP-visible frames, 33 fell outside the L5 model routing region and 17 had ambiguous level. The L5 model supplied a usable correct candidate in 151 frames. In 79 misses, background dominated the thresholded class map on the visible interface.
- The historical complete controller produced 5/8 correct anatomical pairs and 2/4 sessions with both pairs correct. The replay produced 7/8 and 3/4, respectively, on frames acquired by that controller. Its integrated 1/8 target and 0/4 session results remain separate.
- Association is the principal measured loss on the recorded inputs. Perception and vertebral routing still removed usable views. The recording design conditions the comparison on the prior controller's coverage.

### 1.5 Files

| File | Contents |
|---|---|
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_mapping_20261124T204201Z/52640447/results/comparison.json` | All arms and sessions, positions, correctness counts, localization errors, timings, and input hashes |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_mapping_20261124T204201Z/52640447/results/<session>/actor/` | U-Net masks, projected observations, ambiguous-level proposals, both actor maps and competing clusters |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_mapping_20261124T204201Z/52640447/results/<session>/oracle/` | `ORACLE_PERCEPTION_DIAGNOSTIC — NOT AUTONOMOUS` labels, observations and map |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_mapping_20261124T204201Z/52640447/review/visualizations/` | Four actor maps, four oracle maps, fresh actor map, comparison figure and session-00 199-frame MP4, job `52642917` |

## 2. Fixed two-plane acquisition — can the simple controller supply map support?

`workflows/two_stage_navigation/capture_simple_landmark_sweep.py` and `evaluate_simple_landmark_sweep.py`, run by `hpg/jobs/s0010_simple_landmark_sweep.sbatch`, execute and score one development sweep.

**Criterion.** A pass requires valid acquisitions from both planes, no failed motion or invalid frames, a passing projection check and four correct landmarks by §1. A partial result requires a passing projection check and at least one correct map landmark.

### 2.1 Method

- Sweep A uses two Stage 1 reference rows and seven lateral offsets per row, from +2 to +20 in three-unit steps. It commands the longitudinal orientation at 1.571 rad.
- Sweep B uses +5, +11 and +17 lateral offsets and sweeps between six command units below the L5 reference and six above the L4 reference. It evenly spaces each pass at no more than three command units, reverses every second pass, and commands 2.950 rad. Intermediate rotation commands are bounded by 0.35 rad.
- Both use the existing controller and 1 mm / 1° tracking tolerances. The reported calibration projects recorded executed 6DoF poses. Contact, producer freshness, image validity and transition integrity are checked at acquisition.
- U-Net inference records masks and candidates after each valid frame. The map is constructed after the last attempt. Perception has no effect on the motion plan.
- The actor result is saved before offline identity scoring. Privileged labels are recorded separately at acquisition for later evaluation.

### 2.2 Setup

| Setting | Value | Specification |
|---|---|---|
| Fresh executable and job | `9076cf8`, `52641957` | Clean compute checkout |
| Anatomy, seed, Stage 1 | `s0010`, `2026120500`, `cf7_00` | One development reset |
| Stage 1 search references | Home lateral index 133.0. L5 cranial-caudal index 122.831. L4 index 141.425. Ambiguous level interval 130.628–133.795 | §1 routing and §2.1 |
| Resources | 4 CPU cores, 1 NVIDIA L4 and 30 GB requested, `anest-dept` / `hpg-turin` | 2:04 elapsed, batch peak RSS 2742624 KiB, NVIDIA driver 580.178.04 |
| Versions and checkpoints | Same versioned SonoGym environment and job `52498821` checkpoints as §1 | Frozen v6r19 source protocol with the separately specified run seed and survey |

```bash
source_run=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_mbb_single_anatomy_poc_v6r19/52626449
experiment=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_mapping_20261124T204201Z
cd "$experiment/checkout"
sbatch --parsable --dependency=afterok:52641930 --kill-on-invalid-dep=yes \
  --deadline=2026-11-24T16:42:00 --chdir="$experiment/checkout" --output="$experiment/sweep-%j.log" \
  --export=ALL,SONOGYM_CONFIG_FILE="$source_run/context/machine-config.env",SIMPLE_MAPPING_RUN_ROOT="$source_run",SIMPLE_MAPPING_PROTOCOL="$source_run/context/s0010_mbb_single_anatomy_poc_protocol_v6r19.yaml",SIMPLE_MAPPING_OUTPUT_ROOT="$experiment" \
  hpg/jobs/s0010_simple_landmark_sweep.sbatch
# On the compute node, with the environment and Python path loaded by the wrapper:
$SONOGYM_ENV_PREFIX/bin/python workflows/two_stage_navigation/capture_simple_landmark_sweep.py \
  --protocol "$source_run/context/s0010_mbb_single_anatomy_poc_protocol_v6r19.yaml" \
  --seed 2026120500 --stage1-session cf7_00 --stage1-actor "$source_run/stage1/results/actor_results.json" \
  --output "$experiment/${SLURM_JOB_ID}/fresh_s0010_00" --headless --device cuda:0 \
  --evaluator-segmentation "$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
$SONOGYM_ENV_PREFIX/bin/python workflows/two_stage_navigation/evaluate_simple_landmark_sweep.py \
  --session "$experiment/${SLURM_JOB_ID}/fresh_s0010_00" \
  --protocol "$source_run/context/s0010_mbb_single_anatomy_poc_protocol_v6r19.yaml" \
  --label-conversion source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml
```

| Sample accounting | Longitudinal | Transverse | Total |
|---|---:|---:|---:|
| Attempted and acquired | 14 | 36 | 50 |
| Dropped | 0 | 0 | 0 |
| Final | 14 | 36 | 50 |

### 2.3 Results

| Landmark, ref `cf7_00`, seed `2026120500` | U-Net observations correct / total | Selected distinct views | Position in robot base, mm | Distance to visible interface, mm | Outcome |
|---|---:|---:|---|---:|---|
| RIGHT L4 TP | 14/15 | 4 | (−13.519, 535.624, 374.919) | 0.320 | MAPPED, correct |
| RIGHT L4 SAP | 4/4 | 1 | (−14.190, 505.650, 374.816) | 0.500 | CANDIDATE |
| RIGHT L5 TP | 5/5 | 2 | (15.153, 523.195, 372.219) | 0.000 | MAPPED, correct |
| RIGHT L5 SAP | 1/2 | 1 | (4.208, 513.529, 381.616) | 1.000 | CANDIDATE |

**Verdict. PARTIAL. The 50-view trajectory completed with 50 unique images and reached both physical imaging planes. Both TPs mapped correctly. Each SAP remained below the distinct-view requirement.** The run performed 2,559 action steps and the actor measured 64.721 s. The maximum projection residual was 0.000106 mm on 2,869 sampled pixels.

### 2.4 Reading it

- The executed angles were 1.561–1.588 rad in the longitudinal survey and 2.933–2.966 rad in the transverse survey.
- L4 SAP had four visible frames and all four had correct U-Net components. Its four predicted components lay in four separate clusters.
- L5 SAP had 12 visible frames. Four were routed to L4 and three had ambiguous level. Two of those three had a correct L5 proposal. Four other views were dominated by thresholded background and one supplied a usable correct proposal.
- The fresh Oracle diagnostic used the same 50 frames and the same aggregation. It mapped 3/4 landmarks. L4 SAP still had one distinct view in every cluster, which localizes the deficit to coverage and spatial support. L5 SAP had five distinct views in its correct oracle cluster, which localizes its deficit to U-Net perception and Stage 1 routing.
- Retain the simple mapper as the reference. Evaluate a predetermined change in SAP survey coverage and level routing on new seeds before reconsidering ranking rules.

### 2.5 Files

| File | Contents |
|---|---|
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_mapping_20261124T204201Z/52641957/fresh_s0010_00/waypoints.json` | All precomputed commands and scan families |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_mapping_20261124T204201Z/52641957/fresh_s0010_00/actor_result.json` | Acquisition counts and map hypotheses |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_mapping_20261124T204201Z/52641957/fresh_s0010_00/evaluation.json` | Archived fresh evaluation |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_mapping_20261124T204201Z/52641957/fresh_s0010_00/public/` | Images, segmentation masks and synchronized 6DoF poses |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_mapping_20261124T204201Z/52640447/review/fresh_oracle_diagnostic.json` | Offline oracle map and anatomy score for the fresh sweep, job `52642917` |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_mapping_20261124T204201Z/52640447/review/simple-landmark-mapping-review.tar.gz` | Replay, all figures, all offline scores and fresh comparison files |
