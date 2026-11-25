# SAP coverage and level assignment in the simple anatomical mapper

The experiment started at 03:31:28 UTC on 2026-11-25. It traces the two unresolved SAPs in the original `s0010` sweep. It tests a fixed sampling change and actor-visible level assignment on a matched reset and two predetermined development resets.

## Result

**The matched repair produced one correct four-landmark map with zero wrong landmarks.**

The original sweep observed L4 SAP in four distinct transverse views of spatially separated parts of the surface. Projecting the SAP observations before assigning the vertebral level recovered L5 SAP on the original recordings.

| Step | Measurement | Criterion | Result |
|---|---|---|---|
| §1 L4 SAP | Distinct relevant views and distance between adjacent visible interfaces | Two views in a 4 mm cluster | 4 views, four one-view clusters |
| §1 L5 SAP | Accounting for the five selected Oracle views | Explain eligibility under frozen U-Net and routing | All five traced |
| §1 Routing replay | Correct mapped landmarks on the original 50 frames | Improve over 2/4 with zero wrong mappings | 3/4, zero wrong |
| §2 Fresh sweeps | Correct mapped landmarks and wrong mappings | Four correct and zero wrong per map | 8 / 12 correct, 1 / 12 wrong across three first-revision seeds |
| §3 Matched repair | Four-landmark map on the opened seed | Four correct and zero wrong | Pass, 4 / 4 correct |
| §4 Demo | Synchronized sweep replay | Full MP4 decode | Pass, 150 / 150 frames |

## 1. Original sweep — where was the SAP evidence lost?

`diagnose_simple_sap.py` recomputes both checkpoint predictions and traces the five views selected by the original L5 SAP Oracle cluster. It also scores a routing-only replay with the 50 old frames and the unchanged aggregation.

**Criterion.** Account for every selected Oracle view at its first exclusion in the original actor and distinguish later failures revealed by the counterfactual L5 inference. Require unchanged two-view support, 4 mm clustering and correct identity to credit a landmark in the routing-only replay.

### 1.0 Quantities

| Quantity | Definition | Source |
|---|---|---|
| Map and medoid | [Simple anatomical landmark mapping](../notation.md#simple-anatomical-landmark-mapping) | `simple_map` |
| Acceptance and identity | [Anatomical mapping status](../notation.md#anatomical-mapping-status) | `simple_map`, `score_map` |
| Localization | [Distance to the visible interface](../notation.md#distance-to-the-visible-interface) | `score_map` |

### 1.1 Method

- Verify each pose/image join and the public projection. Construct provisional visible interfaces using only evaluator pixels at the first observed cortical hit. Score within 3 pixels as in the original experiment.
- Compare complete projected visible L4 SAP pixel sets between adjacent views. Distinct-view thresholds are 2 mm translation or 1° rotation.
- For each selected L5 Oracle view, record both U-Nets' strongest SAP pixel within 3 pixels of the provisional interface, its winning class, and the sizes of eight-connected eligible SAP components in that neighborhood.
- Both L4 and L5 SAP require a winning SAP pixel with probability at least 0.90. Minimum connected size is three pixels for L4 and two for L5.
- The routing-only replay applies §2.1 SAP assignment to the original acquisitions. Its TP inputs are the original actor's serialized observations.

### 1.2 Setup

| Setting | Value | Specification |
|---|---|---|
| Original run | `52641957`, seed `2026120500`, `cf7_00`, 50 / 50 valid images | Revision `9076cf87997600046707b7f48b210d5285e4c03d` |
| Frozen mapper reference | `61779bfa7183f607a86a2fb78a33b39ab3f26e19` | Prior replay job `52640447`, 15 / 16 correct, 0 / 16 wrong |
| Diagnostic jobs | `52692440`, then `52693340` | 2 CPU cores each, `anest-dept`, `hpg-default` |
| Models | L4 and L5 U-Nets trained in `52498821` | Same checkpoint hashes recorded in the source protocol |
| Analyzed L4 SAP frames | 46, 47, 48, 49 | All four visible and physically distinct |
| Analyzed L5 SAP frames | 0, 1, 21, 31, 43 | All five supports of the selected Oracle cluster |

```bash
original=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_mapping_20261124T204201Z/52641957/fresh_s0010_00
source_run=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_mbb_single_anatomy_poc_v6r19/52626449
results=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_mapping_sap_20261125T033128Z
# In the experimental worktree on a scheduled CPU node:
"$SONOGYM_ENV_PREFIX/bin/python" workflows/two_stage_navigation/diagnose_simple_sap.py \
  --session "$original" \
  --protocol "$source_run/context/s0010_mbb_single_anatomy_poc_protocol_v6r19.yaml" \
  --output "$results/sap_evidence.json"
```

### 1.3 Results

| Adjacent original L4 SAP frames | Probe translation, mm | Closest visible-interface pixels in 3D, mm | Largest-component representative distance, mm |
|---|---:|---:|---:|
| 46–47 | 8.469 | 5.297 | 6.749 |
| 47–48 | 9.061 | 4.124 | 4.273 |
| 48–49 | 6.582 | 4.200 | 4.507 |

| Original L5 SAP Oracle frame | Original level | Peak L5 SAP probability within 3 px | First exclusion and counterfactual result |
|---|---|---:|---|
| 00 | L5 | 0.0000020 | `MISSED_BY_UNET`. Background wins throughout the visible SAP pixels. |
| 01 | L5 | 0.8630 | `REJECTED_BY_CONFIDENCE`. The SAP peak is below 0.90. |
| 21 | L4 | 0.8048 | `WRONG_VERTEBRAL_LEVEL`. L5 would also fail confidence. |
| 31 | L4 | 0.9636 | `WRONG_VERTEBRAL_LEVEL`. L5 has one above-threshold SAP pixel in the nearby component and requires two. |
| 43 | L5 | 0.999996 | Retained. The accepted component has 34 pixels, mean confidence 0.995, and a representative 2 px from the visible interface. |

**Verdict.** All five original Oracle supports are accounted for. With revised routing alone, L5 SAP maps correctly with two distinct views and L4 SAP stays a one-view candidate. The replay has three correct landmarks, zero wrong and one unresolved.

### 1.4 Reading it

- All four L4 SAP observations came from the outermost transverse pass, at lateral command index about 150 and cranial-caudal indices 138.44–146.77. Each occupied a separate cluster. Four correct L4 U-Net predictions also occupied four clusters.
- The projected closest distances exceed 4 mm even for the full labeled interfaces. Repositioning the representative pixel within its component cannot join those adjacent pairs at the frozen radius. Interleaving new slices tests closer repeat coverage of the extended SAP surface.
- All five L5 frames passed acquisition and were mutually distinct in the Oracle cluster. Four were lost before clustering.
- Projecting these L5 SAP pixels puts them at cranial-caudal indices 133.96–135.21. The original center-based rule used an L4/L5 ambiguity band of 130.63–133.79, which excludes L5 in frames 21 and 31 even though the SAP is visible.
- The five Oracle supports identify specific near-misses, not a complete inventory of usable L5 SAP evidence. The routing-only replay can also use a correct SAP component from a previously ambiguous frame outside that five-frame subset.

### 1.5 Files

| File | Contents |
|---|---|
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_mapping_sap_20261125T033128Z/sap_evidence.json` | Full per-frame poses, images by SHA-256, provisional pixel indices, probabilities, classes, components and projected points |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_mapping_sap_20261125T033128Z/original_sap_diagnosis.json` | First offline pass, before the routing-only replay |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_mapping_20261124T204201Z/52641957/fresh_s0010_00` | Immutable original acquisitions and final score |

## 2. Fresh sweeps — can the revised acquisition and routing produce a complete map?

`capture_simple_landmark_sweep.py` executes the frozen scan profile. `evaluate_simple_landmark_sweep.py` applies the same simulator identity and localization evaluation.

**Criterion.** A complete map has all four correct landmarks with at least two mutually distinct views under the established mapper. Acquisition must be valid, synchronized and cover both planes. Any wrong map outcome remains a failure.

### 2.1 Method

- Preserve the original two scan families and their order. Interleave four midpoints only between original outer transverse waypoints whose midpoints fall within six command indices of the Stage 1 L4 reference. Each inserted command satisfies the existing pairwise tracking tolerance check.
- Infer with both existing U-Nets on valid frames. TP retains the original frame-level assignment and thresholds.
- Fit the cranial direction using the calibrated image origin at row zero and the center column from valid executed transverse poses, removing the lateral component. Scale by the existing 1.5 mm per command index. Accept the calibration only if the maximum command-index residual is at most half the original scan spacing, 1.5 indices.
- Score the projected SAP representative along that direction. The Stage 1 L5 and L4 reference positions delimit the superior interval used for L5 SAP. L4 SAP lies cranial to the L4 reference. Preserve uncertain assignments near either reference using the original midpoint-band half width plus the calibration residual. Only a matching checkpoint and a unique position-based level may contribute a view.
- Keep the checkpoints, semantic acceptance, 4 mm complete-link clustering, distinct-view separation and abstention rule fixed. The original sweep is available with no revision flag.

### 2.2 Setup

| Setting | Value | Specification |
|---|---|---|
| Frozen execution revision | `ae3ba30b789f3129192451c4626b262648bd4df0` | Job `52693396` |
| Profile | `s0010_simple_landmark_sap_development_v1.yaml` | SHA-256 `fa9cc6fe4d152e6011a89458f7f8b10d51763b50b66f41779ebb0be276689557` |
| Source protocol | Frozen v6r19 geometry, original mapper settings and checkpoints | Same as §1 |
| Seeds and reference | `2026120500` matched, `2026120501` and `2026120502` new, all with actor Stage 1 `cf7_00` | All development on `s0010` |
| Resource request | Four CPUs, one NVIDIA L4, 30 GB, 13-minute job bound | `anest-dept`, `hpg-turin` |
| Scoped tests | 37 local checks, then 24 job preflight tests | Pass |

```bash
source_run=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_mbb_single_anatomy_poc_v6r19/52626449
results=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_mapping_sap_20261125T033128Z
protocol="$source_run/context/s0010_mbb_single_anatomy_poc_protocol_v6r19.yaml"
revision=experiments/unity_two_stage_transfer/s0010_simple_landmark_sap_development_v1.yaml
# From the experimental worktree, submit after diagnostic job 52693340 completes:
sbatch --dependency=afterok:52693340 --kill-on-invalid-dep=yes \
  --deadline=2026-11-24T23:18:00 --output="$results/sweep-%j.log" \
  --export=ALL,SONOGYM_CONFIG_FILE="$source_run/context/machine-config.env",SIMPLE_MAPPING_RUN_ROOT="$source_run",SIMPLE_MAPPING_PROTOCOL="$protocol",SIMPLE_MAPPING_OUTPUT_ROOT="$results" \
  hpg/jobs/s0010_simple_landmark_sap_development.sbatch
# The wrapper runs each fixed seed sequentially and enables the robot camera only for 2026120500:
"$SONOGYM_ENV_PREFIX/bin/python" workflows/two_stage_navigation/capture_simple_landmark_sweep.py \
  --protocol "$protocol" --revision "$revision" --stage1-session cf7_00 \
  --stage1-actor "$source_run/stage1/results/actor_results.json" --seed "$seed" \
  --output "$results/${SLURM_JOB_ID}/fresh_s0010_$seed" --headless --device cuda:0 \
  --evaluator-segmentation "$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz" \
  "${camera_args[@]}"
```

| Session | Longitudinal valid / attempted | Transverse valid / attempted | Total valid |
|---|---:|---:|---:|
| 2026120500 | 14 / 14 | 40 / 40 | 54 / 54 |
| 2026120501 | 14 / 14 | 40 / 40 | 54 / 54 |
| 2026120502 | 14 / 14 | 40 / 40 | 54 / 54 |

### 2.3 Results

Each landmark entry gives the provisional identity outcome, selected distinct views, and distance to the nearest visible interface in millimeters.

| Outcome | Original 2026120500 | Revised 2026120500 | Revised 2026120501 | Revised 2026120502 |
|---|---|---|---|---|
| RIGHT L4 TP | Correct · 4 · 0.320 | Ambiguous · 4 · 0.320 | Ambiguous · 4 · 0.000 | Correct · 7 · 0.307 |
| RIGHT L4 SAP | Candidate · 1 · 0.500 | Correct · 3 · 1.000 | Correct · 3 · 0.500 | Ambiguous · 2 · 0.500 |
| RIGHT L5 TP | Correct · 2 · 0.000 | Correct · 2 · 0.000 | Wrong · 2 · 1.145 | Correct · 3 · 0.000 |
| RIGHT L5 SAP | Candidate · 1 · 1.000 | Correct · 2 · 1.000 | Correct · 4 · 1.000 | Correct · 3 · 0.697 |
| Correct / wrong / unresolved | 2 / 0 / 2 | 3 / 0 / 1 | 2 / 1 / 1 | 3 / 0 / 1 |
| Complete map | No | No | No | No |
| Oracle correct / complete | 3 / No | 4 / Yes | 3 / No | 3 / No |
| Actor elapsed time after app launch, s | 64.721 | 113.936 | 40.078 | 40.035 |
| Mapping only, ms | Uninstrumented | 3.163 | 3.566 | 3.534 |

**Verdict. The first revision improved SAP coverage and mapped L5 SAP on all three starts. It produced zero complete actor maps and one wrong mapping among twelve landmark attempts.** The revised batch finished `0:0` in 5:58, including all simulator startups and evaluations. The matched seed had robot camera recording enabled and took about 225 s from actor log creation to its evaluation file.

### 2.4 Reading it

- On the matched seed, L4 SAP improved from one to three distinct views and L5 SAP improved from one to two. The matched Oracle produced a complete map.
- L4 SAP was unresolved on seed `2026120502` in both actor and Oracle maps due to equal support in competing clusters.
- The wrong L5 TP on seed `2026120501` had one correct view among two. Its view distances were 1.000 and 8.062 pixels. The small nearest-surface distance alone did not establish identity.
- Actor and native projection agreed within 0.000120 mm in all three sessions. All projection checks passed.
- The matched L4 TP initially occupied a selected cluster supported by frames 13, 11, 9 and 10, all within the identity gate. In the revised sweep, the competing cluster used frames 47, 28, 48 and 24. Frame 48 was an added SAP waypoint. The two clusters tied at four views and differed in confidence by 0.00159, below the existing 0.01 abstention margin.

### 2.5 Files

| File | Contents |
|---|---|
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_mapping_sap_20261125T033128Z/52693396/attempts.tsv` | All three attempted seeds and execution outcomes |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_mapping_sap_20261125T033128Z/52693396/fresh_s0010_<seed>/evaluation.json` | Per-landmark counts, pose support, error and visibility diagnostics |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_mapping_sap_20261125T033128Z/52693396/fresh_s0010_<seed>/oracle_diagnostic.json` | Oracle comparison on the same executed images |

## 3. Matched repair — does holding TP acquisition fixed preserve both TPs?

The repaired profile keeps the four supplemental views for SAP and credits TP observations only at the original planned waypoints. It is a single opened-development repeat on `2026120500`.

**Criterion.** On the matched seed, meet the same 4/4 map criterion with both SAP improvements and zero wrong mapped landmarks.

### 3.1 Method

- Replay the original-waypoint TP selection on the first revised session to test the source of the tie. Preserve the same TP source selection as the original planned sweep and retain the earlier all-frame outcome in the record.
- Freeze the repaired profile before running. Acquire all 54 waypoints again. Keep U-Net predictions at the four extra views for inspection and SAP mapping. Exclude them only from TP evidence.
- Preserve the original mapper, classifier thresholds, calibrated projection and evaluator definitions.

### 3.2 Setup

| Setting | Value | Specification |
|---|---|---|
| Repair executable | `a53cf7be668a7aa47918511eaefeae5fc8d3bfb3` | Job `52694480` |
| Profile | `s0010_simple_landmark_sap_development_v2.yaml` | SHA-256 `3fa38f95921c506cdb683eef6d550a68ec91d06dc61c0eaca595a45d8b03c91b` |
| Seed and reference | `2026120500`, `cf7_00` | Opened development, same anatomy and U-Nets |
| Resource request | Four CPUs, one NVIDIA L4, 30 GB, 10-minute job bound | `anest-dept`, `hpg-turin` |

```bash
revision=experiments/unity_two_stage_transfer/s0010_simple_landmark_sap_development_v2.yaml
# From the same worktree and environment as §2.2:
sbatch --time=00:10:00 --deadline=2026-11-24T23:18:00 --output="$results/repair-%j.log" \
  --export=ALL,SONOGYM_CONFIG_FILE="$source_run/context/machine-config.env",SIMPLE_MAPPING_RUN_ROOT="$source_run",SIMPLE_MAPPING_PROTOCOL="$protocol",SIMPLE_MAPPING_OUTPUT_ROOT="$results",SIMPLE_MAPPING_SAP_REVISION="$revision" \
  hpg/jobs/s0010_simple_landmark_sap_development.sbatch
```

| Session | Longitudinal valid / attempted | Transverse valid / attempted | Total valid |
|---|---:|---:|---:|
| Matched repair 2026120500 | 14 / 14 | 40 / 40 | 54 / 54 |

### 3.3 Results

| Outcome | Original 2026120500 | First revision 2026120500 | Matched repair 2026120500 |
|---|---|---|---|
| Valid acquired frames | 50 / 50 | 54 / 54 | 54 / 54 |
| RIGHT L4 TP | Correct · 4 views | Ambiguous · 4 views | Correct · 4 views · 0.320 mm |
| RIGHT L4 SAP | Candidate · 1 view | Correct · 3 views | Correct · 3 views · 1.000 mm |
| RIGHT L5 TP | Correct · 2 views | Correct · 2 views | Correct · 2 views · 0.000 mm |
| RIGHT L5 SAP | Candidate · 1 view | Correct · 2 views | Correct · 2 views · 1.000 mm |
| Correct mapped landmarks | 2 / 4 | 3 / 4 | 4 / 4 |
| Wrong mapped landmarks | 0 / 4 | 0 / 4 | 0 / 4 |
| Unresolved landmarks | 2 / 4 | 1 / 4 | 0 / 4 |
| Complete map | No | No | Yes |
| Actor elapsed time after app launch | 64.721 s | 113.936 s | 101.869 s |
| Job time | 124 s, one seed | 358 s, three seeds | 200 s, one seed |

**Verdict. PASS on the opened matched seed.** All four landmarks were correct with the same support, confidence and identity thresholds. The projection maximum residual was 0.000106 mm across 3,146 sampled unclipped pixels. The unchanged mapper took 3.075 ms.

### 3.4 Reading it

- All 54 first-revision and repair image hashes, executed relative poses, and recorded 6DoF positions and quaternions matched exactly. The outcome change isolates TP evidence selection on these acquisitions. The original sweep and the repair share 50 requested waypoints and 46 identical image hashes.
- The original-waypoint replay on the first revised recordings yielded 4/4, 3/4, and 3/4 correct in seed order. The wrong L5 TP persisted on seed 2026120501.
- The supplemental views were frames 46, 48, 50 and 52. All 54 views retained their SAP evidence. The selected L4 TP returned to the original four-view cluster. Each of the four supports met the provisional identity gate.
- The repair stopped after one repeat of the matched seed. The two new development seeds at the first revision remain part of the result and establish no evidence for repeatability of the repaired profile.
- The original 816-frame comparison entry point is unchanged. An AST comparison against `61779bf` confirmed that each original mapping, projection, observation extraction and sweep definition is unchanged. The original 15/16 and 0 wrong results remain archived. The 816-frame experiment was not rerun.
- The next comparison should freeze the repaired acquisition/evidence schedule on untouched seeds, and test the L4 SAP ambiguity and L5 TP semantic failure separately.

### 3.5 Files

| File | Contents |
|---|---|
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_mapping_sap_20261125T033128Z/52694480/fresh_s0010_2026120500/actor_result.json` | Four mapped landmarks, observation provenance, executed Stage 1 fit and mapping time |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_mapping_sap_20261125T033128Z/52694480/fresh_s0010_2026120500/evaluation.json` | Correctness, support, localization error and image/pose verification |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_mapping_sap_20261125T033128Z/52694480/fresh_s0010_2026120500/public` | 54 ultrasound/pose acquisitions and 55 robot pictures including reset |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_mapping_sap_20261125T033128Z/review/diagnostic_examples/tp_original_waypoints_diagnostic.json` | Post hoc replay of TP evidence from the original planned waypoints |

## 4. Demo — what did the robot acquire and map?

`render_simple_sap_demo.py` renders the actor replay from actual synchronized acquisitions. `render_simple_sap_evidence.py` produces separate evaluator examples and repeated-position diagnostics.

**Criterion.** Decode all MP4 frames at positive duration. Verify each source image hash and the pose join. Show the matched repair's actual final map with separate offline identity scoring.

### 4.1 Method

- Use the prespecified demonstration seed `2026120500` after the matched repair. The intro shows the robot at reset and the Stage 1 references used to plan the first waypoint.
- Show the actual robot pictures, ultrasound images, both U-Net masks, executed pose and accumulating map at each acquisition. The final card gives the offline evaluation separately.
- Place provisional visible cortical pixels and classifier peaks in a separate figure. Repeated views at original waypoints 31 and 43 show reset, tracking and image variability. The comparison records realized pose differences and image hashes.

### 4.2 Setup

| Setting | Value | Specification |
|---|---|---|
| Demo executable | `68c94608b82486375f6a5a6af1d26ad3b891a4a5` | CPU job `52695837` |
| Video source | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_mapping_sap_20261125T033128Z/52694480/fresh_s0010_2026120500` | 54 valid captures and the reset view |
| Output format | 1600 × 900 MP4 at 6 fps | OpenCV encode/decode and matched image hashes |

```bash
"$SONOGYM_ENV_PREFIX/bin/python" workflows/two_stage_navigation/render_simple_sap_demo.py \
  --session "$results/52694480/fresh_s0010_2026120500" --protocol "$protocol" \
  --output "$results/review/matched-repair/visualization"
```

### 4.3 Results

| Verification | Measured |
|---|---|
| Source ultrasound and executed-pose joins | 54 / 54 valid acquisitions |
| Synchronized robot camera | 54 / 54 acquisition pictures and one reset picture |
| MP4 decode | 150 / 150 frames at 6 fps, 25.0 s |
| Resolution | 1600 × 900 |
| CPU job and archive | 87 s on two CPUs, checksum and tar contents verified |
| Actor map shown | 4 / 4 mapped |

**Verdict. Pass.** All frames decoded. The video uses the matched repair's synchronized acquisitions and shows its final map. The example images were inspected for legible labels and correct actor/evaluator separation.

### 4.4 Reading it

- The video displays acquired views in source order with interim maps rendered by the unchanged mapper.
- The fixed script selected the matched seed for the camera and the video. Static maps and evaluation records retain both new seeds and the first matched attempt.
- At requested waypoint 31, the matched L5 checkpoint produced no accepted SAP pixels. New seeds produced 11 and 10 with executed pose offsets of 0.133 and 0.022 mm and rotation offsets of 0.041° and 0.004°. The different images and near-matched poses show sensitivity at this view. The comparison does not isolate speckle from other reset effects.
- The matched repair still requires testing on unused development seeds. The experiment establishes anatomical mapping only under the provisional simulator score.

### 4.5 Files

| File | Contents |
|---|---|
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_mapping_sap_20261125T033128Z/review/matched-repair/visualization/revised_systematic_sweep.mp4` | Verified fresh acquisition demo |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_mapping_sap_20261125T033128Z/review/matched-repair/visualization/final_actor_map.png` | Final 3D map and executed sweep |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_mapping_sap_20261125T033128Z/review/diagnostic_examples/` | Five-frame L5 trace figure, repeated-pose comparisons and original-TP waypoint replay |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_mapping_sap_20261125T033128Z/simple-landmark-sap-review.tar.gz` | Maps, video, diagnosis and results of every attempted seed |
