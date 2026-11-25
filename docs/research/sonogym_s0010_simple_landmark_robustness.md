# Robustness of simple anatomical mapping on s0010

Four interventions were measured independently, using previously inspected sweeps as development evidence. The capture and scoring definitions remained fixed, and Stage 2 reused the Stage 1 actor reference `cf7_00`.

## Result

**The L5 pilot improved map correctness, but no configuration consistently produced a complete four-landmark actor map.** Denser scanning improved visible coverage and support for L4 SAP. Stabilization reduced tracking error. The surface mapper improved L5 TP on the archived sweeps and lost one L4 TP. The controlled renderer separated spatial speckle from probe geometry.

| Experiment | Valid/planned frames | Correct landmarks | Wrong | Unresolved | Complete maps | Runtime |
|---|---:|---:|---:|---:|---:|---|
| E0 | 216/216 | 10/16 | 2 | 4 | 0/4 | 8.43 ms mapping |
| E1-S | 108/108 | 5/8 | 1 | 2 | 0/2 | 129.6 s acquisition, 3.80 ms mapping |
| E1-M | 204/204 | 4/8 | 1 | 3 | 0/2 | 207.7 s acquisition, 4.99 ms mapping |
| E1-D | 298/300 | 6/8 | 1 | 1 | 0/2 | 292.2 s acquisition, 5.81 ms mapping |
| E2 stabilized | 108/108 | 5/8 | 2 | 1 | 0/2 | 144.1 s acquisition, 3.91 ms mapping |
| E3 fine-tuned L5 | 216/216 | 12/16 | 1 | 3 | 1/4 | 15.90 ms mapping |
| E4 surface | 216/216 | 11/16 | 1 | 4 | 0/4 | 75.59 ms mapping |
| E1-M + E4 | 204/204 | 4/8 | 1 | 3 | 0/2 | 207.7 s acquisition, 120.47 ms mapping |
| E1-D + E4 | 298/300 | 5/8 | 1 | 2 | 0/2 | 292.2 s acquisition, 200.97 ms mapping |
| E1-M + E3 | 204/204 | 6/8 | 1 | 1 | 0/2 | 207.7 s acquisition, 9.76 ms mapping |
| E1-D + E3 | 298/300 | 5/8 | 1 | 2 | 0/2 | 292.2 s acquisition, 12.72 ms mapping |
| E0 Oracle point | 216/216 | 8/16 | 0 | 8 | 0/4 | 19.09 ms mapping |
| E4 Oracle surface | 216/216 | 8/16 | 0 | 8 | 0/4 | 581.03 ms mapping |
| E3 fine-tuned L5 + surface | 216/216 | 12/16 | 0 | 4 | 0/4 | 175.73 ms mapping |
| E1-M all TP | 204/204 | 4/8 | 0 | 4 | 0/2 | 207.7 s acquisition, 6.89 ms mapping |
| E1-D all TP | 298/300 | 6/8 | 0 | 2 | 0/2 | 292.2 s acquisition, 11.18 ms mapping |
| E1-M all TP, Oracle | 204/204 | 8/8 | 0 | 0 | 2/2 | 207.7 s acquisition, 18.16 ms mapping |
| E1-D all TP, Oracle | 298/300 | 5/8 | 0 | 3 | 0/2 | 292.2 s acquisition, 27.04 ms mapping |

The E0, E3, and E4 replay rows use all four old confirmation seeds. Acquisition time runs through the last valid capture, while mapping time covers association from saved observations. Model inference is excluded from replay mapping time. The E1, E2, and combined rows use seeds `2026121700` and `2026121702`. Complete maps require valid acquisition of all planned frames and four correct identities under the frozen evaluator.

## 1. Acquisition density

`capture_simple_landmark_sweep.py` generated the sweep. `evaluate_robust_mapping.py` scored the nested acquisition budgets with separate SAP level fits per subset.

**Criterion.** Compare the original point mapper under the frozen evaluator. Track support and visible coverage in addition to correctness. A complete map requires four correct identities and full capture integrity.

### 1.1 Method

- E1-S used the 54 original poses. E1-M added 48 transverse views on four lateral lanes. E1-D added another 48 views on four lanes.
- The 54 poses form the prefix of every capture. Its 108 recorded image hashes exactly reproduce the corresponding images archived in E0. The offline subsets use the same actual poses and images. Only the 150-target trajectories were executed as standalone dense sweeps.
- Primary comparisons kept the original 50-waypoint TP eligibility. Secondary comparisons let TP use all the views. Each U-Net/Oracle comparison used the same frames and eligibility policy. Oracle labels use simulator anatomy and the same visible-interface test as the frozen evaluator.

### 1.2 Setup

| Field | Value |
|---|---|
| Source capture | E0 Slurm `52699995`, seed set `2026121700:2026121703`, execution `3dceed443fde53d218883a74a36cea7d9ece2d68` |
| New capture | Slurm `52710240`, seeds `2026121700,2026121702`, execution `6230928a0fc4d972c15030ed6e4b9a1ac7f02983` |
| Versioned profile | `experiments/unity_two_stage_transfer/s0010_simple_landmark_robustness_v1.yaml`. SHA-256 `b234c69deb85f709faf4201b1f1d83405ba1429c96bea23b3c091b20eda11ac3` |
| Frozen E0 | SHA-256 `2c4e60945c38b6d501a1fa1e1134c7118fe915a2615e77fd0f32788110fdc669` |
| Source protocol | SHA-256 `c26802d61f211c62acd16fe12208183e3442ebf776e9188d867a77126c6a114e` |
| L4 and L5 checkpoints | Job `52498821`. SHA-256 `c6b7ecc9bc245a88ccc0a5db31dbfd919b57407988dde5ea8211dd7eea2c1b2a` and `ca13c8fb6bf99bd363dd87c5e30dc1eeb5b22da5cf3f87e991893fad96b41acf` |
| Association and identity evaluation | Association radius 4 mm. Distinct viewpoint separation 2 mm or 1 degree, minimum two viewpoints, equal-support confidence margin 0.01. Evaluator requires a strict majority within 3 px of the corresponding visible interface. |
| Input and execution | Exact paths and main commands in the artifact index in §5.5. The frozen files and resolved per-job contexts are retained with the captures. |

```bash
python workflows/two_stage_navigation/capture_simple_landmark_sweep.py --protocol "$protocol" --revision "$reference/context/session_00.yaml" --robustness-arm E1 --robustness-profile "$profile" --stage1-session cf7_00 --stage1-actor "$stage1" --seed 2026121700 --output "$capture/E1_2026121700" --headless --device cuda:0 --record-robot --enable_cameras --evaluator-segmentation "$segmentation"
python workflows/two_stage_navigation/evaluate_robust_mapping.py --protocol "$protocol" --profile "$profile" --reference "$reference" --label-conversion "$conversion" --acquisition "$capture" --extended-tp --output "$review/mapping"
```

| Acquisition budget | Planned | Failed motion | Final valid |
|---|---:|---:|---:|
| E1-S | 108 | 0 | 108 |
| E1-M | 204 | 0 | 204 |
| E1-D | 300 | 2 | 298 |

### 1.3 Results

| E1 budget | TP policy | U-Net correct/wrong/unresolved | Oracle correct/wrong/unresolved | Oracle complete |
|---|---|---|---|---|
| E1-S | Original 50 only | 5/1/2 | 4/0/4 | 0/2 |
| E1-M | Original 50 only | 4/1/3 | 7/0/1 | 1/2 |
| E1-D | Original 50 only | 6/1/1 | 4/0/4 | 0/2 |
| E1-M-TP-all | All available poses | 4/0/4 | 8/0/0 | 2/2 |
| E1-D-TP-all | All available poses | 6/0/2 | 5/0/3 | 0/2 |

**Verdict.** L4 SAP improved on the denser sweeps, with no complete actor map. Both 150-target runs missed waypoint 102 after 360 actions when reduced angular error remained 1.166 or 1.157 degrees. E1-M acquired every planned frame.

### 1.4 Reading it

- L4 SAP had 6/54 oracle-visible frames in each sparse case. The counts rose to 12/102 and 11/102 in E1-M, and to 18/149 and 17/149 in E1-D.
- L4 SAP selected support went from 2/2 distinct viewpoints in the sparse cases to 4/4 in medium and 5/6 in dense, listed per seed. Focusing on mapped L4 SAP, the dense maps stayed correct in 8 of 11 leave-one-view-out removals, while the only correct sparse map stayed correct in 0 of 2.
- Allowing TP evidence from the new viewpoints made the medium oracle correct in both sessions. The same U-Net configuration resolved four correct landmarks in total and left four unresolved. The geometry in these sweeps supports more than the learned masks reliably recover.
- The medium point mapper lost the reliable L5 SAP in both sessions. The dense original-TP variant kept both L5 SAP, while one L5 TP remained wrongly mapped.

### 1.5 Files

| File | Contents |
|---|---|
| `review-52711211/mapping/results.json` | Separate point/Oracle maps, actual coverage, motion failures, runtimes, supporting frames, and leave-one-view-out results |
| `diagnostics-52711846/sweep_trajectories.png` | Commanded and executed acquisition paths in anatomy coordinates |

## 2. Pose stabilization

The comparison used the same two starts, speckle seeds, checkpoints, 54 planned waypoints, and point mapper.

**Criterion.** Compare absolute translation error and rotation geodesic from the nominal surface-constrained command, and compare anatomical correctness and exclusions using the frozen evaluator.

### 2.1 Method

- Use the same planner to compute each nominal surface-relative 6DoF target independently of the live controller state. Measure the synchronized executed pose against this transform.
- E2 used the original controller with 1 mm and 0.5 degree acceptance in the reduced command tracking coordinates, six zero-command action steps, then one bounded retry if needed. Projection used the executed pose.

### 2.2 Setup

| Field | Value |
|---|---|
| Baseline | E1-S, matching original-controller capture prefixes |
| Stabilized | E2, job `52710240`, same two seeds and original 54-waypoint plan |
| Reproduction | The command in §1.2 with `--robustness-arm E2` and a separate run directory |
| Sample accounting | 108/108 valid in each controller arm, zero drops |

### 2.3 Results

| Arm | Seed | Position mean / median / p95 / max (mm) | Rotation mean / median / p95 / max (deg) |
|---|---|---|---|
| E1-S | 2026121700 | 2.641 / 2.008 / 7.312 / 7.545 | 1.839 / 1.412 / 4.580 / 5.699 |
| E1-S | 2026121702 | 2.659 / 2.043 / 7.512 / 9.413 | 1.869 / 1.386 / 4.575 / 5.992 |
| E2 | 2026121700 | 1.880 / 1.229 / 6.003 / 6.979 | 1.677 / 1.308 / 4.368 / 5.416 |
| E2 | 2026121702 | 1.881 / 1.195 / 6.072 / 6.993 | 1.695 / 1.307 / 4.369 / 5.901 |

| Between-seed spread across 54 matched acquisitions | Position mean / median / p95 / max (mm) | Rotation mean / median / p95 / max (deg) |
|---|---|---|
| E1 | 0.324 / 0.101 / 1.196 / 2.769 | 0.130 / 0.037 / 0.679 / 1.081 |
| E2 | 0.185 / 0.056 / 0.926 / 1.878 | 0.077 / 0.015 / 0.403 / 1.182 |

**Verdict.** E2 lowers full-pose tracking error but leaves 5/8 mapped correctly and increases wrong mapped identities from 1/8 to 2/8. The proposed full 6DoF tolerances were exceeded at many valid acquisitions.

### 2.4 Reading it

- E2 produces 3/4 and 2/4 correct in the two seeds. The second seed gains a wrong L4 SAP and retains the wrong L5 TP.
- The two sweeps execute 3,041 and 3,098 actions, compared with 2,531 and 2,496 for the matched control prefixes. Some six-step holds move the end effector more than 3 mm, which indicates the physical pose is still settling.
- Oracle correctness increased from 4/8 to 6/8. Actor correctness remained unchanged despite the additional oracle evidence.

### 2.5 Files

| File | Contents |
|---|---|
| `diagnostics-52711846/matched_probe_pose_errors.png` | Full position and orientation errors at each matched waypoint |
| `acquisition-52710240/attempts.tsv` | Every acquisition and evaluation attempt |

## 3. Speckle and L5 U-Net pilot

`render_robust_speckle.py` replayed explicit transforms through the same convolutional ultrasound simulator. `finetune_robust_l5.py` used a fixed three-epoch experiment.

**Criterion.** An independent renderer intervention must preserve sliced geometry when only its speckle fields change, and reproduce both geometry and image when pose and field repeat. Compare the two checkpoints on the same held-out validation frames and frozen acceptance thresholds.

### 3.1 Method

- Render with separate Torch generators for the simulator's two speckle volumes. Reuse the saved acquisition transform as input and regenerate provisional visible-interface labels for each image.
- Pool 78 nominal poses and 78 bounded 0.5 mm/0.5 degree renderer perturbations in the pilot training set, with three fields per pose. The 44 held-out poses come from different trajectories and a disjoint lateral neighborhood with a 4 mm exclusion buffer. Two further fields cover the validation poses.
- Fine-tune only L5 and retain geographically separated original training examples. Evaluate held-out lateral subsets and run a separate replay of the four original 54-frame sessions, which are development evidence. Model acceptance thresholds and the L4 checkpoint remain frozen.

### 3.2 Setup

| Field | Value |
|---|---|
| Renderer and training | `52711211`, execution `7f53bdb481a4ff9ff66d2301894011f6d354276a` |
| Dataset | `review-52711211/E3-dataset/dataset.npz` and `split_manifest.json` |
| Training sources | Original dataset job `52497655`, original checkpoints and L5 split job `52498821` |
| Optimizer | AdamW, learning rate 0.00003, batch size 8, fixed 3 epochs, seed 393999 |
| Sample accounting | 468 novel training images and 88 validation images, all accepted by the contact proxy and validity checks |
| Minimum train-to-validation distance | 4.428 mm for newly rendered pairs. 32/44 validation poses lie within 4 mm of original checkpoint training poses. Pose neighborhoods are held out from the pilot fine-tune, and the new validation speckle fields are disjoint. |

```bash
python workflows/two_stage_navigation/render_robust_speckle.py --profile "$profile" --protocol "$protocol" --reference "$reference" --acquisition "$capture" --label-conversion "$conversion" --output "$review/E3-dataset" --headless --device cuda:0
python workflows/two_stage_navigation/finetune_robust_l5.py --profile "$profile" --protocol "$protocol" --dataset "$review/E3-dataset" --original-dataset "$original_data" --original-manifest "$original_manifest" --output "$review/E3-training"
```

### 3.3 Results

| At original L5 TP waypoint 5 | Image MAE relative to reference | Visible L5 TP pixels |
|---|---:|---:|
| Fixed pose, new field | 0.321738 | 9 |
| New pose, fixed field | 0.032545 | 0 |
| New pose and field | 0.327921 | 9 |
| Repeat same pose and field | 0, bitwise identical | 0 |

| Matched renderer intervention | Split | Paired geometry entries | Mean native intensity MAE | TP visibility changed | SAP visibility changed |
|---|---|---:|---:|---:|---:|
| same pose different fields | train | 78 | 0.3260 | 11/78 | 2/78 |
| same pose different fields | validation | 44 | 0.3348 | 6/44 | 0/44 |
| different pose same field | train | 78 | 0.3052 | 5/78 | 3/78 |
| different pose and field | train | 78 | 0.4187 | 11/78 | 4/78 |

Fine-tuning used 562 frames including 94 retained original examples. Both CPU masks agreed exactly with the 88 corresponding saved GPU masks per checkpoint.

The pilot checkpoint SHA-256 is `2d92e070ca315f4490129d44be5147aaf91f2a079eba458aba6aa7a7c05dbc1a`.

| Checkpoint | L5 label | Pixel Dice before confidence gating | Frame precision | Frame recall | False acceptance on competing bone | Retained positives across both fields |
|---|---|---:|---:|---:|---:|---:|
| original | TP | 0.702 | 12/13 | 12/28 | 1/60 | 6/11 |
| original | SAP | 0.599 | 8/16 | 8/12 | 8/76 | 4/6 |
| fine tuned | TP | 0.668 | 12/13 | 12/28 | 1/60 | 6/11 |
| fine tuned | SAP | 0.638 | 8/16 | 8/12 | 8/76 | 4/6 |

| Held-out source | Speckle field | Frames | Original L5 TP / SAP | Fine-tuned L5 TP / SAP |
|---|---:|---:|---|---|
| session_01 | 393911 | 22 | U / U | U / U |
| session_03 | 393911 | 22 | U / U | U / U |
| session_01 | 393912 | 22 | U / U | U / U |
| session_03 | 393912 | 22 | U / U | U / U |

**Verdict.** Speckle independently changes ultrasound visibility. The pilot did not improve held-out frame acceptance, recall, or cross-field retention under the same thresholds. Both checkpoints were replayed to measure map-level effects separately.

### 3.4 Reading it

- This fixes the geometry seen by the simulator as well as holding the actor model constant. The effect size in the waypoint-5 comparison is specific to the tested pose pair and field pair.
- A stricter post hoc comparison found 12 poses (24 images) at least 4 mm from both training inventories. Both models missed all 6 visible TP frames, each made 1 TP false acceptance, and this subset contained no SAP-positive frames.
- The archived-pose renders and perturbed slices exercise the simulator acoustic path. The perturbed samples do not represent newly executed robot motion.
- The original run completed all three epochs and stored the final checkpoint and GPU validation masks. The first metrics write failed on NumPy integer serialization. The recovery reads the same checkpoint, checks the source manifest and thresholds, reproduces the saved masks, and performs no additional optimization. The corrected replay unpacks the frozen per-class threshold table from that checkpoint.

### 3.5 Files

| File | Contents |
|---|---|
| `review-52711211/E3-dataset/factorial.json` | Position, rotations, image and geometry SHA-256, and visibility counts for every controlled cell |
| `diagnostics-52711846/controlled_speckle.png` | Side-by-side acoustic render and Oracle contours |
| `diagnostics-52711846/L5_original_vs_fine_tuned.png` | Oracle and both model predictions on identical validation images |
| `review-52711211/E3-training` | Stored checkpoint, paired validation masks, and exact training split |
| `pilot-recovery-52713119` | Segmentation metrics recovered from the stored checkpoint |
| `pilot-replay-52714468` | Matched original sweeps and held-out L5-only validation maps |

## 4. Surface association

`surface_landmark_mapping.py` replaces the pairwise distance between component representatives by the greater of the two directed 25th-percentile nearest distances between the observed component pixels projected into 3D.

**Criterion.** Compare the same eligible observation sets and the original identity evaluator. Preserve separate levels and require the same translation, rotation, confidence, support, ambiguity, and 4 mm association settings.

### 4.1 Method

- The projected clouds use only pixels in each original accepted image component. Complete linkage prevents an unrestricted merge formed by a chain of near neighbors. The existing selected-view rules count each viewpoint once.
- Ranking and representative output retain the original point mapper's support order and medoid of supporting representative points. The historical metric scores those same representative observation pixels.
- The connected-component query on the provisional SAP volume is an offline explanation. The actor surface mapper consumes saved U-Net pixels and poses.

### 4.2 Setup

| Field | Value |
|---|---|
| Inputs | All 216 E0 frames, four saved actor observation sets and corresponding Oracle results |
| Evaluator | Same subregions, pixel calibration, visible-interface support and 3 px scoring as E0 |
| Software | `52711211` for the component comparison, `52712399` for nested-density replay |
| Sample accounting | 216/216 valid frames for the recorded component comparison |

```bash
python workflows/two_stage_navigation/evaluate_robust_mapping.py --protocol "$protocol" --profile "$profile" --reference "$reference" --label-conversion "$conversion" --acquisition "$capture" --density-only --surface-on-density --output "$integration"
```

### 4.3 Results

- Oracle point and surface overall correctness were both 8/16, with zero wrong and 8 unresolved. Both left L4 SAP unresolved in all four cases. Per-landmark differences are in the per-session table.
- The two L4 SAP competitors used frames `[51, 52]` and `[48, 49]` in all four cases. They belonged to the same connected component of the provisional SAP label. The closest observed cross-cluster surface points were 4.040–4.083 mm apart. The largest pairwise observation distances between the two clusters were 6.827–7.012 mm.
- The original U-Net with surface mapping gave 11/16 correct, one wrong, four unresolved. L5 TP improved from 1/4 to 3/4 with one remaining wrong. L4 TP fell from 4/4 to 3/4. L4 SAP remained 1/4 and L5 SAP remained 4/4.
- The surface mapper took 75.6 ms total for the four U-Net maps versus 8.4 ms for the four point maps. Oracle had more eligible observations and took 581.0 ms versus 19.1 ms.

**Verdict.** Surface distance recovers useful L5 TP support on these scans but leaves the L4 SAP coverage gap and introduces a control regression.

### 4.4 Reading it

- The provisional volume suggests the two competing L4 SAP regions are visible portions of the same labeled structure. The recorded ultrasound never bridges their surface gap at the frozen threshold.
- Eight selected surface clusters include representatives that previously supported two different point clusters. Six of those eight contain at least one off-interface view, and all eight pass the strict majority identity criterion. The remaining wrong L5 TP map used its original two-view cluster.
- The 3 px rule identified unsupported views in the selected U-Net clusters. An accepted map can include an individual unsupported view when a strict majority remains correct. The L5 TP in seed 1702 had one unsupported view out of two and was scored wrong.

### 4.5 Files

| File | Contents |
|---|---|
| `diagnostics-52711846/L4_SAP_point_vs_surface.png` | Original and proposed maps with separate Oracle and actor views |
| `review-52711211/mapping/results.json` | Full pair of candidate surface clusters and representative observation support |

## 5. Integration and disposition

**Criterion.** A configuration should improve the matched development outcome while preserving the consistently mapped L4 TP and L5 SAP, avoiding wrong landmarks and maintaining acquisition integrity.

### 5.1 Method

Replay the measured L4 SAP coverage intervention with the L5 TP surface association change on the same nested acquisitions. The E4 comparison kept both original U-Nets. The separate E3 comparison kept the original point mapper and original L4 U-Net. Both retained TP eligibility, level routing, and acceptance thresholds.

### 5.2 Setup

| Field | Value |
|---|---|
| Source and evaluation | E1 acquisitions from `52710240`; density plus E4 replay `52712399`; density plus E3 replay `52715479` |
| Held-out seeds | `2026121800`–`2026121803`. Reserved in a hashed manifest before any execution on those seeds. |

### 5.3 Results

**Verdict.** Neither the density-plus-surface nor the density-plus-fine-tune replay met the criterion. No configuration advanced to independent confirmation. All four reserved seeds remain unused.

- The E1-M plus E4 replay gives 4/8 correct actor identities, 1 wrong and 3 unresolved. The E1-D plus E4 gives 5/8 correct, 1 wrong and 2 unresolved, with the two acquisition exclusions from E1-D. Neither restores the reliable L4 TP in seed 1702.
- E1-M plus the fine-tuned L5 checkpoint gives 6/8 correct, one wrong and one unresolved with 204/204 valid acquisitions. E1-D plus fine-tuning gives 5/8 correct, one wrong and two unresolved with 298/300 valid acquisitions.
- The fine-tuned L5 checkpoint increases the archived-sweep results from 10/16 to 12/16 correct, with one wrong and one complete point-based map. Held-out image-level operating characteristics are unchanged. With surface association it retains 12/16 correct and has zero wrong, but loses the reliable L4 TP for one seed and produces zero complete maps.
- The Oracle plus surface mapper identifies all four structures in both medium subsets. The matched actor result leaves a perception gap even when the geometry and tested association are sufficient.

C, W, and U denote correct, wrong, and unresolved. Parentheses show selected distinct viewpoints.

| Arm | Seed | L4 TP | L4 SAP | L5 TP | L5 SAP | Valid/planned |
|---|---|---|---|---|---|---:|
| E0 | 2026121700 | C (5) | U (2) | U (2) | C (2) | 54/54 |
| E0 | 2026121701 | C (7) | U (2) | C (2) | C (3) | 54/54 |
| E0 | 2026121702 | C (4) | C (2) | W (2) | C (2) | 54/54 |
| E0 | 2026121703 | C (7) | U (2) | W (2) | C (2) | 54/54 |
| E1-S | 2026121700 | C (5) | U (2) | U (2) | C (2) | 54/54 |
| E1-S | 2026121702 | C (4) | C (2) | W (2) | C (2) | 54/54 |
| E1-M | 2026121700 | C (5) | C (4) | U (2) | U (3) | 102/102 |
| E1-M | 2026121702 | C (4) | C (4) | W (2) | U (3) | 102/102 |
| E1-D | 2026121700 | C (5) | C (5) | U (2) | C (7) | 149/150 |
| E1-D | 2026121702 | C (4) | C (6) | W (2) | C (5) | 149/150 |
| E2 stabilized | 2026121700 | C (5) | U (2) | C (2) | C (2) | 54/54 |
| E2 stabilized | 2026121702 | C (6) | W (2) | W (2) | C (3) | 54/54 |
| E3 fine-tuned L5 | 2026121700 | C (5) | U (2) | W (2) | C (3) | 54/54 |
| E3 fine-tuned L5 | 2026121701 | C (7) | U (2) | C (2) | C (3) | 54/54 |
| E3 fine-tuned L5 | 2026121702 | C (4) | C (2) | C (3) | C (2) | 54/54 |
| E3 fine-tuned L5 | 2026121703 | C (7) | U (2) | C (2) | C (2) | 54/54 |
| E4 surface | 2026121700 | C (6) | U (2) | C (3) | C (2) | 54/54 |
| E4 surface | 2026121701 | C (8) | U (2) | C (3) | C (4) | 54/54 |
| E4 surface | 2026121702 | U (4) | C (3) | W (2) | C (2) | 54/54 |
| E4 surface | 2026121703 | C (7) | U (2) | C (3) | C (2) | 54/54 |
| E1-M + E4 | 2026121700 | C (6) | C (4) | C (3) | U (3) | 102/102 |
| E1-M + E4 | 2026121702 | U (4) | C (4) | W (2) | U (3) | 102/102 |
| E1-D + E4 | 2026121700 | C (6) | C (5) | C (3) | U (5) | 149/150 |
| E1-D + E4 | 2026121702 | U (4) | C (6) | W (2) | C (6) | 149/150 |
| E1-M + E3 | 2026121700 | C (5) | C (4) | W (2) | C (4) | 102/102 |
| E1-M + E3 | 2026121702 | C (4) | C (4) | C (3) | U (3) | 102/102 |
| E1-D + E3 | 2026121700 | C (5) | C (5) | W (2) | U (5) | 149/150 |
| E1-D + E3 | 2026121702 | C (4) | C (6) | C (3) | U (4) | 149/150 |
| E0 Oracle point | 2026121700 | C (8) | U (2) | U (3) | C (4) | 54/54 |
| E0 Oracle point | 2026121701 | C (5) | U (2) | C (7) | U (3) | 54/54 |
| E0 Oracle point | 2026121702 | C (7) | U (2) | C (5) | U (2) | 54/54 |
| E0 Oracle point | 2026121703 | C (9) | U (2) | C (6) | U (3) | 54/54 |
| E4 Oracle surface | 2026121700 | C (11) | U (2) | C (4) | C (4) | 54/54 |
| E4 Oracle surface | 2026121701 | C (9) | U (2) | C (7) | U (3) | 54/54 |
| E4 Oracle surface | 2026121702 | U (5) | U (2) | C (5) | U (2) | 54/54 |
| E4 Oracle surface | 2026121703 | C (9) | U (2) | C (6) | U (3) | 54/54 |
| E3 fine-tuned L5 + surface | 2026121700 | C (6) | U (2) | C (3) | C (3) | 54/54 |
| E3 fine-tuned L5 + surface | 2026121701 | C (8) | U (2) | C (3) | C (4) | 54/54 |
| E3 fine-tuned L5 + surface | 2026121702 | U (4) | C (3) | C (3) | C (2) | 54/54 |
| E3 fine-tuned L5 + surface | 2026121703 | C (7) | U (2) | C (3) | C (2) | 54/54 |
| E1-M all TP | 2026121700 | U (5) | C (4) | U (4) | U (3) | 102/102 |
| E1-M all TP | 2026121702 | C (6) | C (4) | C (4) | U (3) | 102/102 |
| E1-D all TP | 2026121700 | C (9) | C (5) | U (5) | C (7) | 149/150 |
| E1-D all TP | 2026121702 | U (8) | C (6) | C (5) | C (5) | 149/150 |
| E1-M all TP, Oracle | 2026121700 | C (11) | C (4) | C (6) | C (6) | 102/102 |
| E1-M all TP, Oracle | 2026121702 | C (8) | C (4) | C (7) | C (5) | 102/102 |
| E1-D all TP, Oracle | 2026121700 | C (12) | C (5) | C (8) | U (7) | 149/150 |
| E1-D all TP, Oracle | 2026121702 | C (10) | U (4) | C (11) | U (6) | 149/150 |

### 5.4 Reading it

- The medium geometry gives the strongest direct evidence for the next controlled perception experiment. Oracle maps 2/2 sessions when allowed TP evidence from all medium views, while the matched actor maps 0/2.
- Deliberately selecting that geometry is justified for comparing perception at the same poses and speckle fields. The results do not support promoting a new mapping configuration.

### 5.5 Files

The artifact root is `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/artifacts/robust-landmark-061929`. In the commands above, `capture` is `acquisition-52710240`, `review` is `review-52711211`, and `integration` is `integration-52712399`, each under that root.

| Input | Exact location |
|---|---|
| Frozen source protocol | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_mbb_single_anatomy_poc_v6r19/52626449/context/s0010_mbb_single_anatomy_poc_protocol_v6r19.yaml` |
| Stage 1 actor reference | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_mbb_single_anatomy_poc_v6r19/52626449/stage1/results/actor_results.json` |
| `reference` | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_simple_landmark_independent_20261125T044110Z/52699995` |
| `conversion` | `source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml` |
| `segmentation` | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/assets/data/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz` |
| `original_data` | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_sonoanatomy_dataset_v5/52497655/dataset/dataset.npz` |
| `original_manifest` | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_sonoanatomy_segmentation_v9/52498821/training/l5_split_manifest.csv` |
| Profile | `experiments/unity_two_stage_transfer/s0010_simple_landmark_robustness_v1.yaml` |

The review includes the synchronized MP4 from the 149 valid acquisitions of seed `2026121700`. It shows the 3/4 correct map and the unresolved L5 TP. The renderer and a separate OpenCV invocation verify 340 video frames at 6 fps and 149 synchronized camera/ultrasound pairs.
