# Complete observed cortex and right L4 medial branch target localization on s0010

This study replays complete cortical-interface reconstructions on four matched simulator sessions. It tests whether local observed cortex can link the right L5 TP root and the lateral SAP neck for a provisional right L4 medial branch bone reference.

## Result

**Complete bone reconstruction recovers nearby observations, but the 408 recorded frames and the single fresh oblique supplement yield no bounded cortex transition region. Anatomical target correctness remains NOT_EVALUABLE.**

| Study | Result |
|---|---|
| Anatomy without ultrasound | Two CT-only cortical hypotheses, 11 face samples each. Their shortest intervening external cortex trace is 7.68 mm. The anatomical reference remains unresolved. |
| Matched 408-frame replay | E0, E1 and E2 returned zero regions. The additional bone gave local search seeds but no qualifying cortical route. |
| Fresh development | 102+24 images, 126/126 valid. The extra views increase surface and view support. Oracle stays at 1/7 and learned at 0/7 on the diagnostic cortex trace. |
| Independent confirmation | Reserved seeds 2026121800 through 2026121803 unopened. |

## 1. Anatomical reference — Does the full L5 cortex support a unique reference?

`workflows.mbb_end_to_end.reference` audits the registered s0010 CT segmentation and the original subregion masks.

**Criterion.** Identify the target on external cortex from the right TP root and lateral SAP neck. Preserve uncertain anatomical assignments. Freeze the resulting interpretation before evaluating observed surfaces.

### 1.1 Method

- Select two anatomy-only sets of exposed L5 face centers from the prior broad cortical uncertainty window. The SAP side requires a right-facing normal and a caudal owner relative to the SAP proposal center. The TP side requires a superior- or medial-facing normal and an owner in the medial half of the TP proposal.
- Measure the shortest edge-connected route on the complete external L5 boundary. Mark the three contributing subregion types on that trace. Retain the old midpoint outputs for historical comparison.
- The posterior lateral-neck boundary and the mammillary/accessory processes matter for delimiting the reference. The available source labels do not identify those boundaries. [Tran et al. 2022](https://pubmed.ncbi.nlm.nih.gov/35589133/) · [Tran et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11372964/)

### 1.2 Setup

| Setting | Value |
|---|---|
| Stimulus and voxel spacing | s0010 CT and full L5 segmentation, 1.5 mm, NIfTI RAS |
| Frozen reference | `reference/reference.json` SHA-256 `2f24e1f01d2f0d8542cf431880921a0b66bbdc47ab6a7d94a121939c4323b36d` |
| Archive and input provenance | `reference/checksums.sha256`; full file hashes embedded in reference descriptor |
| Job and revision | `52800030`, `66dd8dab0f69778820c9c31acd150a3beb3acba2`, 2 CPUs |
| Runtime | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/envs/sonogym` |

Run the audit from the integrated worktree or the frozen anatomy revision.

```bash
project=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym
runtime="$project-runtime"
study="$project/artifacts/mbb-end-to-end-210237"
prior="$project/artifacts/mbb-surface-localization-191322"
manifest="$project/artifacts/mbb-identifiability-114133/shared/manifest.json"
cache_manifest="$project/artifacts/reconstruction-103750/shared/manifest.json"
cd "$study/checkout"
sbatch --output="$study/reference-reproduction-%j.log" \
  hpg/jobs/s0010_mbb_observed_surface.sbatch "$runtime" -m workflows.mbb_end_to_end.reference \
  --manifest "$manifest" --prior-protocol "$prior/checkout/experiments/unity_two_stage_transfer/s0010_mbb_observed_surface_v1.json" \
  --prior-audit "$prior/anatomy/audit.json" --ct "$runtime/assets/data/HumanModels/selected_dataset/s0010/ct.nii.gz" \
  --ct-sha256 eff5e5f5829cf8cc3e413b3ac09ce455e45bc0f91f5ba37fab9b37b1fe683546 \
  --prior-ct-figure "$prior/anatomy-review/ct_review.png" --output "$study/reference-reproduction"
```


### 1.3 Results

**Verdict. Anatomical target correctness is `NOT_EVALUABLE`.** The two hypotheses are 5.30 mm apart at the nearest centers. Their shortest 7.68 mm external trace has seven sampled face centers, of which five belong to lamina, one belongs to SAP and one belongs to TP under the V3 model. The trace fixes a diagnostic coverage test while the intended anatomical boundaries remain uncertain.

### 1.4 Interpretation

- The historical 3 mm separation gate required the two labels to approach more closely than the minimum separation of the complete SAP and TP proposal masks, which was 6.185 mm.
- The outer cortical trace shows that an intervening modeled subregion can connect the vicinity of the anchors. Neither hypothesis is used as an accepted target.
- The trace and reference metadata were sealed without loading the ultrasound or point clouds.

### 1.5 Files

| File | Contents |
|---|---|
| `reference/external_cortex_review.png` | Adjacent physical CT sections, two face hypotheses and external trace |
| `reference/anatomy_hypotheses.npz` | Evaluator-only face and trace positions in RAS millimetres |
| `reference/historical_proxy.npz` | Archived midpoint, separate from the cortical hypotheses |

## 2. Recorded replay — Do the additional observed interfaces support a bounded region?

`workflows.mbb_end_to_end.reconstruct` tracks and projects first-supported cortical samples. `workflows.mbb_end_to_end.evaluate` runs the localizer before opening anatomical data.

**Criterion.** Return one geometrically supported region with diameter at most 6 mm and at least two distinct local views. Both directed nearest-distance P95s must be at most 3 mm to an independently reviewed anatomical region for correctness. The 3 mm distance is a research criterion. All plausible accepted reference hypotheses would have to pass. `NOT_EVALUABLE` applies when the anatomical reference is unresolved.

### 2.1 Method

- Use the same 102 ultrasound frames per seed, executed poses and calibration. Preserve support for all first-visible acoustic-bone labels. For learned reconstruction, use the original L4 and T4 L5 probabilities, the existing accepted foreground and image support. Retain tracker-only samples as uncertain. Do not assign their semantic level from the simulator.
- Use a local graph of measured cortical samples, with edges at most 1.5 mm. Longer edges within that cap require observed support near the middle. A proposed region has an observed route of at most 6 mm to each semantic flank. The exact model uses local plane fits, oriented SAP and TP flanks, and separate support across viewpoints. Ambiguous and neighboring-level points do not supply the primary route.
- Evaluate the same new localizer on complete bone and with additional bone removed. Bounded interpolation is permitted only between neighboring accepted surface samples within 1.5 mm. Use a measured-sample diameter cap of 6 mm, track source frames/pixels, and abstain for competing regions or unstable view removal. Pose stress trials apply 1 mm and 0.5° perturbations. They are sensitivity diagnostics.
- Measure diagnostic cortical coverage within 1.5 mm of each of the seven full-L5 CT trace samples. This is an evaluator-only observability measurement. Report the historical midpoint only as a separate geometric comparison.
- Seal the exact 102-row public pose ledgers. Actor receives these ledgers and cloud only. No raw labels, target reference geometry or hidden simulator pose are passed to the actor process.

### 2.2 Setup

| Setting | Value |
|---|---|
| Development seeds | 2026121700, 2026121702, 2026121900, 2026121901 |
| Stage 1 | Reused `cf7_00` |
| Model digests | L4 `c6b7ecc9bc245a88ccc0a5db31dbfd919b57407988dde5ea8211dd7eea2c1b2a`; T4 L5 `1efc323e9c379bcf657eb59fb793cfa30b98e618033a3abc1966a4f3cc69bf47` |
| Reconstruction | `52800465`, `2ab781e47856c32c2be06c05abf39470098a2153`, 2 CPUs, 408/408 acquisitions |
| Localizer | Worktree commit `fc4c91e32732f53aabe177f296634beb58b5891a`; matched evaluation job `52800942` |
| Matched evaluation revision | `e98eafe7b4cb294afafbf8be700e03a8ee4b1b4a` |
| Acceptance and source hashes | `s0010_mbb_end_to_end_localization_v1.json` and adjacent SHA-256 |

```bash
rerun="$study/reproduction"
mkdir -p "$rerun"
sbatch --output="$rerun/reconstruct-%j.log"   hpg/jobs/s0010_mbb_observed_surface.sbatch "$runtime" -m workflows.mbb_end_to_end.reconstruct   --manifest "$manifest" --cache-manifest "$cache_manifest" --reference "$study/reference"   --output "$rerun/reconstruction"

# Submit after the reconstruction reaches COMPLETED.
sbatch --output="$rerun/evaluate-%j.log" hpg/jobs/s0010_mbb_end_to_end_replay.sbatch   "$runtime" "$rerun" "$rerun/reconstruction" "$study/reference"
```

### 2.3 Results

| Seed | E0 Oracle / learned gap, mm | E1 Oracle seeds / routes | E2 learned seeds / routes | E1 / E2 CT trace coverage | E1 / E2 inference with sensitivity, ms | E0 / E1 / E2 target |
|---|---:|---:|---:|---:|---:|---|
| 2026121700 | 10.37 / 9.61 | 2 / 0 | 0 / 0 | 1/7 / 0/7 | 100.0 / 89.6 | Unresolved / unresolved / unresolved |
| 2026121702 | 9.92 / 8.11 | 1 / 0 | 2 / 0 | 0/7 / 0/7 | 92.2 / 110.1 | Unresolved / unresolved / unresolved |
| 2026121900 | 10.04 / 9.66 | 1 / 0 | 4 / 0 | 0/7 / 0/7 | 90.0 / 116.0 | Unresolved / unresolved / unresolved |
| 2026121901 | 10.29 / 9.01 | 0 / 0 | 1 / 0 | 0/7 / 0/7 | 79.0 / 96.6 | Unresolved / unresolved / unresolved |

Each matched replay arm uses 102/102 frames per session. The 58 focused and regression tests include a synthetic 8.13 mm label-gap case connected by observed OTHER_BONE cortex and its negative without the connector. Each anchor-only ablation returned zero local route seeds. No observed or inferred output region is available. Supporting TP and SAP viewpoints for each method, sensitivity outcomes and calibration checks are in `replay/results.json`.

**Verdict. The additional bone did not provide a connected target region in the four recordings.** The Oracle and learned arms had zero bounded connector routes before the stronger orientation and support gates. No candidate diameter, two-direction error, or anatomical correctness is available for these sessions.

### 2.4 Interpretation

- Oracle reconstructs 11,979 points, including 7,552 other-bone points. It preserves all 3,313 historically admitted Oracle TP/SAP points.
- Learned reconstruction exports 48,600 candidate interface samples. Of these, 40,026 are uncertain, including 39,719 from tracks without accepted foreground, and 5,398 are other bone. It preserves 2,746 of 4,014 historical learned TP/SAP observations under the stricter first-supported rule.
- On the same image ray and within 3 pixels, 810/1,945 learned L5 other-bone points had comparable Oracle L5 evidence. Another 847 matched non-L5 and 288 lacked comparable Oracle depth. This quantifies level ambiguity in the learned supplement.
- New other-bone geometry supplied local search seeds in three Oracle sessions and three learned sessions. Both anchor-only ablations produced none. The extra candidates never supported a complete bounded route.
- Oracle covered 1/28 CT trace samples across sessions, unchanged from the historical TP/SAP cloud. The match is on the TP side. All 20 lamina face-session positions and all four SAP face-session positions lack observed coverage. Learned coverage was 0/28. Within 5 mm of the historical midpoint, learned geometric proximity covers 88/91 Oracle-visible points disregarding semantics. The two additional pose stress directions in both arms retained abstention.

### 2.5 Files

| File | Contents |
|---|---|
| `reconstruction/<seed>/` | PLY and NPZ clouds, original frame and pixel metadata, posterior maps and checksums |
| `replay/actor_prediction_seals.json` | Inputs and outputs hashed before anatomical evaluation |
| `replay/results.json` | Per-arm geometry, uncertainty, viewpoint and two-direction scoring fields |
| `figures/recorded_surfaces.png`, `figures/recorded_ultrasound.png` | Matched 3D and image comparisons with separate evaluator overlay |

## 3. Fresh development — Do 24 oblique views recover the missing cortex?

`capture_simple_landmark_sweep.py` uses the existing ultrasound controller, with a sealed supplement. `workflows.mbb_end_to_end.evaluate_fresh` compares the original prefix to the full acquisition from the same executed session.

**Criterion.** Retrieve new valid synchronized observations and assess cortical trace coverage and candidate geometry under the exact frozen replay rules. A reproducible anatomical localization still requires an independently defined reference. The four reserved seeds are excluded.

### 3.1 Method

- Reuse the Stage 1 reference and original 102 commands. Append 24 oblique views on three lateral lanes and four cranial positions, anchored only to the public L5 Stage 1 region. Yaw is 2.60 or 2.25 radians. Keep workspace, contact, freshness, pose projection and motion limits unchanged.
- Run the exact same geometry on the prefix and full sequence. Compare corresponding original commands to the archived matched development seed. The comparison uses actual executed poses and synchronized hashes.
- Reconstruct with the existing checkpoints and native first-bone Oracle diagnostic. Any target or CT trace overlay appears only after prediction seal.

### 3.2 Setup

| Setting | Value |
|---|---|
| Capture | `52801423`, commit `39f66898ef94adafe7b303ca3fb468b9baebf643`, 4 CPUs, one L4 |
| Video with surface class legend | `52803053`, revision `40189e76e1ebc22dbe70049a859cca1d7e082455`, one CPU, same sealed prediction and images |
| Fresh replay | `52802149`, commit `e60dd9f3d46f7c5a2c0b0d0cc2f05496eaacff9b`, 2 CPUs, zero GPUs |
| Stage 1 and development seed | Recorded `cf7_00`, reset seed 2026121700 |
| Motion and frames | 5,389 controller actions, 126/126 valid synchronized frames, 0 failed moves |
| Runtime after simulator application initialization | 178.32 s for acquisition and capture-side setup |
| Frozen added views | `s0010_mbb_oblique_development_v1.json`, SHA-256 `8c55736e2631d77572d2b7a940a3627eaaf26c906848711127b7289d0b9028e6` |

For capture, use a separate checkout at the exact acquisition revision, then use the replay revision for reconstruction and evaluation.

```bash
stage1="$runtime/outputs/unity_two_stage_transfer/s0010_mbb_single_anatomy_poc_v6r19/52626449/stage1/results/actor_results.json"
machine="$project/artifacts/robust-landmark-061929/acquisition-52710240/context/machine-config.env"
capture_revision=39f66898ef94adafe7b303ca3fb468b9baebf643
fresh_rerun="$study/fresh-reproduction"
mkdir -p "$fresh_rerun"
cd "$study/checkout"
git worktree add --detach "$fresh_rerun/capture-checkout" "$capture_revision"
cd "$fresh_rerun/capture-checkout"
sbatch --export=ALL,SONOGYM_CONFIG_FILE="$machine",MBB_CAPTURE_COMMIT="$capture_revision"   --output="$fresh_rerun/capture-%j.log" hpg/jobs/s0010_mbb_end_to_end_capture.sbatch   "$fresh_rerun" "$manifest" "$stage1"

# Submit after capture succeeds, using the job identifier returned by sbatch.
cd "$study/checkout"
capture_id=52801423  # Replace with the rerun job ID.
sbatch --output="$fresh_rerun/review-%j.log" hpg/jobs/s0010_mbb_end_to_end_replay.sbatch   "$runtime" "$fresh_rerun" "$fresh_rerun/reconstruction" "$study/reference"   "$fresh_rerun/capture-$capture_id/session" "$manifest" "$cache_manifest"
```

### 3.3 Results

| Fresh arm | Valid images | L5 other-bone points | Distinct TP / SAP views | TP–SAP gap, mm | Seeds / routes | CT trace coverage | Inference with sensitivity, ms | Region |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| E1_prefix | 102 | 680 | 19 / 18 | 10.367 | 2 / 0 | 1/7 | 68.9 | Unresolved |
| E1_full | 126 | 839 | 26 / 31 | 10.259 | 3 / 0 | 1/7 | 141.1 | Unresolved |
| E2_prefix | 102 | 491 | 13 / 13 | 10.487 | 0 / 0 | 0/7 | 103.4 | Unresolved |
| E2_full | 126 | 713 | 18 / 24 | 9.794 | 2 / 0 | 0/7 | 164.0 | Unresolved |

**Verdict. The fresh supplement recovered additional observed anatomy but did not recover the diagnostic bridge or localize a bounded region.** All 126 acquisitions were valid. All 102 original waypoint ultrasound hashes matched the earlier acquisition. Executed position differences were 0 mm at median and P95 against the prior prefix.

### 3.4 Interpretation

- L5 other-bone samples rose by 159 in Oracle and by 222 in learned reconstruction. Distinct supporting views grew for both flanks. Neither arm added a trace match.
- Within 5 mm of the historical midpoint, the fresh Oracle prefix covers 30/31 visible samples from the full fresh Oracle cloud. The additional views supply the last observed sample. Learned proximity stays at 29/31 for prefix and full. This coverage is pooled over labels and does not establish anatomical identity or the cortical bridge.
- The primary failures are missing route continuity and unresolved anatomical boundaries. Label confusion contributes additional uncertainty in learned geometry. No correct or wrong anatomical region can be scored. There was no accepted geometric target in any attempted session.
- The added views test one finite Stage 1-relative survey on this simulator. The result does not rule out observability under other views or a reviewed region definition. The next discriminating observation would show the relevant external L5 cortex connecting the TP root and lateral SAP neck, after the posterior neck and mammillary/accessory boundaries are independently identified.

### 3.5 Files

| File | Contents |
|---|---|
| `capture-52801423/session` on HPG | Actual ultrasound, calibrated pose records, 126 robot camera frames and evaluator-only diagnostic evidence |
| `fresh-evaluation/results.json` | Frozen 102-frame prefix and full fresh sequence comparisons, both arms |
| `fresh-evaluation/prediction_seals.json` | Source and actor output hashes |
| `reconstruction-fresh/2026121700` | Raw learned and Oracle point clouds in actor base millimetres |
| `figures-review/recorded_semantic_examples.png` | TP, SAP and other-bone examples over source ultrasound |
| `video-review/fresh_oblique_sweep.mp4` and `video-review/video_verification.json` | Synchronized capture progression and full decode verification |
