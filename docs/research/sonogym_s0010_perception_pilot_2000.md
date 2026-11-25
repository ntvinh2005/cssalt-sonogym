# Ultrasound perception on the frozen s0010 sweep

This study compares level-specific U-Nets on the same executed ultrasound data. The pilot uses a single shared set of exactly 2,000 images across the four possible identities and all training arms.

## Result

**The 2,000-image pilot improved L5 recognition and raised the fixed development result from 10/16 to 12/16 correct landmarks. Only 1/4 maps completed and one L5 TP was wrong. The readiness gate failed and the reserved independent seeds were kept unused.**

| 102-waypoint comparison | Correct | Wrong | Unresolved | Complete |
|---|---:|---:|---:|---:|
| Original U-Net | 10/16 | 0 | 6 | 0/4 |
| E3 U-Net | 9/16 | 0 | 7 | 0/4 |
| 2000-image pilot, original thresholds | 12/16 | 1 | 3 | 1/4 |
| Same checkpoints, calibrated thresholds | 10/16 | 1 | 5 | 0/4 |
| Oracle, all eligible waypoints | 15/16 | 0 | 1 | 3/4 |
| Oracle, actor level routing | 15/16 | 0 | 1 | 3/4 |

Previously inspected acquisition sessions are development evidence. The fixed 102-waypoint geometry, executed-pose calibration, Stage 1 reference, and point mapper were preserved. New independent acquisition is conditioned on the readiness gate in §4.

## 1. Baseline diagnosis

`workflows/perception_s0010/replay.py` replays the actor and the oracle on the exact medium sweep.

**Criterion.** Compare the frozen visible-interface and complete-map definitions. In the primary comparison, both actor and oracle use the full common 102-waypoint pool. The original TP restriction is a separate result.

### 1.1 Method

- Recompute both model outputs with the original preprocessing and connected-component gates. Check each eligible representative against the visible interface within 3 pixels.
- The actor uses the original TP frame-center level and SAP projected-position level routing. The oracle may assign the label's true level. The additional routed Oracle comparison applies the actor's level and ambiguity rules to oracle observations.
- Inspect ultrasound with matched actor and oracle contours. Broad source labels distinguish other bone, background, and uncertain or unsupported interfaces. They do not uniquely identify lamina versus every neighboring articular portion.
- Compare the existing fixed-pose renderer controls and an offline one-frame substitution in the historical map. The repeat control keeps the same geometry and speckle field.

### 1.2 Setup

| Input | Value |
|---|---|
| Frozen source configuration | `c26802d61f211c62acd16fe12208183e3442ebf776e9188d867a77126c6a114e` |
| Medium scan 102 ordered waypoints | `b325d3c5df65ca17bda3584374d3fd6d784c246ae931b7d63c5629fb45d653e9` |
| Original L4 and L5 | Training job `52498821`. SHA-256 `c6b7ecc9bc245a88ccc0a5db31dbfd919b57407988dde5ea8211dd7eea2c1b2a` and `ca13c8fb6bf99bd363dd87c5e30dc1eeb5b22da5cf3f87e991893fad96b41acf`. |
| E3 L5 | SHA-256 `2d92e070ca315f4490129d44be5147aaf91f2a079eba458aba6aa7a7c05dbc1a`. Pilot E3 archive checked against `68474e108bee66efafcd1adba4b58274f9207871663b767c0751580ee026b5a6`. |
| Stage 1 | Recorded actor session `cf7_00`, checksum `1fcfd37760034feaf5eccca586a687713a04181fcb943df293d5b1c213a1f4a8`. |
| Acquisitions | E1-M prefixes from job `52710240` (seeds 2026121700 and 2026121702), fixed fresh medium sweep job `52719450` (2026121900 and 2026121901). |
| Diagnosis | Job `52720256`. |

```bash
python workflows/perception_s0010/replay.py --protocol "$protocol" --profile "$profile" --previous "$previous" --historical "$historical" --development "$development" --output "$out/diagnosis" --diagnose --device cpu
```

### 1.3 Results

| Original checkpoint | Within 3 px / visible frames | Competing background | Competing other bone | Below confidence | Below component size | Representative off interface | Level routing |
|---|---:|---:|---:|---:|---:|---:|---:|
| RIGHT L4 TP | 92/122 | 7 | 1 | 4 | 4 | 14 | 0 |
| RIGHT L4 SAP | 35/47 | 2 | 2 | 4 | 2 | 2 | 0 |
| RIGHT L5 TP | 35/97 | 36 | 7 | 7 | 5 | 5 | 2 |
| RIGHT L5 SAP | 28/73 | 35 | 5 | 1 | 3 | 1 | 0 |

The two historical E1-M prefixes reproduce 4/8 correct, 1 wrong, 3 unresolved with the original 50 TP waypoints. With all TP waypoints they produce 4/8 correct, 0 wrong, 4 unresolved, with 2/2 complete oracle maps. The corrected original-50 replay across all four 102-waypoint cases is 9/16 correct, 1 wrong, 6 unresolved and 0/4 complete. The first evaluation artifact recorded online TP eligibility for its secondary restricted column on the two new captures. The supplementary correction uses the archived original waypoint marker. Primary predictions and checkpoint selection are unchanged.

| Controlled render | Visible L5 TP pixels | Maximum class probability on visible pixels | Accepted-mask pixels | Map after single-frame substitution |
|---|---:|---:|---:|---|
| A0 | 0 | No visible target | 0 | UNRESOLVED |
| A1 | 9 | 0.998 | 35 | UNRESOLVED |
| B1 | 0 | No visible target | 0 | UNRESOLVED |
| C1 | 9 | 0.997 | 33 | UNRESOLVED |
| D0 | 0 | No visible target | 0 | UNRESOLVED |

A0 is the reference. A1 changes only the field. B1 changes only the pose. C1 changes both. D0 repeats A0. A0/D0 reproduce identical model probabilities. None of the single-frame substitutions alone resolved the map.

**Verdict.** Most L5 misses are class competition, especially background, rather than only a threshold or component-size failure. The oracle leaves L5 SAP unresolved in one 102-waypoint session. This limits any attribution solely to perception.

### 1.4 Reading it

- L4 SAP had sufficient misses before acceptance to justify its own pilot. The strongest original L4 TP performance makes preservation of that class explicit in the readiness gate.
- The hard-negative analysis treats a representative on independently labeled other bone as a confirmed negative. It keeps unsupported/uncertain and close borderline cases separately.
- Label presence changes under field variation even with the same pose. Model failures on still-visible anatomy and actual loss of visible evidence are distinct.

### 1.5 Files

| Artifact | Content |
|---|---|
| `benchmark-52720256/diagnosis/mapping.json` | Per-frame failure modes, source labels, distances and overlays |
| `evaluation-52723344/review/factorial_perception.json` | Render controls, probabilities, mask sizes, representatives, and paired map outcomes |
| `evaluation-52723344/review/speckle_controls.png` | Side-by-side render controls |

## 2. Data and split audit

`render_dataset.py` uses the same simulator speckle volumes and regenerates labels for both levels after every render.

**Criterion.** Preserve the frozen provisional labels. The pilot admits exactly 2,000 distinct training images. Combined validation has disjoint source acquisitions, pose hashes, image hashes, and speckle seeds, with at least 3 mm separation in anatomy-frame end-effector position from the pilot training inventory, original checkpoint training inventory, and E3 pose inventory.

### 2.1 Method

- Attach small rigid perturbations to physically executed probe poses. Image-origin displacements are bounded by 3 mm and rotations by 2.5 degrees. Contact is checked through the simulator proxy at each rendered pose. These perturbations are direct simulator slices.
- Draw the training subset uniformly across pose groups before filling extra field replicates. The pilot uses the same image hashes in every arm and adds zero legacy images. It starts from the frozen original weights.
- A larger run started before the 2,000-image instruction. Jobs `52721571` and `52721659` were cancelled. The pilot used no checkpoints from that run, and its dependent evaluation did not open the combined test.
- Use only selected training images for hard-example sampling. T1 uses all pilot images in a random order. T2 and T3 allocate half the draws to ordinary examples and one quarter each to hard positives and hard negatives, balancing source pose groups.

### 2.2 Setup

| Input | Value |
|---|---|
| Benchmark | Job `52720256`, 96/96 images, 19.83 images/s. Rendering took 0.82 s, labeling 3.60 s, and replacing the two speckle fields took 0.008 s in total. |
| Dataset | Job `52720837`, 4,474 provisional labeled images at 22.62 images/s. |
| Label audit | Visually inspected small, median, and large cortical labels of each target before scaled generation. The smallest sample is a 1-pixel SAP annotation. |
| Mining | Job `52720842`, original checkpoints, train split only. |
| Pilot config | `experiments/unity_two_stage_transfer/s0010_perception_pilot_2000_v1.yaml`, SHA-256 `663a20c54df174c1ae42d0ed48b1ae85c3ee1b0c0d92d1832bf5bd8f0b2299aa`. |
| Exact commands | Job context receipts and the reproduction note in §4.5. |

```bash
python workflows/perception_s0010/render_dataset.py --protocol "$protocol" --profile "$profile" --previous "$previous" --historical "$historical" --legacy-training "$legacy_training" --output "$dataset/render" --benchmark-receipt "$benchmark/render/receipt.json" --headless --device cuda:0
```

### 2.3 Results

| Split | Images | Poses | New or shared fields | L4 TP / SAP positives | L5 TP / SAP positives |
|---|---:|---:|---|---:|---:|
| unseen speckle | 340 | 170 | 394310, 394311 | 97 / 28 | 67 / 56 |
| unseen geometry | 110 | 55 | 394300, 394301 | 35 / 15 | 23 / 21 |
| combined | 99 | 33 | 394320, 394321, 394322 | 15 / 3 | 15 / 17 |
| train | 2000 | 785 | 394300, 394301, 394302, 394303, 394304 | 574 / 157 | 386 / 365 |

The combined minimum separation is 3.001 mm from pilot training and 3.006 mm from original and E3 training. The distinct pilot poses include 99 longitudinal and 686 transverse. Image-origin displacement has median 1.46 mm and p95 3.00 mm. Rotation perturbation has median 1.49 degrees and p95 2.50 degrees. Per-image and per-class pixel and connected-component counts are in the data receipt.

| Level | Unique pilot hard-positive frames | Unique pilot hard-negative frames | TP negatives | SAP negatives |
|---|---:|---:|---:|---:|
| L4 | 197 | 24 | 8 | 17 |
| L5 | 413 | 51 | 15 | 40 |

**Verdict.** The 2,000-image training cap was met, and the held-out groups satisfy the documented overlap checks.

### 2.4 Reading it

- The familiar-geometry validation intentionally shares the selected poses and uses new fields. The geometry validation shares fields and uses separate source acquisitions and a spatial buffer.
- The 99-image combined split is small. Sparse L4 SAP has three positive images, so its held-out frame metric is descriptive.
- These are simulator-proxy labels on one anatomy. Bright-interface support may change with speckle and does not give independent anatomical annotation.

### 2.5 Files

| Artifact | Content |
|---|---|
| `dataset-52720837/render/manifest.jsonl` | Per-image calibrated pose, checksum, source grouping, speckle, both label summaries |
| `pilot-training-52723139/fit/training_split.json` | Exact common 2,000 training image records |
| `supplement-52725009/split_audit.json` | Cross-split overlap and selected hard-case counts |
| `benchmark-52720256/render/reviewed_labels.png` | Visually audited provisional label cases |
| `mining-52720842/fit/L4/mining.png`, `L5/mining.png` | Representative hard positives, other-bone negatives, and controls |

## 3. Training and calibration

`fit.py` trains the original architecture. `selection.py` compares both threshold arms and their point maps.

**Criterion.** Checkpoints are selected within each training arm by mean development interface F1 across labels and grouped validation splits. Architecture, inference preprocessing, point mapper, and 3-pixel acceptance remain fixed.

### 3.1 Method

- T1 uses the original weighted cross-entropy, foreground Dice, and negative-frame confidence penalty. T2 changes sampling alone. T3 preserves the T2 sampler and replaces Dice with foreground Tversky overlap using 0.35 false-positive and 0.65 false-negative weight.
- All arms start directly from the original level checkpoint, run up to 22 epochs with a minimum of 14, and use AdamW, learning rate 0.00012 and batch size 12. Checkpoints are saved at each epoch, with the best and final files retained.
- Each model is evaluated at the original thresholds. The calibrated candidate uses the same component minima and tests cutoffs on grouped development validation. The prescribed precision floor is 0.85 or original precision minus 0.02, whichever is greater, with at least eight accepted frames and clear-negative false acceptance limited to baseline plus 0.02. If no cutoff qualifies, retain the original threshold.
- Precision is correct-interface-positive frames divided by frames with any accepted component. Recall is correct-interface-positive frames divided by oracle-visible positive frames. Pixel Dice and IoU use argmax on nonignored pixels. Localization is closest accepted representative distance on frames with visible target.
- Record the selected checkpoint and thresholds before opening the combined test. Only original, E3, and that already selected pair are evaluated on combined geometry and fields.

### 3.2 Setup

| Input | Value |
|---|---|
| Training | Job `52723139`, command below |
| Selection and comparison | Job `52723344`, same design and source data |
| Grouped validation | 340 new-field/familiar-geometry, plus 110 familiar-field/new-geometry images |
| Sealed final combined validation | 99 images, new poses and fields |
| Threshold and readiness definitions | `s0010_perception_campaign_v1.yaml`, SHA-256 `97dd93a3bc3495c8b7c8dd7dc421762e78348aef2f676c0ad8959ff0d0b03821`. |

```bash
python workflows/perception_s0010/fit.py --protocol "$protocol" --profile "$profile" --data "$dataset/render" --legacy-dataset "$legacy_dataset" --legacy-training "$legacy_training" --diagnosis "$benchmark/diagnosis/mapping.json" --mining "$mining/fit" --pilot "$pilot_config" --output "$training/fit" --device cuda:0
python workflows/perception_s0010/selection.py --protocol "$protocol" --profile "$profile" --previous "$previous" --historical "$historical" --development "$development" --data "$dataset/render" --training "$training/fit" --output "$evaluation/metrics" --frozen "$artifacts/frozen" --pilot "$pilot_config" --device cuda:0
```

### 3.3 Results

| Training level and arm | Epochs | Best epoch | Best grouped interface F1 | Checkpoint SHA-256 |
|---|---:|---:|---:|---|
| L4 T1 | 18 | 13 | 0.835 | `83d2e73c6c2f9452664359b09bc9471a26c35c74760901404b49f166fb7343b9` |
| L4 T2 | 14 | 7 | 0.860 | `cd4136dca3db7114bea8e637bb8691aeb449b814c517d27356fd0f869bc922f0` |
| L5 T1 | 21 | 16 | 0.736 | `ea4de45a8df4851eae3cfa92dd82600710ef6bda3549b93a4d050c2b7fb4be5b` |
| L5 T2 | 14 | 9 | 0.732 | `5e8e4aeb6cbcb8006b45734a1da02155efd7167849cea76448d3529a2b4df710` |
| L5 T3 | 16 | 11 | 0.726 | `26b648777b88873a5b7a22aa6727d56f06e305151eba42e40ee36b52f0c5ed63` |

For reference, the earlier interrupted run had admitted 4166 images for L5 T1. Its resulting checkpoint and logs are isolated from the pilot.

| Combined held-out split | Checkpoint | Correct / accepted | Correct / visible | Dice | IoU | FPR on target-absent frames | FNR | Median / p95 representative error (px) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| L4 TP | original | 11/12 | 11/15 | 0.713 | 0.554 | 1/84 | 4/15 | 0.00 / 2.00 |
| L4 TP | E3 | 11/12 | 11/15 | 0.713 | 0.554 | 1/84 | 4/15 | 0.00 / 2.00 |
| L4 TP | selected | 11/12 | 11/15 | 0.713 | 0.554 | 1/84 | 4/15 | 0.00 / 2.00 |
| L4 SAP | original | 3/11 | 3/3 | 0.164 | 0.089 | 8/96 | 0/3 | 0.00 / 2.01 |
| L4 SAP | E3 | 3/11 | 3/3 | 0.164 | 0.089 | 8/96 | 0/3 | 0.00 / 2.01 |
| L4 SAP | selected | 3/11 | 3/3 | 0.164 | 0.089 | 8/96 | 0/3 | 0.00 / 2.01 |
| L5 TP | original | 6/12 | 6/15 | 0.596 | 0.425 | 6/84 | 9/15 | 1.50 / 2.24 |
| L5 TP | E3 | 6/10 | 6/15 | 0.703 | 0.542 | 4/84 | 9/15 | 1.50 / 2.24 |
| L5 TP | selected | 11/14 | 11/15 | 0.861 | 0.756 | 2/84 | 4/15 | 1.00 / 3.09 |
| L5 SAP | original | 6/11 | 6/17 | 0.403 | 0.253 | 4/82 | 11/17 | 1.00 / 3.19 |
| L5 SAP | E3 | 6/8 | 6/17 | 0.383 | 0.237 | 1/82 | 11/17 | 1.00 / 2.82 |
| L5 SAP | selected | 12/13 | 12/17 | 0.632 | 0.462 | 1/82 | 5/17 | 1.00 / 1.78 |

| Robustness check | L5 TP original | L5 TP pilot | L5 SAP original | L5 SAP pilot |
|---|---:|---:|---:|---:|
| unseen speckle development recall | 22/67 | 42/67 | 25/56 | 51/56 |
| unseen geometry development recall | 9/23 | 12/23 | 12/21 | 15/21 |
| Both models tested on repeated speckle at combined held-out poses | 1/3 | 2/3 | 1/5 | 3/5 |

There were six same-field pose pairs in combined validation. No L5 TP pair had visible labels in both poses. For L5 SAP, both poses were visible in all six pairs and both were correctly recognized in 0/6 original versus 1/6 pilot.

Development calibration kept the L5 TP minimum component size fixed and moved its confidence cutoff to 0.975. Grouped development correctness was 54/76 accepted and 54/90 visible at the original threshold, versus 39/45 accepted and 39/90 visible after calibration. Calibration also changed the L4 TP threshold in that arm. Its downstream score dropped to 10/16 and 0/4 complete. The selected operating point retained both original threshold tables. On the grouped development challenge subset, the selected L5 model recovered 27/59 of the original model's missed TP frames.

### 3.4 Reading it

- A higher training or segmentation score needs corroboration from a correct map. The calibration curves expose sensitivity and specificity under the same confidence and region-size definitions.
- Combined test PR curves are descriptive. They were never used to change the selected operating threshold.
- Observations use the unchanged confidence-weighted component representative in the primary comparison. The supplementary fixed-weight comparison tests a maximum-probability representative separately.

### 3.5 Files

| Artifact | Content |
|---|---|
| `evaluation-52723344/metrics/learning_curves.png` | Grouped validation and training curves by arm |
| `evaluation-52723344/metrics/calibration.json` | Original and calibrated grouped-validation results, cutoff rationale, hard-case outcomes |
| `evaluation-52723344/metrics/combined_validation.json` | All four held-out class scores, support ratios, PR curves and interface distances |
| `evaluation-52723344/review/paired_segmentation.png` | Both models and oracle on identical images |
| `evaluation-52723344/review/combined_reliability.png` | Confidence vs observed interface correctness |
| `evaluation-52723344/review/combined_interface_localization.png` | Distance summaries |

## 4. Downstream readiness and handoff

**Criterion.** At least three of the four fixed development maps complete and zero wrong mappable identities, with the two reliable landmarks preserved on the historical sessions. The untouched combined test requires at least 12 L5 TP positive frames, a recall increase of at least 0.10, a precision loss of at most 0.02, and recall loss of at most 0.05 on L4 TP and L5 SAP.

### 4.1 Method

- Compare the original and E3 baselines and train both fixed-confidence and development-calibrated pilot model combinations on exactly the same valid frames and point mapper. Select on correct identity counts, completeness, absence of wrong identities, and grouped development interface validation.
- Independently replay the four historical 54-waypoint cases and the 816 compatible v6r19 acquisitions under that study's original observation policy.
- The reserved independent seeds are `2026121800` through `2026121803`. If readiness passes, the acquisition shell requires the sealed model/threshold and geometry manifest before it can use those seeds.

### 4.2 Setup

| Input | Value |
|---|---|
| Primary and historical replay | `52723344`, point mapper used by every checkpoint pair |
| Additional historical replay and split verification | `52725009` |
| Dataset and assessment receipts | Listed in the review archive and located under the artifact root below |

### 4.3 Results

| Readiness check | Outcome |
|---|---|
| four development sessions | Pass |
| complete development maps | Fail |
| wrong development landmarks | Fail |
| reliable landmarks | Fail |
| combined l5 tp positives | Pass |
| combined l5 tp absolute recall gain | Pass |
| combined l5 tp precision | Pass |
| combined L4 TP recall preservation | Pass |
| combined L5 SAP recall preservation | Pass |

**Verdict.** FAIL. The fixed 2,000-image pilot is a useful perception starting point, but its four-landmark mapping does not meet the frozen criteria. Independent acquisition was not run.

| Development seed | Actor original L4 TP / SAP / L5 TP / SAP | E3 | Pilot | Oracle |
|---|---|---|---|---|
| 2026121700 | U / C / U / U | U / C / C / C | U / C / W / C | C / C / C / C |
| 2026121702 | C / C / C / U | C / C / U / U | C / C / C / C | C / C / C / C |
| 2026121900 | C / U / C / C | C / U / C / U | C / U / C / C | C / C / C / U |
| 2026121901 | U / C / C / C | U / C / U / C | U / C / C / C | C / C / C / C |

C, W and U mean correct, wrong and unresolved under the fixed visible-interface evaluator. Supporting viewpoints, their projected coordinates, candidate alternatives and dispersion are in downstream.json.

| Historical replay | Model | Correct | Wrong | Unresolved | Complete |
|---|---|---:|---:|---:|---:|
| Four original 54-waypoint sessions | Original | 10/16 | 2 | 4 | 0/4 |
| Four original 54-waypoint sessions | E3 | 12/16 | 1 | 3 | 1/4 |
| Four original 54-waypoint sessions | Pilot | 11/16 | 0 | 5 | 1/4 |
| 816-frame v6r19 replay | Original | 15/16 | 0 | 1 | 3/4 |
| 816-frame v6r19 replay | E3 | 15/16 | 0 | 1 | 3/4 |
| 816-frame v6r19 replay | Pilot | 14/16 | 0 | 2 | 2/4 |

### 4.4 Reading it

- The stronger L5 model solves more true interfaces with better held-out precision and recall. In development seed 2026121700 the selected wrong TP cluster still contained only two correct views out of five. The three remaining representatives were 8 px, 6 px, or had no visible TP at all.
- L4 TP in the four 54-waypoint regression cases remains 4/4. L5 SAP drops from 4/4 to 3/4. Training for fewer spurious views on real acquired data is more directly supported than increasing the image count on its own.
- The separate maximum-probability representative gave 13/16 correct, 1 wrong, 2 unresolved and 2/4 complete across the same four sweeps. It played no role in readiness.
- Preserve the current mapper and audit the three off-interface supports before scaling. Keep the four reserved seeds untouched for a future candidate that passes the same readiness gate.

### 4.5 Files

Artifact root `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/artifacts/perception-campaign-073638`.

| Artifact | Content |
|---|---|
| `evaluation-52723344/metrics/downstream.json` | Every attempted development and historical session, maps, competing hypotheses, support and thresholds |
| `evaluation-52723344/metrics/selected_before_combined.json` | Frozen candidate before opening final validation |
| `supplement-52725009/v6r19_regression.json` | Historical 816-frame regression |
| `supplement-52725009/representative_sensitivity.json` | Observation extraction diagnostic outside readiness |
| `evaluation-52723344/review/landmark_sweep.mp4` | Fresh development robot motion with recorded-data replay of the chosen pilot, with an evaluator result separated from actor predictions. |
| `evaluation-52723344/review/video_verification.json` | Actual frame count, decoded size, duration, camera join and SHA-256 |
| `reproduce.txt` | Resolved job commands, input location and scheduler provenance |

No patient generalization, procedural success, or clinical performance is claimed.
