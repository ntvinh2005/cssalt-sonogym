# Observed L5 surfaces and the right L4 medial branch reference on s0010

This experiment audits the historical SAP–TP midpoint against the complete L5 CT segmentation. It tests a bounded surface estimator on recorded, matched Oracle-visible and U-Net observations.

## Result

**Both replay arms leave all four sessions unresolved. The CT audit localizes the historical proposal near the junction but does not establish the required external cortical reference.**

| Step | Evidence | Decision |
|---|---|---|
| CT audit | 3/13 historical midpoint samples assign to full L5 lamina in the source sampling grid. 5/11 SAP core voxels are deeper than 0.75 mm from the full L5 exposed faces. | C. The anatomically appropriate lateral neck reference is unresolved. |
| Matched replay | 0/4 candidates in both arms. Before and after eligibility filtering, the Oracle-visible TP–SAP separation ranges from 9.92 to 10.37 mm. | The observed TP and SAP surfaces do not support a bounded transition under the frozen 3 mm adjacency criterion. Anatomical localization is not evaluable. |
| New confirmation | Recorded development gate only | Reserved seeds unopened. |

The target identity is the **right L4 medial branch** at a **right L5 bony reference**, where the **TP root** and **lateral SAP neck** meet. The historical nearest-mask midpoint is an uncertain geometric sample set. Its relevance to a single cortical region remains unresolved. The historical evaluator and outcome are retained.

## 1. Anatomy audit — Where does the historical geometric target lie?

`workflows.mbb_observed_surface.audit` reproduces the historical construction with the original loader and measures it against the full source L5 and the corresponding CT.

**Criterion.** A new reference requires anatomical justification on the external cortex at the TP root and lateral SAP neck. Anatomical plausibility from a model label and observation coverage alone cannot confirm the boundary. Distances are descriptive. No clinical distance tolerance is applied.

### 1.1 Method

- Reuse the V3 SAP and TP proposal masks within right source L5. Recover the 11 SAP core voxels, 3 TP core voxels and 13 nearest-core midpoint samples at the original 1.50 mm lattice spacing.
- Use the six-neighbour exposed faces of the complete L5 segmentation for distance. For points at a voxel boundary, nearest-neighbour membership follows the source sampling convention. Inspect CT and surface overlays alongside those counts.
- Sample the central 10–90% of each unique nearest-core segment. Count samples separately in full L5, in model SAP/TP and in the remaining model classes.
- Review the physical relationship using RAS CT and full bone. Compare the engineering construction with the lateral-neck course and mammillary–accessory anatomy reported in the primary literature. The work on RFA contributes anatomical description. This experiment uses geometry and sampling scales for its technical rules. [Tran, Peng and Loh 2022](https://pubmed.ncbi.nlm.nih.gov/35589133/) · [Tran et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11372964/) · [Shuang et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5291620/)

### 1.2 Setup

| Setting | Value |
|---|---|
| Original construction | `source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/l4_medial_branch_target.py`, `build_target_contract_geometry` |
| Source CT SHA-256 | `eff5e5f5829cf8cc3e413b3ac09ce455e45bc0f91f5ba37fab9b37b1fe683546` |
| Full L5 source segmentation SHA-256 | `0d435584ed7851736f044c0a9c906231ef3f139b7cda16770d3066773992a1ed`, label 27 |
| Proposal SHA-256 | `a78daeae9159f0df28ba6fffcb8e8d7ef6c2e87f490c9598acfcb82efad1677f`, V3 job `52078067` |
| Audit | Slurm `52790788`, execution `08aea14de4abe405845ecd2bb46a99fb8d1337e5`; two CPUs, zero GPUs |
| Visualization with uncertainty | Slurm `52791353`; one CPU, zero GPUs. Input and numerical audit are identical. |
| Frozen audit SHA-256 | `13f0ef9b6cfa8bc39e8ed60e152a3ca43bf5a159e68dd578d6365a749fbc64ea` |

```bash
project=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym
runtime="${project}-runtime"
study="$project/artifacts/mbb-surface-localization-191322"
manifest="$project/artifacts/mbb-identifiability-114133/shared/manifest.json"
cd "$study/checkout"
sbatch --output="$study/reproduce-audit-%j.log" \
  hpg/jobs/s0010_mbb_observed_surface.sbatch "$runtime" -m workflows.mbb_observed_surface.audit \
  --manifest "$manifest" --ct "$runtime/assets/data/HumanModels/selected_dataset/s0010/ct.nii.gz" \
  --ct-sha256 eff5e5f5829cf8cc3e413b3ac09ce455e45bc0f91f5ba37fab9b37b1fe683546 \
  --output "$study/reproduce-anatomy"
```

All historical samples were included.

### 1.3 Results

| Historical geometry | Count or distance |
|---|---:|
| SAP core in the six-neighbour boundary layer of L5 | 6/11 |
| SAP core deeper than 0.75 mm from exposed L5 faces | 5/11, maximum 2.49 mm |
| TP core in the boundary layer of L5 | 3/3, each 0.75 mm from the nearest exposed face |
| Midpoint inside L5 at native nearest-neighbour sampling | 3/13, all model label 3 (lamina) |
| Midpoint outside L5 at native nearest-neighbour sampling | 10/13 |
| Midpoint unsigned external L5 distance, median / maximum | 0.75 / 1.50 mm |
| Central samples over 13 unique nearest-core segments | 395 |
| Central samples inside complete L5 | 186/395 |
| Of those inside L5, samples outside both SAP and TP proposal masks | 127/186, all lamina proposal |
| Minimum distance between the full SAP and TP proposal surfaces | 6.185 mm |

**Verdict C.** The old proposal lies near the expected bony junction, but the available segmentation and literature cannot identify a defensible replacement contour on this specimen. The midpoint traverses a mixture of non-SAP/TP-labelled bone and space external to the segmented vertebra. It is not used as an accepted anatomical reference for this round.

### 1.4 Reading it

- On the reviewed slices, the SAP core sits on the lower, root-facing part of the model SAP rather than the superior articular cap. Some selected voxels are interior. The exact boundary of the lateral neck remains a review question.
- TP core is on the proximal, superior-facing edge of the TP proposal, toward the root. The three samples alone establish neither the complete root nor the relation to the accessory and mammillary processes.
- The apparent gap includes both model-labelled lamina in full L5 and external space. The unsigned distance and CT partial-volume scale limit interpretation of individual near-boundary samples.
- Anatomical review should trace the lateral neck–root contour across adjacent axial and coronal cuts and mark the mammillary and accessory boundaries independently of local ultrasound quality.
- The uncertainty markers in the 3D figure show the historical expanded core rule. The bone contour, historical proposal and uncertainty are separately identifiable. A new contour is unresolved.

### 1.5 Files

| File | Contents |
|---|---|
| `anatomy/audit.json` | Historical points in NIfTI RAS millimetres, sample occupancy and external-face distances |
| `anatomy-review/ct_review.png` | Nine adjacent CT sections with full L5, old masks, cores and midpoints |
| `anatomy-review/l5_review_3d.png` | Entire L5 and local oblique view with historical uncertainty bands |
| `experiments/unity_two_stage_transfer/s0010_mbb_observed_surface_v1.json` and seal | Reference decision C and actor/evaluator configuration locked before replay |

## 2. Recorded replay — Does the observed surface support a bounded local region?

`workflows.mbb_observed_surface.replay` starts an isolated process per arm for `workflows.mbb_observed_surface.localize`. The process receives a cloud, public frame records and actor rules. The evaluator opens privileged geometry after the eight prediction files are written and hashed.

**Criterion.** The geometrically eligible output is one bounded observed SAP patch adjacent to observed TP. It requires stable evidence from different poses. Anatomical correctness requires an independently reviewed bone reference. A failure to meet the geometric criteria is a measured technical abstention. A candidate, if present, receives descriptive two-direction comparisons with the historical proxy only.

### 2.1 Method

- Use the exact raw dense Oracle-visible and learned clouds from jobs `52739322` and `52739323`. Both use the same 102 eligible frames per session, executed calibrated poses and frozen vertebral-level routing. Prior global coverage was 3,158/3,313 (95.3%) for learned dense and 1,660/3,313 (50.1%) for learned sparse within 1.5 mm of the eligible observed Oracle surfaces. These counts refer to observed samples.
- Reproject every cached pixel from the public executed pose. The parity gate is 0.0015 mm, with 0.5 mm ultrasound image sampling and 200 by 150 pixels. Positive lateral and cranial directions come from fitting the recorded transverse indices through the existing public calibration. The cranial fit permits at most 1.5 index units of residual.
- Consider right L5 TP–SAP pairs separated by at most 3 mm. Fit a local plane within the existing 4 mm spatial-association radius, retaining connected samples linked within 2 mm. Require 0.5 mm of second-axis RMS spread and at most 1.5 mm normal RMS thickness. The two scales represent image and source voxel sampling. Fit weights use confidence normalized per distinct group.
- Use the existing 2 mm or 1° viewpoint separation, distinct image hashes and at least two groups per patch. Orient normals toward the sensor. Require the SAP outward normal to have right component at least 0.55, the TP absolute right component at most 0.75 and the absolute dot product of the two normals at most 0.866. These are exploratory geometric filter values, chosen before opening the replay. Check a 2 mm upward SAP and lateral TP continuation. Allow 1.5 mm in the directional consistency inequalities.
- Take the connected set of observed SAP samples only. A second component, fewer than two samples, or a diameter above 6 mm leaves the region unresolved. Evaluate removing each contributing pose group with a minimum of one remaining view for each patch and a maximum two-direction displacement of 3 mm. Report scan-family removal as a separate sensitivity result. Surface normal and adjacency are inferred. The output contains no extrapolated points.
- If there is an output, report directed nearest-distance median, P95, maximum and proportion within the old 3 mm technical comparison distance in each direction, along with region diameter and observed/inferred provenance. The historical sample set expresses uncertainty, so these diagnostics cannot certify a new target.

### 2.2 Setup

| Setting | Value |
|---|---|
| Shared development cohort | `2026121700`, `2026121702`, `2026121900`, `2026121901`, all with recorded Stage 1 `cf7_00` |
| Source manifest SHA-256 | `805ad43aba2fc99a26cb586fdee84a6de9cf0bd95dc1857260f6e55c8a4c670c` |
| Unchanged L4 checkpoint | Job `52498821`, `c6b7ecc9bc245a88ccc0a5db31dbfd919b57407988dde5ea8211dd7eea2c1b2a` |
| Unchanged L5 T4 checkpoint | Job `52731568`, `1efc323e9c379bcf657eb59fb793cfa30b98e618033a3abc1966a4f3cc69bf47` |
| Actor rules SHA-256 | `d8a39aeef231b03b2718d0447b444da9a8412bc7074da74c4b9e6cba495578b3` |
| Replay protocol SHA-256 | `054719a8cc73407dafa0588915d8c126ecbb5ebf99433849c80940528ca5a092` |
| Replay | Slurm `52791346`, execution `0a0812874eb1f6f3b74cda3f6dc48a62208d76de`, two CPUs, zero GPUs |
| Accounting | Final replay `52792227`, execution `3e6638b2d9862ab9ffc804542b9650c46be4112e`, 24 s elapsed, two CPUs. All four jobs finished with exit 0. Peak concurrent scheduled use was three CPUs and zero GPUs. Total allocated CPU-slot seconds 153. |

```bash
sbatch --output="$study/reproduce-replay-%j.log" \
  hpg/jobs/s0010_mbb_observed_surface.sbatch "$runtime" -m workflows.mbb_observed_surface.replay \
  --manifest "$manifest" --audit "$study/anatomy/audit.json" \
  --rules workflows/mbb_observed_surface/actor_rules.json \
  --protocol experiments/unity_two_stage_transfer/s0010_mbb_observed_surface_v1.json \
  --output "$study/reproduce-replay"
```

The development set is opened evidence. Nothing was dropped from the selected 408-frame cohort in either arm. The archived four-landmark outputs from `52739322` are copied as separate controls.

### 2.3 Results

| Seed | Frames | Oracle nearest gap, mm | Oracle distinct TP / SAP views | U-Net nearest gap, mm | U-Net distinct TP / SAP views | Actor compute Oracle / U-Net, ms | Candidate Oracle / U-Net | Old paired-core observations | Old landmarks correct Oracle / U-Net |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 2026121700 | 102 | 10.367 | 17 / 15 | 9.608 | 14 / 15 | 53.1 / 47.3 | Unresolved / unresolved | 0 | 4/4 / 3/4 |
| 2026121702 | 102 | 9.923 | 17 / 14 | 8.114 | 12 / 17 | 46.6 / 47.8 | Unresolved / unresolved | 0 | 4/4 / 4/4 |
| 2026121900 | 102 | 10.045 | 16 / 17 | 9.662 | 13 / 17 | 48.5 / 47.1 | Unresolved / unresolved | 0 | 3/4 / 3/4 |
| 2026121901 | 102 | 10.290 | 17 / 15 | 9.011 | 13 / 16 | 48.4 / 49.1 | Unresolved / unresolved | 0 | 4/4 / 3/4 |

The eight estimates all stopped before patch fitting. A candidate region, its diameter, distances to the old proxy, and region-specific viewpoint counts are unavailable in all eight cases. Both directions of error remain unevaluated. Each whole-surface support count in the table uses actual separated poses for that structure. All eight actor reprojection checks matched the sealed coordinates exactly. A candidate accepted by the engineering rule appeared in 0/4 sessions in each arm. Anatomical correct and wrong rates have no denominator under reference decision C.

**Verdict. The observed TP and SAP surfaces do not support a bounded transition under the frozen 3 mm adjacency criterion. Anatomical localization is not evaluable.**

### 2.4 Reading it

- The historical paired-core gate yielded 0/4 qualifying sessions on these same acquisitions. Its outputs qualify evidence at fixed endpoint cores. The separate four-landmark maps are landmark estimates. Neither provides a comparable accepted target region on the matched cohort.
- The minimum gap over all image-supported evaluator-labelled Oracle points was unchanged in every session before actor eligibility filtering. Both acquisition families remain separated after either family is held out. Learned gap is smaller at 8.11–9.66 mm, without a connected target-local region. Available scans show each semantic structure across many poses, while the adjacent cortex needed to link the proximal TP and lateral SAP neck remains unsupported in the archived data. This finding applies to the existing visibility labels and sampled views.
- The observability archive retains 766/766 occluded projected core pixels, 900 nominal diagnostic poses, and 0/6 first-visible core images satisfying the frozen acoustic criterion. New poses had no demonstrated executed reach. Oracle support around the old 5 mm diagnostic region was 40/40 eligible points, all TP, and 0/76 of the repeated local SAP CT boundary samples were near visible SAP.
- The independent reference and matched development gates stayed closed. No acquisition adjustment or fresh robot confirmation was justified. The four reserved seeds remain unused. The first replay is retained. A technical correction computes support on each patch’s actual frames to prevent nearby sources assigned to different global groups from inflating coverage. The repeat under unchanged scientific thresholds reproduced all eight decisions, nearest gaps and phase removals. The supplementary before-filter calculation is evaluator-only.
- Focused and regression tests covered separable surfaces, repeated poses and images, competing regions, directional scoring, historic target geometry, dense projection code and the original mapper.

### 2.5 Files

| File | Contents |
|---|---|
| `replay-final/results.json` | Per-session historical controls, new replay evidence, candidate status and timings |
| `replay-final/predictions/` and `prediction_seals.json` | Independent outputs, closest-pair diagnostics and source frame/pixel identities |
| `replay-final/surface_evidence.png` | Observed right L5 surface points from both arms with a separately labeled historical proxy overlay |
| `tests-final.log` and `slurm-accounting.psv` | Executed checks and resource receipts |
