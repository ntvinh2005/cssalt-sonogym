# Setup status

## 2026-11-20: SAP V3 review metric and conflict categories corrected

CPU job `52085026` completed `0:0` in 20 seconds at source revision
`405e7b5d` with one CPU, 8 GB requested memory and zero GPUs. It reused the
frozen V3 raw prediction and did not repeat inference or modify any model mask.

Independent review correctly identified that `segmentation > 0` was not a
bone-only gate. The corrected full-volume audit places `109,139 / 110,736`
(98.56%) model foreground voxels in raw source IDs mapped to acoustic bone
class 13; 1,597 are outside, including 1,576 on source label 120. The prior
100% whole-bone statement is invalid and superseded.

Distance proximity and sampled semantic class are now independent. The compact
package contains numbered MPR callouts for 4 L4 and 7 L5 distance outliers plus
7 L4 and 1 L5 sampled-IAP conflicts, 19 points total. The 22.198 mm L5 outlier
is preserved for review rather than deleted or relabelled.

The proposal verdict remains `MODEL_SAP_PROPOSAL_READY_FOR_REVIEW`; scientific
status remains `SAP_GT_NOT_VALIDATED`, `ULTRASOUND_SAP_NOT_CONFIRMED` and
`MBB_END_TO_END_BLOCKED`. All 11 focused tests, 8/8 checksums and ZIP integrity
pass.

- Result: `experiments/unity_two_stage_transfer/s0010_sap_semantic_review_v3_correction_result.json`.
- Report: `docs/research/sonogym_s0010_sap_semantic_review_v3_correction.md`.
- ZIP: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_sap_semantic_review_v3_correction/52085026/review_packages/s0010_right_l4l5_sap_semantic_review_v3_correction.zip`.
- ZIP SHA-256: `06e7719b228382dcba04dc4dbb5e9b3ca7ec45d8a4204f86ff696c5bacc4405c`.

## 2026-11-20: pretrained CT model produces inspectable RIGHT L4/L5 SAP proposals

GPU job `52078067` completed `0:0` in 5 minutes 29 seconds at source revision
`0bb79e0c` with one NVIDIA L4, four CPUs and 30 GB requested memory. It ran the
official MICN-Lab Spine_Subregions `V1` fold-0 checkpoint once on the original
s0010 CT; no training, fine-tuning, new ultrasound capture or actor change was
performed.

The raw prediction contains labels 0–7 and preserves the input affine exactly.
The original `110,736 / 110,736` whole-bone metric was later invalidated and is
superseded by the explicit-ID correction above. RIGHT L4 SAP has 259 model voxels and supports `51 / 55` V2 core
locations within 1.5 mm. RIGHT L5 SAP has 338 model voxels and supports
`70 / 77`; its 22.198 mm maximum V2/model distance remains an explicit review
outlier despite a 1.5 mm median.

Both structures are **`MODEL_SAP_PROPOSAL_READY_FOR_REVIEW`**, not accepted
semantic ground truth. The four archived ultrasound examples contain only 2–5
projected SAP pixels each, so they do not establish acoustic visibility or SAP
identity. The package passes 49/49 references, 60/60 checksums, ZIP integrity,
7/7 V3 focused tests and 10/10 V1/V2 regressions; third-party weights are not
included.

- Result: `experiments/unity_two_stage_transfer/s0010_sap_semantic_review_v3_result.json`.
- Report: `docs/research/sonogym_s0010_sap_semantic_review_v3.md`.
- ZIP: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_sap_semantic_review_v3/52078067/review_packages/s0010_right_l4l5_sap_semantic_review_v3.zip`.
- ZIP SHA-256: `ce55b17345507a5baa2ee6552daeb63c924e162be285507f3bfb9c8750f3a442`.

## 2026-11-20: SAP review V2 narrows paired-surface evidence but source semantics remain insufficient

The CPU-only `s0010_right_l4l5_sap_review_v2` build completed at commit
`7a91337e` with one CPU thread, zero GPUs, 51.27 seconds elapsed and 696,480
KB peak RSS. Slurm was unavailable (`slurm3,slurm4 is DOWN`), so this bounded
build ran on `login12` and records that execution mode explicitly; it is not
reported as a scheduler job.

All source STL zero-area faces were explained by repeated physical corners
after integer-coordinate quantization. Separately reconstructed label surfaces
have zero degenerate faces and remain aligned at the expected half-voxel-corner
offset. The V2 paired-level review proposes 55 unique L4 and 77 unique L5
surface locations while retaining 172 and 177 unique locations as uncertain;
no location was assigned to likely non-SAP anatomy without source semantics.

Both structures remain **`SOURCE_SEMANTICS_INSUFFICIENT`**. No archived L4
intersection keeps the full proposed core at least 10 pixels inside the image;
L5 has 65 such frames, but neither structure has independently confirmed SAP
cortical visibility or semantic identity. The package passes 57/57 references,
69/69 checksums, ZIP integrity, 7/7 focused tests and 15/15 final regression
tests.

- Result: `experiments/unity_two_stage_transfer/s0010_sap_static_review_v2_result.json`.
- Report: `docs/research/sonogym_s0010_sap_semantic_boundary_correction_v2.md`.
- ZIP: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_sap_static_review_v2/manual_20261120T0120/review_packages/s0010_right_l4l5_sap_review_v2.zip`.
- ZIP SHA-256: `3a05422fbd3197274f56bc6b2539681eba22d4e9c8ba2fc880851fadc542ab80`.

## 2026-11-19: static RIGHT L4/L5 SAP review package is ready

CPU job `52061219` completed `0:0` in 50 seconds with four CPUs, 16 GB
requested memory and zero GPUs at source revision `e1985885`. It generated the
self-contained `s0010_right_l4l5_sap_review_v1` package from the existing
versioned SAP candidates, audit job `52053307` and 336 V3 ultrasound frames.

No annotation geometry was changed. RIGHT L4 SAP remains `UNRESOLVED` with
2,198 candidate/uncertainty vertices and zero proposed semantic vertices;
RIGHT L5 SAP remains `UNRESOLVED` with 2,422 candidate/uncertainty vertices and
zero proposed semantic vertices. The package contains 13 review figures and six
distinct curated ultrasound examples per structure, editable geometry, decision
sheets and a response template.

All 66 referenced paths exist, 79/79 checksums pass, the ZIP integrity test
passes and 5/5 focused tests pass. The 14,792,823-byte ZIP SHA-256 is
`543886058f7d7c6390b5c008ff787c4bdafae8b0bd56682cf2d62dd0276065fd`.
This is review evidence only: no semantic label, MBB target, nerve coordinate,
actor input or intervention capability was created.

- Result: `experiments/unity_two_stage_transfer/s0010_sap_static_review_v1_result.json`.
- ZIP: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_sap_static_review_v1/52061219/review_packages/s0010_right_l4l5_sap_review_v1.zip`.

## 2026-11-19: SAP audit needs review; L4 verifier remains inconclusive

CPU job `52053307` completed `0:0` at source revision `097fda3`. The fixed
`s0010` assets contain whole L4/L5 vertebral meshes and labels, and exact-pose
projection passed on 336 V3 frames with maximum coordinate disagreement
`4.0528e-05` volume index. They do not contain separate SAP/facet semantics:
all 2,198 L4 and 2,422 L5 SAP candidate vertices remain uncertain, no semantic
training label or MBB surrogate coordinate was emitted, and the anatomy verdict
is **`SAP_CANDIDATES_NEED_REVIEW`**.

CPU job `52054026` completed `0:0` at source revision `5a51de3`. The image-only
L4 verifier improved calibration precision from `0.8261` to `0.9474` and reduced
false positives from four to one, but opened-development recall fell from
`0.8065` to `0.5806`. V3 reconstruction confirms sufficient L4 search coverage
in both failed sessions, but the verifier still cannot separate one session and
would reject all four historically selected L4 views. With no unopened
session-separated holdout or physical alternative return/confirmation, its
verdict is **`VERIFIER_INCONCLUSIVE`** and active integration remains on hold.

The project is not ready to construct branch-specific MBB surrogate targets or
start a needle/injection simulator. No GPU, new capture, complete MBB rerun or
intervention was executed.

- Results: `experiments/unity_two_stage_transfer/s0010_sap_geometry_audit_v1_result.json`
  and `experiments/unity_two_stage_transfer/s0010_l4_boundary_verifier_v1_result.json`.
- Report: `docs/research/sonogym_s0010_mbb_anatomy_verifier_investigation.md`.
- Runtime: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_sap_geometry_audit_v1/52053307/`
  and `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_l4_boundary_verifier_v1/52054026/`.

## 2026-11-19: integrated fixed-s0010 MBB POC remains BLOCKED

GPU job `52039283` completed `0:0` in 11 minutes 36 seconds with one NVIDIA L4,
four CPUs and 30 GB requested memory at execution commit `79d3f355`. Stage 1
again passed 4/4 sessions with 28/28 correct landmarks, 3.0 mm median and 10.0
mm maximum localization error.

The integrated Stage 2 actor accepted both RIGHT L4/L5 TP surface-core targets
in all four starts, acquired 336 distinct frames and returned within 0.250 mm
and 0.141 degrees. Independent proxy evaluation passed 6/8 targets: RIGHT L5
was 4/4, while RIGHT L4 was 2/4 because high-probability views overlapped the
provisional boundary-uncertainty region. The run therefore failed its frozen
all-target gate.

No simulated MBB intervention ran. RIGHT L4/L5 SAP boundaries, branch-specific
MBB target-region construction and an MBB needle/injection simulator contract
remain unavailable; the existing guided-surgery environment is a pedicle
drill/screw task and cannot be relabelled as MBB. The milestone verdict is
**`BLOCKED`**, end-to-end success is 0/4, and historical NO_GO results remain
unchanged.

- Result: `experiments/unity_two_stage_transfer/s0010_mbb_single_anatomy_poc_v3_result.json`.
- Report: `docs/research/sonogym_s0010_mbb_single_anatomy_poc.md`.
- Runtime: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_mbb_single_anatomy_poc_v3/52039283/`.

## 2026-11-19: L4 simulator-proxy improvement demonstrated on protocol v16

GPU job `52009585` completed `0:0` in 6 minutes 55 seconds with one NVIDIA L4,
four CPUs and 30 GB requested memory at commit `3be07395`. All 192 observations
across two train, one calibration and one development session passed native-map
replay, synchronized full-pose replay, freshness, viewpoint-diversity and
serialized-array integrity; each session contained 48 distinct image hashes.

The selected 240-tree L4 one-vs-rest proxy classifier improved sealed-development
TP/FP/FN from `10/5/4` to `11/1/3`: precision increased from `0.6667` to `0.9167`,
recall from `0.7143` to `0.7857`, ambiguous activations decreased from three to two,
and clean-negative detections decreased from seven to one. The checkpoint SHA-256
is `6188c7713cac2f21485d030ac79f4e26ac9c4f0f179008aee57a6d891fe9e8c0`, and
save/reload verification passed. L5 remained frozen.

The cycle status is **`IMPROVEMENT_DEMONSTRATED`**. This remains fixed-anatomy
`SIMULATOR_GEOMETRY_PROXY` evidence: `ACTIVE_ACQUISITION_INTEGRATION_HOLD` remains
until researcher review, and no whole-TP, SAP, MBB-target, patient-generalization,
or Stage 3 claim is made.

- Result: `experiments/unity_two_stage_transfer/s0010_tp_surface_core_proxy_l4_improvement_cycle_result.json`.
- Report: `docs/research/sonogym_s0010_tp_surface_core_proxy_l4_improvement_cycle.md`.
- Runtime: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_tp_surface_core_proxy_l4_coverage_repair_v4/52009585/`.

## 2026-11-19: tri-state proxy audit passes, but existing L4 hard negatives are insufficient

CPU job `51956725` completed `0:0` in 70 seconds with four CPUs, 16 GB requested memory and zero
GPUs at commit `0a5ebe54`. It assigned all 5,469 existing proposals to explicit L4 proxy-positive,
proxy-negative or proxy-ambiguous states without fitting, capture or development training use.

The 14 opened-development L4 false positives divide into seven ambiguous and seven clean-negative
images; historical results remain unchanged. Training-only mining found 16 false-trigger proposal
rows but only ten independent clear hard-negative images after retaining at most one patch per
source image. This is below the predeclared 40-image gate, and no competing-L5-core hard negative
qualified.

Readiness is **`DATA_INSUFFICIENT`** for L4 v2: no retraining protocol or checkpoint was produced.
The exact deficit is 30 independent clear hard-negative training images, with supported L5
confusers needed if L4-versus-L5 repair is part of the next experiment. The margin result remains
`FAIL`, proxy v1 remains `EXPLORATORY_TRAINING_PASS`, and active acquisition remains `HOLD`.

- Result: `experiments/unity_two_stage_transfer/s0010_tp_surface_core_proxy_tristate_hard_negative_v1_result.json`.
- Report: `docs/research/sonogym_s0010_tp_surface_core_proxy_tristate_hard_negative_v1.md`.
- Runtime: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_tp_surface_core_proxy_tristate_hard_negative_v1/51956725/`.

## 2026-11-19: L4 margin calibration fails the clean-negative repair gate

CPU job `51953988` completed `0:0` in 39 seconds with four CPUs, 16 GB requested memory and zero
GPUs at commit `a841e25`.  It scored 1,062 proposals across 110 calibration images using the frozen
checkpoint and unchanged 0.50 L4/L5 thresholds; no fitting or development-image access occurred.

The whole-image L4 baseline was TP/FP/FN/TN `9/3/3/95`.  All three false positives had accepted
proposals overlapping uncertainty, so clean-negative FP was already zero.  Margin 0.175 retained all nine TP and reduced
total FP from three to two, but it removed only one of those cases and therefore failed the
predeclared clean-negative reduction criterion.  No margin was selected, no frozen-margin file was
written and no new development acquisition is justified for this rule.

`EXPLORATORY_TRAINING_PASS` and `ACTIVE_ACQUISITION_INTEGRATION_HOLD` remain unchanged.  Subsequent
work should address coverage of proxy-negative bone-interface examples and the simulator-proxy uncertainty/label contract
rather than further margin tuning.

- Result: `experiments/unity_two_stage_transfer/s0010_tp_surface_core_proxy_l4_margin_calibration_v1_result.json`.
- Report: `docs/research/sonogym_s0010_tp_surface_core_proxy_l4_margin_calibration_v1.md`.
- Runtime: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_tp_surface_core_proxy_l4_margin_calibration_v1/51953988/`.

## 2026-11-18: frozen L4 full-image failure audit isolates proxy coverage and model errors

CPU job `51936904` completed `0:0` in 29 seconds with four CPUs, 16 GB requested memory and zero
GPUs at commit `6688af7`.  It reproduced the frozen full-image confusion exactly: L4 TP/FP/FN/TN
`12/14/5/72` and L5 `16/4/3/80`; no fitting or threshold change occurred.

The 14 L4 false positives are heterogeneous: seven contain projected L4 core evidence excluded by
the strict uncertainty/support contract, four are clean proxy-negative morphology confusions, two
overlap the competing L5 core, and one is primarily proposal multiplicity.  Three images have
multiple L4-triggering proposals (18 total triggers), but multiplicity alone does not explain the
precision loss.  L4 errors occur in every development session and are worst in the L4-focused V9
session; L5 false positives are confined to V6.

The researcher decisions remain **`EXPLORATORY_TRAINING_PASS`** and
**`ACTIVE_ACQUISITION_INTEGRATION_HOLD`**.  The next proposed test keeps the checkpoint and base
threshold frozen, selects an L4 margin using calibration only, and evaluates
once on a newly designated session-disjoint development set while reporting proxy-uncertainty
activations separately.

- Result: `experiments/unity_two_stage_transfer/s0010_tp_surface_core_proxy_l4_failure_audit_v1_result.json`.
- Report: `docs/research/sonogym_s0010_tp_surface_core_proxy_l4_failure_audit_v1.md`.
- Runtime: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_tp_surface_core_proxy_l4_failure_audit_v1/51936904/`.

## 2026-11-18: isolated TP simulator-geometry proxy pilot completed

CPU job `51836756` completed `0:0` in 27 seconds with four CPUs, 16 GB requested memory and zero
GPUs at commit `4d0500e`.  The separate `SIMULATOR_GEOMETRY_PROXY` no-fit path audited all 5,469
v3 manifest rows, matched the sealed ZIP/manifest/geometry/source-image hashes, excluded smoke,
found no cross-split exact duplicates and returned `PROXY_TRAINING_READY` with `fit_called=false`.

The 240-tree ExtraTrees pilot then fitted 463 train samples, selected both target thresholds using
151 calibration samples, and evaluated 139 development patches plus every actor-visible proposal
across 103 development images.  Patch accuracy was 125/139 (89.93%); whole-image L4 precision was
12/26 (46.15%) and L5 precision was 16/20 (80.00%).  The checkpoint was saved and reload-verified,
but the result is explicitly simulator-proxy performance, not anatomical recognition or MBB
readiness.  The verified-label status below remains `ENGINEERING_READY_LABEL_REVIEW_PENDING`.

- Result: `experiments/unity_two_stage_transfer/s0010_tp_surface_core_simulator_proxy_exploratory_v1_result.json`.
- Report: `docs/research/sonogym_s0010_tp_surface_core_simulator_proxy_exploratory_v1.md`.
- Runtime: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_tp_surface_core_simulator_proxy_exploratory_v1/51836756/`.

## 2026-11-17: TP support-qualified coverage complete; exact label decisions remain

GPU job `51831979` completed `0:0` in 7:02 on one L4 and four CPUs after CPU preflight job
`51831912` passed 98/98 tests.  All 180 predeclared supplemental observations passed movement,
freshness, synchronized exact-pose replay and serialized-array gates.  No new capture frame was
dropped.

The unchanged support-aware builder now reports background `347/110/103`, L4 RIGHT TP surface core
`55/12/17`, and L5 RIGHT TP surface core `61/29/19` unique source images for
train/calibration/development.  All three classes clear 40/10/10; the prior five coverage deficits
are closed, there are no cross-split exact-image duplicates, and the v3 package contains 180
primary plus 63 reserve exact-patch assets.

Readiness is **`ENGINEERING_READY_LABEL_REVIEW_PENDING`**.  The user has authorized the bounded CPU
pilot once ready, but zero rows are currently eligible because the checksum-bound geometry,
visibility and semantic decision template has not been completed.  No-fit resolved all 5,469 real
features and recorded `fit_called=false`; fitting must follow automatically only after review import
and `TRAINING_READY_FOR_APPROVAL`.

- Result: `experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_v10_result.json`.
- Review ZIP SHA-256: `dd24c07541bf8e5d6ffe07c7904ec104996a3bb11d39f421f90e4c75fbcc423e`.
- Runtime: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_tp_surface_core_dataset_v10/51831979/`.

## 2026-11-17: TP exact-crop reselection exposes a real support-qualified data deficit

CPU jobs `51824166` and `51825014` passed 93/93 regressions and rebuilt the complete review
package from the existing synchronized captures with four CPUs and zero GPUs.  Candidate selection
now compares every eligible proposal in a source image before spending its review budget; the old
L4-06 and L5-52 one/two-pixel border fragments are replaced by proposals with 44/53 unique
projected pixels and 31/28 interior feature cells.  Background primaries are distributed across
six training sessions instead of one.

The new package contains 137 primary and 24 reserve samples.  For each it provides the clean full
ultrasound, context, exact 48x80 raw crop, exact normalized 16x16 input, separate privileged
overlay, feature hash and checksum-bound identity.  All 992 package checksums passed; no-fit
resolved all 3,732 real proposals and recorded `fit_called=false`.

The truthful readiness state is **`DATASET_INCOMPLETE`**.  Support-qualified independent image
counts are L4 `28/12/9` and L5 `18/4/8`, below the frozen `40/10/10` gate, and zero labels have
qualified review approval.  Therefore real fitting remains blocked; closing this result requires
a separately approved supplemental capture targeted at the exact deficits plus review of the new
checksum-bound crops.

- Result: `experiments/unity_two_stage_transfer/s0010_tp_surface_core_review_selection_v2_result.json`.
- Review ZIP SHA-256: `e75a31fb4b600ed901eba7e5deb072c3d11eeda36ae0dc5701f19efb2d7aaf6f`.
- Runtime: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_tp_surface_core_review_v2/51825014/`.

## 2026-11-17: TP candidate coverage complete; exact label review remains

V9 completion job `51818619` completed `0:0` in 2:08 on one L4 and four CPUs.  Its 24/24
development frames passed native-map replay, synchronized full-pose replay, freshness,
viewpoint-diversity and array-hash integrity.  The prior V8 failure remains recorded: its requested
2.985-rad development waypoint saturated physically and never reached the frozen 1-degree tracking
tolerance.

The verified combined dataset now has 384 development observations across ten sessions.  Unique
image coverage is background `234/74/73`, L4 RIGHT TP surface core `49/14/14`, and L5 RIGHT TP
surface core `55/15/16` for train/calibration/development, with no deficits or cross-split exact
duplicates.  No-fit resolved all 3,732 actor-visible proposals to finite features and did not call
fit.

Readiness is **`ENGINEERING_READY_LABEL_REVIEW_PENDING`**, not training success: 180 exact review
rows still need checksum-bound boundary and image-semantic decisions, so zero rows are currently
training eligible and no research model was fitted.  Commit `46e25d9` adds a fail-closed importer;
after a qualified review, import plus no-fit is sufficient to reach `TRAINING_READY_FOR_APPROVAL`
without another simulator capture.

- Result: `experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_v9_result.json`.
- Review ZIP SHA-256: `aec872965c310c8659a483a393f92efefb98ba7eca9e5af26f059b998876bb8b`.
- Runtime: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_tp_surface_core_dataset_v9/51818619/`.

## 2026-11-17: TP V6 captures 300 frames; L4 unique-image coverage remains short

The trainer partitions eligible rows before fitting: only `train` enters `fit()`, calibration
selects operating thresholds, and development evaluation remains read-only.  Actor-visible
background proposals, per-split unique-image gates, checkpoint save/reload verification and the
strict serialized-coordinate hash gate are implemented.

GPU job `51811811` completed `0:0` in 6:58 on one L4 and four CPUs.  The eight-frame smoke and all
six 50-frame development sessions passed native-map replay, synchronized full-pose replay,
freshness, viewpoint-diversity and serialized-hash gates.  The 300 development observations have
300 distinct exact hashes; maximum smoke coordinate disagreement was `3.0929e-05` volume index.

The actor-visible morphology proposal path produced 2,940 finite candidate patches: 414 geometric
positive-review targets, 2,038 hard-negative review targets and 488 ambiguous rows.  Independent
source-image coverage is background `199/50/49`, L4 RIGHT TP core `33/7/8`, and L5 RIGHT TP core
`53/13/16` for train/calibration/development.  L4 is below the declared `40/10/10` planning target
in every split, so correlated patches cannot turn this into a trainable complete dataset.

CPU finalization job `51813176` passed 5 training and 21 capture-contract tests, created 168
split-stratified review assets, and verified 184 / 184 package checksums.  The readiness state is
**`DATASET_INCOMPLETE`**: anatomical and image-semantic review is pending, zero rows are training
eligible, and no real model was fitted.  SAP remains excluded.

- Capture commit: `6ed6259`; finalization commit: `0e5b3c7`.
- V6 result: `experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_v6_result.json`.
- Final review ZIP SHA-256: `259746e83b127c573f47eda31dbecd3d6e8edcb4e62f4953eda442420191402d`.
- Readiness report: `docs/research/sonogym_s0010_tp_surface_core_dataset_readiness.md`.

## 2026-11-17: synchronized TP surface-core smoke passes; dataset remains incomplete

The single authorized GPU job `51793371` captured eight genuinely distinct s0010 viewpoints and
passed reloaded native-map, synchronized full-pose, acquisition-freshness and viewpoint-diversity
gates on 8/8 frames. Raw-label agreement and bone IoU were both 1.0, maximum coordinate error was
`2.8149e-05` index and no sample was clipped. The job then stopped `1:0` because the coordinate
hash had been computed before float64-to-float32 serialization; image and raw-label hashes matched.
The hash source was corrected and the CPU verifier preserved the mismatch as an explicit warning.

No second GPU job was submitted. Consequently the six preassigned sessions (300 frames) are absent,
so the final state is **`DATASET_INCOMPLETE`**. The smoke provides three verified geometric review
intersections (one L4 RIGHT TP surface core, two L5), but zero training-eligible labels because exact
anatomical approval, semantic image review, hard negatives and all train/calibration/development
counts remain missing. No-fit resolved three real crops into 774-value features and returned
`ENGINEERING_PREFLIGHT_ONLY` with `fit_called=false`.

- Report: `docs/research/sonogym_s0010_tp_surface_core_dataset_readiness.md`.
- Result: `experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_v1_result.json`.
- Runtime: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_tp_surface_core_dataset_v1/51793371/`.
- Reviewer ZIP: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/reviewer/s0010_tp_surface_core_dataset_review_51793371.zip`.
- ZIP SHA-256: `ce58a9d375be42f22e76885ffd9fc7bdd3872ff096163f6778415259a3ce037d`.

## 2026-11-17: s0010 exact-pose mapping smoke remains blocked

GPU job `51783444` (`anest-dept`, one L4, four CPUs, 30 GB requested) captured six development
frames after V1 job `51782795` failed before acquisition. All 33 CPU tests passed. Persisted native
pixel-to-volume coordinates reconstructed 6 / 6 raw-label slices exactly, and full-pose replay
passed the historical 0.985 agreement / 0.90 bone-IoU gates on 6 / 6 frames. The sealed protocol
nevertheless failed: full-pose coordinates differed from native coordinates by 0.1918--0.2136
volume index against the frozen 0.001 limit, and only 2 / 6 hashes were distinct because frames
2--5 required zero actions inside the 5 mm movement tolerance.

The result is **`PROJECTION_REPAIR_BLOCKED`** and detector training remains `NOT_RUN`. No third GPU
job was submitted. The next capture requires waypoint spacing larger than the motion tolerance,
mandatory producer advancement, and researcher review of the strict coordinate-level gate; TP
geometry still needs independent approval and SAP boundaries remain unresolved.

- Report: `docs/research/sonogym_s0010_exact_pose_replay_capture.md`.
- Result: `experiments/unity_two_stage_transfer/s0010_exact_pose_replay_v2_result.json`.
- Runtime: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_exact_pose_replay_v2/51783444/`.
- Reviewer ZIP: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/reviewer/s0010_exact_pose_replay_review_51783444.zip`.
- ZIP SHA-256: `0263a99829125431d53c962a6ddcdd25c3f55f56cc08e9b3e2777ced6d79a7f8`.

## 2026-11-17: RIGHT SAP/TP projection and boundary correction

CPU job `51779234` (`anest-dept`, four CPUs, 16 GB requested, zero GPUs) completed `0:0` in 1:29.
All 30 focused and historical regression tests passed. The native width/elevation/depth mismatch
and image-origin rule were corrected, improving median exact raw-label agreement from 0.3002 to
0.8157 and median bone IoU from 0.0000 to 0.5638 across 3,826 frames. Nevertheless, zero frames
passed the unchanged joint 0.985/0.90 gate because existing captures retain only a reduced
three-value pose and not the full end-effector position/quaternion used by the renderer.

The result is **`PROJECTION_REPAIR_BLOCKED`**. L4/L5 RIGHT TP now have editable conservative distal
surface-core proposals with explicit uncertain bases, both
`PROPOSED_REVISION_PENDING_ANATOMICAL_REVIEW`; L4/L5 RIGHT SAP remain `UNRESOLVED_BOUNDARY`.
There are zero verified candidate projections and zero training-eligible samples, so detector
training remains `NOT_RUN` despite researcher authorization to train after annotation. The next
data action is a bounded development capture that records exact per-frame `human_to_ee_pos`,
`human_to_ee_quat`, and preferably sampled pixel-to-volume coordinates, plus qualified review of
the exact TP/SAP geometry version.

- Report: `docs/research/sonogym_s0010_right_sap_tp_correction.md`.
- Result: `experiments/unity_two_stage_transfer/s0010_right_sap_tp_correction_v1_result.json`.
- ZIP: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/reviewer/s0010_right_sap_tp_correction_v1_51779234.zip`.
- ZIP SHA-256: `1dcf71bd463bb53abd72877e831961bb5c70060b2e66cf7229c3ea1ca1f20ebd`.

## 2026-11-17: RIGHT SAP/TP ultrasound-inclusive review v2

CPU job `51764274` (`anest-dept`, four CPUs, 16 GB requested, zero GPUs) completed `0:0` in 1:45
and produced the final reviewer ZIP
`/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/reviewer/s0010_right_sap_tp_review_v2_51764274.zip`
(SHA-256 `9cfc8be656e219043010eea78b2787cd7f4de0c1e7f442c3d5205c0afb5bfc88`). All
16 focused/regression tests passed. The audit joined 3,826 real `conv` frames across 22 opened
development sessions, but zero frames passed the candidate-plane raw-label round-trip gate because
the captures do not preserve the full end-effector transform or per-pixel image coordinates.
Accordingly, candidate overlays were withheld, all four boundaries remain unapproved, and semantic
training remains blocked pending qualified boundary review plus a separately approved targeted
capture with exact image-plane provenance. See
`docs/research/sonogym_s0010_right_sap_tp_review_v2_ultrasound.md`.

Status: **PHASE_4_RUNTIME_READY; PHASE_5A_METRIC_GO; PHASE_5C_STAGE1_GROUP_A_CONFIRMATION_PASS; PHASE_5C_STAGE2_CONFIRMATION_NO_GO; STAGE2_TEMPORAL_REPLAY_EXPLORATORY_NO_GO; STAGE2_CONTRACT_CORRECTED_EXPLORATORY_NO_GO; STAGE2_ACTIVE_ACQUISITION_DEVELOPMENT_PASS; STAGE2_PHYSICAL_GRID_DEMO_DEVELOPMENT_NO_GO; STAGE2_BOUNDED_ORIENTATION_REFINEMENT_DEVELOPMENT_PASS; STAGE2_NEW_START_CONFIRMATION_NO_GO; SELECTION_REPAIR_RECOMMENDED; MBB_STAGE2_BLOCKED_GROUND_TRUTH; MBB_RIGHT_LANDMARK_REVIEW_V2_PACKAGE_READY**

Updated: 2026-11-17 (America/New_York)

## RIGHT SAP/TP expanded-review v2 package

CPU job `51717444` completed `0:0` on `anest-dept` in 39 seconds with four CPUs, 16 GB requested
RAM, and zero GPUs. It passed 10 / 10 tests and 28 / 28 package checksums. The v2 ZIP adds four
15-slice zoomed CT montages, editable CSV/NPZ/PLY geometry for each candidate, source L4/L5 STL,
non-resampled CT/segmentation crops, and a structured reviewer response template.

The package result is **`MBB_RIGHT_LANDMARK_REVIEW_V2_PACKAGE_READY`**; the anatomical result remains
**`MORE_EVIDENCE_REQUIRED`** and Stage 2 remains **`MBB_STAGE2_BLOCKED_GROUND_TRUTH`**. The enlarged
slices reinforce possible anterior contamination in TP candidates and axis-cut/adjacent-posterior
contamination in SAP candidates. No boundary was automatically corrected or promoted; annotation
v2 remains unchanged; ultrasound observability, model training, and target-region construction
remain not run.

- Report: `docs/research/sonogym_s0010_right_sap_tp_review_v2.md`.
- Result: `experiments/unity_two_stage_transfer/s0010_right_sap_tp_review_v2_result.json`.
- ZIP: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/reviewer/s0010_right_sap_tp_review_v2_51717444.zip`.
- ZIP SHA-256: `4e9722d0c51fbe497bba4005caeaa357692ae6e70ac3ddaa6805d06cacf24f83`.

## RIGHT SAP/TP independent-review package

CPU job `51706459` completed `0:0` on `anest-dept` in 17 seconds with four CPUs, 16 GB requested
RAM, and zero GPUs. It verified all source hashes, reconstructed the four RIGHT L4/L5 SAP/TP v2
candidates exactly, passed 5 / 5 focused tests, rendered CT/mesh review figures, and produced a
checksum-sealed ZIP. Peak RSS was 449,608 KiB.

The result is **`MBB_RIGHT_LANDMARK_REVIEW_PENDING`**, not annotation acceptance. L4/L5 TP remain
`PROVISIONAL`; L4/L5 SAP remain `UNRESOLVED`; annotation v2 was not modified; no model was trained;
and no target region was constructed. The next action is a per-landmark independent reviewer
decision (`ACCEPT_BOUNDARY`, `REVISE_BOUNDARY`, or `REJECT_CANDIDATE`) with reviewer provenance and
uncertainty.

- Report: `docs/research/sonogym_s0010_right_sap_tp_review.md`.
- Result: `experiments/unity_two_stage_transfer/s0010_right_sap_tp_review_v1_result.json`.
- ZIP: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/reviewer/s0010_right_sap_tp_review_v1_51706459.zip`.
- ZIP SHA-256: `c7bb77eafcc672493266b785745bee9ea99c0d3c6f62f2c9169662637d293ec5`.

## MBB-oriented Stage 2 contract and readiness audit

CPU job `51697175` completed `0:0` on `anest-dept` in 13 seconds with four CPUs, 16 GB requested
RAM, and zero GPUs. The isolated branch/site taxonomy, causal multi-view landmark records,
candidate memory, information-boundary checks, uncertainty gate, result schema, and fail-closed
target-region interface passed 19 / 19 focused tests. Existing annotation, Stage 1, and frozen Stage
2 regressions passed 61 / 61; the task-specific readiness manifest reproduced byte-for-byte.

The scientific result is **`MBB_STAGE2_BLOCKED_GROUND_TRUTH`**. For both left and right L4-L5
configurations, L4/L5 SAP annotations remain `UNRESOLVED`, L4/L5 TP candidates remain low-confidence
`PROVISIONAL`, medial-branch nerve references are `MISSING`, and existing ultrasound captures do
not establish SAP/TP semantic observability. No detector was trained, no target region was
constructed, no simulator/GPU job ran, and historical `CONFIRMATION_NO_GO` evidence was unchanged.

- Report: `docs/research/sonogym_stage2_mbb_contract_readiness.md`.
- Manifest: `experiments/unity_two_stage_transfer/stage2_mbb_landmark_readiness_v1.json`.
- CPU evidence: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_mbb_contracts/51697175/`.
- Next approval: independent L4/L5 SAP/TP anatomical delineation and review, followed only then by
  a bounded ultrasound-observability pilot.

## Phase 5C Stage 2 angular-selection diagnosis

CPU job `51683029` completed `0:0` on `anest-dept` with four CPUs, 16 GB requested RAM, and zero
GPUs. Focused tests passed 4 / 4 and existing Stage 2 regression tests passed 34 / 34.

All three evaluable new-start runs physically acquired at least one frame meeting the unchanged
position, orientation, L4-visibility, technical, and image gates. All five such frames also passed
the frozen actor threshold and multi-view evidence contract, but the maximum-quality rule selected
an orientation-invalid frame in every run. The primary diagnosis is therefore
`SELECTION_FAILURE`, not search coverage or physical tracking; the recommendation is
`SELECTION_REPAIR_RECOMMENDED`.

The 2.97-rad refinement ceiling was reached closely enough to produce 9.9794, 9.9385, and 9.9379
degree observations without refinement-frame clamping. Start 00 remains a separate technical
freshness failure: a 0.003809-degree clipped waypoint separation was below the 0.25-degree movement
tolerance, produced zero actions, and repeated the producer sequence and hash. The frozen
`CONFIRMATION_NO_GO` result is unchanged and no GPU, simulator, retraining, threshold change, or
retry ran.

- Detailed report: `docs/research/sonogym_stage2_angular_selection_audit.md`.
- Small artifacts: `experiments/unity_two_stage_transfer/stage2_angular_selection_audit/`.
- CPU run: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_angular_selection_audit/51683029/`.

## Phase 5C Stage 2 new-start confirmation no-go

Job `51668777` executed four sealed unused reset seeds sequentially on `anest-dept` with one NVIDIA
L4, four CPUs, and 30 GB requested RAM. The frozen confirmation required all 4 / 4 runs to pass the
unchanged V6R1 actor, return, fresh-confirmation, image, position, and orientation gates.

The result is **`CONFIRMATION_NO_GO`**. Three runs produced sealed actor/evaluator results: all three
actors accepted, returned successfully, passed two fresh confirmations, and met position and image
gates, but orientation errors were 15.2505, 11.8805, and 11.8782 degrees against the unchanged
10-degree gate. The fourth run failed closed before actor sealing when a clipped final refinement
target was only 0.003809 degrees from the previous target, below the 0.25-degree movement tolerance,
so it executed zero actions and could not provide fresh evidence.

The scheduler recorded `FAILED (2:0)` after 8:25 because one evaluator input was absent. The
scientific no-go is nevertheless decisive: 0 / 3 evaluable runs passed and the aggregate required
4 / 4. No retry, retraining, threshold change, V19, or patient-generalization claim followed.

- Detailed report: `docs/research/sonogym_stage2_new_start_confirmation_v1.md`.
- Result manifest: `experiments/unity_two_stage_transfer/stage2_new_start_confirmation_v1_result.json`.
- Run: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_new_start_confirmation_v1/51668777/`.

## Phase 5C Stage 2 bounded-orientation development pass

Technical-retry job `51664534` completed `0:0` on `anest-dept` with one NVIDIA L4, four CPUs,
and 30 GB requested RAM. After the physical 3x3 grid and fine/micro search, the actor sampled five
fresh angular views inside the frozen 2.70--2.97 rad limits, selected frame 43 using the unchanged
validity and quality models, physically returned within 0.2375 mm / 0.0060 degrees, and passed two
fresh confirmations. All 70 tests passed.

The evaluator returned **`DEVELOPMENT_PASS`**: position error was 4.0860 mm, orientation error was
9.9952305295 degrees, and L4 visibility was 0.2570333. The orientation margin is only
0.0047694705 degrees below the frozen 10-degree gate, so this is a numerical fixed-`s0010`
development pass, not evidence of robustness or independent confirmation. V16--V18 remain
`CONFIRMATION_NO_GO`.

Job `51662670` is preserved as a technical replay-subset indexing failure. V6R1 changed only the
replay-local contiguous index while preserving global frame identities and every scientific,
motion, model, and evaluator parameter.

- Detailed report: `docs/research/sonogym_stage2_orientation_refinement_v6r1.md`.
- Result manifest: `experiments/unity_two_stage_transfer/stage2_orientation_refinement_v6r1_result.json`.
- Run: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_orientation_refinement_v6r1/51664534/`.
- Two-video ZIP: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/demo/sonogym_stage2_orientation_refinement_v6r1_20261116.zip` (SHA-256 `1380a7790ab296e9e68e2a1c31a6ebcd3dbad4ae87c601922389523b0fa5793f`).

## Phase 5C Stage 2 physical-grid development demo

Job `51641034` completed `0:0` on `anest-dept` with one L4, four CPUs, and 30 GB requested RAM.
The formalized search used a 3x3 position grid with 6 mm spacing and three executable angles per
position (27 coarse poses). Every coarse pose executed physical actions and produced a fresh,
valid, distinct ultrasound hash. The actor returned `ACCEPT`, used eight supporting viewpoints,
returned within 0.237 mm / 0.004 degrees, and passed two fresh confirmations.

The result remains **`DEVELOPMENT_NO_GO`**: evaluator-only position error was 4.100 mm, but
orientation error was 10.287 degrees, 0.287 degrees above the frozen 10-degree gate. V3's earlier
development pass remains preserved; V5 is a stricter physical-grid demonstration and does not
retroactively change V16-V18 or establish independent confirmation.

- Detailed report: `docs/research/sonogym_stage2_physical_grid_demo_v5.md`.
- Result manifest: `experiments/unity_two_stage_transfer/stage2_active_acquisition_grid_demo_v5_result.json`.
- Run: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_active_acquisition_grid_demo_v5/51641034/`.
- Two-video ZIP: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/demo/sonogym_stage2_physical_grid_demo_20261116.zip` (SHA-256 `efaf0a4a6ec46a5fb62a50b6067d936bbc43a9817685d467d93008c128fb0c08`).

## Phase 5C Stage 2 continuous active-acquisition development pass

Technical-recovery V3 job `51611658` completed `0:0` on `anest-dept` with one NVIDIA L4, four
CPUs, and 30 GB requested RAM. The unchanged fixed-`s0010` actor executed 572 actions and 41
observations, selected a candidate with three distinct supporting viewpoints, physically returned,
and acquired two fresh, distinct, producer-advanced confirmations. Actor outcome was `ACCEPT`.

Return error was 0.1432 mm / 0.0173 degrees. After actor sealing, the independent evaluator found
8.0565 mm position error, 9.5173 degrees orientation error, and L4 visibility 0.25453. All frozen
development gates passed. All 66 tests passed; the job used at most 5,005 MiB GPU memory and 55%
GPU utilization.

V1 `DEVELOPMENT_NO_GO` and V2 technical action-budget failure remain preserved. V3 changed only
the total action budget from 550 to 600 after V2 measured a 552-action requirement; model,
threshold, search, confirmation movement, decision rule, seed, and evaluator gates were unchanged.
This is a fixed-anatomy engineering `DEVELOPMENT_PASS`, not V19 or independent generalization.

- Detailed report: `docs/research/sonogym_stage2_active_acquisition_v3_demo.md`.
- Result manifest: `experiments/unity_two_stage_transfer/stage2_active_acquisition_pilot_v3_result.json`.
- Run artifacts: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_active_acquisition_pilot_v3/51611658/`.
- Demo: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/demo/stage2_active_pilot_v3_51611658/`.
- Review ZIP: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/demo/sonogym_stage2_active_demo_20261116.zip` (SHA-256 `be9dd0d4ae20e89e9df6a6e1fc9607a87c510083e9e960e50bdef5b561fe9d4a`).

## Phase 5C Stage 2 active-acquisition development pilot

Job `51607477` ran one continuous fixed-`s0010` coarse/fine/micro search on `anest-dept` with one
NVIDIA L4, four CPUs, and 30 GB requested RAM. The actor completed 371 action steps and 41
observations, found a candidate supported by three hashes and three viewpoints, and returned to
it with 0.0 mm / 0.0 degree relative error. Evaluator-only geometry (8.197 mm, 9.535 degrees) and
L4 visibility (0.2548) met their development thresholds.

The actor correctly returned `ABSTAIN`: both 0.75 mm confirmation dithers were below the shared
5 mm movement tolerance, so they executed zero actions, did not advance the producer, and repeated
the selected image hash. Fresh confirmation therefore failed and the result is
`DEVELOPMENT_NO_GO`; full Stage 2 remains `NOT_ESTABLISHED`.

The job state is `FAILED (1:0)` because the evaluator initially looked for the wrong actor-seal
filename after the actor completed. Commit `390416b` fixed that packaging bug, and a recovered
evaluation of the sealed artifact produced the same scientific NO_GO without rerunning the GPU.
All 64 tests passed. External output size is 4,916,102 bytes.

- Detailed report: `docs/research/sonogym_stage2_active_acquisition_pilot.md`.
- Result manifest: `experiments/unity_two_stage_transfer/stage2_active_acquisition_pilot_v1_result.json`.
- Artifacts: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_active_acquisition_pilot_v1/51607477/`.
- Next approval: a new development-only confirmation-motion protocol with a dedicated tolerance
  below the frozen 0.75 mm dither; no model, threshold, or search-policy change.

## Phase 5C Stage 2 contract-corrected CPU replay

CPU job `51550463` completed `0:0` on `anest-dept` with four CPUs, 16 GB requested RAM, and
zero GPUs. The replay promoted the audited angle-ceiling exception into the explicit movement
contract while retaining the original frozen behavior as the default. All 64 tests passed and
the original replay had zero decision mismatches.

On the same opened 1,350 frames and 144 trajectories, the explicit contract reproduced 38
temporal accepts and 31 multi-view accepts with zero false accepts. Acceptance occurred in all
four capture sources, but 31 remains below the unchanged 52-accept usefulness condition.
The result therefore remains `EXPLORATORY_NO_GO`; no model, threshold, observation, GPU capture,
V19 protocol, or simulator run was added.

Job `51550144` is retained as a 10-second technical failure after all tests passed. Its source
count assertion referenced the wrong output key; commit `72deeb8` corrected that one-line error
without changing the sealed protocol or scientific parameters.

- Detailed report: `docs/research/sonogym_stage2_contract_corrected_replay.md`.
- Result manifest: `experiments/unity_two_stage_transfer/stage2_contract_corrected_replay_v1_result.json`.
- Successful artifacts: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_contract_corrected_replay_v1/51550463/`.
- Preserved failed run: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_contract_corrected_replay_v1/51550144/`.
- Next approval: one bounded development-only continuous active-acquisition pilot; full Stage 2
  closed loop remains `NOT_RUN`.

## Phase 5C Stage 2 movement-contract audit

CPU job `51542236` completed `0:0` on `anest-dept` with four CPUs, 16 GB requested RAM, and
zero GPUs. It reproduced the frozen `1 / 144` replay exactly, then confirmed that the collector
accepted 315 angle-only-clamped frames which replay rejected through the combined
`action_clamped` flag. No position-clamped, rejected, incomplete, mapping-invalid, unsafe, or
nonfinite frame was admitted.

The narrow collector-consistent diagnostic increased temporal accepts from 1 / 144 to 38 / 144
and multi-view accepts from 1 / 144 to 31 / 144, with zero false accepts on the opened data and
acceptance in all four sources. This is a material root-cause effect but remains below the prior
52-accept usefulness reference, so `EXPLORATORY_NO_GO` is unchanged. The required next action is
Option 1: review and implement the narrow contract correction, then rerun opened CPU replay before
any active-acquisition GPU pilot.

The remaining data limitation is also demonstrated: 766 frames were unchanged-producer,
zero-action, zero-motion, identical-hash duplicates; no existing frame is a fresh same-pose
acquisition. The 144 trajectories are reset-separated 10/5-frame grid segments, not a continuous
Unity-style active search.

- Detailed report: `docs/research/sonogym_stage2_contract_audit.md`.
- Result manifest: `experiments/unity_two_stage_transfer/stage2_contract_audit_v1_result.json`.
- External artifacts: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_contract_audit_v1/51542236/`.

## Phase 5C Stage 2 temporal/multi-view CPU replay

CPU job `51518820` completed `0:0` on `anest-dept` with four CPUs, 16 GB requested RAM, and
zero GPUs. It causally replayed 1,350 opened frames from 18 sessions and 144 trajectories across
V14 and V16--V18 using the unchanged V15 model bundle and historical V18 threshold. The focused
suite passed 19 / 19 tests and the existing Stage 2 suite passed 34 / 34.

The declared multi-view rule reduced false accepted candidates from 2 / 103 for quality-ranked
validity-gated selection to 0 / 1, but accepted only 1 / 144 trajectories and abstained on
143 / 144. It therefore received `EXPLORATORY_NO_GO`; zero errors at 0.6944% coverage is not a
useful repair. Quality-guided validity-gated ranking remained useful (2 versus 3 false accepts at
the same 103 / 144 coverage), while quality-only ranking was unsafe (37 / 144 false accepts).

Replay integrity, causal information boundaries, evidence independence, and implementation
readiness pass. Physical return-to-best, fresh post-return confirmation, and full coarse/fine/micro
Stage 2 remain `NOT_RUN`. No GPU, V19, model fitting, threshold change, or simulator session was
used. The next step requires explicit approval for one bounded development-only physical
acquisition-contract pilot; it is not another confirmation retry.

- Detailed report: `docs/research/sonogym_stage2_temporal_decision_replay.md`.
- Result manifest: `experiments/unity_two_stage_transfer/stage2_temporal_replay_v1_result.json`.
- Successful CPU run: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_temporal_replay_v1/51518820/`.
- Preserved technical failure: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_temporal_replay_v1/51515957/`.

## Phase 5C Stage 2 final confirmation outcome

The final preregistered V18 single-frame validity-classifier confirmation is
**CONFIRMATION_NO_GO**. Capture job `51402759` completed `0:0` on account `anest-dept` using
one NVIDIA L4, four CPUs, and a 30 GB request. It produced four new sessions, 32 trajectories,
300 frames, and 126 distinct image hashes. Evaluator job `51409933` used the unchanged model and
frozen 0.5666666667 threshold; its exit `3:0` is the declared scientific gate failure, not a
runtime crash.

Frame-level validity precision was 0.9496855 against the frozen 0.95 requirement; recall was
0.8934911. Quality ranking remained strong with Spearman 0.8525381 and MSE 0.0082874 below the
0.0284061 constant baseline. No reference-image hash overlapped. Precision must not be rounded
into a pass. Read-only diagnosis showed that seven of eight false positives came from one
session, while unique-hash precision was exactly 0.95. This supports image observability and
quality ranking but rejects the frozen single-frame binary operating point as sufficiently
session-stable.

The sealed V18 protocol prohibits a V19 retry after failure. Closed-loop coarse/fine/micro
search and a full Stage 2 demo therefore remain `NOT_RUN`. The next step requires researcher
review of a material actor-contract redesign, such as temporal fresh-frame confirmation while
retaining the stable quality regressor; it is not another threshold adjustment or capture.

- Detailed report: `docs/research/sonogym_s0010_stage2_dataset_pilot.md`.
- Result manifest: `experiments/unity_two_stage_transfer/stage2_confirmation_v18_result.json`.
- Capture: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_confirmation_v18/51402759/`.
- Evaluation and diagnostics: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_confirmation_evaluation_v18/51409933/`.

## Phase 5C Stage 1 confirmation outcome

The s0010 L4 development result remains PASS, but independent confirmation is **FAIL** under
the frozen four-session criterion. Job `51196094` completed four physical scans and produced
three correct localizations plus one wrong stop at 10.362 mm error. The development-selected
v2 interval-midpoint rule did not resolve the reliability gap: after the Home-tolerance fix,
job `51197643` completed all four scans but produced two correct localizations and two
abstentions. No score threshold, evaluator tolerance, checkpoint, or target reference was
changed after either evaluation.

The v2 engineering attempt `51197194` is preserved separately. It stopped before actor
inference because the collector used a hard-coded 3 degree Home tolerance instead of the
protocol's 5 degree limit. Commit `1f9302036e5a0df9f44040d2bcde97e56e5a5c65` corrected that
protocol-compliance defect; it did not alter the scientific gate.

An auditable engineering demo is available from preregistered v1 session `cf_00`, which
localized L4 with 7.862 mm evaluator-only error. The 211-frame, 15 fps, 14.067 s MP4 uses the
real clean SonoGym ultrasound observations and shows actor evidence separately from the
evaluator-only reference. The artifact PASS does not override the four-session confirmation
FAIL.

- Detailed report: `docs/research/sonogym_s0010_stage1_confirmation.md`.
- V1 run: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage1_confirmation/51196094/`.
- V2 evaluated run: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage1_confirmation/51197643/`.
- Demo MP4: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage1_confirmation/51196094/video/sonogym-phase1-confirmation.mp4`.
- Review ZIP: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/reviewer/sonogym_phase1_demo_20261113.zip`.

The next scientific action is a new development cycle focused on start-dependent image-evidence
support, followed by a new preregistered confirmation set. Phase 2 must not start from the
current result.

## Phase 4 runtime-validation outcome

- GPU smoke test: **PASS**, job `50523856` (`COMPLETED`, exit `0:0`).
- PPO pilot: **PASS**, job `50525848` (`COMPLETED`, exit `0:0`).
- Readiness conclusion: **READY_FOR_REPRODUCTION**.

The successful jobs used account `anest-dept`, partition `hpg-turin`, one NVIDIA L4, four CPU
cores, 30 GB requested RAM, and local Blue storage. The L4 reported 23,034 MiB and driver
580.178.04. Both jobs used the real `Isaac-robot-US-guidance-v0` task, Isaac Sim 4.5.0.0,
Isaac Lab `21f7136325136ca3f6ca4e0a8125edffe5c24f7e`, Python 3.10.22, PyTorch
2.5.1+cu121, skrl 1.4.2, seed `20261106`, model-based `conv` ultrasound, headless mode, and
disabled visualization/external logging. The execution HEAD was `5ab60675`; each run captured
its dirty worktree patch under `context/uncommitted.patch` so the tested fixes remain auditable.

### Smoke evidence

Job `50523856` initialized Isaac Sim, loaded the KUKA and subject `s0010` assets, created and
reset the task, and emitted finite `float32` policy observations with shape
`[1, 1, 200, 150]`. It completed 32 bounded actions. The simulation counter advanced from 200
to 264 (0.533333 simulated seconds at 1/120 s physics dt), maximum joint displacement was
0.476689, and maximum end-effector pose displacement was 0.199634. Rewards were finite at the
first and last recorded steps. The explicit real time-limit probe returned
`terminated=false, truncated=true`; a second reset returned another finite observation of the
same shape. The process exited cleanly after 36.82 wall-clock seconds with 3,329,268 KiB peak
RSS. Nonzero pixels establish observation generation only, not anatomical correctness.

The raw simulated policy observation, distinct from a rendered visualization, is stored in
`outputs/smoke/50523856/ultrasound_original_observations.npz`; the short state/action record is
in `trajectory.json` in the same external run directory.

### PPO evidence

Job `50525848` ran the published `workflows/skrl/train.py` PPO workflow with four environments,
10 configured rollout/update intervals, 320 vector steps, and 1,280 collected transitions. It
finished the timed command in 29.24 seconds (43.7756 transitions/s), recorded nine finite
policy/value-loss points, and recorded finite instantaneous reward summaries. No episode-return
summary was emitted in this short horizon. GPU telemetry peaked at 51% sampled utilization and
5,297 MiB used; `/usr/bin/time` recorded 3,717,120 KiB peak RSS. The 113,760,335-byte
`agent_320.pt` checkpoint loaded on CPU as a dictionary containing `optimizer`, `policy`,
`value`, and `value_preprocessor`.

The first PPO launcher job, `50524765`, failed before training with a Hydra override-path error
and collected zero transitions. The corrected nested paths are
`agent.agent.mini_batches` and `agent.agent.experiment.checkpoint_interval`; the failed log is
preserved. The successful retry kept the same task, seed, and resource request.

### Engineering fixes and evidence locations

The validated fixes execute the environment Python directly instead of activating Conda inside
the GPU job, propagate application failures through Slurm, make the optional Stable-Baselines3
extractor genuinely optional, correct the declared ultrasound space from `[1, 150, 200]` to
`[1, 200, 150]`, validate samples against the declared Gym space, use the correct nested Hydra
paths, and capture GPU telemetry, TensorBoard scalars, and checkpoint structure. NVIDIA EULA
acceptance remains machine-local; the tracked example defaults to `N`.

- Machine report: `reports/phase4_gpu_smoke_ppo.json`.
- Smoke: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/smoke/50523856/`.
- PPO: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/ppo/50525848/`.
- Failed-run logs: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/`.
- CPU preflight: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/records/preflight.json`.

Runtime storage increased from 23,499 MiB before GPU validation to 23,740 MiB after it; the
repository uses 5 MiB. Generated logs, observations, and checkpoints remain outside Git.

This phase establishes simulator initialization, robot/ultrasound transitions, PPO execution,
and checkpoint creation. It does **not** establish useful learned behavior, generalization,
clinical safety, anatomical fidelity, or MBB performance.

## Follow-up paired evaluation and rendered demo

The preregistered no-training follow-up is complete. Job `50553529` (`anest-dept`, one L4,
four CPUs, 30 GB) evaluated `agent_320.pt` and a zero-action control sequentially with seed
`20261110`. Each arm completed 20 episodes on patient `s0010`/`vertebrae_L4`; all observations,
actions, and rewards remained finite. The 20 requested initial commands matched exactly between
arms (`max_abs_difference=0`). Both arms produced 11,980 transitions and all episodes ended by
the native time-limit truncation, with no native task termination.

The checkpoint did not outperform the paired control at this pilot budget. Mean episode return
was -0.654744 for the checkpoint and -0.001471 for zero action. The paired checkpoint-minus-zero
difference averaged -0.653274, and the checkpoint was higher in only 20% of pairs. Composite
distance improved in 20% of checkpoint episodes versus 30% of zero-action episodes. The task
defines neither a native success criterion nor an enabled out-of-bounds termination, so this
result does not support a useful-policy or safety claim. It is evidence against promoting the
10-update pilot checkpoint as a learned navigation result.

Evaluation jobs `50550192` and `50551253` are preserved as protocol failures. The first seeded
before Runner construction; the second re-seeded before reset but allowed Gaussian action
sampling to advance CUDA RNG before later resets. The accepted evaluator restores RNG around
deterministic `mean_actions`, which produced exact pairing in `50553529`.

Rendered demo job `50551365` completed with exit `0:0` on `anest-dept`. It produced a 2.98 s,
179-frame, H.264, 1280x720, 60 fps MP4. The file was decoded successfully and the first/middle
frames were visually checked as nonblank. It is a rendered simulator demonstration, not the raw
ultrasound policy observation and not evidence of policy quality. First camera launch spent about
12 minutes compiling RTX shaders; the completed job used 14:43 wall time.

- Follow-up machine report: `reports/checkpoint_evaluation_demo.json`.
- Accepted evaluation: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/evaluation/50553529/`.
- Video manifest: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/demo/50551365/video/demo-manifest.json`.
- Downloadable MP4: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/demo/50551365/video/sonogym-checkpoint-demo-step-0.mp4`.

From a local terminal, download the MP4 with:

```bash
scp vinhnguyen1@hpg.rc.ufl.edu:/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/demo/50551365/video/sonogym-checkpoint-demo-step-0.mp4 .
```

Runtime storage is now 24,314 MiB, up 574 MiB from the post-Phase-4 value; most follow-up growth
is renderer/shader cache rather than the 390,986-byte MP4.

Next exact action: do not scale this checkpoint. First define and review a native navigation
success criterion and enabled failure/constraint metrics, then preregister a matched-budget
learning-curve protocol with zero/random controls. No additional GPU training is authorized by
this result alone.

## Phase 5A navigation-metric validation

Phase 5A is complete with **METRIC_GO**. The base frozen protocol and manifest remained
unchanged. Approved protocol-v2 commit `907d093` changed only execution lifetime: each frozen
four-start batch ran in a new Isaac Sim process, avoiding the reset stall observed in jobs
`50641608` and `50647528`. No policy training, reward/observation/action/dynamics change, or
threshold adjustment occurred.

Job `50656802` completed `0:0` on `anest-dept` in 10:38 using four CPUs, 30 GB requested
RAM, and one NVIDIA L4. All 32 process-isolated batches exited zero. The merged evaluation
contains 38,400 transitions; paired starts matched exactly across controllers, and no
nonfinite observation/reward/pose or invalid transition was found.

At the primary 10 mm / 10 degree threshold, oracle success was 32/32, while zero action and
bounded random action were both 0/32. Oracle median final error was 0.0246 mm and 8.5375
degrees, versus 65.0751 mm / 26.3056 degrees for zero and 68.3269 mm / 68.9998 degrees for
random. The oracle exceeded each control by 32 successes and had lower median position and
orientation error, satisfying every frozen gate.

The existing Phase 4 PPO checkpoint was diagnostic only: 0/32 final success, 5/32 ever success,
120.7920 mm median final position error, and 7.7488 degrees median final orientation error. It
does not establish useful learned navigation. The target angle is 3.14 rad while the native
command maximum is 3.0 rad, which explains the oracle's approximately 8.54-degree floor and
0/32 success at the secondary 5 mm / 5 degree sensitivity threshold. The native boundary
termination remains disabled, and every episode ended at the 300-step protocol horizon.

Evidence:

- Audit: `reports/phase5a_navigation_objective_audit.md`.
- Human report: `reports/phase5a_navigation_metric_validation.md`.
- Machine report: `reports/phase5a_navigation_metric_validation.json`.
- Completed run: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/navigation_metric_validation_v2/50656802/`.
- Failed initial run: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/navigation_metric_validation/50641608/`.
- Failed sole retry: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/navigation_metric_validation/50647528/`.

The completed run adds approximately 14 MiB (13 MiB run directory plus a 359,968-byte Slurm
log). Peak host RSS was 3,397,904 KiB; sampled GPU memory/utilization peaked at 5,081 MiB/62%.
No raw outputs, logs, plots, videos, or checkpoints were committed.

Next decision: before any separately preregistered Phase 5B learning experiment, the researcher
must decide whether the 3.0-rad command ceiling versus 3.14-rad target is intentional. No
Phase 5B training is authorized by the Phase 5A result.

## Phase 5C current Stage 1 status

G0/G1 capture job 50908807 completed 16 sessions and 3,376 frames. The fixed-anatomy L4
supervised discovery pilot now has a canonical development PASS in CPU job 51192890. The job
used anest-dept, four CPUs, 16 GB requested RAM, zero GPUs, and completed in 3:04 with exit 0.
Image-only seven-level macro recall was 0.844 versus 0.147 after within-session image shuffling;
image plus position/history exceeded position/history-only macro recall by 0.092. Bidirectional
image-derived evidence localized L4 in 4/4 development sessions with median 6.36 mm error, zero
wrong stops, and median 3.5 mm forward/reverse peak difference.

This is fixed-anatomy, prior-inspected development evidence. The acceptance threshold was
selected on the same four development sessions, so independent confirmation and patient
generalization remain unestablished. The checkpoint is marked non-deployment and Phase 2 has
not started. The concise result and report are:

- experiments/unity_two_stage_transfer/stage1_supervised_pilot_result.json
- docs/research/sonogym_s0010_stage1_supervised_pilot.md

Canonical external evidence is under
/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage1_supervised_pilot/51192890/.
The development checkpoint SHA-256 is
bc6df0a1d9e520ac8340789fa751f35e63e0cdcfe4072c130bd70749f22ceb79.
Application timing was 175.659 seconds, maximum measured RSS was 1,117,384 KiB, and the output
directory grew by 72,034,539 bytes. Predecessor job 51192265 is preserved; its scientific
numbers are identical, while job 51192890 adds stage timing and a compact checkpoint.

The user approved a four-session Phase 1 confirmation and short demo on 2026-11-13. The actor
protocol SHA-256 is 965e5cfefd09eace010841f9f720485e48967484ee8df508af48aebfaa5bcce8;
the separately sealed evaluator protocol SHA-256 is
b7fdb067a2b90ac2ae3bcfb5e62c78a62c046b593cd2ee3b9bee6ef7eeafa6cd. The selected checkpoint,
0.50 spatial threshold, four starts, four-session success requirement, and 10 mm localization
tolerance are frozen before GPU capture. The approved bound is one L4, four CPUs, 30 GB, and
15 minutes on anest-dept. Actor output must be written and hashed before evaluator-only L4 data
are opened.

Confirmation job 51196094 completed all four physical scans and produced a valid 211-frame,
14.07-second video, but it failed the frozen scientific gate: three sessions were within 10 mm
and cf_03 emitted a coordinate with 10.362 mm error. The job therefore exited 3 and is retained
as negative evidence. A v2 actor protocol changes only the reported coordinate from the accepted
event's highest grid bin to the midpoint of that event's existing support interval. This rule
passes all four pre-existing development sessions with 6.49 mm median error; job 51196094 was
not used to select the v2 coordinate. The checkpoint, 0.50 acceptance threshold, 10 mm evaluator
tolerance, actions, and actor information boundary remain unchanged. V2 must run on four new
sessions before any demo-readiness claim.

The first v2 execution, job 51197194, stopped before actor inference: cf2_00--cf2_02 completed,
but cf2_03 reached 1.906 mm position error and 3.718 degree orientation error after 80 Home
actions. Those values satisfy the protocol's 5 mm / 5 degree motion tolerance, while the collector
incorrectly retained a 3 degree Home-only constant. The implementation now reads both Home
tolerances from the sealed protocol. This is an engineering retry condition; job 51197194 contains
no perception result and cannot be counted as confirmation evidence.

The frozen Group A development job 51116088 used anest-dept, four CPUs, 16 GB requested
RAM, zero GPUs, and completed in 4:37. It failed the development gate: no threshold produced
a complete seven-structure map, so the reused evaluation split remained unopened and no actor
bundle was released. The concise result and full report are:

- experiments/unity_two_stage_transfer/stage1_group_a_result.json
- docs/research/sonogym_s0010_group_a_stage1_result.md

Reviewer audit found that the six v1 facet candidates were nearest points on adjacent meshes
with 0.0 mm separation, not demonstrated facet-joint locations. Commit `0640c4f` now rejects
that construction; evidence commit `f447616` preserves the six candidates as `REJECTED` audit
records and marks all six actual facet-joint annotations `UNRESOLVED`. Group A remains 7/7;
Group B is now 8/34 `PROVISIONAL` and 26/34 `UNRESOLVED`.

The corrected detailed review package is external to Git at
/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/reviewer/s0010_anatomical_ground_truth_review_20261113_v5.zip.
The 24,564,576-byte ZIP SHA-256 is
2cab0d5ad220e4d5685e4f8c9cda94d11cdd5e1fa4d2fbd4583a1fb6cd290083.
Its 37 PNG files decode successfully, all internal SHA-256 checks pass, and `unzip -t` reports
no errors. The prior v4 archive is retained as superseded review history.
The L4-specific PASS does not override the seven-structure NO-GO. The next scientific action is
a separately approved frozen confirmation capture with new session-held-out starts; Stage 2
has not started.

That next capture is now complete and supersedes the preceding next-action sentence. Job
51213110 ran four preregistered new `s0010` sessions under `anest-dept` with one NVIDIA L4,
four CPUs, 30 GB requested RAM, and a 30-minute bound. Slurm reported `COMPLETED`, exit `0:0`,
after 2:06. The four scans produced 844 fresh frames and the frozen actor emitted 28 / 28
landmarks in 4 / 4 complete maps, with zero wrong stops and zero abstentions. Median
localization error was 3.0 mm, p90 was 6.375000000000001 mm, and the maximum was exactly the
inclusive 10.0 mm limit for L5 in session `cf7_01`.

Actor output SHA-256
`0d62f3c50007a14f1d3d54758e7d359eee4f3a8827ef10d844aa60348d4f5b` was written before the
evaluator opened privileged annotations. Evaluation output SHA-256 is
`6546cfe396cccebee1dd76e988b920db70cb4697a68ebcae8a54ae7bd5199a2a`. Peak sampled GPU
memory was 5,009 MiB, peak sampled utilization was 56%, maximum resident memory was
3,625,692 KiB, and the run produced 94,251,998 bytes. The result and report are:

- experiments/unity_two_stage_transfer/stage1_group_a_confirmation_result.json
- docs/research/sonogym_s0010_seven_landmark_confirmation.md

Canonical evidence is under
`/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage1_group_a_confirmation/51213110/`.
This establishes fixed-anatomy seven-landmark confirmation within the tested scan corridor. It
does not establish patient-held-out generalization, Stage 2 best-view acquisition, or clinical
MBB performance. Phase 2 has not started.

Stage 2 targeted capture job `51377943` (`anest-dept`, one L4, four CPUs, 30 GB) completed all
six sessions, 48 trajectories, and 450 joined frames in 3:15. It passed the preregistered
independent-positive coverage gates with 79 / 21 / 20 unique positive hashes and 32 / 8 / 8
positive trajectories in train / calibration / development. Public ultrasound/motion records
and privileged offline labels joined one-to-one. Precursor job `51374942` is preserved as a
technical failure of a legacy distinct-hash heuristic, not a scientific result.

The unchanged V14 geometric-validity classifier then failed more strongly on CPU job `51378486`:
development precision was 0.5556 and recall 0.1087. Its quality regressor still passed with
Spearman 0.6757 and MSE 0.00989 versus the 0.02908 constant baseline. Diagnosis found that high-
confidence false positives at 10.0678 and 10.4356 degrees were visually continuous with frames
inside the hard 10-degree evaluator boundary. Additional repeated positive coverage therefore
did not repair the mismatch between an image-only classifier and an exact geometric threshold.

After researcher approval, V15 separated image-observable validity from independent geometric
success without changing the 10 mm / 10 degree evaluator. The classifier target is a technically
valid L4 view with raw-label image fraction at least 0.20; only the predeclared 0.18--0.22 band
is excluded from fitting. CPU job `51380423` completed `0:0` in 13 seconds and returned
`DEVELOPMENT_PASS`: development precision 1.00, recall 0.8929, quality Spearman 0.6757, and
quality MSE 0.00989 below 0.02908. This is fixed-`s0010`, post-diagnosis development evidence.
Closed-loop Stage 2 remains `NOT_RUN`; the next scientific action is a newly frozen independent
perception confirmation before coarse/fine/micro search. No new GPU capture was run for V15.

That independent V16 confirmation is now complete and is **CONFIRMATION_NO_GO**. GPU job
`51387027` (`anest-dept`, one L4, four CPUs, 30 GB) completed four new sessions, 32 trajectories,
and 300 frames in 2:29 with zero V14 image-hash overlap. CPU evaluator job `51387434` loaded the
unchanged V15 bundle and threshold without retraining. Precision remained 1.00, but recall was
0.6824 against the frozen 0.80 gate. Quality ranking confirmed strongly with Spearman 0.8644 and
MSE 0.00776 below the 0.02889 frozen-constant baseline.

Read-only diagnosis found 1.00 precision and 0.7031 recall after deduplicating image hashes, so
repeated frames do not explain the failure. A post-hoc 0.60 threshold would have met both
classification gates on V16, but it was not applied and cannot change the NO_GO decision. The
current blocker is session-unstable threshold calibration, not simulator execution or quality
regression. Closed-loop remains `NOT_RUN`. Any repair must use an explicitly reviewed
multi-session calibration protocol and a separate future confirmation set; do not silently tune
on V16 or launch another GPU capture.

The blocked G0/G1 record below is retained as historical recovery evidence and is superseded
by successful capture job 50908807.

## Phase 5C historical G0/G1 observability-capture status

Phase 5C anatomical observability and coordinate-integrity gates are **BLOCKED**, not failed on
scientific evidence. No complete scan and no ultrasound dataset were produced, so neither L4
observability nor Stage 1 recognition has been evaluated.

Job `50853505` (`anest-dept`, one L4, four CPUs, 30 GB) failed after 4:04. All 16 sessions
stopped during physical Home return at action step 2 because the collector incorrectly treated a
small probe-basis command-mapping difference as a native workspace clamp. The maximum observed
difference was 0.068359 command index (approximately 0.103 mm), while the incorrect tolerance
was 0.0002 index. A second launcher defect accepted the simulator process's zero exit status even
though every per-session `summary.json` said `FAILED`, so the job unnecessarily initialized all
16 sessions before the summarizer returned exit 2. No frames were saved.

Commit `eea68bcb8a07506abf96dd9754fbeea078a4b28f` separated true workspace clamping from bounded
command-mapping error and made the launcher fail closed on `summary.status`. The sole justified
retry, job `50889771`, then failed fast on the first train/development session after 0:59 because
the new check attempted to convert a CUDA-resident command-bounds tensor directly to NumPy. Its
application exit code was correctly propagated as 2, demonstrating that the launcher repair
worked. Commit `af543204ed5db3709e62f830c2a9a36a0fccde1c` now explicitly transfers this tensor to CPU.
The current code passes 31 Phase 5C unit tests and all 11 Phase 5A regression tests, plus Python
compilation, Bash syntax, and diff-whitespace checks. The frozen protocol and SHA-256
`dc240b5bb333312f6bf85a9ff509f1beafd81e18f1ed458e75b498b5a017017a` were not changed.

Both jobs used NVIDIA L4 GPUs with driver 580.178.04. Sampled GPU memory peaked at 5,011 MiB and
5,009 MiB. The two external run directories occupy 521,934 and 70,850 bytes; the complete G0/G1
output namespace is 972 KiB, well below the 1.5 GB protocol cap. Evidence is preserved at:

- `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/g0_g1_v1/50853505/`
- `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/g0_g1_v1/50889771/`
- `reports/phase5c_g0_g1_capture_status.json`

No third GPU attempt was submitted. The next scientific action requires researcher approval: run
one bounded single-session engineering smoke on `td_00` with the reviewed `af54320` fix. Only if
that produces a complete 211-frame scan with valid motion and manifests should the frozen
16-session G0/G1 capture be resubmitted. Do not fit Stage 1 or begin Stage 2 while G0/G1 remain
blocked.

## Phase 3 historical record

## Completed

- Preserved upstream history, license, and attribution at SonoGym commit
  `e67be58334d1a5274f0913af36f56e4b0b7ffe5a`.
- Inspected the actual guidance YAML loader, task registration, asset paths, visualization
  construction, W&B calls, skrl configuration, and `max_iterations` mapping.
- Added configurable external asset/model/config paths without replacing the tracked Python
  `assets` directory.
- Added the `conv`, headless, fixed-seed navigation baseline and disabled both OpenCV and
  PyVista visualization before environment construction.
- Added a real-environment smoke script and separate serial Slurm templates for smoke and PPO.
- Pinned the published asset dataset revision and SHA-256 values. Both archives were downloaded
  and extracted to local staging at `D:\CSSALT\SonogymRuntime`; they are not in Git.
- Added resumable download, CPU-safe preflight, host inspection/install, and dry-run/submit tools.
- Resource templates request one node, one task, four CPUs, one L4 GPU through `hpg-turin`,
  32 GB RAM, bounded walltimes, and account `anest-dept`. No GPU job was submitted.
- Local CPU-safe checks passed for parsing 75 Python files, parsing the baseline YAML, shell
  syntax (Git Bash), asset path resolution, Git ignore guards, diff whitespace, and both Slurm
  dry-runs. The full local package preflight correctly failed because the unrelated Windows
  Python 3.11 environment does not contain the pinned Isaac stack; this is an installation
  blocker, not a GPU-only check.
- Published the independent project repository at
  `https://github.com/ntvinh2005/cssalt-sonogym`; GitHub API verification reported private
  visibility, default branch `main`, and a remote head matching the local branch.

## Blockers and unverified work

- HiPerGator SSH reached `hpg.rc.ufl.edu` but noninteractive authentication failed with
  `Permission denied (publickey,keyboard-interactive)`. Therefore host/GLIBC/modules/storage,
  scheduler associations/QOS/GPU metadata, Blue path writability, and `anest-dept` usability
  could not be inspected from this session.
- The pinned Conda/pip environment is not installed on HiPerGator. The installer is ready but
  intentionally refuses to install on a login node or another allocation.
- The downloaded local staging copy has not been transferred to Blue storage.
- Isaac Sim extension downloads, NVIDIA EULA acceptance, shader compilation, L4/driver
  compatibility, simulator smoke, and PPO pilot are `NOT_RUN_NO_GPU`.
- Repository publication is complete; simulator readiness remains separate and partial.

Because the cluster environment and Blue assets are incomplete, the project must not be labeled
`PREPARED_GPU_UNTESTED` yet. No engineering claim that SonoGym runs and no scientific claim about
policy learning is made.

## Pinned inputs

- SonoGym: `e67be58334d1a5274f0913af36f56e4b0b7ffe5a`
- Isaac Lab: tag `v2.1.0`, commit `21f7136325136ca3f6ca4e0a8125edffe5c24f7e`
- Assets: `yunkao/SonoGym_assets_models`, revision
  `b37b080a8673f856266a2306724e48d5e034521a`
- `assets.tar.gz`: 754,262,884 bytes,
  SHA-256 `0d5840da3af2eb97e1ca9ebc7c2a969d32b24eef3f3107bc0c9f7db41e4a8f77`
- `models.tar.gz`: 754,887,112 bytes,
  SHA-256 `545113c94b641eda199c6edee1e3586f8cd3a844f0b600470a828fec6210b14f`

## Next action

Restore SSH key/MFA access, copy this repository to the intended Blue path, run
`scripts/setup_hpg.sh --inspect-only`, confirm `anest-dept`, then perform `--install` inside a
four-CPU development allocation. Do not submit either GPU job during setup.
