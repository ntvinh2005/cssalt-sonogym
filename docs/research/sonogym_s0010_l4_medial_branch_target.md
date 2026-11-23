# s0010 RIGHT L4 medial-branch target localization at RIGHT L5

This experiment asks whether historical ultrasound observations can support a
non-privileged localization of a bony surrogate for the RIGHT L4 medial branch
at RIGHT L5. It does not define a nerve coordinate, needle endpoint or clinical
intervention target.

## Result

**`STAGE2_COMPLETE = FAIL`.** The 2026-11-23 clinician-inspired feasibility
capture executed both requested scan families with 96 / 96 fresh observations,
but produced only 1 / 4 image-supported reversible transitions, no qualifying
longitudinal TP-to-AP transition, and no frozen perception result satisfying
the pair-precision and false-acceptance gates at both levels. A fresh autonomous
closed-loop run therefore remains scientifically ineligible.

| required verdict | result |
|---|---|
| `ANATOMICAL_TARGET_CONTRACT` | `PARTIAL` |
| `ULTRASOUND_TARGET_LOCALIZATION` | `FAIL` |
| `ROBOT_NAVIGATION` | `PARTIAL` (feasibility executed; not autonomous) |
| `RIGHT_L4_STAGE2` | `FAIL` |
| `RIGHT_L5_STAGE2` | `FAIL` |
| `STAGE2_COMPLETE` | `FAIL` |
| `FULL_MBB_INTERVENTION` | `BLOCKED` |

## 1. Target contract — is the endpoint definable

`l4_medial_branch_target.py:build_target_contract_geometry`, evaluated by the
sealed CPU workflow.

**Criterion.** The RIGHT L5 SAP and TP proposals must be aligned, right-sided,
nonempty and measurable in physical millimetres. The contract must keep its
uncertainty explicit and must not infer a nerve location from a generic offset.

### 1.0 Quantities

| quantity | value | source |
|---|---:|---|
| RIGHT L5 SAP voxels | 338 | frozen V3 CT-model proposal |
| RIGHT L5 TP voxels | 445 | frozen V3 CT-model proposal |
| minimum SAP-to-TP surface separation | 6.184658 mm | NIfTI RAS physical distance |
| core endpoint voxels, SAP / TP | 11 / 3 | minimum plus one 1.5 mm CT voxel |
| uncertainty endpoint voxels, SAP / TP | 24 / 12 | minimum plus one voxel diagonal |
| evaluator midpoint samples | 13 | evaluator only |

### 1.1 Method

The evaluator retained the frozen right-sided L5 SAP and TP predictions inside
the source L5 segmentation, measured their closest surface relationship and
formed bounded core and uncertainty bands at the native 1.5 mm CT resolution.
The midpoint samples are used only to measure localization error. They are not
actor inputs and are not nerve or needle coordinates.

The anatomy is consistent with the general relationship described in lumbar
facet guidelines: L1–L4 medial branches cross the superior portion of the
subjacent transverse process near the SAP–TP root junction. That literature
does not provide patient-specific ground truth for `s0010`:

- <https://pmc.ncbi.nlm.nih.gov/articles/PMC7362874/>
- <https://pmc.ncbi.nlm.nih.gov/articles/PMC10193888/>

### 1.2 Setup

| item | value |
|---|---|
| anatomy | fixed `s0010` |
| prediction SHA-256 | `a78daeae9159f0df28ba6fffcb8e8d7ef6c2e87f490c9598acfcb82efad1677f` |
| source segmentation SHA-256 | `0d435584ed7851736f044c0a9c906231ef3f139b7cda16770d3066773992a1ed` |
| TP revision SHA-256 | `01c910898f1463d89ac2e694c0466ff84967a0bc7dd2a464c700a8d62d5bb565` |
| coordinate system | NIfTI RAS millimetres |

### 1.3 Results

The geometry is reproducible and physically measurable, but the CT prediction
does not supply a semantic junction label, nerve model or independently
validated intervention target. The target contract is therefore `PARTIAL`.
The 3.0 mm evaluator gate is a development-resolution gate derived from image
and CT resolution, not a clinically valid targeting tolerance.

### 1.4 Reading it

The contract supports an exploratory observability test only. It does not turn
the L5 TP surface core or the SAP proposal into an L4 medial-branch target.

### 1.5 Files

| file | contents |
|---|---|
| `l4_medial_branch_target.py` | target geometry and actor/evaluator separation |
| runtime `target_contract.json` | exact geometry counts, limits and hashes |
| runtime `anatomy_target_contract.png` | evaluator-only geometry review figure |

## 2. Ultrasound evidence — is the relationship observable

`run_s0010_l4_medial_branch_target.py`, job `52273102`.

**Criterion.** Both target-region endpoint surfaces must be first-visible bone
in at least two distinct frames from each of at least two sessions. A projected
but occluded SAP remains ambiguous rather than becoming a hard negative.

### 2.0 Quantities

| quantity | value | source |
|---|---:|---|
| evaluated search frames | 156 | four historical Stage 2 sessions |
| proxy-positive | 0 | both endpoints first visible |
| proxy-ambiguous | 100 | projected, occluded or noncore evidence |
| proxy-negative | 56 | TP-only or other bone without SAP in plane |
| qualifying sessions | 0 / 4 | frozen visibility gate |

### 2.1 Method

The actor used only clean ultrasound, the historical L5 TP proposal, cortical
image evidence and actor-visible trajectory data. CT masks, raw labels and
native pixel coordinates were opened only after each actor hypothesis was
frozen. The comparison included the historical TP proposal, a fixed
TP-relative control and an image-interface method using image intensity and
posterior-shadow evidence.

### 2.2 Setup

| item | value |
|---|---|
| source job | `52039283` |
| CPU execution job | `52273102`, completed, exit `0:0` |
| execution commit | `e89039dad9d579ae1214cb2e100a6a4a8ac479a6` |
| protocol SHA-256 | `b6a0e30aa027d2af122811d6076d6586c3b4f45f532263124d67ff22e401f4cc` |
| resources | account `anest-dept`; 4 CPU; 16 GB; no GPU; 47 s |
| source status | historically opened development evidence |

```bash
SONOGYM_CONFIG_FILE="$PWD/.sonogym.env" \
  sbatch hpg/jobs/s0010_l4_medial_branch_target_v1.sbatch
```

Fresh return-to-best confirmations were excluded by the sealed protocol. No
search frame was dropped after the declared phase filter.

### 2.3 Results

| method | accepted | wrong accepted | ambiguous accepted | median distance |
|---|---:|---:|---:|---:|
| historical L5 TP proposal | 156 | 56 | 100 | 20.152 mm |
| fixed TP-relative control | 151 | 52 | 99 | 8.389 mm |
| image-interface evidence | 147 | 54 | 93 | 9.974 mm |

The image method did not outperform the simple fixed control and both accepted
unsupported evidence. With zero legitimate positive frame, correct accepted
target selection cannot be demonstrated.

### 2.4 Reading it

This does not invalidate the historical RIGHT L5 TP navigation result. It shows
that those observations do not contain sufficient first-visible SAP/TP
relationship evidence to infer the proposed L4 medial-branch bony surrogate.
Accordingly, the later GPU capture tested observability only; no new robot
target-search execution was submitted.

### 2.5 Files

| file | contents |
|---|---|
| `s0010_l4_medial_branch_target_v1_result.json` | concise immutable result |
| runtime `frame_results.csv` | per-frame labels and evaluator distances |
| runtime `session_results.csv` | session-level evidence counts |
| runtime `demo.mp4` | actual historical observations with separated panels |
| runtime review ZIP | self-contained external-review package |

## 3. Fresh capture — did geometry-designed views reveal both endpoint surfaces

`evaluate_s0010_l4mb_target_observability.py`, capture jobs `52278401` and
`52283924`.

**Criterion.** The frozen protocol requires at least two supported target
frames in each of at least two sessions, zero supported negative controls, and
complete exact-pose verification. This is the approved observability gate, not
a clinical targeting criterion.

### 3.1 Method

The offline geometry supplied frozen probe waypoints only for data acquisition.
The clean-image test detected cortical tracks and posterior shadow before the
evaluator opened the SAP/TP masks, raw labels and native coordinates. A target
frame required both endpoint cores to be first-visible bone and to agree with
the clean-image evidence. The geometry and native coordinates were never actor
inputs.

### 3.2 Setup

| what | value | spec |
|---|---|---|
| initial capture | `52278401`, source `3c7d656ba828df7e2230e4d15435a485e2323ab7` | §3 |
| bounded resume | `52283924`, source `16bcb3b6b4b4197f4eb30501558329ee750a157b` | §3 |
| frozen protocol | SHA-256 `089cc97563c90847d91cc172daf9dd513a670713e75372aa0b660e45f68b540d` | §3 |
| resume resources | one NVIDIA L4, four CPUs, 30 GB, 1 min 51 s | §3 |
| attempted | 3 sessions, 36 frames | §3 |
| completed | 3 sessions, 36 frames | §3 |
| dropped | 0 frames | §3 |

```bash
SONOGYM_CONFIG_FILE="$PWD/.sonogym.env" \
  sbatch hpg/jobs/s0010_l4mb_target_observability_v1.sbatch
```

The first session was captured before the verifier raised a missing metadata
field. The bounded resume preserved that immutable protocol and session, then
captured only the two missing sessions.

### 3.3 Results

| completed-session measurement | value |
|---|---:|
| exact-pose replay | 36 / 36 frames |
| distinct frame hashes | 36 / 36 frames |
| distinct acquisition identifiers | 36 / 36 frames |
| fresh viewpoints | 36 / 36 frames |
| coordinate-array hash mismatches | 0 / 36 frames |
| maximum coordinate error | 0.0000376263 voxel-index |
| planned target frames | 18 |
| both endpoint cores first-visible | 0 / 18 frames |
| ambiguous target frames | 18 / 18 frames |
| negative controls classified negative | 16 / 18 frames |
| negative controls classified ambiguous | 2 / 18 frames |
| image-supported positive frames | 0 / 36 frames |

**Verdict: fail.** The completed three-session protocol supplies no positive
endpoint observation. Robot search therefore remains ineligible.

### 3.4 Reading it

- The acquisition synchronization and exact-pose mapping passed for all saved frames.
- All 18 target candidates intersect SAP and TP outside the accepted endpoint core.
- The SAP core was not the first visible bone surface in any target candidate.
- The result rejects this frozen pose family, not every physically possible probe pose.
- Repeating small offsets around these waypoints is not justified by the result.

### 3.5 Files

| file | contents |
|---|---|
| `s0010_l4mb_target_observability_v1_result.json` | capture, verification and observability counts |
| runtime `sessions/*/verification_report.json` | 36-frame replay and freshness gates |
| runtime `results/evaluation/frame_results.csv` | per-frame clean-image and evaluator evidence |
| runtime `results/evaluation/clean_ultrasound_summary.png` | clean images with evaluator captions |

## 4. Broad probe-angle test — is any SAP–TP surface signal present

`search_s0010_l4mb_target_viewpoints.py` and
`evaluate_s0010_sap_tp_signal.py`, jobs `52299785` and `52300444`.

**Criterion.** The geometry search was frozen over the native command domain.
The fresh-image gate then required at least three first-hit SAP pixels and
three first-hit TP pixels, a four-pixel image margin, cortical-track distance
at most four pixels and posterior-shadow contrast at least 0.05. At least two
target frames in two sessions and zero supported controls were required. These
settings were fixed before opening the fresh images.

### 4.1 Method

The first search tested the strict endpoint cores and returned no valid pose.
The second, distinct question tested whether separate SAP and TP surfaces could
be seen in the same image without calling them the intervention endpoint. The
evaluator-only CT masks supplied eight frozen probe poses; the capture runtime
received only those commands. Clean-image cortical tracks and shadow evidence
were computed independently of the masks, and privileged first-hit anatomy was
opened only for scoring.

The reduced-pose plane model was used only to screen commands. Against the 36
previous full-pose captures it had median point error 0.838 mm, p90 2.602 mm,
p95 5.712 mm and maximum 9.189 mm. Therefore every new frame was rechecked with
the synchronized full pose and native coordinate array.

### 4.2 Setup

| item | value |
|---|---|
| search domain | x 100–180 by 2 indices; z 70–200 by 2; angle 1–3 by 0.025 rad |
| strict-core search | job `52299421`; valid no-candidate result |
| surface search | job `52299785`; 4 CPU; 4 min 10 s |
| fresh capture | job `52300444`; one NVIDIA L4; 4 CPU; 30 GB; 1 min 51 s |
| execution commit | `306b1ae8fbb0c37c98fa35019ab081d0547d8c9c` |
| protocol SHA-256 | `044772307d78218f9b1e79eaeee5d3782fe8bef14359a4ba3049f0b9d4ba453a` |
| storage | approximately 21 MB |

### 4.3 Results

| measurement | result |
|---|---:|
| pose commands with both cores crossing the plane | 1,393 |
| poses with both endpoint cores first-visible | 0 |
| poses with separate SAP and TP surfaces first-visible | 860 |
| fresh sessions / frames | 3 / 36 |
| replay, freshness and integrity gates | 36 / 36 pass |
| maximum full-pose coordinate error | 0.00003617 voxel-index |
| geometry-supported target frames | 24 / 24 |
| clean-image-supported target frames | 19 / 24 |
| supported targets by session | 6 / 8; 6 / 8; 7 / 8 |
| supported controls | 0 / 12 |

**Verdict: `SAP_TP_IMAGE_SIGNAL_SUPPORTED`.** The simulator can render a
repeatable ultrasound configuration in which evaluator-confirmed RIGHT L5 SAP
and TP first-hit surfaces align with separate clean-image cortical/shadow
evidence.

### 4.4 Reading it

- This answers the observability question positively for the broader SAP–TP
  surface configuration.
- It does not rescue the strict endpoint-core test: that result remains 0.
- It does not show that an image-only actor can assign SAP and TP identities.
- It does not define a nerve coordinate, validated MBB target or intervention
  endpoint.
- The new images are opened development evidence, not independent confirmation.

### 4.5 Files

| file | contents |
|---|---|
| `s0010_l4mb_sap_tp_signal_v1_result.json` | concise immutable result |
| runtime `results/evaluation/frame_results.csv` | all image and geometry measurements |
| runtime `results/evaluation/sap_tp_signal_review.png` | clean images beside evaluator overlays |
| runtime `sessions/*/verification_report.json` | synchronized replay, freshness and integrity |

## 5. Image-only pair selection — can the actor distinguish the SAP–TP configuration

`ultrasound_sap_tp_configuration.py`, run by
`hpg/jobs/s0010_sap_tp_pair_development.sbatch`. It selects two local cortical
intervals from clean ultrasound and uses prior actor-visible motion only to
require support from a distinct earlier view.

**Criterion.** Before either CPU comparison, the protocol required at least 6
/ 8 target waypoint groups to pass, no more than 0 / 3 clear-negative waypoint
groups to fail, and at least 75% of accepted target frames to select the
evaluator-confirmed TP/SAP interval assignment. Each waypoint group had to be
evaluated by a model that fit none of its three reset repetitions. Ambiguous
waypoint group 10 was excluded from the clear-negative gate.

### 5.1 Method

The first hypothesis extracted disconnected local cortical components. The
second and final hypothesis partitioned each complete image-derived cortical
track into spatially separated intervals because §4 showed that SAP and TP
often lie on one visually continuous interface. Both used the same geometry
labels, cross-validation folds, 0.50 pair-probability threshold, distinct-view
support rule and final gate. CT masks, native coordinates, absolute waypoints
and evaluator labels were absent from the model inputs.

### 5.2 Setup

| item | value |
|---|---|
| source capture | job `52300444`; 3 sessions; 36 frames |
| disconnected-component protocol | `s0010_sap_tp_pair_development_v1.yaml`; SHA-256 `2a7aae96b7cce2302af1982048f531ca9c84e8bfc3857da3843d1de275f11bd2` |
| complete-track protocol | `s0010_sap_tp_track_pair_development_v1.yaml`; SHA-256 `5ef4cbbbd3136d9a2c5fa2345c5343e4ad62d0ad498e5028510d11d93173d21d` |
| execution commits | `4f3b9f1`; `8630e56` |
| CPU jobs | `52302642`, 20 s; `52303002`, 32 s; 4 CPU each; 0 GPU |
| model | 240-tree ExtraTrees; image-derived pair features only |

```bash
sbatch --export=ALL,SONOGYM_CONFIG_FILE="$PWD/.sonogym.env" \
  hpg/jobs/s0010_sap_tp_pair_development.sbatch

sbatch --export=ALL,SONOGYM_CONFIG_FILE="$PWD/.sonogym.env",SAP_TP_PAIR_PROTOCOL_STEM=s0010_sap_tp_track_pair_development_v1 \
  hpg/jobs/s0010_sap_tp_pair_development.sbatch
```

| sample accounting | count |
|---|---:|
| acquired frames | 36 |
| target frames | 24 |
| clear-negative frames | 9 |
| ambiguous frames excluded from the negative gate | 3 |
| final analyzed frames | 36 |

### 5.3 Results

| measurement | disconnected components | complete-track intervals |
|---|---:|---:|
| target waypoint groups passed | 1 / 8 | 3 / 8 |
| clear-negative waypoint groups failed | 0 / 3 | 0 / 3 |
| accepted target pair precision | 5 / 7 (71.4%) | 6 / 8 (75.0%) |
| accepted clear-negative frames | 0 / 9 | 2 / 9 |
| session top-1 correct pair | not used | 2 / 3 |
| session top-3 contains a correct pair | not used | 3 / 3 |

**Verdict: fail.** The stronger representation improved target coverage from
1 / 8 to 3 / 8 waypoint groups and reached the pair-precision boundary, but
it remained below the frozen 6 / 8 coverage requirement and accepted a clear
negative in two frames of one reset session.

### 5.4 Reading it

- The image contains usable relative SAP/TP surface information in some views.
- Complete cortical tracks match the observed rendering better than requiring two disconnected components.
- Waypoint groups 0, 2, 3, 4 and 6 did not obtain repeatable accepted pairs across resets.
- The second method ranked a correct pair first in 2 / 3 sessions and within the top three in 3 / 3 sessions.
- The third session also placed accepted clear-negative frames in its candidate list, so top-three coverage alone is not a safe navigation rule.
- The data are opened development evidence and repeat the same twelve waypoint commands across resets.
- No junction coordinate, target-view pose, robot action or needle endpoint was released from this experiment.

### 5.5 Files

| file | contents |
|---|---|
| `s0010_sap_tp_pair_development_result_v1.json` | immutable comparison and job provenance |
| runtime job `52302642/analysis/result` | disconnected-component per-frame and group results |
| runtime job `52303002/analysis/result` | complete-track per-frame, group and figure outputs |
| `tests/test_ultrasound_sap_tp_configuration.py` | image-only extraction and leakage-focused checks |

## 6. Engineering checks

The original CPU execution ran 29 focused regressions. The resume preflight ran
44 focused regressions; all passed in 0.116 seconds. Jobs `52302642` and
`52303002` ran 6 and 7 focused tests respectively; all passed. The new checks require
the observability protocol to use synchronized-capture integrity gates, recover
the captured commit from immutable run context when older sealed metadata lack
the field, and prevent a partial session selection from appearing complete.

## 7. Fresh-pose confirmation technical recovery

The first frozen confirmation submission, job `52305012`, stopped before the
first frame. Its 36 code-level tests passed, but the real collector rejected an
adjacent waypoint separated by exactly the movement-tolerance boundary. The
exact-pose verifier also lacked the new protocol identifier. No ultrasound
array was written and no scientific metric was evaluated.

The recovery changes use the same acquisition validator during pose-plan
construction, require strict separation beyond the tolerance boundary, and
register the corrected synchronized protocol in the collector and verifier.
CPU job `52305250` completed the corrected preparation in 49 seconds with 9 / 9
tests passing. It froze 24 target poses and 18 controls across three sessions;
the model checkpoint is unchanged.

| recovery item | value |
|---|---|
| failed GPU job | `52305012`; 45 s; 0 captured frames |
| failed protocol SHA-256 | `4a488ad9d50fc6f74be572deb530e9748dabcfd3d7c96cd3e5cb3b79bdc1661b` |
| corrected CPU preparation | job `52305250`; commit `78947e33fd93fa5ea13bc094473e15b064185d27` |
| unchanged model SHA-256 | `449953374809635726bd0eadfdc8c0686c8310bfd74d967ece4657a577203de3` |
| corrected pose-plan SHA-256 | `a753dec05937b444030c0e51322bc6408c0a5d443d818570a91a070d70e28df7` |
| corrected protocol | `sonogym-s0010-sap-tp-pair-confirmation-v1r1` |
| corrected protocol SHA-256 | `01af044bee4e21fb90c82717755783732de21cc65bcd42344d7f79a05b43d250` |
| corrected regression checks | 61 / 61 pass |

The authorized technical retry, job `52305701`, passed its sealed preflight but
failed at the final waypoint of its first session before arrays were persisted.
The first 13 waypoint trajectories completed; the fourteenth transition was
27.8 mm because dividing each control class into groups of three split an
intended nearby-pose pair. After 100 steps it remained at 1.918 mm position
error against the 1.5 mm acceptance tolerance. No scientific evaluation ran.

CPU job `52305904` corrected the plan without changing the checkpoint or
evaluation gates. Each session now contains seven intact nearby-pose pairs:
four target pairs and three control pairs. All non-reset pair transitions are
within 15 mm and 15 degrees. The corrected plan contains 24 target poses, eight
TP-only controls and ten other-bone/background controls.

| second recovery item | value |
|---|---|
| failed technical retry | `52305701`; 56 s; 0 persisted frames |
| failure action evidence | 1,042 partial action records; 13 completed trajectories |
| pair-preserving CPU preparation | job `52305904`; commit `18f47050c6fc7017a61ebd5580e42d647f245508` |
| pair-preserving pose-plan SHA-256 | `47d6483b7345294592ce1549a20b7328a7b9344773b901aec541725ca1cd3471` |
| pair-preserving protocol | `sonogym-s0010-sap-tp-pair-confirmation-v1r2` |
| pair-preserving protocol SHA-256 | `2a2e55f93a41c54f05c88048174776f007bb0018b523786408142326164fb34d` |

## 8. Fresh-pose confirmation result

Jobs `52306775` and `52308098` produced the final 42-frame confirmation set.
The first job preserved two complete sessions before the third session's
freshness gate correctly rejected a sub-threshold movement. Protocol `v1r3`
carried those verified sessions by their original protocol hash and acquired a
replacement third session with translation-supported pose pairs. The model and
all decision gates remained frozen.

| measurement | result |
|---|---:|
| synchronized, distinct frames | 42 / 42 |
| exact-pose replay pass | 42 / 42 |
| coordinate hash mismatches | 0 |
| maximum coordinate error | 0.0000352 voxel-index |
| target / clear-negative / ambiguous frames | 19 / 18 / 5 |
| session top-1 correct | 3 / 3 |
| session top-3 contains correct | 3 / 3 |
| correct accepted pairs | 9 |
| accepted clear negatives | 7 / 18 |
| pair precision | 56.25% |
| clear-negative false-accept rate | 38.89% |

**Verdict: fail.** Ranking transferred to all three fresh pose groups, but the
unchanged acceptance rule did not control confusing bone views. The frozen
requirements were at least 75% pair precision and at most 10% clear-negative
false accepts.

## 9. Bounded improvement tests

The 42 confirmation frames became opened development evidence after the failed
frozen evaluation. Two CPU hypotheses were tested with each confirmation
session held out in turn.

1. Adding geometry-labelled clear negatives to pair-model training did not
   improve acceptance. False-accept rates remained 40.0%, 33.3% and 42.9%.
2. Keeping the frozen pair ranker and adding an image-only rejection model
   reduced false accepts to 0%, 16.7% and 14.3%. It passed one of three sessions,
   but the third session retained only one correct accepted target and 50% pair
   precision.

No new checkpoint was released. Adjusting a threshold on these opened sessions
would be post-hoc tuning and cannot establish confirmation.

## 10. Final decision and next concrete action

The SAP–TP surface configuration is observable and ranks correctly in the
tested fresh sessions, but the current image representation cannot reliably
decide when to accept it rather than another bony configuration. Do not
integrate this selector into Stage 2 and do not label it as medial-branch target
localization. The strongest presentation-ready autonomous result remains the
historical Stage 1 plus RIGHT L5 TP navigation workflow.

Further SAP–TP work requires a materially different observable signal, such as
calibrated multi-view surface geometry with synchronized actor-visible 6DoF
pose, not another threshold or ExtraTrees retraining pass on these frames.

## 11. Calibrated 6DoF landmark-graph result

### 11.1 Criterion

The actor must rank the correct L5 TP/SAP proposal pair in all four fresh target
sessions, accept the correct pair in all four, make no wrong accepted selection,
obtain at least 75% accepted-pair precision and keep the clear-negative
false-accept rate at or below 10%. L4 must first show independently supported SAP
and TP proposals in every development session before fresh confirmation is
eligible. CT labels, native pixel-to-volume coordinates and absolute waypoint
banks remain evaluator or capture-design inputs only.

### 11.2 Method

The actor reconstructed every proposal from clean ultrasound, synchronized
robot-base-to-end-effector proprioception and public image calibration. A graph
ranker combined image scores with the reconstructed TP-to-SAP displacement and
required support from physically distinct views. Evaluator geometry was opened
only after each session ranking and decision were frozen.

### 11.3 Setup

| item | value |
|---|---|
| L4 focused capture / detector jobs | `52363363` / `52364004` |
| L5 final protocol preparation | job `52368288`; 59 / 59 tests pass |
| L5 final protocol SHA-256 | `1367b9ea5bca59eb98795cd2da6ec97298104e9c90654d6859c9546f83fc1fff` |
| frozen L5 graph checkpoint SHA-256 | `36df29d97e81b95686cb4d93b4872d6652faa68cff2bd0f03d2fb665cc0034fa` |
| L5 fresh capture / CPU evaluation | jobs `52369053` / `52370256` |
| L5 evaluator tests | 60 / 60 pass |
| L5 captured sessions | 4 target and 2 clear-negative sessions |
| actor 6DoF maximum reconstruction error | 0.000113439 mm |

The earlier L5 jobs `52364308` and `52367817` exposed two technical defects
before scientific evaluation: unstable 3.14-rad ceiling assistance and a
14-frame freshness default applied to 11-frame negative sessions. Both defects
were corrected without changing the model, acceptance threshold or scientific
gate. The complete capture from `52369053` passed exact-pose and actor-6DoF
verification for all six sessions.

### 11.4 Results

| measurement | L4 opened development | L5 fresh confirmation |
|---|---:|---:|
| sessions | 4 | 4 target + 2 negative |
| TP-positive frames with correct proposal in top 3 | 28 / 28 | not the release gate |
| SAP-positive frames | 3 / 56 | evaluated through pair ranking |
| target-session top-1 correct | 1 / 4 | 3 / 4 |
| target-session top-3 contains correct | 2 / 4 | 3 / 4 |
| correct accepted target sessions | 0 / 4 | 2 / 4 |
| wrong accepted sessions | 0 | 2 |
| accepted-pair precision | not eligible | 50% |
| clear-negative false-accept rate | not eligible | 50% |

**Result.** `ACTOR_6DOF_AVAILABLE` and the calibrated reconstruction path are
engineering passes. Anatomical identification fails: L4 remains limited by
missing visible SAP evidence in two sessions, while L5 independently reproduces
a wrong accepted target session and a wrong accepted negative session. The L5
precision and false-accept rate miss the frozen 75% and 10% gates.

### 11.5 Reading it

- Correct pose synchronization does not by itself provide anatomical identity.
- L4 TP recognition is strong, but the executed trajectories do not reveal SAP
  often enough to build or confirm a two-landmark graph.
- L5 geometry and ultrasound jointly rank the correct pair in three sessions,
  but a competing bone configuration can receive the same multi-view support.
- Raising the score threshold after opening these sessions would be post-hoc and
  would not repair the incorrect top-ranked pair in session `02`.
- No new closed-loop navigation was eligible. The historical L5 TP navigation
  result remains valid but is not a SAP/TP or medial-branch-target result.
- `STAGE2_COMPLETE` and `FULL_MBB_INTERVENTION` remain blocked.

### 11.6 Files

| file | contents |
|---|---|
| `s0010_multiview_landmark_graph_result_v1.json` | concise L4/L5 result and provenance |
| runtime job `52364004/detectors/result.json` | L4 proposal-detector metrics |
| runtime job `52364004/graph_analysis/result.json` | L4 graph development result |
| runtime job `52369053/results/evaluation_52370256/result.json` | frozen L5 confirmation result |
| runtime job `52369053/results/evaluation_52370256/session_results.csv` | per-session acceptance and correctness |

## 12. Simulator-supervised sonoanatomy campaign checkpoint

This section freezes the incomplete 2026-11-22 campaign before a change of
research direction. It preserves the new implementation and negative results;
it does not revise the historical Stage 2 verdicts above.

### 12.1 Criterion

The weak-label dataset had to contain image-supported, first-visible TP, SAP
and other-bone interfaces from at least three independent sessions per level.
An image-only model then had to distinguish those interfaces on opened,
session-separated development data before multiview or closed-loop promotion.
Ambiguous and occluded pixels remained ignored rather than becoming hard
negative labels.

### 12.2 Method

The label builder combined clean ultrasound tracks with privileged
pixel-to-volume correspondence and first-acoustic-bone labels. The resulting
labels are explicitly `SIMULATOR_WEAK_ANATOMICAL_LABEL`; CT-model-derived SAP
regions remain provisional. A small image-only U-Net predicted TP, SAP,
other-bone and background interface classes. A separate actor-side multiview
module was added to cluster image predictions in calibrated 6DoF coordinates
and abstain unless both TP and SAP have support from distinct frames.

The planned L4 capture broadened the evaluator-designed pose domain to 96
unique poses in four sessions. Absolute waypoints were capture-design inputs
only and were not added to actor observations.

### 12.3 Setup

| item | value |
|---|---|
| final weak-label dataset job | `52385260`; 282 distinct frames |
| dataset SHA-256 | `6393b3981d1b3fafed627cde9de5e14a97544ace1b2f6823168a4ca07ca7493c` |
| segmentation training job | `52387251`; one NVIDIA L4; 4 CPUs; 34 s |
| L4 checkpoint SHA-256 | `38fcf3ba7257ccea9818b50a5013664762a2ccc30843718cfed4a61936d58863` |
| L5 checkpoint SHA-256 | `040b7359852a1917e4a35ba3d239371f49eeacd7ae6c7573a9c55d73796a1bce` |
| widened capture preparation | jobs `52387270`, `52388082` |
| final capture protocol SHA-256 | `dfd039d7405e6031aa541dcc4741841c3fe18a0be5519c1bf67d608302104707` |
| multiview implementation commit | `301562f` |

### 12.4 Results

| measurement | RIGHT L4 | RIGHT L5 |
|---|---:|---:|
| labelled frames / sessions | 112 / 8 | 170 / 13 |
| SAP-positive frames / sessions | 23 / 8 | 76 / 11 |
| SAP interface pixels | 84 | 895 |
| TP-positive frames | 54 | 77 |
| other-bone frames | 101 | 139 |
| opened-development SAP pixel Dice | 0.000 | 0.865 |
| opened-development SAP frame recall | 0 / 4 | 5 / 5 |
| opened-development SAP clear negatives | 8 / 8 | 3 / 3 |

The L4 model collapsed to background and was not promoted. The L5 model was
promising on opened development data, but its TP frame gate accepted the only
negative frame; it is not independent confirmation and was not promoted.

The broadened L4 capture did not produce a completed dataset. Jobs `52387702`
and `52389230` exposed a controller conflict for requested 3.0-rad poses: the
ceiling-assisted controller pursued 3.14 rad while acceptance remained at
3.0 rad. The latter job stopped fail-closed at 0.513 mm position error and
3.530 degrees orientation error. Jobs `52388963` and `52389184` stopped at
launcher startup because an empty `SONOGYM_CONFIG_FILE` was exported; they did
not run the simulator. Ninety of the 96 planned poses are below 2.99 rad and
are unaffected by the identified ceiling-assistance conflict.

### 12.5 Reading it

- The campaign established a reproducible weak visible-interface dataset and
  a working L5 development segmentation signal.
- It did not improve RIGHT L4 recognition or complete a fresh capture.
- The L4 failure remains a combination of sparse interface supervision and an
  unresolved physical capture-plan/controller mismatch, not proof that the
  structure is acoustically unobservable.
- The multiview code passed focused unit tests but has not been scientifically
  evaluated with the planned new capture.
- `RIGHT_L4_STAGE2`, `RIGHT_L5_STAGE2` and `STAGE2_COMPLETE` remain failed or
  blocked under their frozen gates. No new closed-loop run was performed.

### 12.6 Files and resume boundary

| file or runtime path | contents |
|---|---|
| `s0010_sonoanatomy_dataset_v1/52385260/dataset` | frozen weak-label dataset and review overlays |
| `s0010_sonoanatomy_segmentation_v1/52387251/training` | L4/L5 checkpoints and opened-development metrics |
| `s0010_sonoanatomy_capture_prepare/52388082/prepared` | 96-pose capture protocol and seal |
| `s0010_sonoanatomy_capture_v1/52389230` | fail-closed controller trace and partial actions |
| `sonoanatomy_labels.py` | visible-interface weak-label construction |
| `sonoanatomy_segmentation.py` | image-only U-Net implementation |
| `sonoanatomy_multiview.py` | actor-side semantic clustering and pair abstention |

No campaign job was active when this checkpoint was written. A future resume
must either remove or redesign the six 3.0-rad waypoints under an explicitly
versioned protocol; it must not reinterpret the incomplete capture as data.

## 13. Clinician-inspired scan execution — are the two scan families physically feasible and observable

`prepare_s0010_clinician_scan.py`, `collect_stage2_dataset.py` and
`evaluate_s0010_clinician_scan.py`, run by Slurm jobs `52475568`, `52476333`
and `52477256`. They prepare evaluator-assisted waypoints, execute them in the
simulator and measure actual motion, freshness and weak visible-interface
transitions.

**Criterion.** Motion feasibility requires every acquired frame to have valid
contact, a fresh acquisition identity, an accepted action and synchronized
realized pose. Anatomical observability additionally requires the declared
image-only morphology change to agree with the privileged TP loss/return pattern;
SAP observability requires image-supported SAP evidence in both sessions at a
level. These criteria were frozen before job `52477256`.

### 13.1 Method

The v2 protocol corrected the patient-relative lateral direction and separated
the two scan families. The longitudinal sequence starts in a TP region, moves
laterally, reverses on a nearby parallel path, returns through the TP region and
continues medially. The transverse sequence changes the realized orientation,
moves cranially and then reverses caudally. All actor-side image measurements
use clean ultrasound, executed pose, contact, acquisition identity and prior
observations. Weak interface labels are opened only by the evaluator.

| clinical reference | reference timestamps | intended probe movement | executed probe movement | expected ultrasound transition | observed transition | freshness/contact | anatomical interpretation | limitation | verdict |
|---|---|---|---|---|---|---|---|---|---|
| Video A | 01:50–03:47 | longitudinal TP region toward AP and lamina regions | right-side longitudinal medial-lateral sweep | repeated TP interfaces change toward AP-region morphology | no SAP-supported longitudinal endpoint in 4 / 4 sessions | 32 / 32 valid | TP-region boundary evidence only | evaluator-designed waypoints; no lamina identity gate | `FAIL` |
| Video B S03/S05/S07 | 04:48–08:06; 15:18–15:44 | lateral TP loss, medial return, AP-region candidate | outward motion 4 / 4; medial return 4 / 4; parallel turn 1 / 4 | TP loss, return and neighboring AP morphology | privileged loss/return occurred, but actor morphology and SAP endpoint did not jointly pass | 32 / 32 valid | reversible TP-region survey, not SAP identity | simulator morphology did not reproduce the complete sequence | `PARTIAL` |
| Video B S10–S13 | 19:29–25:30 | transverse repositioning, cranial-caudal forward/reverse sweep | cranial motion 4 / 4; caudal return 4 / 4 | posterior-shadow and deep-window change with recovery | actor/evaluator transition agreement in L4 session `00` only | 32 / 32 valid | one repeatable L4 boundary transition | not the identity of a specific rediscovered TP | `PARTIAL` |
| Video B S06/S08/S11 | 08:42–09:34; 16:26–16:48; 21:36–22:16 | local or alternative-view cross-check | 32 additional alternative-view observations | independent TP/SAP regional support | 3 SAP-supported L5 frames and 8 SAP-supported L4 frames over all scan families | 32 / 32 valid | sparse simulator-proxy SAP evidence | facet/AP-region appearance is not SAP identity | `INCONCLUSIVE` |

### 13.2 Setup

| what | value | spec |
|---|---|---|
| code revision | `c635dcf834e02bed7c7d92c4a1d6e0b3047a8faa` plus captured dirty patch | §13 |
| anatomy | fixed `s0010`; RIGHT L4 and RIGHT L5 | §13 |
| prepared protocol | job `52475568`; SHA-256 `97539e710d8602d4cff090a7c98a351c1607ea8bd19c92a876f921d1af717f92` | §13 |
| capture | job `52476333`; 1 NVIDIA L4; 4 CPUs | §13 |
| evaluator | job `52477256`; 4 CPUs | §13 |
| inputs | `s0010_clinician_scan_feasibility_v2/52476333` | §13 |
| image transition filters | intensity below 0.20 in a 42-pixel distal band; deepest 28% image rows | `docs/notation.md` |

```bash
sbatch --export=ALL,SONOGYM_CONFIG_FILE=<machine-config>,CLINICIAN_SCAN_DESIGN_VERSION=v2 \
  hpg/jobs/s0010_clinician_scan_capture_v1.sbatch

python workflows/two_stage_navigation/evaluate_s0010_clinician_scan.py \
  --protocol <capture>/context/s0010_clinician_scan_feasibility_v2.yaml \
  --run-root <capture> --prediction <prediction.nii.gz> \
  --segmentation <segmentation.nii.gz> --label-conversion <labels.json> \
  --output-dir <evaluation>
```

Nothing was dropped: 96 attempted observations, 0 dropped, **96 final**.

### 13.3 Results

| measurement, capture job `52476333` | measured | criterion | verdict |
|---|---:|---:|---|
| image valid | 96 / 96 (100%) | 96 / 96 | pass |
| contact valid | 96 / 96 (100%) | 96 / 96 | pass |
| action accepted | 96 / 96 (100%) | 96 / 96 | pass |
| distinct image hashes | 96 / 96 (100%) | 96 / 96 | pass |
| tracking position error | median 1.185 mm; p90 1.368 mm; maximum 1.715 mm | exact-capture verifier | pass |
| tracking orientation error | median 0.543 degrees; p90 0.924 degrees; maximum 1.196 degrees | exact-capture verifier | pass |
| longitudinal transition | 0 / 4 sessions (0%) | all required sequence elements | fail |
| transverse reversible transition | 1 / 4 sessions (25%) | actor/evaluator agreement | fail |

| weak-label observability, evaluator job `52477256` | RIGHT L4 | RIGHT L5 |
|---|---:|---:|
| TP-supported frames | 31 / 48 (64.6%) | 15 / 48 (31.3%) |
| SAP-supported frames | 8 / 48 (16.7%) | 3 / 48 (6.3%) |
| TP-and-SAP-supported frames | 6 / 48 (12.5%) | 1 / 48 (2.1%) |
| sessions meeting SAP observability | 0 / 2 (0%) | 0 / 2 (0%) |

**Verdict: motion feasibility passes, anatomical transition observability
fails.** Both scan families are physically executable, but the complete
image-supported sequence passes in only 1 / 4 sessions and SAP observability
passes at neither level.

### 13.4 Reading it

- Nominal angle values were not treated as proof of imaging-plane identity.
- Camera motion, commanded pose and synchronized realized pose remain separate.
- A changed image hash is necessary but not sufficient; contact, action and
  producer sequence also passed for every observation.
- The capture is an evaluator-assisted feasibility experiment, not an actor
  policy and not autonomous landmark localization.
- AP-region or facet-like evidence was not promoted to SAP identity.

### 13.5 Files

| file | contents |
|---|---|
| `configs/experiments/s0010_sonoanatomy_dataset_v2.yaml` | v2 source/session selection |
| runtime job `52476333/sessions` | executed robot, ultrasound and synchronized pose records |
| runtime job `52477256/evaluation/frame_results.csv` | per-frame motion and image measurements |
| runtime job `52477256/evaluation/trajectory_results.csv` | per-scan transition verdicts |
| runtime job `52477256/evaluation/weak_interface_labels.npz` | evaluator-only weak labels |

## 14. Perception and information value — do new observations support SAP/TP identity

`prepare_s0010_sonoanatomy_dataset.py`,
`train_s0010_sonoanatomy_segmentation.py` and
`evaluate_s0010_clinician_scan_perception.py`, run by Slurm jobs `52477797`,
`52478035`, `52478290` and `52478433`. They build the expanded dataset, fit
image-only models and compare frame and multiview decisions with provisional
weak labels.

**Criterion.** An eligible pair selector requires accepted-pair precision at
least 75%, clear-negative false-acceptance at most 10%, nontrivial coverage and
both level-specific gates to pass before fresh closed-loop execution. A model
that abstains on all sessions does not satisfy coverage. The scan-v2 replay is
opened development evidence because its frames contributed to the training
dataset.

### 14.1 Method

The dataset builder restricted L4 and L5 source sessions independently and
preserved session-level train, calibration and development separation. The
first bounded 60-epoch run showed that L5 early-stopping occurred before the
historical foreground-learning onset. The single retry kept the data, seed,
architecture, loss and maximum epochs unchanged and prevented patience-based
stopping before 20 completed epochs. No hyperparameter sweep followed.

The perception replay computes image-only TP, SAP and other-bone observations,
projects them through actor-visible synchronized 6DoF and applies the same
physical-viewpoint separation rule. Duplicate viewpoints, repeated hashes,
missing SAP evidence, reversed order and cyclic pose-image mismatch are explicit
controls.

### 14.2 Setup

| what | value | spec |
|---|---|---|
| dataset | job `52477797`; 378 distinct frames; SHA-256 `dd7f744b69f7382efdb5467ae2b984f1adb501208932e25e6865b9a2e5a71939` | §14 |
| first training | job `52478035`; L5 stopped at epoch 11 before foreground learning | §14 |
| bounded retry | job `52478290`; minimum 20 epochs before early stopping | §14 |
| L4 checkpoint | SHA-256 `2e0f7b88ca4a7c815df414d3955215eaf691d6abaf29516254b0b6c1feb912e7` | §14 |
| L5 checkpoint | SHA-256 `e5627eb9096da6e0d466a531531987c5f487fba850247ef1f5e58285bfcab6c9` | §14 |
| scan replay | job `52478433`; capture job `52476333` | §14 |
| resources | each training run: 1 NVIDIA L4 and 4 CPUs; replay: 4 CPUs | §14 |

```bash
sbatch --export=ALL,SONOGYM_CONFIG_FILE=<machine-config>,\
SONOANATOMY_DATASET_JOB_ID=52477797,\
SONOANATOMY_DATASET_NAMESPACE=s0010_sonoanatomy_dataset_v2,\
SONOANATOMY_SEGMENTATION_NAMESPACE=s0010_sonoanatomy_segmentation_v2r1 \
  hpg/jobs/s0010_sonoanatomy_segmentation_v1.sbatch

sbatch --export=ALL,SONOGYM_CONFIG_FILE=<machine-config>,\
CLINICIAN_SCAN_CAPTURE_JOB_ID=52476333,\
CLINICIAN_SCAN_EVALUATION_JOB_ID=52477256,\
CLINICIAN_SCAN_DESIGN_VERSION=v2,\
SONOANATOMY_SEGMENTATION_JOB_ID=52478290,\
SONOANATOMY_SEGMENTATION_NAMESPACE=s0010_sonoanatomy_segmentation_v2r1 \
  hpg/jobs/s0010_clinician_scan_perception_v1.sbatch
```

Nothing was dropped from the scan replay: 96 attempted observations, 0 dropped,
**96 final**. Near-duplicate training frames were excluded by the dataset split
logic and recorded in each checkpoint bundle.

### 14.3 Results

| opened development, training job `52478290` | RIGHT L4 | RIGHT L5 | gate |
|---|---:|---:|---:|
| pair precision | 6 / 11 (54.5%) | 4 / 5 (80.0%) | at least 75% |
| pair recall | 6 / 6 (100%) | 4 / 4 (100%) | report |
| clear-negative false-acceptance | 5 / 18 (27.8%) | 1 / 4 (25.0%) | at most 10% |
| model gate | fail | fail | both levels pass |

| opened scan-v2 replay, job `52478433` | RIGHT L4 | RIGHT L5 |
|---|---:|---:|
| TP frame precision | 28 / 29 (96.6%) | 15 / 20 (75.0%) |
| SAP frame precision | 8 / 33 (24.2%) | 2 / 4 (50.0%) |
| pair frame precision | 6 / 22 (27.3%) | 1 / 3 (33.3%) |
| pair false-positive rate | 16 / 42 (38.1%) | 2 / 47 (4.3%) |
| multiview session acceptance | 0 / 2 (0%) | 0 / 2 (0%) |

| information arm, job `52478433` | observed result | interpretation |
|---|---|---|
| static ultrasound | pair precision 7 / 25 (28.0%) over both levels | insufficient SAP discrimination |
| static ultrasound plus pose | not separately decision-gated | no isolated information-gain claim |
| unordered multiview | 0 / 4 sessions accepted | coverage failure |
| ordered observations | reversing order changed no original abstention; 0 / 4 accepted | current rule uses geometry, not temporal direction |
| active clinician-inspired scanning | not autonomous; evaluator-designed capture only | not measured |

| control, job `52478433` | abstained | residual failure |
|---|---:|---|
| repeated image hash | 4 / 4 (100%) | none |
| duplicate viewpoint | 4 / 4 (100%) | none |
| missing SAP evidence | 4 / 4 (100%) | none |
| reversed observation order | 4 / 4 (100%) | no information gain over original abstentions |
| cyclic pose-image mismatch | 3 / 4 (75%) | one incorrect L4 acceptance |
| competing bone | unresolved in 4 / 4 (100%) | current pair rule does not use other-bone rejection |

**Verdict: perception and information value fail.** The bounded retry repaired
the L5 background collapse, but neither level passed the false-acceptance gate,
the scan-v2 replay had poor SAP precision, and the multiview rule provided no
session coverage.

### 14.4 Reading it

- The new capture contains 11 simulator-proxy SAP-supported frames, so the
  failure is not equivalent to zero signal.
- The present model cannot distinguish that sparse signal from competing bone
  reliably enough for Stage 2 acceptance.
- The scan replay is not independent confirmation and cannot establish
  generalization even if its metrics had passed.
- A second hyperparameter sweep is not justified by the observed competing-bone
  and longitudinal-observability failures.

### 14.5 Files

| file | contents |
|---|---|
| runtime job `52477797/dataset` | expanded weak-label dataset and manifest |
| runtime job `52478290/training/result.json` | level-specific opened-development metrics |
| runtime job `52478290/training/*.pt` | bounded-retry checkpoints |
| runtime job `52478433/evaluation/actor` | actor-only predictions and session decisions |
| runtime job `52478433/evaluation/evaluator/result.json` | frame metrics and controls |

## 15. Stage 2 promotion — is fresh autonomous evaluation scientifically eligible

`build_s0010_clinician_scan_review.py`, run by Slurm job `52478983`. It
produces synchronized full and blinded replays and records the final promotion
decision without exposing evaluator data to the actor.

**Criterion.** `STAGE2_COMPLETE` requires correct RIGHT L4/L5 TP and SAP
identity, accepted-pair precision at least 75%, clear-negative false-acceptance
at most 10%, actor-only navigation, physical return, fresh ultrasound
confirmation and four fresh integrated sessions. Full MBB intervention is out
of scope and remains blocked.

### 15.1 Method

The promotion audit consumes the frozen motion, perception and control results.
It does not run fresh autonomous sessions after a development gate fails. The
full replay places privileged weak labels only in panel 4; the blinded replay
withholds that panel while leaving actor-visible panels 1–3 unchanged.

### 15.2 Setup

| what | value | spec |
|---|---|---|
| review-video job | `52478983`; 4 CPUs | §15 |
| source capture | job `52476333`; 96 observations | §13 |
| source perception | job `52478433` | §14 |
| output | 192 frames per video; 8 fps; 1280 × 720 | §15 |
| fresh closed-loop job | not submitted because §14 failed | §15 criterion |

```bash
sbatch --export=ALL,SONOGYM_CONFIG_FILE=<machine-config>,\
CLINICIAN_SCAN_CAPTURE_JOB_ID=52476333,\
CLINICIAN_SCAN_EVALUATION_JOB_ID=52477256,\
CLINICIAN_SCAN_PERCEPTION_JOB_ID=52478433 \
  hpg/jobs/s0010_clinician_scan_review_v2.sbatch
```

Nothing was dropped: all 96 acquired observations appear in both videos.

### 15.3 Results

```text
RIGHT_L4_TP_IDENTIFICATION = PARTIAL
RIGHT_L4_SAP_IDENTIFICATION = FAIL
RIGHT_L5_TP_IDENTIFICATION = PARTIAL
RIGHT_L5_SAP_IDENTIFICATION = FAIL

LONGITUDINAL_MEDIAL_LATERAL_SWEEP = PARTIAL
TRANSVERSE_CRANIAL_CAUDAL_SWEEP = PARTIAL
REVERSIBLE_TRANSITION_EVIDENCE = PARTIAL

ANATOMICAL_DISCRIMINATION = FAIL
ACTOR_ONLY_NAVIGATION = FAIL
FRESH_CONFIRMATION = FAIL

RIGHT_L4_STAGE2 = FAIL
RIGHT_L5_STAGE2 = FAIL
STAGE2_COMPLETE = FAIL

FULL_MBB_INTERVENTION = BLOCKED
```

**Verdict: Stage 2 promotion fails.** No fresh closed-loop session was run,
because both level-specific perception gates and the anatomical-observability
gate failed before promotion.

### 15.4 Reading it

- The physical scan procedure is feasible on `s0010` under evaluator-assisted
  waypoints.
- The current evidence does not support an autonomous SAP/TP identity claim.
- The remaining blocker is reproducible: sparse longitudinal SAP observation,
  weak SAP/competing-bone discrimination and no multiview coverage.
- This is simulator-proxy validation only, not clinical anatomical validation.

### 15.5 Files

| file | contents |
|---|---|
| runtime job `52478983/review/s0010_clinician_scan_actual_execution_full.mp4` | synchronized four-panel replay; SHA-256 `cc155ff79d7b61a5a40a6eeadbe88f625a0e46a8af64b71d3f3000ca18b60cd8` |
| runtime job `52478983/review/s0010_clinician_scan_actual_execution_blinded.mp4` | blinded replay; SHA-256 `aa9ca00bdcde1098ed7b92bddfd9305ccea2320fef5b7a0d70c8c5dcf2d35949` |

## 16. Hard-negative repair and trajectory-scoped SAP/TP identity

Jobs `52482280`, `52485117`, `52485420`, `52485680`, and `52488270`
replace the failed §14 perception result with an opened-development repair. The
historical failure remains valid for its frozen checkpoints and capture.

**Goal.** Determine whether additional TP-negative/SAP-or-other-bone views and
executed-trajectory boundaries can repair SAP/TP discrimination before a fresh
closed-loop run.

### 16.1 Method

The added capture contains both TP-positive views and actor-mined hard
negatives where TP is absent while SAP or competing bone can remain visible.
Training retained clean ultrasound as the only image input, added a bounded
negative-frame presence loss, and kept level-specific checkpoints. The actor
clusters observations only within one executed trajectory and selects the
highest-scoring supported trajectory. Provisional interface labels are used
only after the actor result is sealed to measure selected-cluster identity.

The fresh controller frozen at commits `876b038` and `f98a2c9` starts from the
working Stage 1 and TP handoff, searches for a longitudinal TP reference,
moves laterally until TP evidence is lost or the budget is reached, reverses
medially, and then executes a distinct transverse cranial-caudal sweep. It
abstains on stale or invalid acquisition and requires physical return plus new
TP-and-SAP ultrasound evidence.

### 16.2 Reproducible setup

| item | value |
|---|---|
| additional SAP/TP capture | job `52482280`; 8 / 8 sessions; 192 / 192 distinct frames |
| hard-negative capture | job `52485117`; 6 / 6 sessions; 144 / 144 distinct frames |
| dataset | job `52485420`; 714 unique frames; SHA-256 `9fb58a09605ddb2a1b8bae4dd612192236de479510cbcd8aabcd5b9a6b245987` |
| L4 checkpoint | job `52485680`; SHA-256 `e8f03ce7dac42329f716ff8f63b0dbe8da50f5fd55d067475f3fb3eef5e35cb3` |
| L5 checkpoint | job `52485680`; SHA-256 `d5688fd3bc8e1fd6703e2c96bc3a310e017ee508ede32c53dcce9ea3f875e1ca` |
| opened replay | job `52488270`; capture job `52482280` |
| fresh protocol | `s0010_mbb_single_anatomy_poc_protocol_v4.yaml`; SHA-256 `c35b04f1820d9b9495fd174d277a19481c5448fbd13600a2b0ce28a6a3b6dd39` |

All captures used one NVIDIA L4 and four CPUs. Dataset assembly, replay, and
evaluation were CPU-only. The hard-negative source added 19 TP-negative L4
frames and 20 TP-negative L5 frames, including TP-negative/SAP-positive views.

### 16.3 Results

| frozen v7 development split | RIGHT L4 | RIGHT L5 |
|---|---:|---:|
| pair precision | 6 / 6 (100%) | 7 / 9 (77.8%) |
| clear-negative false-acceptance | 0 / 10 (0%) | 2 / 12 (16.7%) |
| pair recall | 6 / 12 (50.0%) | 7 / 11 (63.6%) |

The L5 development false-acceptance gate does not pass. The independently
calibrated L5 rule did pass its opened calibration gate, so the checkpoint was
retained for a bounded trajectory-scoped replay rather than declared complete.

| opened trajectory replay, job `52488270` | RIGHT L4 | RIGHT L5 |
|---|---:|---:|
| pair frame precision | 18 / 20 (90.0%) | 31 / 34 (91.2%) |
| pair frame false-positive rate | 2 / 62 (3.2%) | 3 / 58 (5.2%) |
| trajectory-scoped session acceptance | 4 / 4 | 4 / 4 |
| selected pair identity within 3 pixels | 4 / 4 | 4 / 4 |

The old full-scan multiview rule accepted 0 / 8 sessions because it mixed
separate scan trajectories. The trajectory-scoped actor accepted 8 / 8 and
the evaluator found the selected TP and SAP cluster identities correct in
8 / 8. This result includes development sessions used in training and is not
an independent confirmation.

### 16.4 Interpretation

- The fixed `s0010` simulator contains usable TP/SAP ultrasound signal.
- The dominant data problem was not a lack of generic TP-positive frames; it
  was a lack of hard TP-negative views containing SAP or competing bone.
- Executed trajectory boundaries add useful information, but the opened replay
  alone does not prove autonomous navigation.
- The remaining scientific gate is four fresh actor-only sessions with physical
  return and fresh ultrasound confirmation. The review videos must be rebuilt
  from that execution rather than reusing the evaluator-designed capture.

```text
RIGHT_L4_TP_IDENTIFICATION = OPENED_PASS
RIGHT_L4_SAP_IDENTIFICATION = OPENED_PASS
RIGHT_L5_TP_IDENTIFICATION = OPENED_PASS
RIGHT_L5_SAP_IDENTIFICATION = OPENED_PASS

LONGITUDINAL_MEDIAL_LATERAL_SWEEP = IMPLEMENTED_NOT_FRESHLY_EXECUTED
TRANSVERSE_CRANIAL_CAUDAL_SWEEP = IMPLEMENTED_NOT_FRESHLY_EXECUTED
REVERSIBLE_TRANSITION_EVIDENCE = OPENED_PASS

ANATOMICAL_DISCRIMINATION = OPENED_PASS
ACTOR_ONLY_NAVIGATION = NOT_YET_EXECUTED
FRESH_CONFIRMATION = NOT_YET_EXECUTED

RIGHT_L4_STAGE2 = NOT_YET_CONFIRMED
RIGHT_L5_STAGE2 = NOT_YET_CONFIRMED
STAGE2_COMPLETE = NOT_YET_CONFIRMED

FULL_MBB_INTERVENTION = BLOCKED
```

### 16.5 Files

| file | contents |
|---|---|
| `source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/sonoanatomy_active_scan.py` | actor-side observation-driven scan state machine |
| `workflows/two_stage_navigation/run_stage2_tp_active.py` | Stage 1/TP handoff, SAP/TP sweeps, return, and fresh confirmation |
| `workflows/two_stage_navigation/evaluate_stage2_tp_active.py` | privileged selected-cluster and fresh-label evaluation |
| runtime job `52488270/evaluation/actor` | opened actor predictions and trajectory decisions |
| runtime job `52488270/evaluation/evaluator/result.json` | opened frame and selected-pair identity metrics |
| runtime job `52478983/review/manifest.json` | source paths, information boundary and checksums |
