# s0010 RIGHT L4 medial-branch target localization at RIGHT L5

This experiment asks whether historical ultrasound observations can support a
non-privileged localization of a bony surrogate for the RIGHT L4 medial branch
at RIGHT L5. It does not define a nerve coordinate, needle endpoint or clinical
intervention target.

## Result

**`ULTRASOUND_TARGET_LOCALIZATION = FAIL`.** Neither the four historical Stage 2
sessions nor the completed three-session geometry-designed capture contained a
frame in which the provisional RIGHT L5 SAP and TP target-region endpoint
surfaces were both first-visible bone. A new closed-loop run therefore remains
scientifically ineligible.

| required verdict | result |
|---|---|
| `ANATOMICAL_TARGET_CONTRACT` | `PARTIAL` |
| `ULTRASOUND_TARGET_LOCALIZATION` | `FAIL` |
| `ROBOT_NAVIGATION` | `HISTORICAL_REPLAY_ONLY` |
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

## 5. Engineering checks

The original CPU execution ran 29 focused regressions. The resume preflight ran
44 focused regressions; all passed in 0.116 seconds. The new checks require
the observability protocol to use synchronized-capture integrity gates, recover
the captured commit from immutable run context when older sealed metadata lack
the field, and prevent a partial session selection from appearing complete.

## 6. Next concrete action

Do not use the strict endpoint core as the next learning target. The new
surface-configured frames justify a development-only actor experiment that
asks whether clean ultrasound and motion can distinguish the relative SAP–TP
configuration from other bone views. Its labels must retain separate SAP, TP,
control and ambiguous states, and its evaluation must use new session groups.
Until that actor experiment succeeds, the historical L5 TP pipeline remains
the presentation-ready navigation baseline and must continue to be labelled TP
navigation rather than MBB target localization.
