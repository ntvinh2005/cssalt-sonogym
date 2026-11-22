# s0010 RIGHT L4 medial-branch target localization at RIGHT L5

This experiment asks whether historical ultrasound observations can support a
non-privileged localization of a bony surrogate for the RIGHT L4 medial branch
at RIGHT L5. It does not define a nerve coordinate, needle endpoint or clinical
intervention target.

## Result

**`ULTRASOUND_TARGET_LOCALIZATION = FAIL`.** Neither the four historical Stage 2
sessions nor the completed first session of the geometry-designed capture
contained a frame in which the provisional RIGHT L5 SAP and TP target-region
endpoint surfaces were both first-visible bone. The fresh capture ended after
one of three sessions because of a post-capture metadata-verifier defect, so a
new closed-loop run remains scientifically ineligible.

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

`evaluate_s0010_l4mb_target_observability.py`, capture job `52278401`.

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
| capture run | `52278401`, source `3c7d656ba828df7e2230e4d15435a485e2323ab7` | §3 |
| frozen protocol | SHA-256 `089cc97563c90847d91cc172daf9dd513a670713e75372aa0b660e45f68b540d` | §3 |
| resources | one NVIDIA L4, four CPUs, 30 GB, 47 s | §3 |
| attempted | 3 sessions, 36 frames | §3 |
| completed | 1 session, 12 frames | §3 |
| not run after verifier exit | 2 sessions, 24 frames | §3 |

```bash
SONOGYM_CONFIG_FILE="$PWD/.sonogym.env" \
  sbatch hpg/jobs/s0010_l4mb_target_observability_v1.sbatch
```

The first session was captured before the verifier raised a missing metadata
field. CPU replay after the technical repair used the immutable protocol and
run context stored by the job.

### 3.3 Results

| completed-session measurement | value |
|---|---:|
| exact-pose replay | 12 / 12 frames |
| distinct frame hashes | 12 / 12 frames |
| fresh viewpoints | 12 / 12 frames |
| coordinate-array hash mismatches | 0 / 12 frames |
| maximum coordinate error | 0.0000327133 voxel-index |
| planned target frames | 6 |
| both endpoint cores first-visible | 0 / 6 frames |
| ambiguous target frames | 6 / 6 frames |
| negative controls | 6 / 6 frames |
| image-supported positive frames | 0 / 12 frames |

**Verdict: fail.** The completed session supplies no positive endpoint
observation, and the missing two sessions prevent completion of the frozen
observability protocol. Robot search therefore remains ineligible.

### 3.4 Reading it

- The acquisition synchronization and exact-pose mapping passed for all saved frames.
- The six target candidates intersect SAP and TP outside the accepted endpoint core.
- The SAP core was not the first visible bone surface in any target candidate.
- The result does not show that every possible probe pose is unobservable.
- Completing the two missing sessions requires a separately authorized GPU submission.

### 3.5 Files

| file | contents |
|---|---|
| `s0010_l4mb_target_observability_v1_result.json` | capture, verification and observability counts |
| runtime `sessions/s0010l4mbobs_00/verification_report.json` | 12-frame replay and freshness gates |
| runtime `results/evaluation_partial/frame_results.csv` | per-frame clean-image and evaluator evidence |
| runtime `results/evaluation_partial/clean_ultrasound_summary.png` | clean images with evaluator captions |

## 4. Engineering checks

The original CPU execution ran 29 focused regressions. After the fresh-capture
failure, 43 focused regressions passed in 0.150 seconds. The new checks require
the observability protocol to use synchronized-capture integrity gates, recover
the captured commit from immutable run context when older sealed metadata lack
the field, and prevent a partial session selection from appearing complete.

## 5. Next concrete action

Do not train another classifier or start robot target search from this frame
pool. The two frozen offset sessions remain the smallest completion step; their
results can either expose the required relationship or close this exact pose
design without changing its gate. The historical L5 TP pipeline remains the
presentation-ready navigation baseline, but it must continue to be labelled TP
navigation rather than MBB target localization.
