# s0010 RIGHT L4 medial-branch target localization at RIGHT L5

This experiment asks whether historical ultrasound observations can support a
non-privileged localization of a bony surrogate for the RIGHT L4 medial branch
at RIGHT L5. It does not define a nerve coordinate, needle endpoint or clinical
intervention target.

## Result

**`ULTRASOUND_TARGET_LOCALIZATION = FAIL`.** The four historical Stage 2
sessions contained no frame in which the provisional RIGHT L5 SAP and TP
target-region endpoint surfaces were both first-visible bone. A new closed-loop
run was therefore not scientifically eligible.

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
Accordingly, no GPU job or new robot execution was submitted.

### 2.5 Files

| file | contents |
|---|---|
| `s0010_l4_medial_branch_target_v1_result.json` | concise immutable result |
| runtime `frame_results.csv` | per-frame labels and evaluator distances |
| runtime `session_results.csv` | session-level evidence counts |
| runtime `demo.mp4` | actual historical observations with separated panels |
| runtime review ZIP | self-contained external-review package |

## 3. Engineering checks

The final Slurm execution ran 29 focused CPU regressions; all 29 passed in
0.157 seconds. The checks cover target geometry, native coordinate flattening,
actor/evaluator separation, failure on mismatched TP identity and preservation
of occluded SAP evidence as ambiguous. The final review ZIP passed member hash
validation and archive integrity checks.

## 4. Next concrete action

Do not train another classifier or rerun Stage 2 against this frame pool. A
meaningful next experiment requires probe viewpoints deliberately designed to
place the provisional SAP–TP endpoint relationship on the first visible bone
surface, followed by review of whether that relationship is distinguishable in
clean ultrasound. The historical L5 TP pipeline remains the presentation-ready
navigation baseline, but it must continue to be labelled TP navigation rather
than MBB target localization.
