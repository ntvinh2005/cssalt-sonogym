# s0010 single-anatomy MBB simulator proof of concept

## 1. Criterion

The milestone required one causal workflow from ultrasound exploration through
RIGHT L4/L5 landmark-view localization, physical return, fresh confirmation,
validated MBB target construction, and simulated intervention on `s0010`.
`END_TO_END_PASS` required all four starts to complete every component.
`PARTIAL_PASS` required all four Stage 1 maps and all eight provisional TP
surface-core target views, while retaining a fail-closed intervention result.
Anything below those gates is `BLOCKED`.

### Method

- Stage 1 reused the frozen seven-landmark ultrasound sweep, patch recognizer,
  spatial evidence map, ordering constraints, and abstention contract.
- Stage 2 received only the Stage 1 actor result, clean ultrasound, frozen image
  model predictions, executed pose/motion feedback, freshness, and actor-owned
  candidate memory.
- Each RIGHT L4/L5 TP surface-core search executed a 27-cell physical coarse
  grid, six fine views, six micro views, return-to-best, and three fresh
  return-to-best observations.
- The evaluator opened raw labels, native pixel-to-volume coordinates and the
  provisional surface geometry only after actor results were sealed.
- The intervention contract remained fail-closed because the required SAP
  boundaries, branch-specific target construction and MBB needle/injection
  task are not available.

### Setup

| item | value |
|---|---|
| patient | `s0010` only |
| protocol | `s0010_mbb_single_anatomy_poc_protocol_v3.yaml` |
| protocol SHA-256 | `9ed7680856e64e54270d33f36867021732de7de8da77821fcaf951d4f4e65a55` |
| execution commit | `79d3f35560d36addf5185af55993a449118c6043` |
| Slurm job | `52039283`, `COMPLETED`, exit `0:0` |
| resources | account `anest-dept`; 1 NVIDIA L4; 4 CPU; 30 GB; 11 min 36 s |
| L4 model | SHA-256 `6188c7713cac2f21485d030ac79f4e26ac9c4f0f179008aee57a6d891fe9e8c0` |
| L5 model | SHA-256 `338351fdee5a8444dee5a6ec250d828b8f1eedbc3f76e79495aa1aafbdeb2f8b` |
| thresholds | 0.50 for both models; unchanged |
| TP geometry | provisional L4 SHA `f60c11...`; provisional L5 SHA `01c910...` |

The V3 confirmation rule was frozen before job `52039283`: acquire three
physically returned fresh frames and require at least two evaluator
proxy-positive frames. The actor still required all three image predictions to
pass the unchanged model threshold. Two earlier technical submissions,
`52038551` and `52038792`, were cancelled before Stage 2 evidence collection
after protocol-binding and protocol-allowlist errors were identified.

### Results

#### Stage 1 anatomical discovery

| metric | result |
|---|---:|
| sessions | 4 / 4 complete maps |
| frames | 844 |
| correct landmark events | 28 / 28 |
| event precision / recall | 1.000 / 1.000 |
| median / p90 / maximum error | 3.0 / 6.375 / 10.0 mm |
| abstentions | 0 |

#### Stage 2 RIGHT L4/L5 TP surface-core acquisition

| metric | result |
|---|---:|
| actor-accepted sessions | 4 / 4 |
| acquired / distinct-hash frames | 336 / 336 |
| action steps | 13,592 |
| evaluator-confirmed targets | 6 / 8 (75%) |
| RIGHT L4 target success | 2 / 4 |
| RIGHT L5 target success | 4 / 4 |
| fresh proxy-positive / ambiguous frames | 17 / 7 of 24 |
| mean / maximum interface-to-core error | 1.460 / 2.697 mm |
| mean / maximum return position error | 0.239 / 0.250 mm |
| mean / maximum return orientation error | 0.027 / 0.141 degrees |
| actor session abstention | 0 / 4 |

Sessions `mbbpoc_v3_00` and `mbbpoc_v3_01` passed both target-view gates.
In `mbbpoc_v3_02`, the actor-selected L4 view and all three confirmations were
proxy-ambiguous. In `mbbpoc_v3_03`, the selected L4 view was proxy-positive,
but only one of three confirmations remained proxy-positive. Offline diagnosis
found 11 and 9 L4 proxy-positive search views respectively in those sessions,
so the failure is candidate discrimination, not absence of physical coverage.
The strongest image probability can favor a morphology-compatible view that
touches the evaluator's unresolved TP boundary zone.

#### MBB intervention and aggregate

| metric | result |
|---|---:|
| validated SAP targets | 0 / 2 |
| validated branch-specific MBB target regions | 0 / 2 |
| simulated MBB interventions | 0 / 4 |
| end-to-end successes | 0 / 4 |
| end-to-end success / failure rate | 0% / 100% |
| final verdict | **`BLOCKED`** |

### Reading it

The engineering path is integrated through exploration, L4/L5 handoff,
physical active search, return-to-best and fresh ultrasound acquisition. Stage
1 is reproducible on the four fixed-anatomy starts, L5 TP surface-core
acquisition is 4/4, and physical return tracking is sub-millimetric. These facts
do not establish MBB target localization: L4 TP selection is only 2/4 under the
provisional proxy evaluator, both RIGHT SAP boundaries remain unresolved, and
no validated mapping converts SAP/TP evidence into the L3/L4 medial-branch
surrogate regions.

The repository's `robot_US_guided_surgery` task uses a `standard_right_traj_L4`
pedicle drill/screw objective and a `screw_6_65` tool. Reusing it as an MBB
needle/injection would silently change the anatomy, instrument and success
criterion, so no intervention was executed. The result is simulator-only,
single-anatomy development evidence; it makes no patient-generalization,
clinical-validation or injection-readiness claim. Historical V16-V18 and
new-start `CONFIRMATION_NO_GO` verdicts remain unchanged.

### Files

| file | contents |
|---|---|
| `experiments/unity_two_stage_transfer/s0010_mbb_single_anatomy_poc_protocol_v3.yaml` | sealed protocol |
| `experiments/unity_two_stage_transfer/s0010_mbb_single_anatomy_poc_v3_result.json` | concise immutable result |
| `workflows/two_stage_navigation/run_stage2_tp_active.py` | physical actor workflow |
| `workflows/two_stage_navigation/evaluate_stage2_tp_active.py` | privileged evaluator and visual evidence |
| `workflows/two_stage_navigation/evaluate_s0010_mbb_poc.py` | aggregate verdict |
| runtime job `52039283/results/end_to_end_report.json` | full aggregate report |
| runtime job `52039283/stage2/*/video/` | robot/probe and ultrasound videos |
| runtime job `52039283/stage2/*/visual_evidence/` | trajectories and evaluator-only overlays |

The smallest scientifically meaningful next change is not another threshold or
confirmation-count retry. It is a reviewed RIGHT L4/L5 SAP boundary plus a
branch-specific bony surrogate-region definition, followed by a dedicated MBB
needle/injection simulator contract. Separately, the L4 image model needs a
session-separated verifier experiment that distinguishes proxy-positive core
views from boundary-ambiguous views before another active-acquisition claim.
