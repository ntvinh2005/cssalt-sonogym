# SonoGym Stage 2 active-acquisition development pilot

This report records the first continuous fixed-`s0010` Stage 2 active-acquisition pilot after the
contract-corrected CPU replay. It is development evidence, not V19 and not independent scientific
confirmation.

## Result

**The search found a validity-gated candidate supported by three physical viewpoints and returned
to it accurately, but the fresh-confirmation contract failed; the result is
`DEVELOPMENT_NO_GO`.**

| capability | measured result | verdict |
|---|---:|---|
| continuous coarse/fine/micro acquisition | 39 observations in one episode | executed |
| multi-view candidate evidence | 3 distinct hashes and 3 distinct viewpoints | PASS |
| return to remembered best | 0.0 mm; 0.0 degrees | PASS |
| evaluator-only final geometry | 8.197 mm; 9.535 degrees | PASS at 10 mm / 10 degrees |
| evaluator-only L4 visibility | 0.2548 | PASS at 0.20 |
| fresh post-return confirmation | 0 / 2 fresh frames | FAIL |
| actor outcome | `ABSTAIN` | correct fail-closed behavior |
| full Unity-equivalent Stage 2 | not established | **DEVELOPMENT_NO_GO** |

The evaluator-only successes do not override actor abstention. The actor did not read the target
pose, raw L4 labels, geometric error, or visibility score.

## 1. Frozen method and provenance

The actor used the sealed Phase 1 `cf_00` handoff, the unchanged V15 validity classifier and
quality regressor, the historical V18 probability threshold, and the contract-corrected temporal
decision rule. No model fitting or threshold selection occurred.

| item | identity |
|---|---|
| execution commit | `94e312026923d6b25d442a7fbd3cdd200f708589` |
| implementation commit named by protocol | `d7509f39bcdc310f0630116e896e117646bd68af` |
| evaluator recovery commit | `390416b71ffcaa1606b88c962429e0ad16621b30` |
| protocol | SHA-256 `85d09795e9a47a402077f5f066bae9f2353b389c03e2447cc20821548cf30dd6` |
| Phase 1 actor handoff | SHA-256 `9d3532f06f15443d7a0d3ac7c32d67430c93de45f1fe8d61bbdbe8ac7c58c13e` |
| frozen Stage 2 model | SHA-256 `7d3b113ba33b1b80448264a0d2e7f2c2cd5561b26021707648c4eef43d586672` |
| actor result | SHA-256 `7561b86aba518b8fd9f5989e65d02fd1c36669c37b697c11e13f836b6118fa69` |

The frozen search comprised 27 coarse poses, six fine offsets, and six micro offsets. Candidate
selection required probability at least 0.5666666666666667, quality drop at most 0.08, three
distinct hashes, and three physical viewpoints separated by at least 1 mm or 1 degree inside the
4.5 mm / 6 degree neighborhood.

## 2. Execution

Slurm job `51607477` ran on `anest-dept`, partition `hpg-turin`, node `c0604a-s8`, with four CPU
cores, 30 GB requested RAM, and one NVIDIA L4. The driver was 580.178.04. The job ran from
13:00:09 to 13:00:53 EST on 2026-11-16.

The simulator actor completed successfully in 37.86 seconds wall time and used 3,678,568 KiB peak
RSS. GPU telemetry sampled a peak of 5,009 MiB and 39% utilization. The job-level state is
`FAILED (1:0)` only because the original evaluator looked for `actor_result.sha256` while the
writer correctly produced `actor_result.json.sha256`. Commit `390416b` fixes that filename bug;
the recovered evaluator was run against the already sealed actor artifact, without another GPU
run.

All 64 preflight tests passed:

| suite | passed | failed |
|---|---:|---:|
| active-pilot boundary | 7 | 0 |
| temporal decision | 23 | 0 |
| existing Stage 2 regression | 34 | 0 |

## 3. Actor trajectory and selection

The actor executed 371 action steps: 51 for the Phase 1 handoff, 302 during coarse search, and 18
during fine search. No action was position-clamped, terminated, or truncated; 75 action rows used
the declared angle-ceiling clamp. Micro poses were already inside the broad 5 mm / 5 degree
movement tolerance, so they produced observations but no additional movement.

The run stored 41 observations: 27 coarse, six fine, six micro, and two confirmation attempts.
There were 17 distinct hashes, 17 producer-advanced observations, and 17 technically valid image
observations. The multi-view rule selected frame 29 with probability 0.9222222222 and predicted
quality 0.4671614310. It had three supporting hashes and viewpoints. The robot returned to the
remembered pose with 0.0 mm / 0.0 degree measured relative error.

Evaluator-only inspection after actor sealing found 8.19697 mm position error, 9.53460 degrees
orientation error, and L4 visible fraction 0.2548. This shows that the chosen candidate happened
to satisfy this development evaluator; it does not convert the actor outcome into success.

## 4. Why fresh confirmation failed

The protocol used confirmation offsets of plus/minus 0.5 label-map index. At 1.5 mm per index,
each requested dither was only 0.75 mm. The same motion helper accepted any pose already within
5 mm and 5 degrees, so it returned before calling `env.step`.

Consequently, both confirmation records had:

- zero action steps;
- unchanged producer sequence 372;
- the same exact image hash as selected frame 29;
- `transition_valid=false`, `producer_advanced=false`, and `image_valid=false`.

The fresh-confirmation guard therefore rejected them, as designed. This is a demonstrated
motion-contract mismatch, not evidence that repeated stale images corroborated the candidate.

The original evaluator packaging failure is independent: it concerned only the sidecar filename.
Recovered report SHA-256 is
`3647b60f1fcb0d15b98f02b90fa260d64285382ab34cd875c1e5f7d3054c5a4b`.

## 5. Evidence boundary

- This run uses one fixed patient, one seed, one sealed Phase 1 handoff, and opened development
  components.
- V16, V17, and V18 remain `CONFIRMATION_NO_GO`; the temporal replay remains
  `EXPLORATORY_NO_GO`.
- The result does not establish independent confirmation, patient generalization, clinical
  validity, or MBB performance.
- No V19, PPO training, model retraining, or threshold adjustment occurred.
- `FULL_STAGE2_CLOSED_LOOP` is now attempted at development level but remains `NOT_ESTABLISHED`.

## 6. Artifacts and next approval

External artifacts occupy 4,916,102 bytes at:

`/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_active_acquisition_pilot_v1/51607477/`

The most important files are `results/actor_result.json`,
`results/evaluation_report_recovered.json`, `public/frames.jsonl`, `public/ultrasound.npz`, and the
separate evaluator-only `privileged/annotations.jsonl` and `privileged/actions.jsonl`.

The smallest next change requiring approval is a new development protocol, not a retry of this
sealed result. Keep the 0.75 mm confirmation dither but give confirmation motion its own tolerance
strictly below the dither magnitude (proposed 0.25 mm), then require verified outbound motion,
verified return, advancing producer sequence, distinct hashes, and two fresh confirmations. Add a
CPU invariant that rejects any protocol where confirmation displacement is not greater than its
movement tolerance. With no model or decision-parameter change, one L4, four CPUs, 30 GB, and at
most 15 minutes should be sufficient for one bounded engineering run.
