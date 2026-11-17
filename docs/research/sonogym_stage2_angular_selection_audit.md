# SonoGym Stage 2 angular reachability and candidate-selection audit

This record diagnoses the frozen `s0010` new-start confirmation using only job `51668777`
artifacts. It does not change the actor, threshold, search, movement controller, evaluator gate, or
the permanent `CONFIRMATION_NO_GO` verdict.

## Result

**All three evaluable starts physically acquired at least one fully valid, actor-eligible view, but the frozen maximum-quality rule selected a geometrically invalid view; `SELECTION_REPAIR_RECOMMENDED`.**

| step | what was measured | criterion | measured | verdict |
|---|---|---|---|---|
| artifact integrity | public frames, privileged annotations/actions, actor/evaluator seals | exact frame/hash/phase join; canonical wrapped angle agrees within `1e-9` degrees | 142 / 142 complete-run frames joined; Start 00 retained as explicitly partial | PASS |
| physical reachability | acquired frames meeting position <=10 mm, orientation <=10 degrees, L4 visibility >=0.20, and technical validity | at least one acquired fully valid frame per evaluable start | 3 / 3 starts; counts 3, 1, 1 | PASS |
| actor eligibility | fully valid frames satisfying the frozen threshold and multi-view contract | determine whether perception/evidence excluded the valid pose | all 5 / 5 fully valid frames were actor-eligible | PASS |
| frozen selection | selected frame meets the complete evaluator gate | selected frame fully valid | 0 / 3 | FAIL |
| physical return | selected geometry remains stable through return and fresh confirmation | no material angular degradation | changes were `-0.011899`, `-0.019329`, and `-0.019343` degrees | PASS |

The diagnosis is selection failure in all three complete runs. Start 00 is a separate technical
freshness failure and has no privileged annotation artifact from which to infer final geometry.

## 1. Frozen-artifact audit — was a valid candidate acquired and then lost by selection?

`workflows/two_stage_navigation/audit_stage2_angular_selection.py`, run by Slurm job `51683029` and
then rerun after adding the reporting-only waterfall fields. It joins privileged evaluator records
only after the historical actor decision is fixed.

**Criterion.** The user-approved classification is: no fully valid acquired frame implies
`SEARCH_COVERAGE_FAILURE`; a fully valid but ineligible frame implies
`ACTOR_ELIGIBILITY_FAILURE`; a fully valid and eligible frame followed by selection of another
frame implies `SELECTION_FAILURE`; loss only after return implies `POST_RETURN_DEGRADATION`.

### 1.0 Quantities

| symbol | definition | source |
|---|---|---|
| Stage 2 bounded orientation refinement | the frozen bounded angular continuation | [`docs/notation.md`](../notation.md#stage-2-bounded-orientation-refinement) |
| Stage 2 multi-view evidence | the frozen distinct-view eligibility contract | [`docs/notation.md`](../notation.md#stage-2-multi-view-evidence) |
| Stage 2 fresh confirmation | post-return reacquisition evidence | [`docs/notation.md`](../notation.md#stage-2-fresh-confirmation) |

### 1.1 Method

- Join `public/frames*.jsonl` to `privileged/annotations.jsonl` by frame sequence, hash, and phase.
- Recompute orientation error with the canonical `[-pi, pi)` convention and reject a join that
  differs from the stored evaluator value by more than `1e-9` degrees.
- Reconstruct each frame's action group from causal phase order and `action_step_count`.
- Reapply the frozen `0.5666666666666667` classifier threshold, 4.5 mm / 6-degree neighborhood,
  0.08 maximum quality drop, three hashes, and three physically distinct viewpoints.
- Apply the complete evaluator gate only after actor eligibility and selection are fixed.
- Compare the frozen selected frame against every physically acquired frame as a retrospective
  evaluator-only diagnosis; no hindsight candidate becomes an actor result.

### 1.2 Setup

| what | value | spec |
|---|---|---|
| source run | GPU job `51668777`; execution commit `ae45f65c927115e1fd5a6319ad6669430fa8432b` | frozen confirmation |
| diagnostic job | CPU job `51683029`; implementation commit `5f274c53e6bc4e70aece0070d89427122ddf2784` | 4 CPUs, 16 GB, 0 GPU |
| input protocol | `stage2_new_start_confirmation_protocol_v1.yaml`; SHA-256 `e3a98e42090df0f02e3e29e1caf25ea8c0b1a3a013fbd233059b1769b774b99c` | frozen |
| model | frozen V15 bundle; threshold `0.5666666666666667` | unchanged |
| patient | `s0010`; one anatomy | fixed-anatomy diagnosis |
| Python | pinned SonoGym Python 3.10.22 | job context |
| resources | `anest-dept`; 4 CPUs; 16 GB; 0 GPU | job `51683029` |

```bash
sbatch --parsable --account=anest-dept --qos=anest-dept --partition=hpg-default \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:30:00 \
  --job-name=sg-s2-angle-audit \
  --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/sg-s2-angle-audit-%j.out \
  --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env \
  hpg/jobs/two_stage_stage2_angular_selection_audit.sbatch

# Resolved diagnostic command inside the job
$SONOGYM_ENV_PREFIX/bin/python \
  workflows/two_stage_navigation/audit_stage2_angular_selection.py \
  --run-root /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_new_start_confirmation_v1/51668777 \
  --protocol experiments/unity_two_stage_transfer/stage2_new_start_confirmation_protocol_v1.yaml \
  --output-dir "$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/stage2_angular_selection_audit/51683029/results"
```

| | value |
|---|---:|
| attempted starts | 4 |
| partial, no evaluator annotation or sealed actor (`confirm_new_00`) | 1 |
| **complete evaluable starts** | **3** |
| complete-run frames joined | 142 |
| complete-run search frames | 136 |

Job `51683029` completed `0:0` in 12 seconds. The audit used 2.96 seconds and 112,792 KiB
peak RSS; final reporting-only regeneration used 2.40 seconds and 117,640 KiB peak RSS. No GPU or
simulator was used.

### 1.3 Results

#### Acquired valid frames and frozen selection

| start | fully valid acquired frames | best acquired orientation | selected frame | selected orientation | final confirmation orientation | primary cause |
|---|---|---:|---|---:|---:|---|
| `confirm_new_01` | 32 (fine), 38 (micro), 43 (refinement) | frame 32: 9.051216 degrees | 40 (refinement) | 15.262413 degrees | 15.250515 degrees | `SELECTION_FAILURE` |
| `confirm_new_02` | 45 (refinement) | frame 45: 9.938485 degrees | 44 (refinement) | 11.899842 degrees | 11.880513 degrees | `SELECTION_FAILURE` |
| `confirm_new_03` | 45 (refinement) | frame 45: 9.937912 degrees | 44 (refinement) | 11.897534 degrees | 11.878191 degrees | `SELECTION_FAILURE` |

Every listed valid frame also passed the frozen actor threshold and multi-view evidence rule. All
selected frames passed position and L4-visibility gates but failed the 10-degree orientation gate.

#### Upper-bound reachability

| start | upper requested angle (rad) | realized angle (rad) | tracking error (degrees) | evaluator error (degrees) | position (mm) | L4 fraction | clamp |
|---|---:|---:|---:|---:|---:|---:|---|
| `confirm_new_01` | 2.970000 | 2.965826 | 0.239151 | 9.979439 | 4.202761 | 0.256200 | none |
| `confirm_new_02` | 2.970000 | 2.966541 | 0.198197 | 9.938485 | 3.638186 | 0.253967 | none |
| `confirm_new_03` | 2.970000 | 2.966551 | 0.197623 | 9.937912 | 3.651592 | 0.253867 | none |

The 2.97-rad bound left little margin but was physically sufficient in all three runs. The robot
reached the bound without a refinement-frame angle clamp and remained below the 10-degree gate.

#### Frozen refinement waterfall

| start | refinement candidates | technical and fresh | classifier accepted | neighborhood | quality consistency | distinct hashes | distinct viewpoints |
|---|---:|---:|---:|---:|---:|---:|---:|
| `confirm_new_00` | 5 | 4 | 4 | 4 | 4 | 4 | 4 |
| `confirm_new_01` | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| `confirm_new_02` | 7 | 7 | 6 | 6 | 6 | 6 | 6 |
| `confirm_new_03` | 7 | 7 | 6 | 6 | 6 | 6 | 6 |

The selection error occurred after evidence eligibility: all complete runs retained multiple
refinement candidates through the full actor contract.

#### Image-model ranking versus evaluator geometry

| start | selected quality | fully valid upper-view quality | quality difference (selected minus valid) | selected probability | valid probability |
|---|---:|---:|---:|---:|---:|
| `confirm_new_01` | 0.550657 | 0.539271 | 0.011386 | 0.966667 | 0.950000 |
| `confirm_new_02` | 0.554276 | 0.532666 | 0.021610 | 1.000000 | 0.994444 |
| `confirm_new_03` | 0.543246 | 0.524927 | 0.018318 | 0.997222 | 0.975000 |

The quality regressor ranked the angularly invalid candidate above the fully valid upper view in
3 / 3 complete runs. The classifier accepted both alternatives, so threshold eligibility did not
cause the loss.

#### Refinement-only pool

| start | valid earlier-phase frame excluded? | valid refinement frame remained? | would maximum-quality ranking over all history repair it? |
|---|---|---|---|
| `confirm_new_01` | yes: frames 32 and 38 | yes: frame 43 | no; all-history maximum was invalid frame 37 |
| `confirm_new_02` | no | yes: frame 45 | no; frozen selected frame remained maximum |
| `confirm_new_03` | no | yes: frame 45 | no; all-history maximum was invalid frame 10 |

Restricting final selection to refinement observations removed two valid earlier views in Start 01,
but it is not the common root cause: a valid refinement view remained in every complete run.

#### Start 00 freshness

| item | value |
|---|---:|
| penultimate requested angle | 2.969933509827 rad |
| final clipped request | 2.970000000000 rad |
| separation | 0.000066490173 rad / 0.003809 degrees |
| controller tolerance | 0.25 degrees |
| final executed actions | 0 |
| producer sequence | 1964 -> 1964 |
| exact hash | unchanged |

The target generator deduplicated only numerical equality (`1e-12` rad), not physical
indistinguishability under the 0.25-degree controller tolerance. The cached zero-action frame was
correctly rejected as non-fresh; no final pose is inferred from the absent annotation file.

**Verdict: `SELECTION_REPAIR_RECOMMENDED`**, while the historical experiment remains
`CONFIRMATION_NO_GO`.

### 1.4 Reading it

- Search coverage was sufficient in 3 / 3 evaluable starts.
- Tracking was sufficient: upper-bound errors were 0.197623--0.239151 degrees and no evaluated
  refinement upper view was clamped.
- Starts 02 and 03 followed nearly identical schedules because their preliminary selected frames
  were both frame 10 at realized angles 2.866238 and 2.866279 rad; both quality maxima occurred at
  frame 44 near 2.93233 rad.
- Return and confirmation did not cause failure: their orientation changes were under 0.020 degrees.
- The current quality regressor is useful for image quality but did not order these candidates by
  evaluator orientation correctness.
- Expanding selection to all earlier phases while retaining maximum-quality ranking does not repair
  any start.
- The smallest single next change worth evaluating is selection-only: among frozen-valid,
  multi-view-eligible refinement candidates within the existing 0.08 quality-consistency band,
  prefer the greatest physically realized angle in the already-declared upward refinement sweep.
- That rule is a retrospective proposal from opened data; it must be frozen and reviewed before any
  new GPU confirmation and must not be described as validated.
- Independently, future target generation should reject adjacent requested angles separated by less
  than the frozen movement tolerance before acquisition; that engineering guard does not repair the
  three scientific failures.

### 1.5 Files

| file | contents |
|---|---|
| `experiments/unity_two_stage_transfer/stage2_angular_selection_audit/per_frame_audit.csv` | joined per-frame requested/realized angles, scores, eligibility, clamp, selection, and evaluator-only gates |
| `experiments/unity_two_stage_transfer/stage2_angular_selection_audit/per_start_summary.csv` | concise per-start classification |
| `experiments/unity_two_stage_transfer/stage2_angular_selection_audit/audit_result.json` | machine-readable inputs, hashes, candidate details, waterfall, and preserved verdicts |
| `experiments/unity_two_stage_transfer/stage2_angular_selection_audit/angular_coverage.png` | requested/realized angular coverage with evaluator-only overlays |
| `experiments/unity_two_stage_transfer/stage2_angular_selection_audit/quality_vs_orientation.png` | frozen quality versus evaluator orientation error |
| `experiments/unity_two_stage_transfer/stage2_angular_selection_audit/failure_decomposition.png` | acquired -> eligible -> final pool -> selected -> confirmed table |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_angular_selection_audit/51683029/final_results/` | Blue-storage copy and `SHA256SUMS` |

Focused diagnostic tests passed 4 / 4 and existing Stage 2 regression tests passed 34 / 34. The
raw confirmation artifacts, seals, model, protocol, and historical reports were not modified.
