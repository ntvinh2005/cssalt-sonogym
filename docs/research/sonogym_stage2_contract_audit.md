# SonoGym Stage 2 movement-contract and abstention audit

This report diagnoses the `1 / 144` temporal/multi-view acceptance result on opened fixed-`s0010`
captures. It is a CPU-only audit; no actor contract, model, threshold, capture, or simulator state
was changed.

## Result

**OPTION 1 — CONTRACT CORRECTION FIRST: the collector accepted 315 angle-only-clamped frames that the frozen replay rejected, and applying only the collector-authorized exception raised multi-view acceptance from 1 / 144 to 31 / 144; the correction remains below the prior 52-accept usefulness condition, so active acquisition is still required after the contract is made consistent.**

| step | what was measured | criterion | measured | verdict |
|---|---|---|---|---|
| 1 | collector versus replay movement contract | identical accepted movement semantics | 315 accepted angle-only frames rejected only by replay's combined clamp check | MISMATCH_CONFIRMED |
| 2 | original abstention waterfall | exact reproduction of job `51518820` | 1 / 144 temporal and 1 / 144 multi-view accepts; zero mismatches | PASS |
| 3 | narrow clamp-consistent diagnostic | same frozen scores and decision parameters; no false accept; prior usefulness reference 52 accepts across at least three sources | 38 temporal; 31 multi-view; zero false accepts; four sources, but 31 < 52 | EXPLORATORY_IMPROVEMENT_BELOW_GATE |
| 4 | freshness and acquisition structure | producer advance and distinct physical observations | 766 stale zero-action duplicates; zero fresh same-pose acquisitions; 144 reset-separated segments | ACTIVE_ACQUISITION_MISSING |

The mismatch is the largest demonstrated cause of the original `1 / 144` collapse. It does not
eliminate the remaining classifier-support, duplicate-frame, short-trajectory, and spatial-view
limitations. `EXPLORATORY_NO_GO`, V16--V18 `CONFIRMATION_NO_GO`, and
`FULL_STAGE2_CLOSED_LOOP = NOT_RUN` remain unchanged.

## 1. Movement contract — did collection and replay accept the same actions?

`collect_stage2_dataset.py`, `stage2_sim.py`, `coordinates.py`, and
`stage2_temporal_decision.py`, audited by `audit_stage2_contract.py`. The audit compares the
frozen collector record with the exact replay predicate and verifies low-level action provenance.

**Criterion.** The contracts are consistent only if every collector-accepted frame is either
accepted by replay or rejected for a safety condition also present in the frozen acquisition
protocol. Position clamps, rejected/incomplete actions, invalid mapping, unsafe contact/workspace,
nonfinite motion, termination, and truncation remain disallowed.

### 1.0 Quantities

| object | registry | use |
|---|---|---|
| temporal support | [Stage 2 temporal evidence](../notation.md#stage-2-temporal-evidence) | candidate support without physical-view separation |
| multi-view support | [Stage 2 multi-view evidence](../notation.md#stage-2-multi-view-evidence) | candidate support with executed-pose separation |
| clamp diagnostic | [Clamp-consistent diagnostic](../notation.md#clamp-consistent-diagnostic) | narrow opened-data counterfactual |

### 1.1 Method

The collector defines frame transition validity as:

```python
all(finite and surface_proxy_ok and command_mapping_valid
    and not terminated and not truncated for action in trajectory)
and all(not command_position_clamped for action in trajectory)
```

The frozen replay instead defines it as:

```python
action_accepted and action_completed and not action_clamped
and contact_ok and workspace_ok
```

- `action_clamped` combines angular and positional clamp states.
- All four source protocols explicitly enable bounded angle-ceiling capture.
- The diagnostic admits an angle-only frame only when the source protocol enables that exception,
  the public record is accepted/completed, the position clamp is false, and every action is finite,
  mapping-valid, surface-valid, nonterminal, nontruncated, and position-unclamped.
- Evaluator labels are joined only after actor selection.
- The deployed replay predicate remains unchanged.

### 1.2 Setup

| what | value | spec |
|---|---|---|
| job / commit | `51542236`; `5d34103e9f03fbc132a116381a3354a991bd1a38` | CPU audit v1 |
| audit protocol | SHA-256 `2487f0759609c60b0c2ad45cd1cc3862e68909e59ed535297fe9775fef9a0efb` | sealed before replay |
| frozen scores | job `51518820`; SHA-256 `0dcd15ebc79e4ccd7cba32f1561aaa114e40617c50f7a61beae4ad456b62a346` | no model execution or fitting |
| frozen trajectory decisions | SHA-256 `f467162d51a3f4da11eae11c32265c7c5c16e93689204c4a8276bb882b844e32` | original reproduction oracle |
| data | V14 `51377943`; V16 `51387027`; V17 `51400282`; V18 `51402759` | 18 sessions; 144 trajectories; 1,350 frames |
| environment | Python 3.10.22; NumPy 1.26.4; SciPy 1.15.2; scikit-learn 1.5.2; joblib 1.4.2 | pinned CPU environment |
| resources | `anest-dept`; `hpg-default`; 4 CPUs; 16 GB; 0 GPUs | 30-minute bound |

```bash
sbatch --parsable --account=anest-dept --qos=anest-dept --partition=hpg-default \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:30:00 \
  --job-name=sg-s2-contract \
  --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/sg-s2-contract-%j.out \
  --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env \
  hpg/jobs/two_stage_stage2_contract_audit.sbatch
```

| | value |
|---|---:|
| attempted frames | 1,350 |
| dropped frames | 0 |
| attempted trajectories | 144 |
| dropped trajectories | 0 |
| **final** | **1,350 frames / 144 trajectories** |

### 1.3 Results

| clamp category, fixed 1,350-frame denominator | frames | distinct source/hash identities |
|---|---:|---:|
| no clamp | 1,035 / 1,350 (76.66666666666667%) | 443 |
| angle only | 315 / 1,350 (23.333333333333332%) | 315 |
| position only | 0 / 1,350 (0%) | 0 |
| angle and position | 0 / 1,350 (0%) | 0 |
| rejected or incomplete | 0 / 1,350 (0%) | 0 |

All 315 angle-only frames were collector-accepted and rejected by the original replay solely
because `action_clamped` was true. Their complete action logs passed the diagnostic's finite,
mapping, surface, terminal, and position-clamp checks. Only 2 / 315 carry the final
`ceiling_limited_capture` fallback marker; the other 313 reached the collector's acceptance state
after angle clamp occurred during coupled control.

| source | frames | accepted angle-only frames rejected by replay |
|---|---:|---:|
| V14 | 450 | 108 |
| V16 | 300 | 69 |
| V17 | 300 | 67 |
| V18 | 300 | 71 |

**Verdict: MISMATCH_CONFIRMED**, because the replay applies a stronger combined-clamp exclusion
than the frozen collection protocols and rejects 315 otherwise accepted source frames.

### 1.4 Reading it

- The mismatch is specific: collector acceptance excludes positional clamp but not every angular
  clamp; replay excludes both through the combined flag.
- This does not justify globally ignoring angular clamps in a future controller.
- A future contract must declare the ceiling exception and retain the per-action safety checks.
- No position-clamped, rejected, incomplete, nonfinite, mapping-invalid, contact-invalid, or
  workspace-invalid frame was rescued.

### 1.5 Files

| file | contents |
|---|---|
| [audit helper](../../source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/stage2_contract_diagnostic.py) | original and narrow diagnostic movement predicates |
| [audit workflow](../../workflows/two_stage_navigation/audit_stage2_contract.py) | identity verification, waterfall, counterfactual, and exports |
| external `frame_clamp_audit.csv` | one row per frame with clamp and freshness diagnostics |

## 2. Abstention waterfall — which rule eliminated each trajectory?

`audit_stage2_contract.py`, run by the command in §1.2. It records per-trajectory candidate counts
and maximum support after movement validity, neighborhood, quality, hash, and viewpoint filters.

**Criterion.** The original arm must reproduce job `51518820` exactly. The diagnostic must retain
the frozen threshold `0.5666666666666667`, 4.5 mm / 6 degree neighborhood, 0.08 quality drop,
three hashes, and 1 mm or 1 degree viewpoint separation.

### 2.1 Method

- Every waterfall row retains the fixed 144-trajectory denominator.
- The sequential table applies conditions in the sealed order until the temporal/multi-view branch.
- Nonexclusive counts independently mark every violated condition; one trajectory may appear in
  multiple failure categories.
- The actor chooses without labels. Image-validity labels are read after selection to count false
  accepts.

### 2.2 Setup

| arm | changed field | unchanged fields |
|---|---|---|
| original frozen | none | all job `51518820` behavior |
| clamp-consistent diagnostic | only movement admissibility for verified angle-only source frames | images, scores, threshold, order, freshness, neighborhoods, quality, hashes, viewpoints, labels |

```bash
/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/envs/sonogym/bin/python \
  workflows/two_stage_navigation/audit_stage2_contract.py \
  --protocol experiments/unity_two_stage_transfer/stage2_contract_audit_protocol_v1.yaml \
  --output-dir /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_contract_audit_v1/51542236/results
```

Nothing was dropped: 144 / 144 trajectories entered both arms.

### 2.3 Results

| sequential condition, fixed denominator 144 | original remaining | diagnostic remaining |
|---|---:|---:|
| all trajectories | 144 | 144 |
| at least one collector-accepted observation | 144 | 144 |
| at least one classifier-accepted collector observation | 118 | 118 |
| at least two raw distinct hashes | 117 | 117 |
| at least three raw distinct hashes | 87 | 87 |
| at least three producer-advanced observations | 87 | 87 |
| at least three movement-valid candidates | 6 | 46 |
| three observations in one candidate neighborhood | 2 | 40 |
| three quality-consistent observations | 1 | 38 |
| three distinct qualifying hashes / temporal accept | 1 | 38 |
| three physically separated qualifying viewpoints / multi-view accept | 1 | 31 |

The machine report serializes the multi-view waterfall and therefore places its displayed
`final_temporal_accept` row after the viewpoint filter. The correct temporal branch counts are the
separately exported decision metrics: 1 original and 38 diagnostic.

| nonexclusive failure, fixed denominator 144 | original | diagnostic |
|---|---:|---:|
| no collector classifier candidate | 26 | 26 |
| fewer than three raw hashes | 34 | 34 |
| fewer than three producer-advanced observations | 34 | 34 |
| fewer than three movement-valid candidates | 138 | 98 |
| no three-observation neighborhood | 142 | 104 |
| no three quality-consistent observations | 143 | 106 |
| no three distinct qualifying hashes | 143 | 106 |
| no three separated viewpoints | 143 | 113 |

| source | original multi-view accept | diagnostic temporal accept | diagnostic multi-view accept | diagnostic false accepts |
|---|---:|---:|---:|---:|
| V14 | 1 / 48 | 14 / 48 | 12 / 48 | 0 |
| V16 | 0 / 32 | 6 / 32 | 5 / 32 | 0 |
| V17 | 0 / 32 | 9 / 32 | 6 / 32 | 0 |
| V18 | 0 / 32 | 9 / 32 | 8 / 32 | 0 |
| **total** | **1 / 144** | **38 / 144** | **31 / 144** | **0** |

The correction increases multi-view coverage by 30 / 144 trajectories, from
0.6944444444444444% to 21.52777777777778%. It accepts in all four sources, but remains below the
predeclared prior-D3 reference of 52 accepts.

**Verdict: EXPLORATORY_IMPROVEMENT_BELOW_GATE**, because the mismatch has a large measured effect
but 31 < 52 required accepts. The frozen `EXPLORATORY_NO_GO` is not changed.

### 2.4 Reading it

- The largest original sequential loss is movement admissibility: 87 to 6 trajectories.
- The narrow correction restores 40 trajectories at that stage and 30 final multi-view accepts.
- 107 / 144 trajectories contain potentially useful qualifying evidence rejected solely because
  of the combined clamp predicate, although not all have three complete supporting views.
- After correction, 98 / 144 still have fewer than three movement-valid classifier candidates.
- Zero observed false accepts on opened data at 31 accepts is exploratory, not confirmation.
- `s2v18cf_01` is not movement-distinctive: it has 17 angle-clamped and 43 stale frames versus
  18 angle-clamped and 43--44 stale frames in the other V18 sessions. It yields 3 / 8 diagnostic
  multi-view accepts, compared with 2 / 8, 1 / 8, and 2 / 8 elsewhere.

### 2.5 Files

| file | contents |
|---|---|
| external `trajectory_contract_diagnostics.jsonl` | candidate and maximum-support counts for every trajectory and arm |
| external `abstention_waterfall.png` | fixed-denominator original/diagnostic multi-view waterfall |
| [result manifest](../../experiments/unity_two_stage_transfer/stage2_contract_audit_v1_result.json) | jobs, counts, interpretation, boundaries, and artifact hashes |

## 3. Freshness and acquisition structure — can the captures emulate Unity active search?

`collect_stage2_dataset.py` and Unity commit
`f7e182b5e2ac71cc73ac41f8b2a928ce10e220c0` are audited for acquisition ordering and explicit
fresh-frame behavior.

**Criterion.** A recorded observation can provide new temporal evidence only if producer identity
advances. Unity-equivalent evidence additionally requires continuous active search, remembered
candidate return, and post-return fresh capture in one physical episode.

### 3.1 Method

- Each frame is compared with its previous frame inside the same trajectory.
- Zero action, zero executed movement, producer advance, and exact-hash change are measured
  separately.
- Evidence is never joined across trajectory reset boundaries.
- Unity source is inspected at the verified commit; its geometry and thresholds are not copied.

### 3.2 Setup

| source | acquisition structure |
|---|---|
| SonoGym V14/V16--V18 | 18 sessions; eight segments/session; seven 10-frame segments plus one 5-frame segment |
| Unity verified demo | one continuous Phase 2 episode; 237 observations; coarse, fine, micro, return-best, two fresh confirmations |

The frame and trajectory population is the same 1,350 / 144 used in §1.

### 3.3 Results

| freshness diagnostic | frames |
|---|---:|
| producer sequence unchanged | 766 / 1,350 (56.74074074074074%) |
| unchanged producer and zero actions | 766 / 1,350 (56.74074074074074%) |
| unchanged producer and zero executed movement | 766 / 1,350 (56.74074074074074%) |
| unchanged producer and identical previous hash | 766 / 1,350 (56.74074074074074%) |
| unchanged producer with nonzero movement | 0 / 1,350 (0%) |
| producer-advanced same-pose acquisition | 0 / 1,350 (0%) |
| producer-advanced same-pose acquisition with a distinct hash | 0 / 1,350 (0%) |

The SonoGym collector reads `clean_ultrasound` after waypoint handling. When no action is needed,
it does not explicitly request an additional producer frame; all 766 such records remain the same
producer sequence and hash. The capture set contains 126 ten-frame trajectories and 18 five-frame
trajectories, with a simulator reset between adjacent segments.

Unity instead requests a producer-fresh frame after each successful physical target movement,
remembers the highest-quality valid candidate, returns physically, and requests two additional
fresh frames before independent evaluation.

**Verdict: ACTIVE_ACQUISITION_MISSING**, because no existing SonoGym record demonstrates fresh
same-pose reacquisition, return-to-best, or a continuous Unity-style coarse/fine/micro episode.

### 3.4 Reading it

- The 766 stale frames are duplicate records, not independent observations.
- No stale frame has nonzero executed motion; potentially moved-but-unacquired viewpoints were not
  silently discarded by the freshness rule.
- Reset separation caps evidence at ten or five frames and forbids accumulating support across
  neighboring segments.
- The current capture can test local causal replay but cannot test reacquisition or confirmation.
- SonoGym's short grid segments and Unity's 237-observation active episode are not equivalent.

### 3.5 Files

| file | contents |
|---|---|
| [SonoGym collector](../../workflows/two_stage_navigation/collect_stage2_dataset.py) | segmented waypoint acquisition and public record contract |
| Unity `ThoracicUSCoachController.Phase2BestView.cs` | continuous search, return, fresh capture, and independent evaluation at verified commit |
| external `frame_clamp_audit.csv` | zero-action, producer, movement, and hash comparison per frame |

## 4. Root-cause decision — what must change before GPU work?

The evidence from §§1--3 is classified and mapped to one required next action. No additional
experiment is run in this step.

**Criterion.** Choose exactly one requested outcome. A confirmed mismatch with a material measured
effect requires Option 1 before any active-acquisition GPU pilot.

### 4.1 Method

- Implementation/contract findings require source predicates and measured counterfactual effect.
- Data limitations require missing or reset-separated recorded evidence.
- Model weaknesses require frozen prediction evidence and are not inferred from missing movement.
- Claims without direct evidence remain hypotheses.

### 4.2 Setup

| evidence set | status |
|---|---|
| historical V16--V18 | opened; immutable `CONFIRMATION_NO_GO` |
| temporal replay | opened; immutable `EXPLORATORY_NO_GO` |
| contract counterfactual | opened exploratory diagnostic |
| fresh physical confirmation | `NOT_RUN` |

No samples were added or removed.

### 4.3 Results

| classification | measured finding |
|---|---|
| **Demonstrated implementation/contract issue** | replay rejected 315 collector-accepted angle-only frames; fixing only this predicate changed multi-view accepts from 1 to 31 |
| **Demonstrated acquisition-data limitation** | 766 stale zero-action duplicates; 34 trajectories had fewer than three raw hashes/producer-advanced observations; zero return/fresh acquisitions; reset-separated 10/5-frame segments |
| **Demonstrated model weakness** | 26 / 144 trajectories had no classifier-accepted collector observation; historical V16--V18 threshold behavior remained session-unstable |
| **Unverified hypothesis** | a continuous active search will raise coverage above the usefulness gate or preserve zero false accepts on new data |

The sealed machine rule reused the prior D3 requirement and therefore labels 31 < 52 as not
materially restored. That is retained as the scientific coverage result. For root-cause ordering,
the 30-trajectory increase and 20.833333333333336 percentage-point coverage change are a material
contract effect; carrying the known mismatch into a GPU pilot would invalidate its movement
semantics.

**Verdict: OPTION 1 — CONTRACT CORRECTION FIRST.** The next approved change should replace the
blanket combined-clamp rejection with the narrow, protocol-declared angle-ceiling rule, retain all
position/safety failures, and rerun the opened CPU replay. No GPU work should precede that review.

### 4.4 Reading it

- Contract correction is necessary but not sufficient: diagnostic multi-view coverage is only
  31 / 144.
- After a reviewed CPU correction, the next likely step is a bounded active-acquisition pilot;
  this audit does not authorize it.
- No new threshold, model, or evidence count should be selected on the opened sets.
- A future independent protocol must use new physical observations and preserve evaluator separation.

### 4.5 Files

| file | contents |
|---|---|
| [audit protocol](../../experiments/unity_two_stage_transfer/stage2_contract_audit_protocol_v1.yaml) | frozen arms, waterfall, interpretation reference, and resources |
| [focused tests](../../tests/test_stage2_contract_diagnostic.py) | narrow exception, safety, freshness, reset, leakage, and frozen-behavior checks |
| [result manifest](../../experiments/unity_two_stage_transfer/stage2_contract_audit_v1_result.json) | final audit decision and exact evidence |
| [result digest](../../experiments/unity_two_stage_transfer/stage2_contract_audit_v1_result.sha256) | result-manifest SHA-256 |

Job `51542236` completed `0:0` in 12 seconds. The audit used 2.82 seconds and 108,544 KiB peak
RSS. Focused tests passed 7 / 7, temporal tests passed 19 / 19, and existing Stage 2 regression
tests passed 34 / 34. The 1.1 MiB artifact tree passed all three stored SHA-256 checks.
