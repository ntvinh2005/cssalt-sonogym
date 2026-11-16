# SonoGym Stage 2 clamp-consistent diagnostic CPU replay

This document records the reviewed CPU implementation of the collector-authorized angle-ceiling
exception. It covers opened-data replay only; live acquisition and closed-loop Stage 2 are out of
scope.

## Result

**The explicit contract reproduced 38 / 144 temporal accepts and 31 / 144 multi-view accepts with zero false accepts, but 31 remains below the frozen 52-accept usefulness condition; the result is `EXPLORATORY_NO_GO`.**

| step | what was measured | criterion | measured | verdict |
|---|---|---|---|---|
| 1 | contract implementation | original replay unchanged; narrow exception matches the sealed contract | zero original mismatches; 64 / 64 tests pass | PASS |
| 2 | opened CPU replay | reproduce 38 temporal and 31 multi-view accepts; multi-view usefulness requires at least 52 accepts | 38 temporal; 31 multi-view; zero false accepts | REPRODUCED; NO_GO |

The implementation defect found by the prior audit is corrected. Existing reset-separated
captures still do not provide enough accepted multi-view trajectories for a live Stage 2 claim.

## 1. Contract implementation — does the narrow correction preserve frozen behavior?

`stage2_temporal_decision.py:transition_valid_for_contract`, run by the CPU test suites. The
function keeps the original contract as its default and requires complete low-level feedback for
the angle-ceiling exception.

**Criterion.** The frozen original decisions must have zero mismatches. Angle-only movement may
pass only under a source protocol that explicitly enables it; position clamps and all unsafe,
incomplete, nonfinite, mapping-invalid, terminal, or truncated movement must fail.

### 1.0 Quantities

| symbol | definition | source |
|---|---|---|
| contract arm | [Clamp-consistent diagnostic](../notation.md#clamp-consistent-diagnostic) | project notation |
| evidence | [Stage 2 temporal evidence](../notation.md#stage-2-temporal-evidence) and [Stage 2 multi-view evidence](../notation.md#stage-2-multi-view-evidence) | project notation |

### 1.1 Method

- Preserve `original_frozen` as the default reconstruction contract.
- Admit angle-only clamps only when the frozen source protocol declares the exception.
- Require accepted/completed public feedback, finite executed movement, exact action-step count,
  valid surface and mapping, no terminal/truncation, and no position clamp.
- Pass only the required action-feedback fields into replay; evaluator labels are joined after the
  decision.

### 1.2 Setup

| what | value | spec |
|---|---|---|
| implementation commit | `6e18eb0738dfb46ec631021047bc86e4f2997c11` | core contract |
| execution commit | `72deeb82c26b4324af5d5bef8e52b9acd00da99c` | job `51550463` |
| protocol | SHA-256 `6bf15d9bee4b13dc40fe0801792afdb72220c5a5a003e02772e49762b463452f` | sealed before execution |
| environment | Python 3.10.22; NumPy 1.26.4; SciPy 1.15.2; scikit-learn 1.5.2; joblib 1.4.2 | pinned CPU environment |

```bash
python tests/test_stage2_temporal_decision.py
python tests/test_stage2_contract_diagnostic.py
python tests/test_two_stage_stage2.py
```

| | value |
|---|---:|
| attempted tests | 64 |
| dropped tests | 0 |
| **final** | **64** |

### 1.3 Results

| suite | passed | failed |
|---|---:|---:|
| temporal decision | 23 | 0 |
| contract diagnostic | 7 | 0 |
| existing Stage 2 regression | 34 | 0 |

The original frozen replay had zero outcome or selected-frame mismatches.

**Verdict: PASS**, because all 64 tests passed and original behavior was reproduced exactly.

### 1.4 Reading it

- The correction is explicit; callers cannot activate it by setting only `action_clamped=false`.
- A source without the frozen angle-ceiling declaration remains fail-closed.
- This engineering PASS does not establish live acquisition or Stage 2 task completion.

### 1.5 Files

| file | contents |
|---|---|
| `stage2_temporal_decision.py` | selected movement-contract validation and replay reconstruction |
| `test_stage2_temporal_decision.py` | default preservation, narrow exception, safety, and leakage tests |
| `test_stage2_contract_diagnostic.py` | prior diagnostic and evidence-boundary regression tests |

## 2. Opened CPU replay — does the correction restore useful multi-view coverage?

`replay_stage2_contract_corrected.py`, run by Slurm job `51550463`. It consumes frozen scores and
capture/action records, makes actor decisions, then joins evaluator labels.

**Criterion.** Reproduce the audit's 38 temporal and 31 multi-view accepts with zero false accepts.
The unchanged exploratory usefulness condition requires at least 52 multi-view accepts and
acceptance in at least three capture sources.

### 2.1 Method

- Use the unchanged V18 probability threshold, neighborhood, quality-drop, hash-count, and
  physical-separation values.
- Use no model fitting, threshold search, or new observation.
- Compare selected candidates with evaluator labels only after replay selection is fixed.

### 2.2 Setup

| what | value | spec |
|---|---|---|
| successful run | `51550463`; `anest-dept`; `hpg-default` | 4 CPUs; 16 GB; 0 GPUs |
| inputs | V14 `51377943`; V16 `51387027`; V17 `51400282`; V18 `51402759` | 1,350 frames; 144 trajectories |
| frozen scores | SHA-256 `0dcd15ebc79e4ccd7cba32f1561aaa114e40617c50f7a61beae4ad456b62a346` | job `51518820` |
| filters | probability 0.5666666666666667; 4.5 mm; 6 degrees; quality drop 0.08; 3 hashes; 1 mm / 1 degree viewpoints | sealed protocol |

```bash
sbatch --parsable --account=anest-dept --qos=anest-dept --partition=hpg-default \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:30:00 \
  --job-name=sg-s2-contract-corrected \
  --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env \
  hpg/jobs/two_stage_stage2_contract_corrected_replay.sbatch
```

| | value |
|---|---:|
| attempted frames | 1,350 |
| dropped frames | 0 |
| attempted trajectories | 144 |
| dropped trajectories | 0 |
| **final** | **1,350 frames; 144 trajectories** |

### 2.3 Results

| actor decision | accepted | abstained | false accepts | coverage |
|---|---:|---:|---:|---:|
| temporal | 38 / 144 (26.38888888888889%) | 106 / 144 | 0 | 26.38888888888889% |
| multi-view | 31 / 144 (21.52777777777778%) | 113 / 144 | 0 | 21.52777777777778% |

| source | multi-view accepts | trajectories |
|---|---:|---:|
| V14 development | 12 | 48 |
| V16 opened confirmation | 5 | 32 |
| V17 opened confirmation | 6 | 32 |
| V18 opened confirmation | 8 | 32 |

Job `51550463` completed `0:0` in 10 seconds. Replay used 1.82 seconds wall time and
60,416 KiB peak RSS. Output size was 220 KiB and both stored artifact checksums passed.

**Verdict: REPRODUCED; EXPLORATORY_NO_GO**, because exact audited counts were reproduced but
31 < 52 multi-view accepts.

### 2.4 Reading it

- Zero false accepts are descriptive on opened fixed-anatomy data, not independent confirmation.
- Acceptance appears in all four sources, but coverage remains 21.52777777777778%.
- The contract correction is necessary and complete; the remaining blocker is acquisition
  structure rather than another movement-validity ambiguity.
- V16, V17, V18, and the original temporal replay remain NO_GO; full closed loop remains NOT_RUN.
- Failed job `51550144` stopped after tests because a source-count assertion used the wrong result
  key. Commit `72deeb8` corrected that implementation-only error; no protocol value changed.

### 2.5 Files

| file | contents |
|---|---|
| [protocol](../../experiments/unity_two_stage_transfer/stage2_contract_corrected_replay_protocol_v1.yaml) | immutable inputs, contract, rules, and expected reproduction |
| [result manifest](../../experiments/unity_two_stage_transfer/stage2_contract_corrected_replay_v1_result.json) | jobs, tests, metrics, limits, and artifact hashes |
| external `replay_report.json` | aggregate and source-level opened-data results |
| external `trajectory_results.jsonl` | one corrected decision per policy and trajectory |

The smallest next experiment requiring approval is one bounded development-only continuous
active-acquisition pilot. It should test whether coarse/fine/micro physical movement can provide
three genuine viewpoints, return to the selected candidate, and acquire fresh confirmation frames;
it must not be called V19 or an independent confirmation run.
