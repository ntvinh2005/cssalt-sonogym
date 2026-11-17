# SonoGym Stage 2 new-start confirmation V1

This record covers four protocol-sealed reset starts of the unchanged fixed-`s0010` Stage 2
bounded orientation refinement. It tests start-state sensitivity; it does not test patient-held-out
generalization or clinical performance.

## Result

**Job `51668777` establishes `CONFIRMATION_NO_GO`: 0 / 3 technically evaluable starts passed the unchanged per-run gate, one start failed closed before actor sealing, and the frozen aggregate required 4 / 4 passes.**

| step | what was measured | criterion | measured | verdict |
|---|---|---|---|---|
| four new reset starts | actor acceptance, physical return, fresh confirmation, image visibility, and final geometry | all 4 / 4 starts independently pass the unchanged V6R1 per-run gate | 0 / 3 evaluable starts passed; 1 / 4 had a technical freshness failure | `CONFIRMATION_NO_GO` |

All three evaluable actors accepted, returned, and captured fresh confirmations. Their position and
image gates passed, but their final orientation errors were 15.25051468069617,
11.880513004590219, and 11.878190740272414 degrees against the unchanged 10-degree gate.

## 1. New-start confirmation — does the development result hold across four unused reset seeds?

`workflows/two_stage_navigation/run_stage2_active_pilot.py`, run sequentially by
`hpg/jobs/two_stage_stage2_new_start_confirmation_v1.sbatch`. Each execution resets the simulator
with one sealed unused seed, moves through the unchanged physical search, freezes the actor result,
then invokes the evaluator.

**Criterion.** The sealed protocol requires all 4 / 4 starts to independently meet actor `ACCEPT`,
at least three supporting viewpoints, two fresh confirmations, return error <=5 mm / <=5 degrees,
final error <=10 mm / <=10 degrees, and L4 visible fraction >=0.20. Any per-run failure makes the
aggregate `CONFIRMATION_NO_GO`.

### 1.0 Quantities

| symbol | definition | source |
|---|---|---|
| Stage 2 new-start confirmation | four sealed unused reset starts with an all-start success rule | [`docs/notation.md`](../notation.md#stage-2-new-start-confirmation) |
| Stage 2 bounded orientation refinement | bounded image-guided angular continuation | [`docs/notation.md`](../notation.md#stage-2-bounded-orientation-refinement) |
| Stage 2 fresh confirmation | post-return fresh-image evidence | [`docs/notation.md`](../notation.md#stage-2-fresh-confirmation) |

### 1.1 Method

- Freeze seeds `2026112200`, `2026112201`, `2026112202`, and `2026112203` before GPU execution.
- Preserve the V6R1 Phase 1 handoff, model bundle, threshold, 3x3 grid, coarse/fine/micro search,
  angular offsets and limits, motion tolerances, confirmation motion, and evaluator gates.
- Record the realized reset pose before moving to the common sealed Phase 1 handoff.
- Seal each actor result before reading evaluator-only geometry and L4 visibility.
- Require 4 / 4 per-run passes; do not tune or retry after a scientific failure.

### 1.2 Setup

| what | value | spec |
|---|---|---|
| run | Slurm `51668777`; execution commit `ae45f65c927115e1fd5a6319ad6669430fa8432b` | §1 |
| protocol | `experiments/unity_two_stage_transfer/stage2_new_start_confirmation_protocol_v1.yaml`; SHA-256 `e3a98e42090df0f02e3e29e1caf25ea8c0b1a3a013fbd233059b1769b774b99c` | §1 |
| model | frozen V15 bundle; SHA-256 `7d3b113ba33b1b80448264a0d2e7f2c2cd5561b26021707648c4eef43d586672` | §1 |
| input handoff | Phase 1 `cf_00`; actor-result SHA-256 `9d3532f06f15443d7a0d3ac7c32d67430c93de45f1fe8d61bbdbe8ac7c58c13e` | §1 |
| task | `Isaac-robot-US-guidance-v0`, `conv`, one environment at a time, patient `s0010` | §1 |
| resources | `anest-dept`; one NVIDIA L4; four CPU cores; 30 GB; 30-minute limit | §1 |
| pre-submit tests | 74 / 74 passed | §1 |

```bash
sbatch --parsable --account=anest-dept --qos=anest-dept --partition=hpg-turin \
  --constraint=l4 --gres=gpu:l4:1 --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=30G --time=00:30:00 --job-name=sg-s2-newstarts-v1 \
  --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/sg-s2-newstarts-v1-%j.out \
  --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env \
  hpg/jobs/two_stage_stage2_new_start_confirmation_v1.sbatch
```

| | value |
|---|---:|
| attempted starts | 4 |
| dropped, fail-closed technical freshness error before actor sealing | 1 |
| **technically evaluable starts** | **3** |

### 1.3 Results

| sealed start | actor | position error (mm) | orientation error (degrees) | L4 visible fraction | per-run verdict |
|---|---|---:|---:|---:|---|
| `confirm_new_00` | no sealed actor; technical freshness failure | not evaluated | not evaluated | not evaluated | technical failure |
| `confirm_new_01` | `ACCEPT`; return and 2 / 2 confirmations passed | 4.8557366151216685 | 15.25051468069617 | 0.2632 | `DEVELOPMENT_NO_GO` |
| `confirm_new_02` | `ACCEPT`; return and 2 / 2 confirmations passed | 3.8447238086796434 | 11.880513004590219 | 0.2617333333333333 | `DEVELOPMENT_NO_GO` |
| `confirm_new_03` | `ACCEPT`; return and 2 / 2 confirmations passed | 3.8584854593679183 | 11.878190740272414 | 0.26153333333333334 | `DEVELOPMENT_NO_GO` |

| resource measurement | job `51668777` |
|---|---:|
| elapsed | 8:25 |
| sampled Slurm MaxRSS | 6,823,376 KiB |
| maximum per-actor `/usr/bin/time` RSS | 6,993,260 KiB |
| peak GPU utilization | 48% |
| peak GPU memory | 7,317 MiB |
| output storage | 38,426,848 bytes |

**Verdict: fail**, because the frozen rule requires 4 / 4 passes, while no technically evaluable
start passed and one start did not produce a sealable actor result.

### 1.4 Reading it

- Four seeds and four realized reset poses were distinct; pairwise reset translations spanned
  18.661551302325087--137.26200560768842 mm.
- Position, image visibility, return, fresh confirmation, and supporting-viewpoint gates passed
  in all three evaluable runs.
- Orientation failed in all three evaluable runs, demonstrating that the earlier 9.99523052950496-
  degree development pass did not reproduce across new starts.
- `confirm_new_00` reached a clipped final angular target only 0.003809 degrees beyond the previous
  target, below the frozen 0.25-degree motion tolerance; the zero-action observation correctly
  failed freshness.
- Fixing that technical edge case cannot change the aggregate verdict because the other three
  starts already failed the scientific gate.
- The scheduler state is `FAILED (2:0)` because one evaluator input was absent; this is distinct
  from the scientifically decisive `CONFIRMATION_NO_GO`.
- V16, V17, and V18 remain `CONFIRMATION_NO_GO`; no V19, retraining, threshold change, or retry ran.

### 1.5 Files

| file | contents |
|---|---|
| `experiments/unity_two_stage_transfer/stage2_new_start_confirmation_v1_result.json` | machine-readable aggregate and per-start evidence |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_new_start_confirmation_v1/51668777/` | raw run, partial technical-failure evidence, three sealed actors/evaluations, videos, logs, and telemetry |
| `experiments/unity_two_stage_transfer/stage2_new_start_confirmation_protocol_v1.yaml` | pre-GPU frozen seeds, actor contract, resources, and 4 / 4 gate |
