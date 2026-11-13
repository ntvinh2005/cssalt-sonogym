# SonoGym s0010 seven-landmark confirmation

This report evaluates whether the frozen [Unity Stage 1](../notation.md#unity-stage-1) adaptation reconstructs sacrum, S1, and L1--L5 on four new physical SonoGym scans. It covers fixed-anatomy confirmation only; patient generalization, [Unity Stage 2](../notation.md#unity-stage-2), and medial branch block validity are out of scope.

## Result

**PASS: the frozen actor reconstructed 28 / 28 (100%) expected landmarks in 4 / 4 (100%) new sessions, with no wrong stop or abstention.**

| step | what was measured | criterion | measured | verdict |
|---|---|---|---|---|
| 1 | seven-landmark reconstruction on new sessions | 4 / 4 complete maps; 28 / 28 correct within 10 mm; zero wrong stops; zero abstentions; median error at most 10 mm | 4 / 4 complete; 28 / 28 correct; zero wrong stops; zero abstentions; median 3.0 mm; p90 6.375000000000001 mm; maximum 10.0 mm | PASS |

The result confirms fixed-`s0010` reconstruction within the tested scan corridor. One L5 estimate reached the inclusive 10 mm limit exactly, so that case has no margin beyond the frozen criterion.

## 1. Frozen reconstruction — does the selected actor recover all seven landmarks on four new scans?

`workflows/two_stage_navigation/run_stage1_group_a_confirmation.py`, run by `hpg/jobs/two_stage_stage1_group_a_confirmation.sbatch`. The actor reads clean ultrasound, executed Home-relative motion, and the frozen model bundle; it writes and hashes its output before the evaluator opens privileged annotations.

**Criterion.** The evaluation protocol requires 4 / 4 complete physical captures, 4 / 4 complete maps, 28 / 28 landmark estimates within the inclusive 10 mm tolerance, zero wrong actor stops, zero abstentions, and median localization error at most 10 mm. Both protocol files and their SHA-256 digests were committed before job `51213110`.

### 1.1 Method

- Each session physically moves from the native reset to the declared Home, then performs one caudal and one cranial scan through the native environment step interface.
- Each complete session contributes 211 fresh 200 x 150 model-based ultrasound frames over a 157.5 mm longitudinal corridor.
- The frozen bundle from development job `51209243` supplies the ExtraTrees models, coordinate offsets, adjacent-landmark separations, and selected thresholds.
- The actor aggregates forward/reverse image evidence in Home-relative millimetres and emits one anatomically ordered coordinate for sacrum, S1, L5, L4, L3, L2, and L1.
- Raw labels, segmentation, reference coordinates, target distance, reward, and evaluator score are forbidden actor inputs.
- The evaluator reads privileged Group A annotations only after `actor_results.json` is written and hashed.

### 1.2 Setup

| what | value | spec |
|---|---|---|
| job / code | `51213110`; `2a4b31e17455593a6fec79e97afd87e9d34b0aa1` | clean recorded worktree |
| actor protocol | `sonogym-stage1-group-a-confirmation-v1` | SHA-256 `d3723417bc56f39c7d99ffb39ac3df5d2541e2126f90fa728cd4cf0c787aa4f2` |
| evaluation protocol | `sonogym-stage1-group-a-confirmation-evaluation-v1` | SHA-256 `f4619d12c2c246af9a5a0d27ffa982f123682c835ef5cc8532ad0806c204791c` |
| model bundle | development job `51209243` | SHA-256 `474c02174b7d8cd47ac448e2fab1d50da559514e3fb6aa62c2640f2b7f2933d4` |
| task | `Isaac-robot-US-guidance-v0`; `conv`; one environment | upstream SonoGym `e67be58334d1a5274f0913af36f56e4b0b7ffe5a` |
| software | Isaac Sim 4.5.0.0; Isaac Lab `21f7136325136ca3f6ca4e0a8125edffe5c24f7e`; Python 3.10.22; PyTorch 2.5.1+cu121 | recorded run context |
| resources | `anest-dept`; one NVIDIA L4; four CPUs; 30 GB; 30-minute bound | node `c0605a-s10`; driver 580.178.04 |
| execution | 2026-11-13 15:35:31--15:37:37 EST | Slurm `COMPLETED`, exit `0:0` |

~~~bash
sbatch --parsable \
  --account=anest-dept --qos=anest-dept --partition=hpg-turin \
  --constraint=l4 --gres=gpu:l4:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=30G --time=00:30:00 \
  --job-name=sg-stage1-confirm7 \
  --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/sg-stage1-confirm7-%j.out \
  --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env \
  /blue/lampotan/share/vinhnguyen1/cssalt-sonogym/hpg/jobs/two_stage_stage1_group_a_confirmation.sbatch
~~~

| | value |
|---|---:|
| attempted sessions | 4 |
| dropped sessions | 0 |
| **final sessions** | **4** |
| attempted landmarks | 28 |
| dropped landmarks | 0 |
| **final evaluated landmarks** | **28** |

### 1.3 Results

| session | seed | Home x index | angle (rad) | frames | action steps | correct landmarks | median error (mm) | maximum error (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cf7_00` | 2026111400 | 133.0 | 2.75 | 211 | 297 | 7 / 7 (100%) | 2.75 | 4.25 |
| `cf7_01` | 2026111401 | 137.0 | 2.85 | 211 | 278 | 7 / 7 (100%) | 3.25 | 10.0 |
| `cf7_02` | 2026111402 | 141.0 | 2.95 | 211 | 279 | 7 / 7 (100%) | 2.75 | 7.5 |
| `cf7_03` | 2026111403 | 143.0 | 2.75 | 211 | 259 | 7 / 7 (100%) | 3.5 | 5.75 |

| landmark | successful sessions | median error (mm) | maximum error (mm) |
|---|---:|---:|---:|
| sacrum | 4 / 4 (100%) | 3.75 | 6.0 |
| S1 | 4 / 4 (100%) | 2.5 | 7.25 |
| L5 | 4 / 4 (100%) | 5.75 | 10.0 |
| L4 | 4 / 4 (100%) | 2.0 | 3.0 |
| L3 | 4 / 4 (100%) | 2.0 | 4.75 |
| L2 | 4 / 4 (100%) | 3.75 | 5.75 |
| L1 | 4 / 4 (100%) | 2.0 | 5.75 |

| resource | measured |
|---|---:|
| elapsed | 126 s |
| maximum resident memory | 3,625,692 KiB |
| peak GPU memory | 5,009 MiB / 23,034 MiB |
| peak sampled GPU utilization | 56% |
| run size | 94,251,998 bytes |

**Verdict: PASS**, because all frozen criteria passed: 4 / 4 maps, 28 / 28 landmarks, zero wrong stops, zero abstentions, and 3.0 mm median error. The maximum error was exactly the inclusive 10 mm criterion for L5 in `cf7_01`.

### 1.4 Reading it

- This is stronger evidence than the earlier development result because the four start configurations were frozen and newly captured before the actor ran.
- The actor result SHA-256 is `0d62f3c50007a14f1d3d54758e7d359eee4f3a8827ef10d844aa60348d4f5b`.
- The evaluation result SHA-256 is `6546cfe396cccebee1dd76e988b920db70cb4697a68ebcae8a54ae7bd5199a2a`.
- The evidence supports fixed-anatomy reconstruction within the tested Home and scan corridor.
- The result does not establish patient-held-out generalization, Stage 2 best-view acquisition, or clinical MBB performance.
- These four sessions are now inspected confirmation evidence and cannot be reused as an unseen test set.

### 1.5 Files

| file | contents |
|---|---|
| [result manifest](../../experiments/unity_two_stage_transfer/stage1_group_a_confirmation_result.json) | concise job, protocol, resource, metric, and limitation evidence |
| [result digest](../../experiments/unity_two_stage_transfer/stage1_group_a_confirmation_result.sha256) | result-manifest SHA-256 |
| [actor protocol](../../experiments/unity_two_stage_transfer/stage1_group_a_confirmation_actor_protocol_v1.yaml) | frozen capture and actor inputs |
| [evaluation protocol](../../experiments/unity_two_stage_transfer/stage1_group_a_confirmation_evaluation_protocol_v1.yaml) | frozen privileged evaluation gate |
| [runner](../../workflows/two_stage_navigation/run_stage1_group_a_confirmation.py) | actor and evaluator separation |
| external run root | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage1_group_a_confirmation/51213110` |
| external actor result | `results/actor_results.json` |
| external evaluation result | `results/evaluation_report.json` |
| external telemetry | `metrics/gpu-telemetry.csv`, `metrics/slurm.csv`, and `metrics/time-*.txt` |
