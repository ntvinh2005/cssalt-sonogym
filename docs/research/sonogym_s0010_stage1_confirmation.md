# SonoGym s0010 Phase 1 confirmation

This report evaluates the frozen L4 [Phase 1 confirmation](../notation.md#phase-1-confirmation) after the session-separated development result. It covers physical SonoGym scans, actor/evaluator separation, and the auditable [Phase 1 demo](../notation.md#phase-1-demo); it does not cover Phase 2 or patient generalization.

## Result

**The independent confirmation failed its frozen 4 / 4 criterion, while one preregistered successful session provides a valid engineering demo of scan, image-evidence aggregation, and L4 inference.**

| step | what was measured | criterion | measured | verdict |
|---|---|---|---|---|
| 1 | confirmation v1 | 4 / 4 correct; zero wrong stops and abstentions; median localization error and direction difference at most 10 mm | 3 / 4 correct; one wrong stop; median error 7.862 mm; direction difference 3.25 mm | FAIL |
| 2 | confirmation v2 | same gate after a development-selected coordinate rule; four complete sessions | 2 / 4 correct; zero wrong stops; two abstentions; median error among correct sessions 4.862 mm | FAIL |
| 3 | selected preregistered session visualization | sealed inputs; 211 real frames; valid MP4; actor result within 10 mm | `cf_00`; 211 frames; 14.067 s; 7.862 mm error | PASS, artifact only |

The evidence establishes that actor-visible ultrasound and executed relative motion can produce correct L4 estimates on some new starts. It does not establish a reliable confirmed Phase 1 method because neither frozen four-session gate passed.

## 1. Confirmation v1 — does the development-selected method pass on four new physical scans?

`workflows/two_stage_navigation/run_stage1_confirmation.py`, run by `hpg/jobs/two_stage_stage1_confirmation.sbatch`. It writes and seals actor output before opening evaluator-only L4 reference data.

**Criterion.** Four complete scans must all localize within 10 mm, with zero wrong stops, zero abstentions, median [L4 localization error](../notation.md#l4-localization-error) at most 10 mm, and median forward/reverse peak difference at most 10 mm. The thresholds were frozen in `stage1_confirmation_evaluation_protocol.yaml` before job `51196094`.

### 1.1 Method

- One physical forward and reverse scan is acquired for each new start.
- Clean ultrasound patches are scored by the frozen development checkpoint.
- Image evidence is projected through executed Home-relative motion onto the frozen 0.5 mm grid.
- The actor result is hashed before the evaluator reads raw label 28 and its reference coordinate.

### 1.2 Setup

| what | value | spec |
|---|---|---|
| job / commit | `51196094`; `2bb98417192e26cb3c331da3c2286f2c917d263e` | §1 |
| checkpoint | `stage1_l4_development.joblib`; SHA-256 `bc6df0a1d9e520ac8340789fa751f35e63e0cdcfe4072c130bd70749f22ceb79` | §1 |
| actor protocol | SHA-256 `965e5cfefd09eace010841f9f720485e48967484ee8df508af48aebfaa5bcce8` | §1 |
| evaluation protocol | SHA-256 `b7fdb067a2b90ac2ae3bcfb5e62c78a62c046b593cd2ee3b9bee6ef7eeafa6cd` | §1 |
| resources | `anest-dept`; one NVIDIA L4; four CPUs; 30 GB; 15 min bound | §1 |

```bash
sbatch --account=anest-dept --qos=anest-dept --partition=hpg-turin \
  --constraint=l4 --gres=gpu:l4:1 --cpus-per-task=4 --mem=30gb \
  --time=00:15:00 --job-name=sg-stage1-confirm \
  --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env \
  hpg/jobs/two_stage_stage1_confirmation.sbatch
```

| | value |
|---|---:|
| attempted sessions | 4 |
| dropped sessions | 0 |
| **final sessions** | **4** |

### 1.3 Results

| session | actor coordinate (mm) | evaluator-only error (mm) | result |
|---|---:|---:|---|
| `cf_00` | -75.5 | 7.862 | correct |
| `cf_01` | -74.5 | 8.862 | correct |
| `cf_02` | -77.5 | 5.862 | correct |
| `cf_03` | -73.0 | 10.362 | wrong stop |

The job used 4 CPUs and one NVIDIA L4 under `anest-dept`, ran for 149 seconds, sampled at most 5,009 MiB GPU memory and 54% GPU utilization, and produced 102,103,959 bytes.

**Verdict: FAIL**, because 3 / 4 sessions were correct and `cf_03` exceeded the 10 mm tolerance by 0.362 mm.

### 1.4 Reading it

- All four scans and actor/evaluator stages completed.
- The failure is scientific, not a simulator or scheduler failure.
- Three new starts support the bounded s0010 capability.
- One wrong stop prevents confirmation and deployment readiness.

### 1.5 Files

| file | contents |
|---|---|
| [v1 result manifest](../../experiments/unity_two_stage_transfer/stage1_confirmation_v1_result.json) | concise job, resource, session, and artifact evidence |
| external run root | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage1_confirmation/51196094/` |
| external actor result | `51196094/results/actor_results.json`; SHA-256 `9d3532f06f15443d7a0d3ac7c32d67430c93de45f1fe8d61bbdbe8ac7c58c13e` |
| external evaluation result | `51196094/results/evaluation_report.json`; SHA-256 `857c8c35e1f4cedf6b4686a79c09fe4048759c3af0ede6c0fb3671fcc656015d` |

## 2. Confirmation v2 — does the development-selected interval midpoint resolve the v1 miss?

`workflows/two_stage_navigation/run_stage1_confirmation.py`, run by `hpg/jobs/two_stage_stage1_confirmation.sbatch`. It keeps the checkpoint, 0.50 acceptance threshold, scan geometry, and 10 mm evaluator tolerance fixed while reporting the accepted event interval midpoint.

**Criterion.** The same v1 confirmation criterion applies. The coordinate rule was selected only on `cd_00`--`cd_03`; v1 confirmation sessions were excluded from selection.

### 2.1 Method

- Four different starts replace all v1 starts.
- The accepted event support-interval midpoint replaces the highest-scoring grid coordinate.
- Job `51197194` exposed a collector defect: Home acceptance used a hard-coded 3 degree limit instead of the protocol's 5 degree limit.
- Commit `1f9302036e5a0df9f44040d2bcde97e56e5a5c65` made Home acceptance use the protocol values; job `51197643` is the evaluated retry.

### 2.2 Setup

| what | value | spec |
|---|---|---|
| engineering-failure job | `51197194`; exit `2:0` before actor inference | §2 |
| evaluated job / commit | `51197643`; `1f9302036e5a0df9f44040d2bcde97e56e5a5c65` | §2 |
| actor protocol | SHA-256 `28d531dd664160c1f1a44eef7a578dc0d74a10f746d8e22a0c66135f9650e380` | §2 |
| evaluation protocol | SHA-256 `644a1a4abaee079b215c80eeb2859eabea2b08fb1d503f2f2c1a9c0d2e6a61a8` | §2 |
| resources | `anest-dept`; one NVIDIA L4; four CPUs; 30 GB; 15 min bound | §2 |

```bash
sbatch --account=anest-dept --qos=anest-dept --partition=hpg-turin \
  --constraint=l4 --gres=gpu:l4:1 --cpus-per-task=4 --mem=30gb \
  --time=00:15:00 --job-name=sg-stage1-confirm-v2r \
  --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env \
  hpg/jobs/two_stage_stage1_confirmation.sbatch
```

| | value |
|---|---:|
| attempted v2 jobs | 2 |
| dropped before evaluation, collector defect | 1 |
| **evaluated jobs** | **1** |
| attempted sessions in evaluated job | 4 |
| dropped sessions in evaluated job | 0 |
| **final sessions** | **4** |

### 2.3 Results

| session | actor status | actor coordinate (mm) | evaluator-only error (mm) | result |
|---|---|---:|---:|---|
| `cf2_00` | abstain | n/a | n/a | abstention |
| `cf2_01` | success | -75.0 | 8.362 | correct |
| `cf2_02` | success | -82.0 | 1.362 | correct |
| `cf2_03` | abstain | n/a | n/a | abstention |

Job `51197643` completed all capture, actor, and rendering stages; evaluator exit 3 correctly made Slurm report `FAILED`. It ran for 131 seconds, sampled at most 5,009 MiB GPU memory and 57% utilization, and produced 102,050,660 bytes.

**Verdict: FAIL**, because only 2 / 4 sessions emitted correct coordinates and the frozen criterion permits zero abstentions.

### 2.4 Reading it

- The Home-tolerance defect was fixed without changing the scientific protocol.
- Both abstentions were caused by insufficient forward support at the selected image-evidence event.
- Post-hoc broader support counting was inspected but not retained because it still produced an error above 10 mm on one session.
- No threshold, tolerance, target reference, or checkpoint was changed after evaluation.

### 2.5 Files

| file | contents |
|---|---|
| [v2 result manifest](../../experiments/unity_two_stage_transfer/stage1_confirmation_v2_result.json) | evaluated v2 job and per-session outcomes |
| [v2 capture-failure manifest](../../experiments/unity_two_stage_transfer/stage1_confirmation_v2_capture_failure.json) | preserved `51197194` engineering failure |
| external evaluated run | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage1_confirmation/51197643/` |

## 3. Phase 1 demo — can a reviewer inspect a real successful scan and frozen inference?

`workflows/two_stage_navigation/render_stage1_confirmation_demo.py`, run inside job `51196094`. It renders the preregistered `cf_00` session from sealed actor and evaluator results without altering inference.

**Criterion.** The MP4 must contain all 211 captured ultrasound frames, use the frozen 15 frames/s rate, decode successfully, display actor and evaluator channels separately, and correspond to an actor result within 10 mm.

### 3.1 Method

- The left panel is the original clean SonoGym ultrasound observation supplied to the actor.
- The right panel accumulates image-derived evidence in Home-relative millimetres.
- The green marker is the frozen actor output.
- The red marker is evaluator-only and was added after the actor result was sealed.

### 3.2 Setup

| what | value | spec |
|---|---|---|
| source job / session | `51196094` / `cf_00` | §3 |
| actor result | -75.5 mm | §3 |
| evaluator-only reference | -83.36209568080575 mm | §3 |
| video | 1,050 x 600; 15 frames/s; 211 frames | §3 |

```bash
/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/envs/sonogym/bin/python \
  workflows/two_stage_navigation/render_stage1_confirmation_demo.py \
  --actor-protocol experiments/unity_two_stage_transfer/stage1_confirmation_actor_protocol.yaml \
  --capture-root /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage1_confirmation/51196094 \
  --actor-result /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage1_confirmation/51196094/results/actor_results.json \
  --evaluation-result /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage1_confirmation/51196094/results/evaluation_report.json \
  --output-video /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage1_confirmation/51196094/video/sonogym-phase1-confirmation.mp4 \
  --output-preview /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage1_confirmation/51196094/video/sonogym-phase1-confirmation-preview.png
```

Nothing was dropped: all 211 frames are present in the MP4.

### 3.3 Results

| measure | result |
|---|---:|
| actor localization error | 7.862095680805751 mm |
| frame count | 211 |
| duration | 14.066666666666666 s |
| file size | 3,852,895 bytes |
| MP4 SHA-256 | `d411bd364a2e955d2bc48ad60678c7d7997be8a614ce8260523d346318956b34` |

**Verdict: PASS, artifact only**, because the selected preregistered session is a correct frozen inference and the complete video decodes; the four-session confirmation remains FAIL (§1).

### 3.4 Reading it

- The demo is suitable for inspecting the implemented Phase 1 data flow.
- It is not a cherry-picked scientific replacement for the four-session result because `cf_00` was selected before execution.
- It does not establish full sacrum/S1/L1--L5 map reconstruction, Phase 2, or anatomy-held-out performance.

### 3.5 Files

| file | contents |
|---|---|
| external MP4 | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage1_confirmation/51196094/video/sonogym-phase1-confirmation.mp4` |
| external preview | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage1_confirmation/51196094/video/sonogym-phase1-confirmation-preview.png` |
| review package | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/reviewer/sonogym_phase1_demo_20261113.zip` |
