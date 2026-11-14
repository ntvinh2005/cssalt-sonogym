# SonoGym s0010 Stage 2 dataset pilot

This report tests whether the fixed-`s0010` simulator can supply the labeled local-view data required for the supervised classifier and regressor in [Unity Stage 2](../notation.md#unity-stage-2). It covers physical data acquisition only; model fitting and closed-loop search were gated off when capture failed.

## Result

**BLOCKED: the dataset had zero validity-positive frames, and the approved orientation-only retry remained 3.7475 degrees from its command after the frozen 120-step bound.**

| step | what was measured | criterion | measured | verdict |
|---|---|---|---|---|
| 1 | six-session supervised dataset acquisition | 6 / 6 complete sessions; 450 joined frames; positive and negative validity support; all waypoint poses within 5 mm and 5 degrees | 1 / 6 complete; 75 / 450 frames; 0 positives; second session ended at 1.052 mm and 5.497 degrees | BLOCKED |
| 2 | orientation-only physical convergence | command orientation error at most 1 degree, then 10 / 10 frames meeting all pose, image, and L4 criteria | 3.747533552289236 degrees after 120 / 120 steps; 0 / 10 frames acquired | FAIL |

The frozen 10 mm / 10 degree validity definition was not relaxed. Training and closed-loop evaluation were not run because the dataset gate failed.

## 1. Physical acquisition — can the frozen local grid produce trainable positive and negative views?

`workflows/two_stage_navigation/collect_stage2_dataset.py`, run by `hpg/jobs/two_stage_stage2_dataset_v3.sbatch`. It moves through a privileged offline sampling grid, records public clean-ultrasound frames separately from privileged labels, and stops before model fitting when capture is incomplete.

**Criterion.** Protocol v3 requires all six sessions and 450 frames, one-to-one public/privileged joins, physically completed waypoints within 5 mm and 5 degrees, and both validity-positive and negative support. The validity label requires a technically valid image, position error at most 10 mm, orientation error at most 10 degrees, and at least 0.05 L4 raw-label image fraction.

### 1.1 Method

- Four train sessions, one calibration session, and one development-evaluation session were frozen by complete session.
- Each session requested a 5 x 5 x 3 local position/angle grid around the evaluator-only L4 pose for offline data generation.
- Public records contained clean ultrasound, hashes, sequences, executed relative motion, and action feedback.
- Privileged records contained raw label 28, realized and goal poses, geometric errors, and supervised labels.
- Protocols v1--v3 changed only the maximum waypoint-convergence actions: 8, 16, then 24.
- The v3 protocol declared itself the final retry for this failure mode.

### 1.2 Setup

| what | value | spec |
|---|---|---|
| v1 job / protocol | `51231922`; SHA-256 `95596e33c899444737d3320a7e7c4c53e5ca49eb25a882f79f1de0c0660b9486` | 8 actions per waypoint |
| v2 job / protocol | `51245393`; SHA-256 `f420938b3d0d05190ee1b35ead8dceca128739864420993d6ac94fd5952b27c8` | 16 actions per waypoint |
| v3 job / protocol | `51245827`; SHA-256 `8a2a308c4e24b2508606f82dbe3bd156dd54ec07c0d1a4fee7035cbeee9f0d29` | 24 actions per waypoint; final retry |
| v3 code | `330495073c494df1160a20af4c9d3d8d1d74e26f` | clean worktree recorded by job |
| resources per job | `anest-dept`; one NVIDIA L4; four CPUs; 30 GB; 30-minute bound | no concurrent project job |
| software | Isaac Sim 4.5.0.0; driver 580.178.04 | unchanged pinned stack |

~~~bash
sbatch --parsable \
  --account=anest-dept --qos=anest-dept --partition=hpg-turin \
  --constraint=l4 --gres=gpu:l4:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=30G --time=00:30:00 \
  --job-name=sg-stage2-data3 \
  --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/sg-stage2-data3-%j.out \
  --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env \
  hpg/jobs/two_stage_stage2_dataset_v3.sbatch
~~~

| | value |
|---|---:|
| attempted v3 sessions | 2 |
| incomplete after physical convergence failure | 1 |
| **complete v3 sessions** | **1** |
| required frames | 450 |
| **complete frames** | **75** |

### 1.3 Results

| v3 completed-session property | measured |
|---|---:|
| technically valid images | 75 / 75 (100%) |
| distinct hashes | 52 / 75 (69.33333333333333%) |
| position error at most 10 mm | 25 / 75 (33.33333333333333%) |
| orientation error at most 10 degrees | 1 / 75 (1.3333333333333333%) |
| L4 fraction at least 0.05 | 45 / 75 (60%) |
| position and orientation conjunction | 0 / 75 (0%) |
| complete validity positives | 0 / 75 (0%) |
| minimum position error | 1.5780634810904883 mm |
| minimum orientation error | 8.62467111027186 degrees |
| maximum L4 visible fraction | 0.26253333333333334 |

| retry | final waypoint position error | final waypoint orientation error | result |
|---|---:|---:|---|
| v1, 8 actions | 1.917335218795949 mm | 6.571863765790991 degrees | failed physical tolerance |
| v2, 16 actions | 1.5999197991511402 mm | 5.103181848941489 degrees | failed physical tolerance |
| v3, 24 actions | 1.0518428672133602 mm | 5.49662150890932 degrees | failed physical tolerance |

Job `51245827` ran for 38 seconds, sampled 5,003 MiB peak GPU memory and 47% peak GPU utilization, and wrote 8,929,478 bytes.

**Verdict: BLOCKED**, because only 1 / 6 sessions completed and the sole complete session contained zero validity-positive frames.

### 1.4 Reading it

- Ultrasound generation, raw-label visibility, hashing, and 75 / 75 technical image checks worked in the completed session.
- L4 raw-label support was present in 45 / 75 frames, so visibility alone was not the blocker.
- The requested 3.0-rad command remained physically offset from the 3.14-rad evaluator goal; capture stopped after reaching the looser 5-degree command tolerance, leaving almost every frame outside the frozen 10-degree goal tolerance.
- Relaxing the 10-degree scientific label would change the approved target and was not done.
- ExtraTrees fitting was not run because zero positive labels cannot support the classifier.
- Closed-loop coarse/fine/micro search was not run because no valid frozen model exists.
- No result here changes the successful Phase 1 confirmation.

### 1.5 Files

| file | contents |
|---|---|
| [v3 failure manifest](../../experiments/unity_two_stage_transfer/stage2_dataset_v3_failure.json) | jobs, labels, resources, failure and next-review boundary |
| [v3 failure digest](../../experiments/unity_two_stage_transfer/stage2_dataset_v3_failure.sha256) | failure-manifest SHA-256 |
| [v3 protocol](../../experiments/unity_two_stage_transfer/stage2_dataset_protocol_v3.yaml) | final frozen dataset acquisition settings |
| [v2 failure manifest](../../experiments/unity_two_stage_transfer/stage2_dataset_v2_failure.json) | preserved 16-action failure |
| [v1 failure manifest](../../experiments/unity_two_stage_transfer/stage2_dataset_v1_failure.json) | preserved 8-action failure |
| external v3 run | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_dataset_v3/51245827` |

## 2. Single-session orientation convergence check

### 2.1 Question and criterion

The follow-up asked whether the physical robot can settle within 1 degree of the supported
3.0-radian command and then produce ten fresh L4-positive frames that also satisfy the
unchanged 10 mm and 10 degree evaluator criteria. Protocol SHA-256
`49dbaa61413663f7e615ec1a4f415036b163cb2dc040bfbbe1132ab2821151ba` froze one job,
seed `2026111600`, at most 120 positioning actions, and no dataset capture or training.

### 2.2 Setup and execution

Job `51274046` ran commit `395e3a05b59a8caf8104db90442cdac08086caa7` on `anest-dept`
with one NVIDIA L4, four CPUs, 30 GB requested RAM, and a 15-minute limit. It started at
2026-11-14 03:32:45 EST and ended after 40 seconds with Slurm exit `3:0`. The process sampled
5,009 MiB peak GPU memory and used 3,337,272 KiB maximum resident memory; the artifact tree
grew by 48,938 bytes.

### 2.3 Result

**NOT_EVALUATED because of a technical positioning error.** The direct reset-to-target move
stopped on action four. The transition was finite, unclamped, progressing toward the target,
and retained valid surface state, but the predicted-to-observed command mapping exceeded the
frozen mapping tolerance while a saturated translation and saturated rotation were applied in
the same action. The robot was still 111.256 mm and 76.245 degrees from the requested pose.
No settled ultrasound frame was acquired, so this job provides no evidence for or against
orientation reachability and no Stage 2 scientific gate was evaluated.

The application result initialized its decision field as `FAIL`, but this report classifies the
run as `TECHNICAL_ERROR / NOT_EVALUATED` because the evaluator never received the required ten
frames. The Slurm wrapper still failed closed and no downstream work ran.

### 2.4 Interpretation and next reviewed action

- Phase 1 remains successful and unchanged.
- The Stage 2 dataset remains blocked; no ExtraTrees model or closed-loop controller exists.
- The 10 mm, 10 degree, and 0.05 L4-fraction criteria remain unchanged.
- A justified retry would separate initial positioning into rotation at the current location,
  followed by translation at the settled orientation. It requires a new sealed protocol and
  review because v1 allowed exactly one execution and forbade automatic retry.

| file | contents |
|---|---|
| [orientation protocol](../../experiments/unity_two_stage_transfer/stage2_orientation_smoke_protocol_v1.yaml) | frozen command, criteria, resources, and stop policy |
| [orientation error evidence](../../experiments/unity_two_stage_transfer/stage2_orientation_smoke_v1_error.json) | exact job, resource, transition, and interpretation record |
| [orientation evidence digest](../../experiments/unity_two_stage_transfer/stage2_orientation_smoke_v1_error.sha256) | error-evidence SHA-256 |
| external run | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_orientation_smoke/51274046` |

## 3. Orientation-only bounded retry — can the robot meet the 1 degree command criterion?

`workflows/two_stage_navigation/run_stage2_orientation_smoke.py`, run by
`hpg/jobs/two_stage_stage2_orientation_smoke_v2.sbatch`. It rotates without requesting
translation, then permits positioning and acquisition only after the realized orientation meets
the frozen command criterion.

**Criterion.** Protocol v2 requires realized orientation error at most 1 degree from the
3.0-radian command within 120 orientation actions. It then requires all ten fresh frames to be
technically valid, within 5 mm / 1 degree of the command, within 10 mm / 10 degrees of the
evaluator goal, and to contain raw-label-28 fraction at least 0.05. The criterion was frozen
before job `51276637`.

### 3.1 Method

- Each orientation action retains the current translation command and changes only angle.
- Finite state, command mapping, progress, surface state, termination, and truncation are checked
  after every action.
- Translation, frame acquisition, dataset capture, and fitting remain gated until orientation
  convergence passes.
- The actor-visible record remains separate from privileged goal pose and raw labels.

### 3.2 Setup

| what | value | spec |
|---|---|---|
| job / code | `51276637`; `a4997121106270afd57698f4c6d547b72341061d` | clean recorded worktree |
| implementation | `4c8ee26a231c1d107c9d800168a66ab88d37f44c` | orientation and translation separated |
| protocol | SHA-256 `bc4cbc9ec6d95fb342b104039370d7342223a345afe0bc722a0081d7450a6176` | seed `2026111600` |
| task | `Isaac-robot-US-guidance-v0`; `conv`; `s0010`; one environment | real Isaac Lab task |
| resources | `anest-dept`; one NVIDIA L4; four CPUs; 30 GB; 15 minutes | one job |

~~~bash
sbatch --parsable \
  --account=anest-dept --qos=anest-dept --partition=hpg-turin \
  --constraint=l4 --gres=gpu:l4:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=30G --time=00:15:00 \
  --job-name=sg-s2-orient2 \
  --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/sg-s2-orient2-%j.out \
  --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env \
  hpg/jobs/two_stage_stage2_orientation_smoke_v2.sbatch

/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/envs/sonogym/bin/python \
  workflows/two_stage_navigation/run_stage2_orientation_smoke.py \
  --protocol experiments/unity_two_stage_transfer/stage2_orientation_smoke_protocol_v2.yaml \
  --output-dir /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_orientation_smoke_v2/51276637/results \
  --assets-data-dir /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/assets/data \
  --headless --device cuda:0
~~~

| | value |
|---|---:|
| allowed orientation actions | 120 |
| completed orientation actions | 120 |
| required settled frames | 10 |
| acquired settled frames | 0 |
| dropped because orientation did not converge | 10 |
| **final analyzed frames** | **0** |

### 3.3 Results

| v2 setting | criterion | measured | verdict |
|---|---:|---:|---|
| command orientation after 120 actions | at most 1 degree | 3.747533552289236 degrees | FAIL |
| required fresh-frame conjunction | 10 / 10 | 0 / 10 acquired | NOT_RUN |
| dataset and fitting | only after convergence PASS | not started | NOT_RUN |

Job `51276637` ran for 39 seconds, sampled 5,009 MiB peak GPU memory and 47% peak GPU
utilization, used 3,340,360 KiB maximum resident memory, and wrote 48,656 bytes.

**Verdict: FAIL**, because the physical orientation remained 2.747533552289236 degrees above
the frozen command-error limit after the complete 120-action bound.

### 3.4 Reading it

- No transition failed before the 120-step limit; otherwise the loop would have stopped earlier.
- The runtime artifact recorded `ERROR`, but the audited result is `FAIL` because exhausting the
  predeclared convergence bound is an acceptance failure, not an infrastructure error.
- The task config sets goal angle 3.14 rad and command maximum 3.0 rad, a gap of
  8.021409131831525 degrees before robot tracking error.
- Phase 5A's privileged oracle reached median final goal-orientation error
  8.537524726651668 degrees over 300 steps; that is feasibility evidence, not a matched
  comparison to this 120-step orientation-only check.
- Phase 1 remains successful; Stage 2 data collection and model fitting remain blocked.
- No threshold, task goal, command range, reward, observation, or dynamics was changed.

### 3.5 Files

| file | contents |
|---|---|
| [v2 protocol](../../experiments/unity_two_stage_transfer/stage2_orientation_smoke_protocol_v2.yaml) | frozen two-phase motion, criteria, resources, and stop policy |
| [v2 failure evidence](../../experiments/unity_two_stage_transfer/stage2_orientation_smoke_v2_failure.json) | exact job, bounded result, source audit, and next decision |
| [v2 evidence digest](../../experiments/unity_two_stage_transfer/stage2_orientation_smoke_v2_failure.sha256) | failure-evidence SHA-256 |
| external run | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_orientation_smoke_v2/51276637` |
