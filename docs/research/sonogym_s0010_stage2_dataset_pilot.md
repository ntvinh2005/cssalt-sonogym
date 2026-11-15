# SonoGym s0010 Stage 2 dataset pilot

This report tests whether the fixed-`s0010` simulator can supply the labeled local-view data and supervised perception required for [Unity Stage 2](../notation.md#unity-stage-2). It covers physical acquisition and session-separated development evaluation; closed-loop search remains gated off.

## Result

**DEVELOPMENT_PASS after target-contract repair: targeted capture supplied independent positive support, and an image-observable L4-view classifier reached 1.00 precision and 0.8929 recall while the unchanged quality regressor passed. Closed-loop evaluation remains NOT_RUN.**

| step | what was measured | criterion | measured | verdict |
|---|---|---|---|---|
| 1 | six-session supervised dataset acquisition | 6 / 6 complete sessions; 450 joined frames; positive and negative validity support; all waypoint poses within 5 mm and 5 degrees | 1 / 6 complete; 75 / 450 frames; 0 positives; second session ended at 1.052 mm and 5.497 degrees | BLOCKED |
| 2 | initial orientation check | acquire ten settled frames after bounded positioning | no frame acquired because of a command-mapping technical error | NOT_EVALUATED |
| 3 | orientation-only physical convergence | command orientation error at most 1 degree, then 10 / 10 frames meeting all pose, image, and L4 criteria | 3.747533552289236 degrees after 120 / 120 steps; 0 / 10 frames acquired | FAIL |
| 4 | extended orientation-only physical convergence | same criteria as step 3; orientation bound extended to 300 using Phase 5A evidence | 3.7384630610714593 degrees after 300 / 300 steps; only 0.0090704912177767 degree improvement; 0 / 10 frames acquired | FAIL |
| 5 | coupled six-session dataset acquisition | 6 / 6 sessions; 450 joined frames; positive and negative support; public/privileged one-to-one | 6 / 6; 450 frames; 58 positives; one-to-one true | PASS |
| 6 | supervised development evaluation | precision at least 0.95; recall at least 0.80; Spearman at least 0.30; regressor MSE below the training-mean constant | 1.00 precision; 0.50 recall; 0.7634 Spearman; 0.00785 versus 0.02418 MSE | NO_GO |
| 7 | targeted independent-positive coverage | unique geometric-valid hashes at least 60 / 20 / 20 and positive trajectories at least 8 / 4 / 4 | 79 / 21 / 20 hashes and 32 / 8 / 8 trajectories | PASS |
| 8 | unchanged geometric-validity classifier on targeted capture | precision at least 0.95; recall at least 0.80 | 0.5556 precision; 0.1087 recall | NO_GO |
| 9 | image-observable validity plus unchanged quality model | validity precision at least 0.95 and recall at least 0.80; quality criteria unchanged | 1.00 precision; 0.8929 recall; 0.6757 Spearman; 0.00989 versus 0.02908 MSE | DEVELOPMENT_PASS |

The frozen 10 mm / 10 degree geometric success definition was not relaxed. V14 demonstrated
that pixels could not reproduce its sharp orientation boundary. V15 therefore separates the
Unity-style image-validity classifier from independent geometric evaluation: the classifier
predicts whether the L4 view has sufficient image support, while the continuous quality model
ranks candidates. This repair was selected using development evidence, so it requires a new
frozen confirmation before closed-loop execution can support a scientific claim.

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

## 4. Final extended orientation check — does a 300-step bound remove the residual error?

`workflows/two_stage_navigation/run_stage2_orientation_smoke.py`, run by
`hpg/jobs/two_stage_stage2_orientation_smoke_v3.sbatch`. It repeats the frozen
orientation-only procedure with a 300-step bound supported by the existing Phase 5A horizon.

**Criterion.** The realized orientation must be within 1 degree of the 3.0-radian command within
300 actions. Only then may the unchanged ten-frame pose, image, and L4 conjunction be evaluated.
The criterion and all settings except the orientation-action bound were frozen before job
`51279543`.

### 4.1 Method

- Protocol v3 changes the orientation bound from 120 to 300 actions.
- Seed, requested angle, motion sequence, 120-step positioning bound, acceptance criteria,
  information boundary, resources, and walltime match protocol v2.
- Translation, frame acquisition, dataset capture, and fitting remain gated on orientation.

### 4.2 Setup

| what | value | spec |
|---|---|---|
| job / code | `51279543`; `4cff08da052b74373824378d011ad8d3497a0596` | clean recorded worktree |
| implementation | `344b02b27a0eea0a494e2141199ab6961124d753` | v3 protocol support |
| protocol | SHA-256 `d4407cff5f17dfcbdbd5577e7ff670173d48ea81ebfc789c04b2b49998aa4c46` | seed `2026111600` |
| task | `Isaac-robot-US-guidance-v0`; `conv`; `s0010`; one environment | real Isaac Lab task |
| resources | `anest-dept`; one NVIDIA L4; four CPUs; 30 GB; 15 minutes | one job |

~~~bash
sbatch --parsable \
  --account=anest-dept --qos=anest-dept --partition=hpg-turin \
  --constraint=l4 --gres=gpu:l4:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=30G --time=00:15:00 \
  --job-name=sg-s2-orient3 \
  --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/sg-s2-orient3-%j.out \
  --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env \
  hpg/jobs/two_stage_stage2_orientation_smoke_v3.sbatch

/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/envs/sonogym/bin/python \
  workflows/two_stage_navigation/run_stage2_orientation_smoke.py \
  --protocol experiments/unity_two_stage_transfer/stage2_orientation_smoke_protocol_v3.yaml \
  --output-dir /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_orientation_smoke_v3/51279543/results \
  --assets-data-dir /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/assets/data \
  --headless --device cuda:0
~~~

| | value |
|---|---:|
| allowed orientation actions | 300 |
| completed orientation actions | 300 |
| required settled frames | 10 |
| acquired settled frames | 0 |
| dropped because orientation did not converge | 10 |
| **final analyzed frames** | **0** |

### 4.3 Results

| setting | command-error criterion | measured | verdict |
|---|---:|---:|---|
| v2, 120 actions | at most 1 degree | 3.747533552289236 degrees | FAIL |
| v3, 300 actions | at most 1 degree | 3.7384630610714593 degrees | FAIL |
| v3 fresh-frame conjunction | 10 / 10 | 0 / 10 acquired | NOT_RUN |

Job `51279543` ran for 43 seconds, sampled 5,009 MiB peak GPU memory and 43% peak GPU
utilization, used 3,338,912 KiB maximum resident memory, and wrote 48,940 bytes.

**Verdict: FAIL**, because 180 additional actions improved the command error by only
0.0090704912177767 degrees and left it 2.7384630610714593 degrees above the frozen limit.

### 4.4 Reading it

- Orientation-only control has plateaued under the tested robot/IK state; another horizon-only
  retry is not justified.
- At step 300, the Phase 5A oracle's 32 episodes all remained at the command bound while applying
  median angle correction 0.008940439671278 rad and median translation norm
  0.000559029247180037 index; median realized angle was 2.9909921884536743 rad.
- The oracle comparison shows that small simultaneous pose corrections are feasible, but it uses
  privileged target-relative state and is not an image-only result.
- A transferred controller may use robot kinematics and its own inferred relative target, but it
  must not receive `goal_cmd_pose` or evaluator annotations at inference.
- At this historical checkpoint, Phase 1 remained successful and Stage 2 dataset collection and
  fitting remained blocked. Section 5 records the later coupled-motion capture that cleared this
  data gate.
- No task goal, command range, threshold, reward, observation, or dynamics was changed.

### 4.5 Files

| file | contents |
|---|---|
| [v3 protocol](../../experiments/unity_two_stage_transfer/stage2_orientation_smoke_protocol_v3.yaml) | frozen 300-step bound, criteria, resources, and stop policy |
| [v3 failure evidence](../../experiments/unity_two_stage_transfer/stage2_orientation_smoke_v3_failure.json) | exact job, plateau measurement, oracle comparison, and next decision |
| [v3 evidence digest](../../experiments/unity_two_stage_transfer/stage2_orientation_smoke_v3_failure.sha256) | failure-evidence SHA-256 |
| external run | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_orientation_smoke_v3/51279543` |

## 5. Coupled dataset capture — did the reviewed grid produce complete labeled support?

`workflows/two_stage_navigation/collect_stage2_dataset.py`, run by
`hpg/jobs/two_stage_stage2_dataset_v12.sbatch`. The reviewed coupled-motion protocol preserved
the frozen validity definition while using the task's physically supported simultaneous pose
correction. Public ultrasound records and privileged offline annotations remained separate and
joined one-to-one by acquisition identity.

**Criterion.** Protocol v12 requires all six complete sessions and 450 joined frames, positive
and negative validity support in every supervised split, and one-to-one public/privileged
records. The 10 mm / 10 degree validity thresholds and 0.05 L4 raw-label fraction were unchanged.

### 5.1 Method

- Four training sessions, one calibration session, and one development session were frozen by
  complete acquisition trajectory; adjacent frames were not randomly split.
- Forty-eight trajectories sampled the reviewed local position/orientation grid through the real
  Isaac Lab action and IK path.
- Ultrasound pixels and executed-motion history were actor-visible. Goal pose, raw anatomy label,
  and validity/quality annotations remained evaluator-only.
- No model was fitted by the capture job.

### 5.2 Setup

~~~bash
sbatch --account=anest-dept --qos=anest-dept --partition=hpg-turin \
  --constraint=l4 --gres=gpu:l4:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=30G --time=00:30:00 \
  --job-name=sg-stage2-data12 \
  --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/sg-stage2-data12-%j.out \
  --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env \
  hpg/jobs/two_stage_stage2_dataset_v12.sbatch
~~~

| what | measured |
|---|---:|
| job / capture commit | `51359047` / `28e1bc870c68097d254be622797b0a4797f72dde` |
| protocol SHA-256 | `cb0c60de6546440bb69a08bdaaa8f8abc580de22e856067a0e4720066bb38862` |
| sessions / trajectories | 6 / 48 |
| attempted / dropped / final frames | 450 / 0 / 450 |
| distinct image hashes | 347 |
| train / calibration / development positives | 40 / 10 / 8 |
| public/privileged joins | 450 / 450, one-to-one |

### 5.3 Results

**Verdict: PASS / DATA_READY.** Job `51359047` completed with Slurm exit `0:0` in 4:28.
It used one NVIDIA L4, four CPUs, and a 30 GB request; sampled peak GPU memory was 5,009 MiB,
peak utilization was 57%, and maximum resident memory was 3,397,008 KiB. The capture tree grew
by 80,303,699 bytes.

This establishes complete fixed-`s0010` supervised acquisition support. It does not establish
image-only recognition or closed-loop acquisition performance.

### 5.4 Files

| file | contents |
|---|---|
| [v12 protocol](../../experiments/unity_two_stage_transfer/stage2_dataset_protocol_v12.yaml) | frozen capture and supervised-development criteria |
| [v12 capture result](../../experiments/unity_two_stage_transfer/stage2_dataset_v12_result.json) | job, dataset, resource, and artifact evidence |
| external capture | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_dataset_v12/51359047` |
| reviewer ZIP | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/reviewer/sonogym_stage2_dataset_review_20261114_job51359047.zip` |

## 6. Supervised development evaluation — do both perception models clear the frozen gate?

`workflows/two_stage_navigation/train_stage2.py`, run by
`hpg/jobs/two_stage_stage2_train_v12.sbatch`. It fits the frozen ExtraTrees validity classifier
and quality regressor from training sessions, selects the validity threshold on calibration only,
then evaluates the untouched development session.

**Criterion.** Validity precision must be at least 0.95 and recall at least 0.80. Quality
Spearman correlation must be at least 0.30 and regressor MSE must be below the training-mean
constant baseline. All four conditions must hold before closed-loop evaluation.

### 6.1 Setup

~~~bash
sbatch --account=anest-dept --qos=anest-dept --partition=hpg-default \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:30:00 \
  --job-name=sg-stage2-train12 \
  --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/sg-stage2-train12-%j.out \
  --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env \
  hpg/jobs/two_stage_stage2_train_v12.sbatch

export PYTHONPATH=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/envs/stage1-python:$PWD/source/spinal_surgery/spinal_surgery/evaluation
/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/envs/sonogym/bin/python \
  workflows/two_stage_navigation/train_stage2.py \
  --protocol experiments/unity_two_stage_transfer/stage2_dataset_protocol_v12.yaml \
  --capture-root /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_dataset_v12/51359047 \
  --output-dir /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_development_v12/51367445/results
~~~

| accounting | attempted | dropped | final |
|---|---:|---:|---:|
| all frames | 450 | 0 | 450 |
| training | 300 | 0 | 300 |
| calibration | 75 | 0 | 75 |
| development | 75 | 0 | 75 |

No image hash crossed a split boundary. Training, calibration, and development contained 232,
58, and 57 distinct hashes, respectively.

### 6.2 Results

| measurement | criterion | measured | verdict |
|---|---:|---:|---|
| calibration validity precision | at least 0.95 | 1.00 | PASS |
| calibration validity recall | at least 0.80 | 0.20 | FAIL |
| development validity precision | at least 0.95 | 1.00 | PASS |
| development validity recall | at least 0.80 | 0.50 (4 TP, 4 FN) | FAIL |
| development quality Spearman | at least 0.30 | 0.7634057971 | PASS |
| development regressor MSE | below 0.0241841637 | 0.0078532917 | PASS |

The calibration-selected validity threshold was 0.6666666667. Job `51367445` returned the
declared gate-failure exit `3:0` after 12 seconds; the application itself took 4.946393 seconds
and used 119,752 KiB maximum resident memory.

**Verdict: DEVELOPMENT_NO_GO.** The quality regressor passed both frozen requirements, but the
validity classifier failed the recall requirement. The fitted bundle is evidence for this
development result only and is not eligible for closed-loop execution.

### 6.3 Reading it

- Lowering the threshold to 0.3 would give calibration precision 0.8333 and recall 1.0, so it
  would violate the frozen precision gate. Development recall would still be only 0.75.
- Four of eight development-positive images were false negatives despite valid geometry and
  L4 visibility. The current evidence therefore points to incomplete image feature/model
  coverage, not an absence of labeled positives.
- No threshold, label, split, feature, model hyperparameter, or gate was changed after seeing
  the result.
- Closed-loop coarse/fine/micro search, return-to-best, and fresh confirmation remain NOT_RUN.
- Phase 1 evidence and the Phase 5A frozen evaluation remain unchanged.

### 6.4 Files

| file | contents |
|---|---|
| [development result](../../experiments/unity_two_stage_transfer/stage2_development_v12_result.json) | concise immutable scientific and resource evidence |
| [development result digest](../../experiments/unity_two_stage_transfer/stage2_development_v12_result.sha256) | result-manifest SHA-256 |
| external development report | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_development_v12/51367445/results/development_report.json` |
| external model bundle | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_development_v12/51367445/results/stage2_models.joblib` |

## 7. Targeted capture — did additional positive morphology resolve data coverage?

`workflows/two_stage_navigation/collect_stage2_dataset.py`, run by
`hpg/jobs/two_stage_stage2_dataset_v14.sbatch`. Protocol v14 concentrated the same physical
sampling mechanism around the positive region while preserving labels, features, model families,
development gates, information boundaries, action semantics, and simulator dynamics.

**Criterion.** Each session required at least 20 distinct frame hashes. Train, calibration, and
development required at least 60 / 20 / 20 unique geometric-valid hashes and at least 8 / 4 / 4
positive trajectories, respectively. Public and privileged records still had to join one-to-one.

### 7.1 Method and setup

- Four train, one calibration, and one development session remained trajectory-separated.
- The capture used one real SonoGym environment and its real IK/action path.
- Job `51374942` is preserved as a technical precursor failure: the legacy half-of-session hash
  heuristic stopped it before any scientific gate. V14 replaced only that precondition with the
  declared per-session minimum of 20.
- No model was fitted by the GPU job.

~~~bash
sbatch --account=anest-dept --qos=anest-dept --partition=hpg-turin \
  --constraint=l4 --gres=gpu:l4:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=30G --time=00:30:00 \
  --job-name=sg-stage2-data14 \
  --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/sg-stage2-data14-%j.out \
  --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env \
  hpg/jobs/two_stage_stage2_dataset_v14.sbatch
~~~

### 7.2 Results

| measurement | required | measured | verdict |
|---|---:|---:|---|
| complete sessions / trajectories / frames | 6 / 48 / 450 | 6 / 48 / 450 | PASS |
| unique positive hashes, train / calibration / development | 60 / 20 / 20 | 79 / 21 / 20 | PASS |
| positive trajectories, train / calibration / development | 8 / 4 / 4 | 32 / 8 / 8 | PASS |
| public/privileged join | one-to-one | 450 / 450 | PASS |

**Verdict: PASS / DATA_READY.** Job `51377943` completed with exit `0:0` in 3:15 on
one NVIDIA L4, four CPUs, and a 30 GB request. Sampled peak GPU memory was 5,009 MiB, peak
utilization was 61%, and maximum resident memory was 3,368,292 KiB. The capture added
63,588,359 bytes. This proves coverage only, not classifier or closed-loop performance.

### 7.3 Files

| file | contents |
|---|---|
| [v14 protocol](../../experiments/unity_two_stage_transfer/stage2_dataset_protocol_v14.yaml) | sealed targeted capture and unchanged model gate |
| [v14 result](../../experiments/unity_two_stage_transfer/stage2_v14_result.json) | capture, V14 fit, diagnosis, resources, and boundaries |
| external capture | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_dataset_v14/51377943` |
| reviewer ZIP | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/reviewer/sonogym_stage2_dataset_review_20261114_job51377943.zip` |

## 8. V14 development — did more positive coverage repair geometric-validity recognition?

The unchanged V14 fit again used `workflows/two_stage_navigation/train_stage2.py`. Its classifier
target was still the conjunction of exact 10 mm position, exact 10 degree orientation, image
validity, and L4 visibility. The quality target and both model families were unchanged.

**Criterion.** Validity precision at least 0.95 and recall at least 0.80; quality Spearman at
least 0.30 and MSE below the training-mean constant.

### 8.1 Results

| measurement | criterion | measured | verdict |
|---|---:|---:|---|
| calibration precision / recall | at least 0.95 / 0.80 | 1.00 / 0.1667 | FAIL |
| development precision / recall | at least 0.95 / 0.80 | 0.5556 / 0.1087 | FAIL |
| quality Spearman | at least 0.30 | 0.6757 | PASS |
| quality MSE | below 0.02908 | 0.00989 | PASS |

**Verdict: DEVELOPMENT_NO_GO.** CPU job `51378486` returned the declared scientific-failure
exit `3:0` in 14 seconds. It used four CPUs, a 16 GB request, and 119,472 KiB maximum resident
memory. No threshold change could satisfy both validity gates.

Read-only diagnosis found no exact image-hash or feature-vector label conflicts, but the two
highest-confidence development false-positive image hashes were only just outside the geometric
orientation boundary at 10.0678 and 10.4356 degrees. One-sample-per-hash evaluation still gave
only 0.80 precision and 0.40 recall. This rejected the hypothesis that more repeated positive
coverage alone could fix the classifier. The continuous quality model remained informative.

### 8.2 Files

| file | contents |
|---|---|
| [V14 combined result](../../experiments/unity_two_stage_transfer/stage2_v14_result.json) | immutable metrics and diagnosis |
| external development | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_development_v14/51378486` |
| external diagnosis | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_development_v14/51378486/diagnostics` |

## 9. Target-contract repair — can image validity be separated from exact pose success?

Protocol v15 keeps geometric success at 10 mm / 10 degrees for independent evaluation. The
classifier instead predicts an image-observable target: a technically valid frame with L4 raw
label fraction at least 0.20 during offline annotation. The actor still receives only ultrasound
pixels. A predeclared 0.18--0.22 band is excluded from classifier fitting, while calibration and
development score every frame against the exact 0.20 target. The unchanged quality regressor
continues to encode continuous visibility and geometric proximity for candidate ranking.

**Criterion.** The same numerical development gate applies: classifier precision at least 0.95
and recall at least 0.80; quality Spearman at least 0.30 and MSE below the constant baseline.

### 9.1 Setup

~~~bash
sbatch --account=anest-dept --qos=anest-dept --partition=hpg-default \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:30:00 \
  --job-name=sg-stage2-train15 \
  --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/sg-stage2-train15-%j.out \
  --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env,STAGE2_CAPTURE_JOB_ID=51377943 \
  hpg/jobs/two_stage_stage2_train_v15.sbatch
~~~

The fit used 276 unambiguous training frames: 153 positives and 123 negatives; 24 boundary
frames were excluded only from fitting. All 75 calibration and 75 development frames remained
in evaluation.

### 9.2 Results

| measurement | criterion | measured | verdict |
|---|---:|---:|---|
| calibration validity precision / recall | at least 0.95 / 0.80 | 1.00 / 0.8333 | PASS |
| development validity precision / recall | at least 0.95 / 0.80 | 1.00 / 0.8929 | PASS |
| development quality Spearman | at least 0.30 | 0.6757 | PASS |
| development quality MSE | below 0.02908 | 0.00989 | PASS |

**Verdict: DEVELOPMENT_PASS.** Job `51380423` completed with exit `0:0` in 13 seconds; the
application used 6.61 wall seconds and 118,020 KiB maximum resident memory. The model bundle was
loaded successfully and its protocol, target, threshold, and feature width matched the report.

This establishes that the label-contract problem is repairable on the inspected fixed-`s0010`
development dataset. It does not establish independent confirmation, closed-loop search,
patient-held-out generalization, or geometric target attainment. Because V15 was selected after
V14 diagnosis, the next scientific step must freeze it and confirm perception on new sessions
before any closed-loop success claim.

### 9.3 Files

| file | contents |
|---|---|
| [v15 protocol](../../experiments/unity_two_stage_transfer/stage2_dataset_protocol_v15.yaml) | sealed classifier target, ambiguity band, unchanged evaluator, and CPU bound |
| [v15 result](../../experiments/unity_two_stage_transfer/stage2_development_v15_result.json) | metrics, resources, hashes, and interpretation |
| external report | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_development_v15/51380423/results/development_report.json` |
| external model bundle | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_development_v15/51380423/results/stage2_models.joblib` |
