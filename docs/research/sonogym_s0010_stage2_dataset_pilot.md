# SonoGym s0010 Stage 2 dataset pilot

This report tests whether the fixed-`s0010` simulator can supply the labeled local-view data required for the supervised classifier and regressor in [Unity Stage 2](../notation.md#unity-stage-2). It covers physical data acquisition only; model fitting and closed-loop search were gated off when capture failed.

## Result

**BLOCKED: the frozen acquisition produced one complete 75-frame session with zero validity-positive frames, then failed the physical orientation tolerance in the second session.**

| step | what was measured | criterion | measured | verdict |
|---|---|---|---|---|
| 1 | six-session supervised dataset acquisition | 6 / 6 complete sessions; 450 joined frames; positive and negative validity support; all waypoint poses within 5 mm and 5 degrees | 1 / 6 complete; 75 / 450 frames; 0 positives; second session ended at 1.052 mm and 5.497 degrees | BLOCKED |

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
