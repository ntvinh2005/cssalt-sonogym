# Phase 5A — SonoGym navigation metric validation

Date: 2026-11-10 (America/New_York)  
Decision: **METRIC_GO**
Training performed: **No**

## Outcome

The original SonoGym navigation task has an achievable and discriminative
technical evaluation objective under the frozen Phase 5A engineering
criterion. The privileged oracle reached final 10 mm / 10 degree success in
32/32 frozen starts. Zero action and bounded random action each reached 0/32.
The oracle also had lower median final position and orientation error than both
controls. All frozen gate conditions passed.

This result validates a navigation metric and the task's action pathway. It
does not show that the Phase 4 PPO checkpoint learned useful navigation, that
the result generalizes to another anatomy, or that SonoGym can perform an MBB
procedure.

## Source and objective audit

PAPER_METRICS and code geometry are detailed in
reports/phase5a_navigation_objective_audit.md. The paper evaluates final
2-D probe-position error in the patient frontal plane and rotation error in
degrees. The code represents the frontal plane with patient label-volume x-z
indices at 0.0015 m/index; patient y is the surface-height/normal direction.
The learned policy sees ultrasound only. Current pose and target pose are
privileged state used only by evaluation and the oracle.

Paper-versus-code discrepancies were preserved:

- Paper initialization is 130 x 130 mm; code initialization is 120 x 195 mm.
- Reset samples angle 1.5--3.5 rad, while the command range clamps at 3.0 rad.
- Paper navigation horizon is 300 steps; native code is 600 action steps. Phase
  5A applies a separate 300-step protocol truncation.
- The appendix states position reward weight 0.045 with millimetres; code uses
  0.03 on label-index distance.
- Task reward/logging use unwrapped angular distance; the approved evaluator
  uses the frozen half-open [-pi, pi) wrap.
- Native boundary termination is disabled, and no native goal-success
  termination exists. All 128 episodes therefore ended at the protocol horizon,
  not by native success or timeout.

The target angle is 3.14 rad but the command maximum is 3.0 rad. Consequently,
the oracle settles at approximately 8.54 degrees orientation error. It passes
the primary 10-degree threshold but cannot reach the secondary 5-degree
threshold with the current controller and native command bound. This was
reported rather than changing task dynamics.

## Frozen metrics and protocol

Position error:

norm((current_xz_index - goal_xz_index) * 0.0015) * 1000 millimetres.

Orientation error:

abs(wrap[-pi,pi)(current_angle - goal_angle)) * 180 / pi degrees.

Primary final success requires position <= 10 mm **and** orientation <= 10
degrees. Secondary sensitivity requires <= 5 mm and <= 5 degrees. These are
engineering thresholds, not published SonoGym or clinical thresholds.

- Base frozen-protocol commit: d73c46739bab88f115ef57bb00aa8af596b9dfeb
- Protocol-v2 execution commit: 907d093afc6d984421eecefb5c50984fd6eeacd5
- Protocol-v2 SHA-256:
  d9d77e59d4893375d5b4695863c7a0afba9a160c27b7487cdbd475b4b2d70c81
- Manifest SHA-256:
  f5ff5bd9f5e254bad0720d3e0f1a6f91b66a025c3bb725f1589e7d08e7bd83dc
- PPO checkpoint SHA-256:
  72f342f61371d4dee3e9728babd7c3b84674e4c4a5bc51dc827d56eac204d01f
- 32 PCG64 starts, generation seed 20261110; 300 actions/start.
- Controller seeds and ordering remained frozen.

Protocol-v2 changed only process lifetime: each four-start batch ran in a fresh
Isaac Sim process. No reset occurred between batches in one process. The 32
processes were sequential within one Slurm allocation, and aggregation failed
closed unless all starts/controllers, raw step counts, artifact identities,
and realized paired resets matched.

## Validation and execution

CPU preflight passed before GPU submission. All 11 focused metric tests passed,
including unit conversion, patient-frame transformation, frontal projection,
angle wrap, success conjunction, final extraction, done classification, and
invalid-state handling. The preflight artifact is:

/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/records/phase5a-preflight-v2.json

GPU job 50656802 completed 0:0 on account anest-dept, partition hpg-turin,
node c1105a-s6, with four CPUs, 30 GB requested RAM, and one NVIDIA L4. It ran
from 12:52:15 to 13:02:53 EST (10:38). Software was Python 3.10.22, Isaac Sim
4.5.0.0, Isaac Lab 21f7136325136ca3f6ca4e0a8125edffe5c24f7e, PyTorch
2.5.1+cu121, skrl 1.4.2, and driver 580.178.04.

All 32 batch processes exited zero. Aggregation found 38,400 raw step records
and 38,400 episode transitions, with 0.0 maximum cross-controller difference
in realized starting position, orientation, and goal pose. No nonfinite
observation, reward, pose, or invalid transition was found.

Peak host RSS was 3,397,904 KiB. Peak sampled GPU memory was 5,081 MiB and peak
sampled utilization was 62%. Total CPU time was 10:15.416 over 10:38 wall time.
The run directory is 13 MiB; its Slurm log is 359,968 bytes. This adds roughly
14 MiB to runtime storage. A full runtime-tree du scan exceeded the bounded
60-second metadata check, so the prior total was not replaced with an
unverified number.

## Controller comparison

| Controller | Final success | Median / p90 position (mm) | Median / p90 orientation (deg) | Ever success | Mean / median return |
|---|---:|---:|---:|---:|---:|
| Oracle | 32/32 (100%) | 0.0246 / 0.0928 | 8.5375 / 8.6315 | 32/32 | 1.8168 / 1.7211 |
| Zero action | 0/32 (0%) | 65.0751 / 90.1571 | 26.3056 / 82.5001 | 0/32 | -0.00126 / -0.00098 |
| Bounded random | 0/32 (0%) | 68.3269 / 99.6983 | 68.9998 / 113.3805 | 0/32 | -0.5581 / -0.4458 |
| PPO pilot | 0/32 (0%) | 120.7920 / 120.7971 | 7.7488 / 7.7492 | 5/32 | -0.5827 / -0.6795 |

Oracle first reached the primary success region between steps 41 and 167
(median 47). Secondary 5 mm / 5 degree final success was 0/32 for every
controller because of the orientation command-bound issue described above.

## Paired discrimination and gate

Relative to zero action, the oracle had 32 additional final successes, a
65.0588 mm median paired position-error reduction, and a 17.7532 degree median
paired orientation-error reduction. Relative to bounded random actions, the
corresponding values were 32 successes, 68.2396 mm, and 60.4625 degrees.
Oracle final errors were lower on all 32 starts for both position controls and
all 32 random-control orientations; oracle orientation was lower than zero
action on 24/32 starts.

The frozen gate passed:

- Metric correctness: PASS (tests, finite artifacts, exact cross-process
  pairing, and termination accounting).
- Task feasibility: PASS (oracle 32/32, required at least 26/32).
- Discrimination versus zero: PASS (32-success advantage and lower medians).
- Discrimination versus random: PASS (32-success advantage and lower medians).

Decision: **METRIC_GO**.

## PPO diagnostic

The existing agent_320.pt checkpoint was loaded without retraining and used
deterministic mean actions. It had 0/32 final successes, although 5/32 episodes
temporarily entered the primary success region. Its median final position error
was 120.792 mm, while its median orientation error was 7.749 degrees. It spent
9,441 of 9,600 episode-steps at a native command bound. The pilot therefore
does not support a useful-policy claim and should not be scaled as-is.

## Reset failure resolution

Jobs 50641608 and 50647528 are retained as failed protocol-v1 executions.
Both completed the first oracle batch and then stalled entering the next reset.
The sole retry changed from direct wrapper reset to native time-limit
auto-reset and reproduced the same stall. Protocol-v2 made no scientific
change; it removed inter-batch reset by closing Isaac Sim after every batch.
Job 50656802 completed all 32 process launches in 10:38, confirming that the
blocker was process-lifetime/reset behavior rather than metric computation,
asset loading, GPU allocation, or the first-batch controller pipeline.

## Artifacts

Run root:

/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/navigation_metric_validation_v2/50656802/

Important files:

- results/final/evaluation.json — complete merged machine-readable evaluation.
- results/final/summary.json — gate and controller summaries.
- results/final/episodes.csv — 128 per-episode records.
- results/final/per_step.jsonl.gz — 38,400 raw step records.
- results/final/final_error_scatter.png — final position/orientation plot.
- results/final/paired_oracle_controls.png — paired oracle/control plot.
- results/final/representative_trajectories.json — preregistered trajectory set.
- results/batches/ — all 32 process-isolated raw batch results.
- metrics/gpu-telemetry.csv and context/ — resources and provenance.

The final summary SHA-256 is
5d3fbb0f8c56611562e2efcd7d0f8a8d4673dc1267acbb325107403df2eedb07.
The merged evaluation SHA-256 is
0d72a1bb26ddde0a036c171e2b13b3a835f5f50ba0b55c6c8d255182e3228ac7.
Large logs and raw outputs remain on Blue Storage and are not committed.

## Next researcher decision

Stop after Phase 5A. The next proposed scientific decision is whether to
approve a separately preregistered Phase 5B learning experiment that trains a
new policy and evaluates it against the now-validated oracle/zero/random
protocol. Before approval, resolve whether the 3.0 rad command ceiling versus
3.14 rad target is an intended task property; changing it would alter task
dynamics and requires explicit review. Do not infer from METRIC_GO that the
existing PPO checkpoint is useful.
