# Phase 5A pre-execution audit: SonoGym navigation objective

Date: 2026-11-10  
Task: `Isaac-robot-US-guidance-v0`  
Upstream revision: `e67be58334d1a5274f0913af36f56e4b0b7ffe5a`  
Project baseline before Phase 5A: `19e3976fa22d32cf4614319e84aaa0b6d8f21c57`

This audit freezes interpretation before controller results are observed. It does
not claim that the Phase 4 PPO checkpoint learned useful navigation.

## PAPER_METRICS

The paper defines navigation as three-degree-of-freedom tangential probe motion
over the patient's skin. The probe frame is `{U}` and the fixed target frame is
`{G}` above the target anatomy. The low-level controller maintains contact and
aligns probe z with the local surface normal. The policy action is translation
along probe x/y plus rotation about probe z. The state is the privileged
relative probe-to-goal pose, while the learned policy observes only the current
ultrasound image.

For navigation evaluation, the authors report final two-dimensional position
error in the patient's frontal plane and the corresponding rotation error in
degrees. Results are continuous errors averaged over trials, not a published
binary success rate. Appendix C.1 reports a 130 x 130 mm initial region,
orientation 1.5--3.5 rad, 200 x 150 ultrasound images at 0.5 mm/pixel, reward
weight `w1=0.045` with position in mm, and 300-step episodes.

Primary source: [SonoGym paper and appendix](https://arxiv.org/html/2507.01152).

## ADDITIONAL_ENGINEERING_METRICS

Phase 5A defines, without changing the task, the following engineering-only
metrics:

- Position error: Euclidean distance between current and goal x-z coordinates
  in the patient frame, multiplied by `0.0015 m/index` and `1000 mm/m`.
- Orientation error: absolute current-minus-goal angle wrapped to `[-pi, pi)`,
  then converted to degrees.
- Primary final success: position <= 10 mm AND orientation <= 10 degrees.
- Secondary sensitivity: position <= 5 mm AND orientation <= 5 degrees.
- Ever-reached success and first successful action step are trajectory
  diagnostics; neither substitutes for final success.
- Native termination, native time-limit truncation, and the protocol's
  300-action horizon are recorded separately.

The 10/10 and 5/5 thresholds are not SonoGym paper thresholds and are not
clinical medial-branch-block criteria.

## Coordinate frames and units verified from source

- `SurfaceMotionPlanner` stores task commands as
  `[patient_x_label_index, patient_z_label_index, angle_rad]`.
- Patient x-z is the plane used to index the dorsal surface map; patient y is
  the surface-height/normal direction. Therefore x-z is the source-level
  realization of the paper's patient frontal plane.
- `GTDiscreteMotionGenerator.human_cmd_state_from_ee_pose` transforms the real
  robot end-effector pose into the patient frame, recovers the surface point by
  adding the probe z-axis times probe height, divides x/z metres by label
  resolution, and computes angle as `atan2(x_axis_z, x_axis_x)` in `[0, 2*pi)`.
- `label_res=0.0015` metres per label index. Pose x/z values are indices, not
  millimetres. Robot/world translations are metres. Angles and angular actions
  are radians.
- The target x-z is the mean of the L4 STL points selected by
  `VertebraViewer`, plus the configured zero offset. The target angle is 3.14
  rad. The current fixed anatomy is patient `s0010`, target `vertebrae_L4`.
- Continuous policy actions are dimensionless values bounded by `[-2, 2]` in
  every dimension. The environment multiplies them by `[2, 2, 0.1]` and clamps
  physical command increments to `[4 indices, 4 indices, 0.2 rad]`.

There is no implemented pi-periodic probe-orientation symmetry. The frozen
metric therefore uses only the requested 2*pi wrap and does not introduce
180-degree equivalence post hoc.

## Reset, reward, and completion behavior

- Reset draws each native command component uniformly from
  `[[100, 70, 1.5], [180, 200, 3.5]]`, then `USSlicer.update_cmd` clamps it to
  operational range `[[100, 70, 1.0], [180, 200, 3.0]]` before surface pose
  construction and 200 IK-settling physics steps.
- With 1.5 mm/index, the configured x-z reset rectangle is 120 x 195 mm, not
  the paper's stated 130 x 130 mm. Nine of the 32 frozen draws exceed the
  operational 3.0-rad upper bound and are expected to clamp to 3.0 rad.
- The code reward is the one-step decrease in
  `0.03 * ||current_xz_index - goal_xz_index|| +
  |current_angle - goal_angle|`. It uses unwrapped angular error and index-space
  position with `w_pos=0.03`; this is not the appendix's stated `w1=0.045`
  applied to millimetres.
- `termination_direct=True` enables the native time limit. At sim dt 1/120 s
  and decimation 2, the action rate is 60 Hz. The configured 10 seconds gives
  `max_episode_length=600`, while the done check fires on observed action step
  599 because it tests `episode_length_buf >= max_episode_length - 1`.
- Native `out_of_bounds` is explicitly replaced by zeros. Therefore terminated
  is false and boundary termination is disabled; `USSlicer.no_collide` is
  retained only as a diagnostic sensor. Phase 5A does not re-enable it.
- There is no goal-success termination. A native timeout or protocol horizon is
  not classified as success unless the final pose independently satisfies both
  frozen error thresholds.

## Oracle and policy information boundary

The original repository supplies `ExpertGuidance`, used by its expert-recording
workflow with ratio `[0.02, 0.02, 0.6]`. It uses privileged current pose, goal
pose, and patient/probe transforms to emit a valid action, which is then applied
through the same `env.step`, surface command, IK, and robot dynamics as every
other controller. It does not teleport or directly set the target pose. It is a
diagnostic proportional expert, not a proof of optimality.

The PPO policy receives only the model-based ultrasound tensor. It is loaded
with the task registry's SKRL architecture and preprocessing, `agent.load`, eval
mode, and deterministic `mean_actions`. It never receives the oracle's
privileged pose/goal fields.

## Frozen replay and horizon

The 32 requested commands are generated once with NumPy PCG64 seed 20261110.
The evaluation-only replay hook substitutes those commands for `torch.rand`
inside reset but leaves the native clamp, surface construction, robot IK,
observation, reward, action, and done logic unchanged. All four controllers use
the same manifest order. Cross-controller realized reset poses must match the
oracle arm within 1 mm and 1 degree, with identical goal state.

Each controller is evaluated for at most 300 action steps per start (5 seconds
at 60 Hz), matching the paper's stated navigation horizon. Reaching this
protocol horizon is reported as `protocol_truncated`, separately from the
native 599-step time limit.
