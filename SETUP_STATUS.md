# Setup status

Status: **PHASE_4_RUNTIME_READY; PHASE_5A_METRIC_GO; PHASE_5C_STAGE1_GROUP_A_FAIL**

Updated: 2026-11-13 (America/New_York)

## Phase 4 runtime-validation outcome

- GPU smoke test: **PASS**, job `50523856` (`COMPLETED`, exit `0:0`).
- PPO pilot: **PASS**, job `50525848` (`COMPLETED`, exit `0:0`).
- Readiness conclusion: **READY_FOR_REPRODUCTION**.

The successful jobs used account `anest-dept`, partition `hpg-turin`, one NVIDIA L4, four CPU
cores, 30 GB requested RAM, and local Blue storage. The L4 reported 23,034 MiB and driver
580.178.04. Both jobs used the real `Isaac-robot-US-guidance-v0` task, Isaac Sim 4.5.0.0,
Isaac Lab `21f7136325136ca3f6ca4e0a8125edffe5c24f7e`, Python 3.10.22, PyTorch
2.5.1+cu121, skrl 1.4.2, seed `20261106`, model-based `conv` ultrasound, headless mode, and
disabled visualization/external logging. The execution HEAD was `5ab60675`; each run captured
its dirty worktree patch under `context/uncommitted.patch` so the tested fixes remain auditable.

### Smoke evidence

Job `50523856` initialized Isaac Sim, loaded the KUKA and subject `s0010` assets, created and
reset the task, and emitted finite `float32` policy observations with shape
`[1, 1, 200, 150]`. It completed 32 bounded actions. The simulation counter advanced from 200
to 264 (0.533333 simulated seconds at 1/120 s physics dt), maximum joint displacement was
0.476689, and maximum end-effector pose displacement was 0.199634. Rewards were finite at the
first and last recorded steps. The explicit real time-limit probe returned
`terminated=false, truncated=true`; a second reset returned another finite observation of the
same shape. The process exited cleanly after 36.82 wall-clock seconds with 3,329,268 KiB peak
RSS. Nonzero pixels establish observation generation only, not anatomical correctness.

The raw simulated policy observation, distinct from a rendered visualization, is stored in
`outputs/smoke/50523856/ultrasound_original_observations.npz`; the short state/action record is
in `trajectory.json` in the same external run directory.

### PPO evidence

Job `50525848` ran the published `workflows/skrl/train.py` PPO workflow with four environments,
10 configured rollout/update intervals, 320 vector steps, and 1,280 collected transitions. It
finished the timed command in 29.24 seconds (43.7756 transitions/s), recorded nine finite
policy/value-loss points, and recorded finite instantaneous reward summaries. No episode-return
summary was emitted in this short horizon. GPU telemetry peaked at 51% sampled utilization and
5,297 MiB used; `/usr/bin/time` recorded 3,717,120 KiB peak RSS. The 113,760,335-byte
`agent_320.pt` checkpoint loaded on CPU as a dictionary containing `optimizer`, `policy`,
`value`, and `value_preprocessor`.

The first PPO launcher job, `50524765`, failed before training with a Hydra override-path error
and collected zero transitions. The corrected nested paths are
`agent.agent.mini_batches` and `agent.agent.experiment.checkpoint_interval`; the failed log is
preserved. The successful retry kept the same task, seed, and resource request.

### Engineering fixes and evidence locations

The validated fixes execute the environment Python directly instead of activating Conda inside
the GPU job, propagate application failures through Slurm, make the optional Stable-Baselines3
extractor genuinely optional, correct the declared ultrasound space from `[1, 150, 200]` to
`[1, 200, 150]`, validate samples against the declared Gym space, use the correct nested Hydra
paths, and capture GPU telemetry, TensorBoard scalars, and checkpoint structure. NVIDIA EULA
acceptance remains machine-local; the tracked example defaults to `N`.

- Machine report: `reports/phase4_gpu_smoke_ppo.json`.
- Smoke: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/smoke/50523856/`.
- PPO: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/ppo/50525848/`.
- Failed-run logs: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/`.
- CPU preflight: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/records/preflight.json`.

Runtime storage increased from 23,499 MiB before GPU validation to 23,740 MiB after it; the
repository uses 5 MiB. Generated logs, observations, and checkpoints remain outside Git.

This phase establishes simulator initialization, robot/ultrasound transitions, PPO execution,
and checkpoint creation. It does **not** establish useful learned behavior, generalization,
clinical safety, anatomical fidelity, or MBB performance.

## Follow-up paired evaluation and rendered demo

The preregistered no-training follow-up is complete. Job `50553529` (`anest-dept`, one L4,
four CPUs, 30 GB) evaluated `agent_320.pt` and a zero-action control sequentially with seed
`20261110`. Each arm completed 20 episodes on patient `s0010`/`vertebrae_L4`; all observations,
actions, and rewards remained finite. The 20 requested initial commands matched exactly between
arms (`max_abs_difference=0`). Both arms produced 11,980 transitions and all episodes ended by
the native time-limit truncation, with no native task termination.

The checkpoint did not outperform the paired control at this pilot budget. Mean episode return
was -0.654744 for the checkpoint and -0.001471 for zero action. The paired checkpoint-minus-zero
difference averaged -0.653274, and the checkpoint was higher in only 20% of pairs. Composite
distance improved in 20% of checkpoint episodes versus 30% of zero-action episodes. The task
defines neither a native success criterion nor an enabled out-of-bounds termination, so this
result does not support a useful-policy or safety claim. It is evidence against promoting the
10-update pilot checkpoint as a learned navigation result.

Evaluation jobs `50550192` and `50551253` are preserved as protocol failures. The first seeded
before Runner construction; the second re-seeded before reset but allowed Gaussian action
sampling to advance CUDA RNG before later resets. The accepted evaluator restores RNG around
deterministic `mean_actions`, which produced exact pairing in `50553529`.

Rendered demo job `50551365` completed with exit `0:0` on `anest-dept`. It produced a 2.98 s,
179-frame, H.264, 1280x720, 60 fps MP4. The file was decoded successfully and the first/middle
frames were visually checked as nonblank. It is a rendered simulator demonstration, not the raw
ultrasound policy observation and not evidence of policy quality. First camera launch spent about
12 minutes compiling RTX shaders; the completed job used 14:43 wall time.

- Follow-up machine report: `reports/checkpoint_evaluation_demo.json`.
- Accepted evaluation: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/evaluation/50553529/`.
- Video manifest: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/demo/50551365/video/demo-manifest.json`.
- Downloadable MP4: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/demo/50551365/video/sonogym-checkpoint-demo-step-0.mp4`.

From a local terminal, download the MP4 with:

```bash
scp vinhnguyen1@hpg.rc.ufl.edu:/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/demo/50551365/video/sonogym-checkpoint-demo-step-0.mp4 .
```

Runtime storage is now 24,314 MiB, up 574 MiB from the post-Phase-4 value; most follow-up growth
is renderer/shader cache rather than the 390,986-byte MP4.

Next exact action: do not scale this checkpoint. First define and review a native navigation
success criterion and enabled failure/constraint metrics, then preregister a matched-budget
learning-curve protocol with zero/random controls. No additional GPU training is authorized by
this result alone.

## Phase 5A navigation-metric validation

Phase 5A is complete with **METRIC_GO**. The base frozen protocol and manifest remained
unchanged. Approved protocol-v2 commit `907d093` changed only execution lifetime: each frozen
four-start batch ran in a new Isaac Sim process, avoiding the reset stall observed in jobs
`50641608` and `50647528`. No policy training, reward/observation/action/dynamics change, or
threshold adjustment occurred.

Job `50656802` completed `0:0` on `anest-dept` in 10:38 using four CPUs, 30 GB requested
RAM, and one NVIDIA L4. All 32 process-isolated batches exited zero. The merged evaluation
contains 38,400 transitions; paired starts matched exactly across controllers, and no
nonfinite observation/reward/pose or invalid transition was found.

At the primary 10 mm / 10 degree threshold, oracle success was 32/32, while zero action and
bounded random action were both 0/32. Oracle median final error was 0.0246 mm and 8.5375
degrees, versus 65.0751 mm / 26.3056 degrees for zero and 68.3269 mm / 68.9998 degrees for
random. The oracle exceeded each control by 32 successes and had lower median position and
orientation error, satisfying every frozen gate.

The existing Phase 4 PPO checkpoint was diagnostic only: 0/32 final success, 5/32 ever success,
120.7920 mm median final position error, and 7.7488 degrees median final orientation error. It
does not establish useful learned navigation. The target angle is 3.14 rad while the native
command maximum is 3.0 rad, which explains the oracle's approximately 8.54-degree floor and
0/32 success at the secondary 5 mm / 5 degree sensitivity threshold. The native boundary
termination remains disabled, and every episode ended at the 300-step protocol horizon.

Evidence:

- Audit: `reports/phase5a_navigation_objective_audit.md`.
- Human report: `reports/phase5a_navigation_metric_validation.md`.
- Machine report: `reports/phase5a_navigation_metric_validation.json`.
- Completed run: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/navigation_metric_validation_v2/50656802/`.
- Failed initial run: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/navigation_metric_validation/50641608/`.
- Failed sole retry: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/navigation_metric_validation/50647528/`.

The completed run adds approximately 14 MiB (13 MiB run directory plus a 359,968-byte Slurm
log). Peak host RSS was 3,397,904 KiB; sampled GPU memory/utilization peaked at 5,081 MiB/62%.
No raw outputs, logs, plots, videos, or checkpoints were committed.

Next decision: before any separately preregistered Phase 5B learning experiment, the researcher
must decide whether the 3.0-rad command ceiling versus 3.14-rad target is intentional. No
Phase 5B training is authorized by the Phase 5A result.

## Phase 5C current Stage 1 status

G0/G1 capture job 50908807 completed 16 sessions and 3,376 frames. The fixed-anatomy L4
supervised discovery pilot now has a canonical development PASS in CPU job 51192890. The job
used anest-dept, four CPUs, 16 GB requested RAM, zero GPUs, and completed in 3:04 with exit 0.
Image-only seven-level macro recall was 0.844 versus 0.147 after within-session image shuffling;
image plus position/history exceeded position/history-only macro recall by 0.092. Bidirectional
image-derived evidence localized L4 in 4/4 development sessions with median 6.36 mm error, zero
wrong stops, and median 3.5 mm forward/reverse peak difference.

This is fixed-anatomy, prior-inspected development evidence. The acceptance threshold was
selected on the same four development sessions, so independent confirmation and patient
generalization remain unestablished. The checkpoint is marked non-deployment and Phase 2 has
not started. The concise result and report are:

- experiments/unity_two_stage_transfer/stage1_supervised_pilot_result.json
- docs/research/sonogym_s0010_stage1_supervised_pilot.md

Canonical external evidence is under
/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage1_supervised_pilot/51192890/.
The development checkpoint SHA-256 is
bc6df0a1d9e520ac8340789fa751f35e63e0cdcfe4072c130bd70749f22ceb79.
Application timing was 175.659 seconds, maximum measured RSS was 1,117,384 KiB, and the output
directory grew by 72,034,539 bytes. Predecessor job 51192265 is preserved; its scientific
numbers are identical, while job 51192890 adds stage timing and a compact checkpoint.

The frozen Group A development job 51116088 used anest-dept, four CPUs, 16 GB requested
RAM, zero GPUs, and completed in 4:37. It failed the development gate: no threshold produced
a complete seven-structure map, so the reused evaluation split remained unopened and no actor
bundle was released. The concise result and full report are:

- experiments/unity_two_stage_transfer/stage1_group_a_result.json
- docs/research/sonogym_s0010_group_a_stage1_result.md

Reviewer audit found that the six v1 facet candidates were nearest points on adjacent meshes
with 0.0 mm separation, not demonstrated facet-joint locations. Commit `0640c4f` now rejects
that construction; evidence commit `f447616` preserves the six candidates as `REJECTED` audit
records and marks all six actual facet-joint annotations `UNRESOLVED`. Group A remains 7/7;
Group B is now 8/34 `PROVISIONAL` and 26/34 `UNRESOLVED`.

The corrected detailed review package is external to Git at
/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/reviewer/s0010_anatomical_ground_truth_review_20261113_v5.zip.
The 24,564,576-byte ZIP SHA-256 is
2cab0d5ad220e4d5685e4f8c9cda94d11cdd5e1fa4d2fbd4583a1fb6cd290083.
Its 37 PNG files decode successfully, all internal SHA-256 checks pass, and `unzip -t` reports
no errors. The prior v4 archive is retained as superseded review history.
The L4-specific PASS does not override the seven-structure NO-GO. The next scientific action is
a separately approved frozen confirmation capture with new session-held-out starts; Stage 2
has not started.

The blocked G0/G1 record below is retained as historical recovery evidence and is superseded
by successful capture job 50908807.

## Phase 5C historical G0/G1 observability-capture status

Phase 5C anatomical observability and coordinate-integrity gates are **BLOCKED**, not failed on
scientific evidence. No complete scan and no ultrasound dataset were produced, so neither L4
observability nor Stage 1 recognition has been evaluated.

Job `50853505` (`anest-dept`, one L4, four CPUs, 30 GB) failed after 4:04. All 16 sessions
stopped during physical Home return at action step 2 because the collector incorrectly treated a
small probe-basis command-mapping difference as a native workspace clamp. The maximum observed
difference was 0.068359 command index (approximately 0.103 mm), while the incorrect tolerance
was 0.0002 index. A second launcher defect accepted the simulator process's zero exit status even
though every per-session `summary.json` said `FAILED`, so the job unnecessarily initialized all
16 sessions before the summarizer returned exit 2. No frames were saved.

Commit `eea68bcb8a07506abf96dd9754fbeea078a4b28f` separated true workspace clamping from bounded
command-mapping error and made the launcher fail closed on `summary.status`. The sole justified
retry, job `50889771`, then failed fast on the first train/development session after 0:59 because
the new check attempted to convert a CUDA-resident command-bounds tensor directly to NumPy. Its
application exit code was correctly propagated as 2, demonstrating that the launcher repair
worked. Commit `af543204ed5db3709e62f830c2a9a36a0fccde1c` now explicitly transfers this tensor to CPU.
The current code passes 31 Phase 5C unit tests and all 11 Phase 5A regression tests, plus Python
compilation, Bash syntax, and diff-whitespace checks. The frozen protocol and SHA-256
`dc240b5bb333312f6bf85a9ff509f1beafd81e18f1ed458e75b498b5a017017a` were not changed.

Both jobs used NVIDIA L4 GPUs with driver 580.178.04. Sampled GPU memory peaked at 5,011 MiB and
5,009 MiB. The two external run directories occupy 521,934 and 70,850 bytes; the complete G0/G1
output namespace is 972 KiB, well below the 1.5 GB protocol cap. Evidence is preserved at:

- `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/g0_g1_v1/50853505/`
- `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/g0_g1_v1/50889771/`
- `reports/phase5c_g0_g1_capture_status.json`

No third GPU attempt was submitted. The next scientific action requires researcher approval: run
one bounded single-session engineering smoke on `td_00` with the reviewed `af54320` fix. Only if
that produces a complete 211-frame scan with valid motion and manifests should the frozen
16-session G0/G1 capture be resubmitted. Do not fit Stage 1 or begin Stage 2 while G0/G1 remain
blocked.

## Phase 3 historical record

## Completed

- Preserved upstream history, license, and attribution at SonoGym commit
  `e67be58334d1a5274f0913af36f56e4b0b7ffe5a`.
- Inspected the actual guidance YAML loader, task registration, asset paths, visualization
  construction, W&B calls, skrl configuration, and `max_iterations` mapping.
- Added configurable external asset/model/config paths without replacing the tracked Python
  `assets` directory.
- Added the `conv`, headless, fixed-seed navigation baseline and disabled both OpenCV and
  PyVista visualization before environment construction.
- Added a real-environment smoke script and separate serial Slurm templates for smoke and PPO.
- Pinned the published asset dataset revision and SHA-256 values. Both archives were downloaded
  and extracted to local staging at `D:\CSSALT\SonogymRuntime`; they are not in Git.
- Added resumable download, CPU-safe preflight, host inspection/install, and dry-run/submit tools.
- Resource templates request one node, one task, four CPUs, one L4 GPU through `hpg-turin`,
  32 GB RAM, bounded walltimes, and account `anest-dept`. No GPU job was submitted.
- Local CPU-safe checks passed for parsing 75 Python files, parsing the baseline YAML, shell
  syntax (Git Bash), asset path resolution, Git ignore guards, diff whitespace, and both Slurm
  dry-runs. The full local package preflight correctly failed because the unrelated Windows
  Python 3.11 environment does not contain the pinned Isaac stack; this is an installation
  blocker, not a GPU-only check.
- Published the independent project repository at
  `https://github.com/ntvinh2005/cssalt-sonogym`; GitHub API verification reported private
  visibility, default branch `main`, and a remote head matching the local branch.

## Blockers and unverified work

- HiPerGator SSH reached `hpg.rc.ufl.edu` but noninteractive authentication failed with
  `Permission denied (publickey,keyboard-interactive)`. Therefore host/GLIBC/modules/storage,
  scheduler associations/QOS/GPU metadata, Blue path writability, and `anest-dept` usability
  could not be inspected from this session.
- The pinned Conda/pip environment is not installed on HiPerGator. The installer is ready but
  intentionally refuses to install on a login node or another allocation.
- The downloaded local staging copy has not been transferred to Blue storage.
- Isaac Sim extension downloads, NVIDIA EULA acceptance, shader compilation, L4/driver
  compatibility, simulator smoke, and PPO pilot are `NOT_RUN_NO_GPU`.
- Repository publication is complete; simulator readiness remains separate and partial.

Because the cluster environment and Blue assets are incomplete, the project must not be labeled
`PREPARED_GPU_UNTESTED` yet. No engineering claim that SonoGym runs and no scientific claim about
policy learning is made.

## Pinned inputs

- SonoGym: `e67be58334d1a5274f0913af36f56e4b0b7ffe5a`
- Isaac Lab: tag `v2.1.0`, commit `21f7136325136ca3f6ca4e0a8125edffe5c24f7e`
- Assets: `yunkao/SonoGym_assets_models`, revision
  `b37b080a8673f856266a2306724e48d5e034521a`
- `assets.tar.gz`: 754,262,884 bytes,
  SHA-256 `0d5840da3af2eb97e1ca9ebc7c2a969d32b24eef3f3107bc0c9f7db41e4a8f77`
- `models.tar.gz`: 754,887,112 bytes,
  SHA-256 `545113c94b641eda199c6edee1e3586f8cd3a844f0b600470a828fec6210b14f`

## Next action

Restore SSH key/MFA access, copy this repository to the intended Blue path, run
`scripts/setup_hpg.sh --inspect-only`, confirm `anest-dept`, then perform `--install` inside a
four-CPU development allocation. Do not submit either GPU job during setup.
