# Setup status

Status: **READY_FOR_REPRODUCTION**

Updated: 2026-11-09 (America/New_York)

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

Next exact experiment: run one preregistered single-seed navigation evaluation using the saved
pilot checkpoint and a fixed held-out initial-condition list, reporting episode returns,
success/termination rates, and trajectory safety diagnostics. This is an evaluation step, not
full reproduction, multi-seed training, anatomy generalization, or MBB procedure development.

This phase establishes simulator initialization, robot/ultrasound transitions, PPO execution,
and checkpoint creation. It does **not** establish useful learned behavior, generalization,
clinical safety, anatomical fidelity, or MBB performance.

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
