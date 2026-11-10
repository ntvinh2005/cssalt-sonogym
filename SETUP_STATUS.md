# Setup status

Status: **RUNTIME_BLOCKED**

Updated: 2026-11-09 (America/New_York)

## Phase 4 runtime-validation outcome

- GPU smoke test: **FAIL**.
- PPO pilot: **NOT_RUN** because the smoke acceptance gate did not pass.
- Readiness conclusion: **RUNTIME_BLOCKED**.

The pinned environment and assets are installed on HiPerGator, but neither GPU attempt reached
Isaac Sim initialization. Job `50515488` failed while resolving a Slurm-spooled helper path. The
one permitted retry, job `50515705`, failed because the batch environment had not loaded Conda.
Both failures occurred before the simulator, robot, anatomy, or ultrasound pipeline started. No
observation, transition, reset-cycle, clean simulator exit, or PPO claim is available.

Verified current state:

- Project: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym`.
- Runtime: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime` (23,499 MiB).
- Execution base commit: `a34ebe3ceee18a8637f7d71ec0d4265260332742`.
- Isaac Lab: `21f7136325136ca3f6ca4e0a8125edffe5c24f7e` (`v2.1.0`).
- Python 3.10.22, Isaac Sim 4.5.0.0, PyTorch 2.5.1+cu121, and skrl 1.4.2.
- Full isolated CPU preflight: **PASS** with `PYTHONNOUSERSITE=1`.
- Asset and model archives match the pinned byte counts and SHA-256 values.
- Both GPU jobs used account `anest-dept`, four CPUs, 30 GB, and one allocated L4 on
  `c1105a-s20`. The driver was not captured because both launchers stopped before `nvidia-smi`.

Infrastructure fixes include the GLIBC `pipefail` issue, pinned bootstrap tooling, Python
user-site isolation, CUDA local-version validation, Slurm-spooled helper resolution, explicit
`conda/25.7.0` module loading, 30 GB memory, 90-minute PPO walltime, 32-step smoke diagnostics,
and per-job reproducibility/resource capture. The final launcher state passes shell syntax,
Python compilation, static preflight, Git diff, and smoke/PPO dry-runs. It has not received
another GPU run because Phase 4 allowed only one retry.

Evidence:

- Machine report: `reports/phase4_gpu_smoke_ppo.json`.
- Preflight: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/records/preflight.json`.
- Logs: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/`.

Next exact experiment: after explicit authorization for a new execution phase, run exactly one
corrected smoke-only job with one L4, four CPUs, 30 GB, seed `20261106`, 32 bounded actions,
`conv` ultrasound, headless mode, and visualization disabled. PPO remains gated on a complete
`SMOKE_PASS` with a zero Slurm exit code and clean simulator shutdown.

No policy usefulness, generalization, clinical safety, or MBB performance conclusion is supported.

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
