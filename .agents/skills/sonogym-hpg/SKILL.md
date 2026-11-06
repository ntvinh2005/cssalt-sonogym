---
name: sonogym-hpg
description: >
  Prepare, validate, run, and diagnose SonoGym workflows on UF HiPerGator.
  Use for environment installation, asset preparation, launch readiness,
  approved Slurm jobs, and simulator or PPO runtime failures.
  Do not use this skill to define clinical MBB technique or to authorize
  experiments beyond the user's approved scope.
---

# SonoGym on HiPerGator

## Establish the task

Read the repository AGENTS.md, the current approved phase, and the relevant
environment/configuration records.

Identify whether the request is:
- Preparation without GPU execution.
- Execution of an approved experiment.
- Diagnosis of an existing failure.

Use the latest user instruction to resolve stale project-state notes.
Do not infer execution authorization from the presence of a GPU, a job
script, or a status flag.

Inspect existing scripts before adding new ones. Reuse the repository's
documented entry points where possible.

## Prepare or repair the environment

1. Inspect the actual host, operating system, GLIBC, available modules,
   writable storage, and relevant scheduler associations.
2. Read the pinned dependency and source revisions.
3. For the initial published baseline, use Python 3.10, Isaac Sim 4.5.0,
   and Isaac Lab 2.1.0 as the starting stack.
4. Check platform compatibility before installing. The Isaac Sim 4.5 pip
   route requires GLIBC >= 2.34.
5. Use the existing supported installation route. Prefer an isolated native
   environment when compatible; use an appropriate Apptainer environment
   when the host userspace requires it.
6. Resolve dependency conflicts without silently replacing the approved
   simulator stack with newer major versions.
7. Record the versions actually installed and make setup safe to rerun.

Use package metadata and CPU-safe checks during preparation.
Do not initialize Isaac Sim or a GUI simply to prove that a package exists.

Record first-launch downloads, shader compilation, and GPU compatibility
checks that remain untested.

## Prepare assets and configuration

Use the published asset source and record its revision and checksums.
Store large assets and model weights outside Git.

Verify paths against the actual code and configuration loaders.
Preserve tracked Python files within the source assets directory.

For the initial navigation baseline:
- Task: `Isaac-robot-US-guidance-v0`.
- Model-based ultrasound: `sim.us = 'conv'`.
- Fixed seed.
- Headless execution.
- Visualization disabled before environment construction.
- Local logging without mandatory external authentication.

Verify how overrides are applied. Do not invent Hydra keys or CLI flags.
Do not assume that `--headless` disables every visualization component.

Keep policy observations, reward calculations, and simulator dynamics
unchanged unless the approved task includes modifying them.

## Check launch readiness

Before a future submission, verify:
- The intended account and QOS are usable.
- The requested GPU is compatible with the pinned stack.
- CPU, GPU, RAM, and walltime requests fit the approved budget.
- Relevant existing jobs will not cause the concurrency cap to be exceeded.
- The environment, assets, configuration, and output paths resolve.
- Logging directories exist and the job uses appropriate working storage.
- Thread and worker counts match the allocated CPUs.
- The launch command is recorded and its dry-run path does not submit work.

Respect Slurm-assigned GPU visibility.

Use existing nonallocating preflight checks where available.
Describe unverified runtime assumptions instead of declaring them passed.

## Execute an approved run

Follow the approved experiment plan and its resource and run-count limits.

For an unvalidated environment:
1. Run the smallest real simulator smoke test.
2. Inspect reset, stepping, observation/reward validity, and shutdown.
3. Run a bounded PPO pilot when included in the approved scope.
4. Inspect logs and checkpoint creation before progressing to larger runs.

For an already validated setup, rerun only the checks needed for the
changes made. Do not repeat a full validation campaign automatically.

Record:
- Revision, resolved configuration, seed, and data/anatomy identifiers.
- GPU model, relevant software versions, and requested resources.
- Slurm job identifier, exit status, runtime, and output paths.
- Available peak memory and throughput measurements.

Distinguish vector environment steps, collected transitions, and PPO updates
when reporting training budgets or speed.

Treat a pilot as an execution check. Evaluate learning quality using the
approved scientific protocol.

## Diagnose failures

Read the actual failing command, error, logs, and resource records first.

Determine whether the failure concerns:
- Access or scheduler configuration.
- Environment or dependency compatibility.
- Asset/path resolution.
- Simulator initialization or rendering.
- Resource exhaustion.
- Training or evaluation behavior.

Change one likely cause at a time where practical.

Do not resubmit the same failing GPU job without a concrete diagnostic
reason. Stop repetitive attempts when they stop producing new information.

Keep scientific changes separate from installation/runtime fixes.
A reward or observation change requires the appropriate experimental scope.

## Report readiness and evidence

Use check statuses:
- PASS: the stated check was executed successfully.
- FAIL: the stated check was executed and failed.
- NOT_RUN: the check has not been executed; state why.
- BLOCKED: an identified prerequisite prevents progress.

Use `PREPARED_GPU_UNTESTED` for setup only when installation, required
downloaded assets, and non-GPU checks are complete.

Report simulator smoke-test results and PPO-pilot results separately.
Do not infer either result from package installation or syntax checks.

Update the designated project-status document after material progress.
Keep it concise and avoid copying entire logs into it.

Finish with the actual outcome, evidence locations, unresolved issues,
and exact commands for the next authorized step.
