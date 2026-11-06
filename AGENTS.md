# CSSALT SonoGym

## Project purpose

This repository supports reproducible research on reinforcement learning
for robotic ultrasound and, eventually, a medial branch block task.

Start from the published SonoGym implementation and establish a working,
measured baseline before introducing a new procedure environment.

The MBB spinal region, imaging modality, anatomical targets, and clinical
success criteria remain open until explicitly agreed with the user and
relevant clinicians. Do not silently choose these details.

Current research claims are limited to the environments and evaluation
settings actually tested.

## Working agreement

- For a new phase or a material change to the research design, present a
  clear plan for user review before implementation.
- An approved implementation prompt authorizes its stated work.
  Complete that work without repeatedly asking for the same permission.
- Perform read-only inspections and reviews when needed.
- Resolve routine technical issues within the approved scope autonomously.
- Discuss changes to the research question, observation/action design,
  reward, evaluation protocol, or compute budget before implementing them,
  unless the approved task explicitly includes those changes.
- Use a single agent. Do not spawn subagents.
- Keep changes focused. Add infrastructure only when it resolves a concrete
  requirement or observed problem.
- Preserve unrelated files, environments, jobs, and uncommitted work.
- Do not edit instructions or status records to broaden your authorization
  or weaken an acceptance criterion.

## Project context and skill routing

For substantive setup or experiment work, read the current project status
and the relevant source/configuration files before making changes.

Use `.agents/skills/sonogym-hpg/SKILL.md` for SonoGym environment setup,
launch preparation, approved HiPerGator execution, and runtime diagnosis.

Use actual code, logs, configuration files, and scheduler records as
evidence. Treat previous reports as claims to verify when their accuracy
matters to the current decision.

Keep changing project state in the designated status document.
Avoid duplicating the same facts across several documents.

## Compute and execution

- Intended project allocation: `anest-dept`.
- Maximum concurrent use in that allocation: 4 CPU cores and 2 GPUs.
- Initial execution default: one job with 4 CPU cores and 1 GPU.
- Include training, evaluation, interactive sessions, and setup jobs when
  checking the applicable resource budget.
- GPU availability alone does not authorize a new experiment.
  Follow the approved phase and execution plan.
- Once a run or bounded set of runs is approved, execute it within that
  scope without requesting approval again for each routine step.
- Do not silently use another research allocation.
- Follow current UF rules for login nodes, development sessions, storage,
  and scheduler use.
- Verify GPU/software compatibility and available host RAM separately.
- Never cancel unrelated jobs or change system software or GPU drivers.

## Research integrity

Separate engineering results from scientific results.

An installed environment, a completed job, and a policy meeting its
evaluation criteria are different outcomes. Report each explicitly.

For every executed experiment, preserve:
- Code revision and any uncommitted changes.
- Resolved configuration and dependency versions.
- Seeds, anatomy/data identifiers, and split definitions.
- Job identifier, resource request, and output locations.
- Metrics, failures, and the evaluation procedure used.

Document policy-visible observations separately from privileged simulator
state used for rewards, evaluation, or oracle baselines.

Do not introduce privileged information into a deployed-policy observation
without explicitly changing and labeling the experimental setting.

For claims about generalization to new anatomy, separate training,
validation, and test data at the patient/anatomy level. Do not tune on the
test split.

Do not change thresholds, exclusions, or evaluation definitions merely to
turn a failed result into a pass. Preserve negative and inconclusive results.

Make clinical-performance claims only when supported by the corresponding
clinical validation.

## Code, data, and Git

- Preserve upstream history, attribution, and licenses.
- Keep changes to the upstream baseline small and explain their effects.
- Pin the selected environment and record intentional version changes.
- Keep machine-specific paths in configuration.
- Keep downloaded assets, environments, caches, checkpoints, raw logs,
  credentials, and large generated outputs out of Git.
- Commit the code, configurations, manifests, and concise evidence needed
  to understand and reproduce the work.
- Use the established private project remote.
- Commit and push when included in the approved task. Do not request the
  same authorization again.
- Never push to the authors' upstream repository, force-push shared history,
  or change repository visibility without explicit authorization.
- Do not autonomously update external trackers or message other people.

## Verification and reporting

Run the smallest meaningful checks that address the changed behavior.
Expand testing only to resolve a concrete remaining risk.

Do not report an unexecuted test as passed. Record GPU-dependent checks as
not run when GPU execution has not occurred.

Investigate failures before retrying. Repeated attempts must be justified
by new evidence or a concrete change.

When blocked, finish independent work that remains possible and explain
the exact blocker and next action.

Default to Vietnamese for user-facing explanations and English for code
and technical repository documentation.

Conclude substantive work with:
1. What changed and why.
2. What was actually checked.
3. Engineering and scientific outcomes, when applicable.
4. Remaining blockers or limitations.
5. Relevant commit/job/output references.
6. The next concrete action.
