#!/usr/bin/env bash
#SBATCH --job-name=sg_mbb_identifiability
#SBATCH --partition=hpg-default
#SBATCH --qos=anest-dept-b
#SBATCH --account=anest-dept
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --time=00:12:00

set -euo pipefail

manifest="${1:?Specify the sealed manifest}"
output="${2:?Specify a new output directory}"
runtime="${3:?Specify the pinned SonoGym runtime}"
[[ "$SLURM_JOB_ACCOUNT" = anest-dept ]]
[[ "${SLURM_CPUS_PER_TASK:-0}" -le 4 ]]
[[ "${SLURM_GPUS_ON_NODE:-0}" = 0 ]]
[[ -z "$(git status --short)" ]]
export PYTHONPATH="$PWD:$PWD/source/spinal_surgery/spinal_surgery/evaluation:$PWD/workflows/two_stage_navigation:$PWD/workflows/perception_s0010:$runtime/envs/stage1-python"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg MPLCONFIGDIR="${TMPDIR:-/tmp}/mbb-identifiability-$SLURM_JOB_ID"
py="$runtime/envs/sonogym/bin/python"
git rev-parse HEAD
"$py" -m workflows.mbb_target_identifiability.analyze --manifest "$manifest" --output "$output"
"$py" -m workflows.mbb_target_identifiability.figures --results "$output"
