#!/usr/bin/env bash
#SBATCH --job-name=sg_mbb_observed_surface
#SBATCH --partition=hpg-default
#SBATCH --qos=anest-dept-b
#SBATCH --account=anest-dept
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --time=00:12:00

set -euo pipefail
runtime="${1:?Specify the pinned SonoGym runtime}"
shift
[[ "$SLURM_JOB_ACCOUNT" = anest-dept ]]
[[ "${SLURM_CPUS_PER_TASK:-0}" -le 4 ]]
[[ "${SLURM_GPUS_ON_NODE:-0}" = 0 ]]
export PYTHONPATH="$PWD:$PWD/source/spinal_surgery/spinal_surgery/evaluation:$PWD/workflows/two_stage_navigation:$PWD/workflows/perception_s0010:$runtime/envs/stage1-python"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg MPLCONFIGDIR="${TMPDIR:-/tmp}/mbb-observed-surface-$SLURM_JOB_ID"
git rev-parse HEAD
git status --short
"$runtime/envs/sonogym/bin/python" "$@"
