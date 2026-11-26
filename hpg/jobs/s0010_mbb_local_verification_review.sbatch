#!/usr/bin/env bash
#SBATCH --job-name=sg_mbb_local_review
#SBATCH --partition=hpg-default
#SBATCH --qos=anest-dept-b
#SBATCH --account=anest-dept
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --time=00:08:00

set -euo pipefail
runtime="${1:?pinned runtime}"
session="${2:?sealed capture}"
protocol="${3:?frozen acquisition protocol}"
archived="${4:?archived public recordings}"
[[ "$SLURM_JOB_ACCOUNT" = anest-dept ]]
[[ "$SLURM_CPUS_PER_TASK" -le 4 ]]
[[ "${SLURM_GPUS_ON_NODE:-0}" = 0 ]]
export CUDA_VISIBLE_DEVICES= MPLBACKEND=Agg MPLCONFIGDIR="${TMPDIR:-/tmp}/mbb-local-review-$SLURM_JOB_ID"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$PWD:$PWD/source/spinal_surgery/spinal_surgery/evaluation:$PWD/workflows/two_stage_navigation:$PWD/workflows/perception_s0010:$runtime/envs/stage1-python"
py="$runtime/envs/sonogym/bin/python"
"$py" -m workflows.mbb_local_active.evaluate --session "$session" --protocol-path "$protocol" \
  --archived "$archived" --output "$session/evaluation"
"$py" -m workflows.mbb_local_active.review --session "$session" --evidence-dir "$session/evaluation"
