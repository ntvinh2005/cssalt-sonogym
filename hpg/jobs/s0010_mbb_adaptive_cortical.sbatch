#!/usr/bin/env bash
#SBATCH --job-name=sg_mbb_adaptive
#SBATCH --partition=hpg-default
#SBATCH --qos=anest-dept-b
#SBATCH --account=anest-dept
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:06:00

set -euo pipefail
runtime="${1:?runtime directory required}"
study="${2:?prior study directory required}"
output="${3:?unused output path required}"
[[ "$SLURM_JOB_ACCOUNT" = anest-dept ]]
[[ "$SLURM_CPUS_PER_TASK" -le 4 ]]
[[ "${SLURM_GPUS_ON_NODE:-0}" = 0 ]]
export CUDA_VISIBLE_DEVICES= MPLBACKEND=Agg MPLCONFIGDIR="${TMPDIR:-/tmp}/mbb-adaptive-$SLURM_JOB_ID"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$PWD:$PWD/source/spinal_surgery/spinal_surgery/evaluation:$PWD/workflows/two_stage_navigation:$PWD/workflows/perception_s0010:$runtime/envs/stage1-python"
py="$runtime/envs/sonogym/bin/python"
git rev-parse HEAD
git status --short
"$py" -m workflows.mbb_adaptive_cortical.replay --source "$study/frozen-run" \
    --frozen-config experiments/unity_two_stage_transfer/mbb_direct_sonographic_v2.json \
    --adaptive-config experiments/unity_two_stage_transfer/mbb_adaptive_cortical_tracking_v1.json \
    --review-zip "$study/s0010-sonographic-observability-visual-review.zip" --destination "$output"
