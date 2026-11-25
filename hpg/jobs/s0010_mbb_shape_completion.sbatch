#!/usr/bin/env bash
#SBATCH --job-name=sg_mbb_shape_completion
#SBATCH --partition=hpg-default
#SBATCH --qos=anest-dept-b
#SBATCH --account=anest-dept
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00

set -euo pipefail
runtime="${1:?Specify the pinned SonoGym runtime}"
output="${2:?Specify a new output directory}"
phase="${3:?Specify A or B}"
action="${4:-all}"
[[ "$SLURM_JOB_ACCOUNT" = anest-dept ]]
[[ "${SLURM_CPUS_PER_TASK:-0}" -le 4 ]]
[[ "${SLURM_GPUS_ON_NODE:-0}" = 0 ]]
export PYTHONPATH="$PWD:$PWD/source/spinal_surgery/spinal_surgery/evaluation:$PWD/workflows/two_stage_navigation:$PWD/workflows/perception_s0010:$runtime/envs/stage1-python"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg MPLCONFIGDIR="${TMPDIR:-/tmp}/mbb-shape-completion-$SLURM_JOB_ID"
protocol="experiments/unity_two_stage_transfer/s0010_mbb_shape_completion_v1.json"
python_exe="$runtime/envs/sonogym/bin/python"
git rev-parse HEAD
git status --short
if [[ "$action" != score ]]; then
  "$python_exe" -m workflows.mbb_shape_completion.prepare --protocol "$protocol" --phase "$phase" --output "$output"
  "$python_exe" -m workflows.mbb_shape_completion.predict --protocol "$protocol" --inputs "$output/inputs" --output "$output/predictions"
fi
if [[ "$action" != predict ]]; then
  "$python_exe" -m workflows.mbb_shape_completion.score --protocol "$protocol" --phase "$phase" --output "$output"
fi
