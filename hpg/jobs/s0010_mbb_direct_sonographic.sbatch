#!/usr/bin/env bash
#SBATCH --job-name=sg_direct_sonographic
#SBATCH --partition=hpg-default
#SBATCH --qos=anest-dept-b
#SBATCH --account=anest-dept
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --time=00:12:00

set -euo pipefail
root="$PWD"
runtime="${1:?Specify the pinned SonoGym runtime}"
source_study="${2:?Specify the previous observed-bone study}"
cache="${3:?Specify the frozen U-Net probability cache}"
output="${4:?Specify an unused output directory}"
profile="$root/experiments/unity_two_stage_transfer/mbb_direct_sonographic_v2.json"

[[ "$SLURM_JOB_ACCOUNT" = anest-dept ]]
[[ "${SLURM_CPUS_PER_TASK:-0}" -le 4 ]]
[[ "${SLURM_GPUS_ON_NODE:-0}" = 0 ]]
mkdir "$output"
export PYTHONPATH="$root:$root/source/spinal_surgery/spinal_surgery/evaluation:$root/workflows/two_stage_navigation:$root/workflows/perception_s0010:$runtime/envs/stage1-python"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg MPLCONFIGDIR="${TMPDIR:-/tmp}/mbb-direct-sonographic-$SLURM_JOB_ID"
export CUDA_VISIBLE_DEVICES=""
git rev-parse HEAD
git status --short

py="$runtime/envs/sonogym/bin/python"
"$py" -m workflows.mbb_direct_sonographic.prepare --source "$source_study" \
  --cache "$cache" --config "$profile" --output "$output/inputs"
"$py" -m workflows.mbb_direct_sonographic.replay --inputs "$output/inputs" \
  --config "$profile" --output "$output/replay"
"$py" -m workflows.mbb_direct_sonographic.review --inputs "$output/inputs" \
  --replay "$output/replay" --config "$profile" --figures "$output/figures" \
  --reference-audit "$source_study/reference/reference.json"
