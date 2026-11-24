#!/usr/bin/env bash
#SBATCH --job-name=sg_simple_landmarks
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept-b
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:10:00

set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?}"
: "${SIMPLE_MAPPING_RUN_ROOT:?}"
: "${SIMPLE_MAPPING_PROTOCOL:?}"
: "${SIMPLE_MAPPING_OUTPUT_ROOT:?}"
experiment_checkout="$PWD"
source "$SONOGYM_CONFIG_FILE"

[[ "$SLURM_JOB_ACCOUNT" == anest-dept && "$SLURM_CPUS_PER_TASK" -le 4 ]]
[[ "${SLURM_GPUS_ON_NODE:-0}" -eq 0 ]]
[[ "$PWD" == /blue/* ]]
running_cpus="$(squeue -A anest-dept --states=RUNNING,COMPLETING,CONFIGURING -h -o '%C' | awk '{sum+=$1} END{print sum+0}')"
[[ "$running_cpus" -le 4 ]]

export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$experiment_checkout/source/spinal_surgery/spinal_surgery/evaluation:$experiment_checkout"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
python="$SONOGYM_ENV_PREFIX/bin/python"
destination="$SIMPLE_MAPPING_OUTPUT_ROOT/$SLURM_JOB_ID"
mkdir "$destination"
export MPLCONFIGDIR="$destination/matplotlib"
date --iso-8601=seconds > "$destination/start-time.txt"
scontrol show job "$SLURM_JOB_ID" > "$destination/slurm-job.txt"
git -C "$experiment_checkout" rev-parse HEAD > "$destination/git-commit.txt"
git -C "$experiment_checkout" status --short > "$destination/git-status.txt"
"$python" -m pip freeze > "$destination/pip-freeze.txt"
"$python" -m unittest -q tests.test_simple_landmark_mapping tests.test_incremental_landmark_map \
  > "$destination/tests.log" 2>&1
"$python" workflows/two_stage_navigation/run_simple_landmark_mapping.py \
  --run-root "$SIMPLE_MAPPING_RUN_ROOT" --protocol "$SIMPLE_MAPPING_PROTOCOL" \
  --label-conversion source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml \
  --output "$destination/results" --batch-size 8
date --iso-8601=seconds > "$destination/end-time.txt"
