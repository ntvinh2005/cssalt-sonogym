#!/usr/bin/env bash
#SBATCH --job-name=sg_ind_replay
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept-b
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:10:00
set -euo pipefail
source "$SONOGYM_CONFIG_FILE"
[[ "$SLURM_JOB_ACCOUNT" == anest-dept && "${SLURM_GPUS_ON_NODE:-0}" -eq 0 ]]
[[ "$(squeue -A anest-dept --states=RUNNING,COMPLETING,CONFIGURING -h -o '%C' | awk '{s+=$1} END{print s+0}')" -le 4 ]]
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$PWD/source/spinal_surgery/spinal_surgery/evaluation:$PWD"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
destination="$SIMPLE_MAPPING_OUTPUT_ROOT/replay-$SLURM_JOB_ID"
mkdir "$destination"
git rev-parse HEAD > "$destination/git-commit.txt"
scontrol show job "$SLURM_JOB_ID" > "$destination/slurm-job.txt"
python="$SONOGYM_ENV_PREFIX/bin/python"
"$python" -m unittest -q tests.test_simple_landmark_mapping tests.test_incremental_landmark_map tests.test_acquisition_integrity >"$destination/tests.log" 2>&1
"$python" workflows/two_stage_navigation/run_simple_landmark_mapping.py \
  --run-root "$SIMPLE_MAPPING_RUN_ROOT" --protocol "$SIMPLE_MAPPING_PROTOCOL" \
  --label-conversion source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml \
  --output "$destination/results" --batch-size 8
"$python" workflows/two_stage_navigation/replay_independent_sap.py \
  --protocol "$SIMPLE_MAPPING_PROTOCOL" --baseline "$destination/results" --output "$destination/revised_sap.json"
