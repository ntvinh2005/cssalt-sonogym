#!/usr/bin/env bash
#SBATCH --job-name=sg_simple_sweep
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=30G
#SBATCH --time=00:12:00

set -euo pipefail
: "${SONOGYM_CONFIG_FILE:?}"
: "${SIMPLE_MAPPING_RUN_ROOT:?}"
: "${SIMPLE_MAPPING_PROTOCOL:?}"
: "${SIMPLE_MAPPING_OUTPUT_ROOT:?}"
experiment_checkout="$PWD"
running_cpus="$(squeue -A anest-dept --states=RUNNING,COMPLETING,CONFIGURING -h -o '%C' | awk '{sum+=$1} END{print sum+0}')"
[[ "$running_cpus" -le 4 ]]
source "$experiment_checkout/hpg/jobs/common.sh"
SONOGYM_PROJECT_DIR="$experiment_checkout"
SONOGYM_GUIDANCE_CONFIG="$experiment_checkout/config/guidance_baseline.yaml"
export SONOGYM_GUIDANCE_CONFIG
export PYTHONPATH="$experiment_checkout/source/spinal_surgery:$SONOGYM_RUNTIME_DIR/envs/stage1-python:$experiment_checkout/source/spinal_surgery/spinal_surgery/evaluation:$experiment_checkout"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/sonogym-simple-sweep-$SLURM_JOB_ID"

destination="$SIMPLE_MAPPING_OUTPUT_ROOT/$SLURM_JOB_ID"
capture_run_context "$destination"
"$PYTHON" -m unittest -q tests.test_simple_landmark_mapping tests.test_acquisition_integrity \
  > "$destination/context/tests.log" 2>&1
"$PYTHON" workflows/two_stage_navigation/capture_simple_landmark_sweep.py \
  --protocol "$SIMPLE_MAPPING_PROTOCOL" --seed 2026120500 --stage1-session cf7_00 \
  --stage1-actor "$SIMPLE_MAPPING_RUN_ROOT/stage1/results/actor_results.json" \
  --output "$destination/fresh_s0010_00" --headless --device cuda:0 \
  --evaluator-segmentation "$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz" \
  > "$destination/actor.log" 2>&1
"$PYTHON" workflows/two_stage_navigation/evaluate_simple_landmark_sweep.py \
  --session "$destination/fresh_s0010_00" --protocol "$SIMPLE_MAPPING_PROTOCOL" \
  --label-conversion source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml
capture_run_end "$destination"
