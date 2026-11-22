#!/usr/bin/env bash
#SBATCH --job-name=sg_graph_eval
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept-b
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/%x_%j.out

set -euo pipefail
: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported}"
: "${CONFIRMATION_JOB_ID:?CONFIRMATION_JOB_ID was not exported}"
: "${LANDMARK_GRAPH_PROTOCOL:?LANDMARK_GRAPH_PROTOCOL was not exported}"
: "${EXPECTED_PROTOCOL_SHA:?EXPECTED_PROTOCOL_SHA was not exported}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck disable=SC1090
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_l5_landmark_graph_confirmation/$CONFIRMATION_JOB_ID"
PREDICTION="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_semantic_review_v3/52078067/raw_prediction/s0010.nii.gz"
SEGMENTATION="$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
LABEL_CONVERSION="$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery:$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/evaluation:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

[[ -d "$RUN_DIR/sessions" ]] || { echo "capture run is missing" >&2; exit 2; }
[[ "$(sha256sum "$LANDMARK_GRAPH_PROTOCOL" | cut -d' ' -f1)" == "$EXPECTED_PROTOCOL_SHA" ]] || {
  echo "protocol SHA-256 changed" >&2
  exit 2
}
mkdir -p "$RUN_DIR/results/evaluation" "$RUN_DIR/cpu_evaluation_context"
capture_cpu_run_context "$RUN_DIR/cpu_evaluation_context"
"$PYTHON" -m unittest -q \
  tests.test_acquisition_integrity tests.test_ultrasound_spatial_map \
  tests.test_two_stage_contracts tests.test_s0010_landmark_graph_confirmation \
  > "$RUN_DIR/results/cpu-evaluation-tests.log" 2>&1
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/evaluate_s0010_landmark_graph_confirmation.py" \
  --protocol "$LANDMARK_GRAPH_PROTOCOL" --run-root "$RUN_DIR" \
  --prediction "$PREDICTION" --segmentation "$SEGMENTATION" \
  --label-conversion "$LABEL_CONVERSION" --output-dir "$RUN_DIR/results/evaluation" \
  > "$RUN_DIR/results/cpu-evaluation.log" 2>&1
capture_cpu_run_end "$RUN_DIR/cpu_evaluation_context"
find -L "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
