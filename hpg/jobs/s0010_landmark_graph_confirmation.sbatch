#!/usr/bin/env bash
#SBATCH --job-name=sg_graph_confirm
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/%x_%j.out

set -euo pipefail
: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported}"
: "${LANDMARK_GRAPH_PROTOCOL:?LANDMARK_GRAPH_PROTOCOL was not exported}"
: "${EXPECTED_PROTOCOL_SHA:?EXPECTED_PROTOCOL_SHA was not exported}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck disable=SC1090
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common.sh"

PROTOCOL="$LANDMARK_GRAPH_PROTOCOL"
PROTOCOL_SEAL="${PROTOCOL%.yaml}.sha256"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_l5_landmark_graph_confirmation/$SLURM_JOB_ID"
PREDICTION="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_semantic_review_v3/52078067/raw_prediction/s0010.nii.gz"
SEGMENTATION="$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
LABEL_CONVERSION="$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery:$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/evaluation:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$SONOGYM_CACHE_DIR/matplotlib"

[[ "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)" == "$EXPECTED_PROTOCOL_SHA" ]] || {
  echo "protocol SHA-256 changed" >&2
  exit 2
}
mkdir -p "$RUN_DIR"/{sessions,results,metrics,context} "$MPLCONFIGDIR"
capture_run_context "$RUN_DIR"
cp "$PROTOCOL" "$PROTOCOL_SEAL" "$RUN_DIR/context/"
(cd "$RUN_DIR/context" && sha256sum -c "$(basename "$PROTOCOL_SEAL")")
"$PYTHON" -m unittest -q \
  tests.test_acquisition_integrity tests.test_ultrasound_spatial_map \
  tests.test_two_stage_contracts tests.test_s0010_landmark_graph_confirmation \
  > "$RUN_DIR/results/preflight-tests.log" 2>&1

nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
  --format=csv --loop=2 > "$RUN_DIR/metrics/gpu-telemetry.csv" &
TELEMETRY_PID=$!
trap 'kill "$TELEMETRY_PID" 2>/dev/null || true' EXIT

for session_id in \
  s0010l5graph_00 s0010l5graph_01 s0010l5graph_02 \
  s0010l5graph_03 s0010l5graph_04 s0010l5graph_05; do
  SESSION_DIR="$RUN_DIR/sessions/$session_id"
  mkdir -p "$SESSION_DIR"
  "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/collect_stage2_dataset.py" \
    --protocol "$PROTOCOL" --session-id "$session_id" --output-dir "$SESSION_DIR" \
    --assets-data-dir "$SONOGYM_ASSETS_DATA_DIR" --headless --device cuda:0 \
    > "$SESSION_DIR/launcher.log" 2>&1
  "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/verify_exact_pose_capture.py" \
    --protocol "$PROTOCOL" --session-dir "$SESSION_DIR" \
    --assets-data-dir "$SONOGYM_ASSETS_DATA_DIR" \
    --output "$SESSION_DIR/verification_report.json" \
    > "$SESSION_DIR/verifier.log" 2>&1
  "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/validate_actor_6dof_capture.py" \
    --protocol "$PROTOCOL" --run-root "$RUN_DIR" --session-id "$session_id" \
    --output "$SESSION_DIR/actor_6dof_validation.json" \
    > "$SESSION_DIR/actor_6dof_validator.log" 2>&1
done

"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/evaluate_s0010_landmark_graph_confirmation.py" \
  --protocol "$PROTOCOL" --run-root "$RUN_DIR" \
  --prediction "$PREDICTION" --segmentation "$SEGMENTATION" \
  --label-conversion "$LABEL_CONVERSION" --output-dir "$RUN_DIR/results/evaluation" \
  > "$RUN_DIR/results/evaluation.log" 2>&1

kill "$TELEMETRY_PID" 2>/dev/null || true
wait "$TELEMETRY_PID" 2>/dev/null || true
trap - EXIT
capture_run_end "$RUN_DIR"
du -sbL "$RUN_DIR" > "$RUN_DIR/metrics/storage-bytes.txt"
find -L "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
