#!/usr/bin/env bash
#SBATCH --job-name=sg_l4mb_obs_r1
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
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck disable=SC1090
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common.sh"

PARENT_RUN="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_l4mb_target_observability_v1/52278401"
PROTOCOL="$PARENT_RUN/context/s0010_l4mb_target_observability_v1.yaml"
PROTOCOL_SEAL="$PARENT_RUN/context/s0010_l4mb_target_observability_v1.sha256"
EXPECTED_PROTOCOL_SHA="089cc97563c90847d91cc172daf9dd513a670713e75372aa0b660e45f68b540d"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_l4mb_target_observability_v1_resume/$SLURM_JOB_ID"
PREDICTION="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_semantic_review_v3/52078067/raw_prediction/s0010.nii.gz"
SEGMENTATION="$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
LABEL_CONVERSION="$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery:$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/evaluation:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$SONOGYM_CACHE_DIR/matplotlib"

[[ -f "$PROTOCOL" && -f "$PROTOCOL_SEAL" ]] || {
  echo "sealed parent protocol is unavailable" >&2
  exit 2
}
[[ "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)" == "$EXPECTED_PROTOCOL_SHA" ]] || {
  echo "parent protocol SHA-256 changed" >&2
  exit 2
}
[[ "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$PARENT_RUN/sessions/s0010l4mbobs_00/verification_report.json")" == "PROJECTION_PARITY_PASS" ]] || {
  echo "completed parent session lacks replay verification" >&2
  exit 2
}

mkdir -p "$RUN_DIR"/{sessions,results,metrics,context} "$MPLCONFIGDIR"
capture_run_context "$RUN_DIR"
cp "$PROTOCOL" "$PROTOCOL_SEAL" "$RUN_DIR/context/"
cp -a "$PARENT_RUN/sessions/s0010l4mbobs_00" "$RUN_DIR/sessions/"
(cd "$RUN_DIR/context" && sha256sum -c "$(basename "$PROTOCOL_SEAL")")

"$PYTHON" -m unittest -q \
  tests.test_s0010_l4mb_target_observability_plan \
  tests.test_s0010_l4mb_target_observability_protocol \
  tests.test_s0010_l4mb_target_observability_evaluation \
  tests.test_l4_medial_branch_target \
  tests.test_acquisition_integrity \
  tests.test_s0010_sap_observability \
  > "$RUN_DIR/results/preflight-tests.log" 2>&1

nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
  --format=csv --loop=2 > "$RUN_DIR/metrics/gpu-telemetry.csv" &
TELEMETRY_PID=$!
trap 'kill "$TELEMETRY_PID" 2>/dev/null || true' EXIT

capture_session() {
  local session_id="$1"
  local session_dir="$RUN_DIR/sessions/$session_id"
  mkdir -p "$session_dir"
  "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/collect_stage2_dataset.py" \
    --protocol "$PROTOCOL" --session-id "$session_id" --output-dir "$session_dir" \
    --assets-data-dir "$SONOGYM_ASSETS_DATA_DIR" --headless --device cuda:0 \
    > "$session_dir/launcher.log" 2>&1
  "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/verify_exact_pose_capture.py" \
    --protocol "$PROTOCOL" --session-dir "$session_dir" \
    --assets-data-dir "$SONOGYM_ASSETS_DATA_DIR" \
    --output "$session_dir/verification_report.json" \
    > "$session_dir/verifier.log" 2>&1
}

capture_session s0010l4mbobs_01
capture_session s0010l4mbobs_02

"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/evaluate_s0010_l4mb_target_observability.py" \
  --protocol "$PROTOCOL" --run-root "$RUN_DIR" \
  --prediction "$PREDICTION" --segmentation "$SEGMENTATION" \
  --label-conversion "$LABEL_CONVERSION" \
  --output-dir "$RUN_DIR/results/evaluation" \
  > "$RUN_DIR/results/evaluation.log" 2>&1

kill "$TELEMETRY_PID" 2>/dev/null || true
wait "$TELEMETRY_PID" 2>/dev/null || true
trap - EXIT
capture_run_end "$RUN_DIR"
du -sb "$RUN_DIR" > "$RUN_DIR/metrics/storage-bytes.txt"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
