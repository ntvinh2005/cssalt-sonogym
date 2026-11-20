#!/usr/bin/env bash
#SBATCH --job-name=sg_sap_obs_v2
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

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit command}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common.sh"

PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_sap_observability_protocol_v2r1.yaml"
PROTOCOL_SEAL="${PROTOCOL%.yaml}.sha256"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_observability_v2r1/$SLURM_JOB_ID"
REFERENCE_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_mbb_single_anatomy_poc_v3/52039283/stage2"
PREDICTION="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_semantic_review_v3/52078067/raw_prediction/s0010.nii.gz"
SEGMENTATION="$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$SONOGYM_CACHE_DIR/matplotlib"
mkdir -p "$RUN_DIR"/{sessions,results,metrics,context,review} "$MPLCONFIGDIR"
capture_run_context "$RUN_DIR"
cp "$PROTOCOL" "$PROTOCOL_SEAL" "$RUN_DIR/context/"
(cd "$RUN_DIR/context" && sha256sum -c s0010_sap_observability_protocol_v2r1.sha256)

"$PYTHON" -m unittest -v \
    tests.test_s0010_sap_observability \
    tests.test_s0010_sap_observability_contract \
    tests.test_s0010_sap_observability_followup \
    tests.test_ultrasound_plane_geometry \
    tests.test_acquisition_integrity \
    > "$RUN_DIR/results/preflight-tests.log" 2>&1

"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/run_s0010_sap_observability.py" plan \
    --protocol "$PROTOCOL" --reference-root "$REFERENCE_ROOT" \
    --prediction "$PREDICTION" --segmentation "$SEGMENTATION" \
    --output "$RUN_DIR/results/waypoint_plan.json" \
    > "$RUN_DIR/results/waypoint-plan.log" 2>&1

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

capture_session s0010sapobs2_00
capture_session s0010sapobs2_01

"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/run_s0010_sap_observability.py" prepare \
    --protocol "$PROTOCOL" --run-root "$RUN_DIR" --prediction "$PREDICTION" \
    --segmentation "$SEGMENTATION" \
    --output-root "$RUN_DIR/review/s0010_right_l4l5_sap_observability_v2" \
    > "$RUN_DIR/results/review-prepare.log" 2>&1

kill "$TELEMETRY_PID" 2>/dev/null || true
wait "$TELEMETRY_PID" 2>/dev/null || true
trap - EXIT
capture_run_end "$RUN_DIR"
du -sb "$RUN_DIR" > "$RUN_DIR/metrics/storage-bytes.txt"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
