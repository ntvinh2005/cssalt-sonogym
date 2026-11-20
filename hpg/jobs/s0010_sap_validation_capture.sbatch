#!/usr/bin/env bash
#SBATCH --job-name=sg_sap_validate
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
: "${SAP_VALIDATION_PHASE:?SAP_VALIDATION_PHASE must be development or holdout}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common.sh"

case "$SAP_VALIDATION_PHASE" in
    development)
        PROTOCOL_BASENAME="s0010_sap_validation_development_protocol_v1"
        EXPERIMENT_BASENAME="s0010_sap_validation_development_v1"
        SESSIONS=(
            s0010sapval_train_00
            s0010sapval_train_01
            s0010sapval_train_02
            s0010sapval_calibration_00
        )
        ;;
    holdout)
        PROTOCOL_BASENAME="s0010_sap_validation_holdout_protocol_v1"
        EXPERIMENT_BASENAME="s0010_sap_validation_holdout_v1"
        SESSIONS=(s0010sapval_evaluation_00 s0010sapval_evaluation_01)
        ;;
    followup)
        PROTOCOL_BASENAME="s0010_sap_validation_followup_protocol_v1"
        EXPERIMENT_BASENAME="s0010_sap_validation_followup_v1"
        SESSIONS=(
            s0010sapval_train_03
            s0010sapval_train_04
            s0010sapval_train_05
            s0010sapval_calibration_01
            s0010sapval_evaluation_00
            s0010sapval_evaluation_01
        )
        ;;
    *)
        echo "unsupported SAP_VALIDATION_PHASE=$SAP_VALIDATION_PHASE (expected development, holdout, or followup)" >&2
        exit 2
        ;;
esac

PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/${PROTOCOL_BASENAME}.yaml"
PROTOCOL_SEAL="${PROTOCOL%.yaml}.sha256"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/$EXPERIMENT_BASENAME/$SLURM_JOB_ID"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$SONOGYM_CACHE_DIR/matplotlib"
mkdir -p "$RUN_DIR"/{sessions,results,metrics,context} "$MPLCONFIGDIR"
capture_run_context "$RUN_DIR"
cp "$PROTOCOL" "$PROTOCOL_SEAL" "$RUN_DIR/context/"
(cd "$RUN_DIR/context" && sha256sum -c "${PROTOCOL_BASENAME}.sha256")

"$PYTHON" -m unittest -q \
    tests.test_s0010_sap_validation_protocols \
    tests.test_s0010_sap_correspondence \
    tests.test_ultrasound_plane_geometry \
    tests.test_acquisition_integrity \
    > "$RUN_DIR/results/preflight-tests.log" 2>&1

nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv --loop=2 > "$RUN_DIR/metrics/gpu-telemetry.csv" &
TELEMETRY_PID=$!
trap 'kill "$TELEMETRY_PID" 2>/dev/null || true' EXIT

for session_id in "${SESSIONS[@]}"; do
    session_dir="$RUN_DIR/sessions/$session_id"
    mkdir -p "$session_dir"
    /usr/bin/time -v -o "$session_dir/time-capture.txt" \
        "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/collect_stage2_dataset.py" \
        --protocol "$PROTOCOL" --session-id "$session_id" --output-dir "$session_dir" \
        --assets-data-dir "$SONOGYM_ASSETS_DATA_DIR" --headless --device cuda:0 \
        > "$session_dir/launcher.log" 2>&1
    /usr/bin/time -v -o "$session_dir/time-verify.txt" \
        "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/verify_exact_pose_capture.py" \
        --protocol "$PROTOCOL" --session-dir "$session_dir" \
        --assets-data-dir "$SONOGYM_ASSETS_DATA_DIR" \
        --output "$session_dir/verification_report.json" \
        > "$session_dir/verifier.log" 2>&1
done

kill "$TELEMETRY_PID" 2>/dev/null || true
wait "$TELEMETRY_PID" 2>/dev/null || true
trap - EXIT
capture_run_end "$RUN_DIR"
du -sb "$RUN_DIR" > "$RUN_DIR/metrics/storage-bytes.txt"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
