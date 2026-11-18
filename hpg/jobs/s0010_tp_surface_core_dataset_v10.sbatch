#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit command}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common.sh"

PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_protocol_v10.yaml"
SEAL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_protocol_v10.sha256"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_dataset_v10/$SLURM_JOB_ID"
V6_RUN="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_dataset_v6/51811811"
V7_RUN="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_dataset_v7/51815016"
V8_RUN="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_dataset_v8/51817719"
V9_RUN="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_dataset_v9/51818619"
GEOMETRY_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_right_sap_tp_correction/51779234"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p "$RUN_DIR"/{sessions,results,metrics,context}
capture_run_context "$RUN_DIR"
cp "$PROTOCOL" "$SEAL" "$RUN_DIR/context/"
(cd "$RUN_DIR/context" && sha256sum -c s0010_tp_surface_core_dataset_protocol_v10.sha256)

nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv --loop=2 > "$RUN_DIR/metrics/gpu-telemetry.csv" &
TELEMETRY_PID=$!
trap 'kill "$TELEMETRY_PID" 2>/dev/null || true' EXIT

(
    cd "$SONOGYM_PROJECT_DIR"
    "$PYTHON" tests/test_acquisition_integrity.py
    "$PYTHON" tests/test_ultrasound_plane_geometry.py
    "$PYTHON" tests/test_two_stage_contracts.py
    "$PYTHON" tests/test_tp_surface_core_training.py
    "$PYTHON" tests/test_tp_surface_core_capture_contract.py
) > "$RUN_DIR/results/preflight-tests.log" 2>&1

capture_session() {
    local session_id="$1"
    local session_dir="$RUN_DIR/sessions/$session_id"
    mkdir -p "$session_dir"
    /usr/bin/time -v -o "$RUN_DIR/metrics/time-capture-$session_id.txt" \
        "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/collect_stage2_dataset.py" \
        --protocol "$PROTOCOL" --session-id "$session_id" \
        --output-dir "$session_dir" --assets-data-dir "$SONOGYM_ASSETS_DATA_DIR" \
        --headless --device cuda:0 > "$session_dir/launcher.log" 2>&1
    /usr/bin/time -v -o "$RUN_DIR/metrics/time-verify-$session_id.txt" \
        "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/verify_exact_pose_capture.py" \
        --protocol "$PROTOCOL" --session-dir "$session_dir" \
        --assets-data-dir "$SONOGYM_ASSETS_DATA_DIR" \
        --output "$session_dir/verification_report.json" \
        > "$session_dir/verifier.log" 2>&1
}

capture_session s0010tpscv10_train_l4_00
capture_session s0010tpscv10_train_l5_00
capture_session s0010tpscv10_cal_l5_00
capture_session s0010tpscv10_dev_l4l5_00

/usr/bin/time -v -o "$RUN_DIR/metrics/time-dataset.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/prepare_tp_surface_core_dataset.py" \
    --protocol "$PROTOCOL" --capture-root "$V6_RUN" --capture-root "$V7_RUN" \
    --capture-root "$V8_RUN" --capture-root "$V9_RUN" --capture-root "$RUN_DIR" \
    --geometry-root "$GEOMETRY_ROOT" \
    --prior-dataset-dir "$V9_RUN/dataset" \
    --output-dir "$RUN_DIR/dataset" > "$RUN_DIR/results/dataset-preparation.log" 2>&1

set +e
/usr/bin/time -v -o "$RUN_DIR/metrics/time-preflight.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/train_tp_surface_core.py" \
    --dataset-dir "$RUN_DIR/dataset" --output-dir "$RUN_DIR/preflight" --no-fit \
    > "$RUN_DIR/results/training-preflight.log" 2>&1
PREFLIGHT_RC=$?
set -e
[[ "$PREFLIGHT_RC" -eq 3 ]] || {
    echo "Expected fail-closed review-pending result, got $PREFLIGHT_RC" >&2
    exit 4
}

ln -s "$V8_RUN/sessions/s0010tpscv8_smoke" "$RUN_DIR/sessions/s0010tpscv8_smoke"
ZIP="$RUN_DIR/s0010_tp_surface_core_review_v3_${SLURM_JOB_ID}.zip"
/usr/bin/time -v -o "$RUN_DIR/metrics/time-package.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/package_tp_surface_core_review.py" \
    --run-dir "$RUN_DIR" --dataset-dir "$RUN_DIR/dataset" \
    --preflight-dir "$RUN_DIR/preflight" --geometry-root "$GEOMETRY_ROOT" \
    --smoke-session-id s0010tpscv8_smoke --output-zip "$ZIP" \
    > "$RUN_DIR/results/review-package.log" 2>&1

STAGING="${ZIP%.zip}"
(cd "$STAGING" && sha256sum -c checksums.sha256)
sha256sum "$ZIP" > "$ZIP.sha256"
"$PYTHON" - "$RUN_DIR/dataset/dataset_summary.json" "$RUN_DIR/results/coverage-gate.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
result = {
    "candidate_coverage_deficits": summary["candidate_coverage_deficits"],
    "candidate_unique_image_counts": summary["candidate_unique_image_counts"],
    "passes_support_coverage_gate": not summary["candidate_coverage_deficits"],
    "status": summary["status"],
}
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2)
    handle.write("\n")
PY

kill "$TELEMETRY_PID" 2>/dev/null || true
wait "$TELEMETRY_PID" 2>/dev/null || true
trap - EXIT
sstat --jobs="${SLURM_JOB_ID}.batch" --format=JobID,AveCPU,MaxRSS,MaxVMSize -P \
    > "$RUN_DIR/metrics/slurm.csv" || true
du -sb "$RUN_DIR" > "$RUN_DIR/metrics/storage-bytes.txt"
capture_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
