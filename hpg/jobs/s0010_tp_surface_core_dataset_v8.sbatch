#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit command}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common.sh"

PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_protocol_v8.yaml"
SEAL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_protocol_v8.sha256"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_dataset_v8/$SLURM_JOB_ID"
BASE_RUN="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_dataset_v6/51811811"
PARTIAL_RUN="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_dataset_v7/51815016"
GEOMETRY_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_right_sap_tp_correction/51779234"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery:${PYTHONPATH:-}"
mkdir -p "$RUN_DIR"/{sessions,results,metrics,context}
capture_run_context "$RUN_DIR"
cp "$PROTOCOL" "$SEAL" "$RUN_DIR/context/"
(cd "$RUN_DIR/context" && sha256sum -c s0010_tp_surface_core_dataset_protocol_v8.sha256)

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

capture_and_verify() {
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

capture_and_verify s0010tpscv8_smoke
"$PYTHON" - "$RUN_DIR/sessions/s0010tpscv8_smoke/verification_report.json" <<'PY'
import json, sys
r=json.load(open(sys.argv[1]))
required={"NATIVE_MAP_REPLAY", "SYNCHRONIZED_FULL_POSE_REPLAY", "ACQUISITION_FRESHNESS", "VIEWPOINT_DIVERSITY", "ARRAY_HASH_INTEGRITY"}
if r.get("frame_count") != 8 or set(r.get("gates", {})) != required or not all(r["gates"].values()):
    raise SystemExit("sealed smoke gates did not all pass")
PY

for session_id in \
    s0010tpscv8_train_l4_00 s0010tpscv8_cal_l4_00 s0010tpscv8_dev_l4_00; do
    capture_and_verify "$session_id"
done

"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/prepare_tp_surface_core_dataset.py" \
    --protocol "$PROTOCOL" --capture-root "$BASE_RUN" --capture-root "$PARTIAL_RUN" \
    --capture-root "$RUN_DIR" --geometry-root "$GEOMETRY_ROOT" \
    --output-dir "$RUN_DIR/dataset" > "$RUN_DIR/results/dataset-preparation.log" 2>&1

"$PYTHON" - "$RUN_DIR/dataset/dataset_summary.json" <<'PY'
import json, sys
r=json.load(open(sys.argv[1]))
if r.get("missing_sessions") or r.get("candidate_coverage_deficits"):
    raise SystemExit("supplement did not close the frozen candidate coverage gate")
if r.get("status") != "ENGINEERING_READY_LABEL_REVIEW_PENDING":
    raise SystemExit(f"unexpected combined dataset status: {r.get('status')}")
PY

set +e
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/train_tp_surface_core.py" \
    --dataset-dir "$RUN_DIR/dataset" --output-dir "$RUN_DIR/preflight" --no-fit \
    > "$RUN_DIR/results/training-preflight.log" 2>&1
PREFLIGHT_RC=$?
set -e
[[ "$PREFLIGHT_RC" -eq 3 ]] || { echo "Expected pending-review no-fit result, got $PREFLIGHT_RC"; exit 4; }

"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/package_tp_surface_core_review.py" \
    --run-dir "$RUN_DIR" --dataset-dir "$RUN_DIR/dataset" --preflight-dir "$RUN_DIR/preflight" \
    --geometry-root "$GEOMETRY_ROOT" --smoke-session-id s0010tpscv8_smoke \
    --output-zip "$RUN_DIR/s0010_tp_surface_core_review_${SLURM_JOB_ID}_final.zip" \
    > "$RUN_DIR/results/review-package.log" 2>&1

STAGING="$RUN_DIR/s0010_tp_surface_core_review_${SLURM_JOB_ID}_final"
(cd "$STAGING" && sha256sum -c checksums.sha256)
sha256sum "$RUN_DIR/s0010_tp_surface_core_review_${SLURM_JOB_ID}_final.zip" \
    > "$RUN_DIR/s0010_tp_surface_core_review_${SLURM_JOB_ID}_final.zip.sha256"

kill "$TELEMETRY_PID" 2>/dev/null || true
wait "$TELEMETRY_PID" 2>/dev/null || true
trap - EXIT
sstat --jobs="${SLURM_JOB_ID}.batch" --format=JobID,AveCPU,MaxRSS,MaxVMSize -P \
    > "$RUN_DIR/metrics/slurm.csv" || true
du -sb "$RUN_DIR" > "$RUN_DIR/metrics/storage-bytes.txt"
capture_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
