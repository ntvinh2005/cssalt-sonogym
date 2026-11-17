#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit command}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common.sh"

PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_protocol_v1.yaml"
SEAL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_protocol_v1.sha256"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_dataset_v2/$SLURM_JOB_ID"
GEOMETRY_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_right_sap_tp_correction/51779234"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery:${PYTHONPATH:-}"
[[ -d "$SONOGYM_RUNTIME_DIR/envs/stage1-python/sklearn" ]] || {
    echo "Pinned supervised-learning packages are missing" >&2
    exit 2
}
mkdir -p "$RUN_DIR"/{sessions,results,metrics,reviewer,context}
capture_run_context "$RUN_DIR"
cp "$PROTOCOL" "$SEAL" "$RUN_DIR/context/"
(cd "$RUN_DIR/context" && sha256sum -c s0010_tp_surface_core_dataset_protocol_v1.sha256)

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
    "$PYTHON" -m py_compile \
        workflows/two_stage_navigation/collect_stage2_dataset.py \
        workflows/two_stage_navigation/verify_exact_pose_capture.py \
        workflows/two_stage_navigation/prepare_tp_surface_core_dataset.py \
        workflows/two_stage_navigation/train_tp_surface_core.py
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

# Stage A: all eight synchronized views must pass after files are reloaded.
capture_and_verify s0010tpscv2_smoke
"$PYTHON" - "$RUN_DIR/sessions/s0010tpscv2_smoke/verification_report.json" <<'PY'
import json, sys
r=json.load(open(sys.argv[1]))
required={"NATIVE_MAP_REPLAY", "SYNCHRONIZED_FULL_POSE_REPLAY", "ACQUISITION_FRESHNESS", "VIEWPOINT_DIVERSITY", "ARRAY_HASH_INTEGRITY"}
if r.get("frame_count") != 8 or set(r.get("gates", {})) != required or not all(r["gates"].values()):
    raise SystemExit("sealed smoke gates did not all pass")
PY

# Stage B: six preassigned development sessions continue automatically.
for session_id in \
    s0010tpscv2_train_00 s0010tpscv2_train_01 s0010tpscv2_train_02 \
    s0010tpscv2_train_03 s0010tpscv2_cal_00 s0010tpscv2_dev_00; do
    capture_and_verify "$session_id"
done

"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/prepare_tp_surface_core_dataset.py" \
    --protocol "$PROTOCOL" --capture-root "$RUN_DIR" --geometry-root "$GEOMETRY_ROOT" \
    --output-dir "$RUN_DIR/dataset" > "$RUN_DIR/results/dataset-preparation.log" 2>&1

set +e
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/train_tp_surface_core.py" \
    --dataset-dir "$RUN_DIR/dataset" --output-dir "$RUN_DIR/preflight" --no-fit \
    > "$RUN_DIR/results/training-preflight.log" 2>&1
PREFLIGHT_RC=$?
set -e
[[ "$PREFLIGHT_RC" -eq 3 ]] || { echo "Expected pending-review no-fit result, got $PREFLIGHT_RC"; exit 4; }
"$PYTHON" - "$RUN_DIR/preflight/preflight_result.json" <<'PY'
import json, sys
r=json.load(open(sys.argv[1]))
if r.get("status") != "ENGINEERING_PREFLIGHT_ONLY" or r.get("fit_called"):
    raise SystemExit("no-fit preflight contract failed")
PY

cp "$RUN_DIR/dataset/candidate_sample_manifest.csv" "$RUN_DIR/reviewer/"
cp "$RUN_DIR/dataset/dataset_summary.json" "$RUN_DIR/reviewer/"
cp "$RUN_DIR/dataset/label_schema.json" "$RUN_DIR/reviewer/"
cp "$RUN_DIR/dataset/session_split_manifest.csv" "$RUN_DIR/reviewer/"
cp "$RUN_DIR/preflight/preflight_result.json" "$RUN_DIR/reviewer/"
cp "$RUN_DIR/sessions/s0010tpscv2_smoke/verification_report.json" "$RUN_DIR/reviewer/smoke_verification_report.json"
cp "$RUN_DIR/sessions/s0010tpscv2_smoke/public/ultrasound.npz" "$RUN_DIR/reviewer/smoke_ultrasound.npz"
cp "$RUN_DIR/sessions/s0010tpscv2_smoke/privileged/raw_labels.npz" "$RUN_DIR/reviewer/smoke_raw_labels.npz"
cp "$RUN_DIR/sessions/s0010tpscv2_smoke/privileged/pixel_to_volume_coordinates.npz" "$RUN_DIR/reviewer/smoke_pixel_to_volume_coordinates.npz"
cp -r "$RUN_DIR/dataset/review" "$RUN_DIR/reviewer/verified_overlays"
(
    cd "$RUN_DIR/reviewer"
    find . -type f -print0 | sort -z | xargs -0 sha256sum > checksums.sha256
    zip -qr "$RUN_DIR/s0010_tp_surface_core_review_${SLURM_JOB_ID}.zip" .
)

kill "$TELEMETRY_PID" 2>/dev/null || true
wait "$TELEMETRY_PID" 2>/dev/null || true
trap - EXIT
sstat --jobs="${SLURM_JOB_ID}.batch" --format=JobID,AveCPU,MaxRSS,MaxVMSize -P \
    > "$RUN_DIR/metrics/slurm.csv" || true
du -sb "$RUN_DIR" > "$RUN_DIR/metrics/storage-bytes.txt"
capture_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"

