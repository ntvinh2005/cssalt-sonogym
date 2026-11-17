#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit command}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common.sh"

PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_exact_pose_replay_protocol_v1.yaml"
PROTOCOL_SEAL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_exact_pose_replay_protocol_v1.sha256"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_exact_pose_replay_v1/$SLURM_JOB_ID"
SESSION_ID="s0010epr_00"
SESSION_DIR="$RUN_DIR/session"
mkdir -p "$SESSION_DIR" "$RUN_DIR/results" "$RUN_DIR/metrics" "$RUN_DIR/reviewer"
capture_run_context "$RUN_DIR"
cp "$PROTOCOL" "$PROTOCOL_SEAL" "$RUN_DIR/context/"
(
    cd "$RUN_DIR/context"
    sha256sum -c s0010_exact_pose_replay_protocol_v1.sha256
)

nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv > "$RUN_DIR/metrics/gpu-before.csv"
nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv --loop=2 > "$RUN_DIR/metrics/gpu-telemetry.csv" &
TELEMETRY_PID=$!

set +e
(
    cd "$SONOGYM_PROJECT_DIR"
    "$PYTHON" tests/test_ultrasound_plane_geometry.py
    PYTHONPATH="$SONOGYM_PROJECT_DIR/source/spinal_surgery" \
        "$PYTHON" tests/test_two_stage_contracts.py
) > "$RUN_DIR/results/preflight-tests.log" 2>&1
TEST_RC=$?

CAPTURE_RC=99
if [[ "$TEST_RC" -eq 0 ]]; then
    /usr/bin/time -v -o "$RUN_DIR/metrics/time-capture.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/collect_stage2_dataset.py" \
        --protocol "$PROTOCOL" --session-id "$SESSION_ID" \
        --output-dir "$SESSION_DIR" --assets-data-dir "$SONOGYM_ASSETS_DATA_DIR" \
        --headless --device cuda:0 > "$SESSION_DIR/launcher.log" 2>&1
    CAPTURE_RC=$?
fi

VERIFY_RC=99
if [[ -f "$SESSION_DIR/privileged/full_pose_replay.jsonl" ]]; then
    /usr/bin/time -v -o "$RUN_DIR/metrics/time-verify.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/verify_exact_pose_capture.py" \
        --protocol "$PROTOCOL" --session-dir "$SESSION_DIR" \
        --assets-data-dir "$SONOGYM_ASSETS_DATA_DIR" \
        --output "$RUN_DIR/results/exact_pose_replay_report.json" \
        > "$RUN_DIR/results/verifier.log" 2>&1
    VERIFY_RC=$?
fi
set -e

kill "$TELEMETRY_PID" 2>/dev/null || true
wait "$TELEMETRY_PID" 2>/dev/null || true
nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv > "$RUN_DIR/metrics/gpu-after.csv"
sstat --jobs="${SLURM_JOB_ID}.batch" --format=JobID,AveCPU,MaxRSS,MaxVMSize -P \
    > "$RUN_DIR/metrics/slurm.csv" || true
{
    echo "test_exit_code=$TEST_RC"
    echo "capture_exit_code=$CAPTURE_RC"
    echo "verify_exit_code=$VERIFY_RC"
} > "$RUN_DIR/context/application-exit-codes.env"
capture_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"

if [[ "$VERIFY_RC" -eq 0 ]]; then
    cp "$RUN_DIR/results/exact_pose_replay_report.json" "$RUN_DIR/reviewer/"
    cp "$SESSION_DIR/public/frames.jsonl" "$RUN_DIR/reviewer/"
    cp "$SESSION_DIR/public/ultrasound.npz" "$RUN_DIR/reviewer/"
    cp "$SESSION_DIR/privileged/raw_labels.npz" "$RUN_DIR/reviewer/"
    cp "$SESSION_DIR/privileged/full_pose_replay.jsonl" "$RUN_DIR/reviewer/"
    cp "$SESSION_DIR/privileged/pixel_to_volume_coordinates.npz" "$RUN_DIR/reviewer/"
    (
        cd "$RUN_DIR/reviewer"
        sha256sum ./* > checksums.sha256
        zip -q "$RUN_DIR/s0010_exact_pose_replay_${SLURM_JOB_ID}.zip" ./*
    )
fi

if [[ "$TEST_RC" -ne 0 ]]; then exit "$TEST_RC"; fi
if [[ "$VERIFY_RC" -ne 0 ]]; then exit "$VERIFY_RC"; fi
exit "$CAPTURE_RC"
