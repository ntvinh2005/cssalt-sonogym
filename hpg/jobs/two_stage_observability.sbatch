#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit helper}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common.sh"

PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/protocol.yaml"
PROTOCOL_SEAL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/protocol.sha256"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/g0_g1_v1/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR/sessions" "$RUN_DIR/results" "$RUN_DIR/metrics"
capture_run_context "$RUN_DIR"
cp "$PROTOCOL" "$RUN_DIR/context/protocol.yaml"
cp "$PROTOCOL_SEAL" "$RUN_DIR/context/protocol.sha256"
(
    cd "$RUN_DIR/context"
    sha256sum -c protocol.sha256
)

nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv > "$RUN_DIR/metrics/gpu-before.csv"
nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv --loop=2 > "$RUN_DIR/metrics/gpu-telemetry.csv" &
TELEMETRY_PID=$!

mapfile -t SESSION_IDS < <(
    "$PYTHON" -c 'import sys,yaml; p=yaml.safe_load(open(sys.argv[1])); [print(x["session_id"]) for x in p["sessions"]]' "$PROTOCOL"
)
[[ "${#SESSION_IDS[@]}" -eq 16 ]] || {
    echo "Frozen protocol must contain 16 sessions; found ${#SESSION_IDS[@]}" >&2
    exit 2
}

CAPTURE_RC=0
SUMMARY_RC=99
set +e
for session_id in "${SESSION_IDS[@]}"; do
    session_dir="$RUN_DIR/sessions/$session_id"
    mkdir -p "$session_dir"
    echo "G0_G1_SESSION_START session=$session_id" | tee "$session_dir/launcher.log"
    /usr/bin/time -v -o "$RUN_DIR/metrics/time-${session_id}.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/collect_observability.py" \
        --protocol "$PROTOCOL" \
        --session-id "$session_id" \
        --output-dir "$session_dir" \
        --assets-data-dir "$SONOGYM_ASSETS_DATA_DIR" \
        --headless \
        --device cuda:0 \
        >> "$session_dir/launcher.log" 2>&1
    session_rc=$?
    echo "$session_rc" > "$session_dir/application-exit-code.txt"
    if [[ "$session_rc" -ne 0 ]]; then
        CAPTURE_RC=$session_rc
        echo "G0_G1_SESSION_FAIL session=$session_id rc=$session_rc" | tee -a "$session_dir/launcher.log"
        break
    fi
    echo "G0_G1_SESSION_DONE session=$session_id" | tee -a "$session_dir/launcher.log"
done

if [[ "$CAPTURE_RC" -eq 0 ]]; then
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/summarize_observability.py" \
        --protocol "$PROTOCOL" \
        --sessions-dir "$RUN_DIR/sessions" \
        --output-dir "$RUN_DIR/results" \
        > "$RUN_DIR/results/summarizer.log" 2>&1
    SUMMARY_RC=$?
fi
set -e

kill "$TELEMETRY_PID" 2>/dev/null || true
wait "$TELEMETRY_PID" 2>/dev/null || true
nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv > "$RUN_DIR/metrics/gpu-after.csv"
sstat --jobs="${SLURM_JOB_ID}.batch" --format=JobID,AveCPU,MaxRSS,MaxVMSize -P \
    > "$RUN_DIR/metrics/slurm.csv" || true
{
    echo "capture_exit_code=$CAPTURE_RC"
    echo "summary_exit_code=$SUMMARY_RC"
} > "$RUN_DIR/context/application-exit-codes.env"
capture_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"

if [[ "$CAPTURE_RC" -ne 0 ]]; then
    echo "G0/G1 capture failed with exit code $CAPTURE_RC" >&2
    exit "$CAPTURE_RC"
fi
if [[ "$SUMMARY_RC" -ne 0 ]]; then
    echo "G0/G1 summarization failed with exit code $SUMMARY_RC" >&2
    exit "$SUMMARY_RC"
fi
