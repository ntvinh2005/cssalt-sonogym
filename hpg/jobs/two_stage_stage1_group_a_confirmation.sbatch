#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit helper}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common.sh"

ACTOR_PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/stage1_group_a_confirmation_actor_protocol_v1.yaml"
ACTOR_PROTOCOL_SEAL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/stage1_group_a_confirmation_actor_protocol_v1.sha256"
EVALUATION_PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/stage1_group_a_confirmation_evaluation_protocol_v1.yaml"
EVALUATION_PROTOCOL_SEAL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/stage1_group_a_confirmation_evaluation_protocol_v1.sha256"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/stage1_group_a_confirmation/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR/sessions" "$RUN_DIR/results" "$RUN_DIR/metrics"
capture_run_context "$RUN_DIR"
cp "$ACTOR_PROTOCOL" "$ACTOR_PROTOCOL_SEAL" "$EVALUATION_PROTOCOL" \
    "$EVALUATION_PROTOCOL_SEAL" "$RUN_DIR/context/"
(
    cd "$RUN_DIR/context"
    sha256sum -c stage1_group_a_confirmation_actor_protocol_v1.sha256
    sha256sum -c stage1_group_a_confirmation_evaluation_protocol_v1.sha256
)

nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv > "$RUN_DIR/metrics/gpu-before.csv"
nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv --loop=2 > "$RUN_DIR/metrics/gpu-telemetry.csv" &
TELEMETRY_PID=$!

mapfile -t SESSION_IDS < <(
    "$PYTHON" -c 'import sys,yaml; p=yaml.safe_load(open(sys.argv[1])); [print(x["session_id"]) for x in p["sessions"]]' "$ACTOR_PROTOCOL"
)
[[ "${#SESSION_IDS[@]}" -eq 4 ]] || {
    echo "Confirmation protocol must contain four sessions" >&2
    exit 2
}

CAPTURE_RC=0
ACTOR_RC=99
EVALUATION_RC=99
set +e
for session_id in "${SESSION_IDS[@]}"; do
    session_dir="$RUN_DIR/sessions/$session_id"
    mkdir -p "$session_dir"
    /usr/bin/time -v -o "$RUN_DIR/metrics/time-${session_id}.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/collect_observability.py" \
        --protocol "$ACTOR_PROTOCOL" \
        --session-id "$session_id" \
        --output-dir "$session_dir" \
        --assets-data-dir "$SONOGYM_ASSETS_DATA_DIR" \
        --headless \
        --device cuda:0 \
        > "$session_dir/launcher.log" 2>&1
    session_rc=$?
    session_status="MISSING"
    if [[ -f "$session_dir/summary.json" ]]; then
        session_status=$("$PYTHON" -c \
            'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
            "$session_dir/summary.json")
    fi
    echo "$session_rc" > "$session_dir/application-exit-code.txt"
    if [[ "$session_rc" -ne 0 || "$session_status" != "PASSED" ]]; then
        CAPTURE_RC=$([[ "$session_rc" -ne 0 ]] && echo "$session_rc" || echo 2)
        echo "GROUP_A_CONFIRMATION_CAPTURE_FAIL session=$session_id rc=$CAPTURE_RC status=$session_status"
        break
    fi
done

if [[ "$CAPTURE_RC" -eq 0 ]]; then
    export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/evaluation"
    /usr/bin/time -v -o "$RUN_DIR/metrics/time-actor.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/run_stage1_group_a_confirmation.py" actor \
        --actor-protocol "$ACTOR_PROTOCOL" \
        --capture-root "$RUN_DIR" \
        --output "$RUN_DIR/results/actor_results.json" \
        > "$RUN_DIR/results/actor.log" 2>&1
    ACTOR_RC=$?
fi

if [[ "$ACTOR_RC" -eq 0 ]]; then
    /usr/bin/time -v -o "$RUN_DIR/metrics/time-evaluator.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/run_stage1_group_a_confirmation.py" evaluate \
        --actor-protocol "$ACTOR_PROTOCOL" \
        --evaluation-protocol "$EVALUATION_PROTOCOL" \
        --capture-root "$RUN_DIR" \
        --actor-result "$RUN_DIR/results/actor_results.json" \
        --output "$RUN_DIR/results/evaluation_report.json" \
        > "$RUN_DIR/results/evaluator.log" 2>&1
    EVALUATION_RC=$?
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
    echo "actor_exit_code=$ACTOR_RC"
    echo "evaluation_exit_code=$EVALUATION_RC"
} > "$RUN_DIR/context/application-exit-codes.env"
capture_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"

if [[ "$CAPTURE_RC" -ne 0 ]]; then exit "$CAPTURE_RC"; fi
if [[ "$ACTOR_RC" -ne 0 ]]; then exit "$ACTOR_RC"; fi
if [[ "$EVALUATION_RC" -ne 0 ]]; then exit "$EVALUATION_RC"; fi
