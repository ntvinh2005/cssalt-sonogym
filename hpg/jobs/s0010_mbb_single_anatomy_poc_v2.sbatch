#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit command}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common.sh"

export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/evaluation"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_mbb_single_anatomy_poc_protocol_v2.yaml"
STAGE1_ACTOR_PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/stage1_group_a_confirmation_actor_protocol_v1.yaml"
STAGE1_EVALUATION_PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/stage1_group_a_confirmation_evaluation_protocol_v1.yaml"
GEOMETRY_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_right_sap_tp_correction/51779234"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_mbb_single_anatomy_poc_v2/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"/{context,metrics,results,stage1/sessions,stage1/results,stage2}
capture_run_context "$RUN_DIR"
cp "$PROTOCOL" "${PROTOCOL%.yaml}.sha256" "$STAGE1_ACTOR_PROTOCOL" \
   "$STAGE1_EVALUATION_PROTOCOL" "$RUN_DIR/context/"
(cd "$RUN_DIR/context" && sha256sum -c s0010_mbb_single_anatomy_poc_protocol_v2.sha256)

nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv > "$RUN_DIR/metrics/gpu-before.csv"
nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv --loop=2 > "$RUN_DIR/metrics/gpu-telemetry.csv" &
TELEMETRY_PID=$!

set +e
"$PYTHON" "$SONOGYM_PROJECT_DIR/tests/test_stage2_tp_active.py" > "$RUN_DIR/results/tests-tp-active.log" 2>&1
TP_TEST_RC=$?
if [[ "$TP_TEST_RC" -eq 0 ]]; then
    "$PYTHON" "$SONOGYM_PROJECT_DIR/tests/test_stage2_active_pilot.py" > "$RUN_DIR/results/tests-active-regression.log" 2>&1
    ACTIVE_TEST_RC=$?
else
    ACTIVE_TEST_RC=99
fi
if [[ "$TP_TEST_RC" -eq 0 && "$ACTIVE_TEST_RC" -eq 0 ]]; then
    "$PYTHON" "$SONOGYM_PROJECT_DIR/tests/test_stage2_mbb_landmarks.py" > "$RUN_DIR/results/tests-mbb-contract.log" 2>&1
    MBB_TEST_RC=$?
else
    MBB_TEST_RC=99
fi

STAGE1_CAPTURE_RC=0
STAGE1_ACTOR_RC=99
STAGE1_EVALUATOR_RC=99
if [[ "$TP_TEST_RC" -eq 0 && "$ACTIVE_TEST_RC" -eq 0 && "$MBB_TEST_RC" -eq 0 ]]; then
    mapfile -t STAGE1_IDS < <(
        "$PYTHON" -c 'import sys,yaml; [print(x["session_id"]) for x in yaml.safe_load(open(sys.argv[1]))["sessions"]]' "$STAGE1_ACTOR_PROTOCOL"
    )
    for session_id in "${STAGE1_IDS[@]}"; do
        session_dir="$RUN_DIR/stage1/sessions/$session_id"
        mkdir -p "$session_dir"
        "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/collect_observability.py" \
            --protocol "$STAGE1_ACTOR_PROTOCOL" --session-id "$session_id" \
            --output-dir "$session_dir" --assets-data-dir "$SONOGYM_ASSETS_DATA_DIR" \
            --headless --device cuda:0 > "$session_dir/launcher.log" 2>&1
        rc=$?
        if [[ "$rc" -ne 0 ]]; then STAGE1_CAPTURE_RC=$rc; break; fi
    done
else
    STAGE1_CAPTURE_RC=99
fi

if [[ "$STAGE1_CAPTURE_RC" -eq 0 ]]; then
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/run_stage1_group_a_confirmation.py" actor \
        --actor-protocol "$STAGE1_ACTOR_PROTOCOL" --capture-root "$RUN_DIR/stage1" \
        --output "$RUN_DIR/stage1/results/actor_results.json" \
        > "$RUN_DIR/stage1/results/actor.log" 2>&1
    STAGE1_ACTOR_RC=$?
fi
if [[ "$STAGE1_ACTOR_RC" -eq 0 ]]; then
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/run_stage1_group_a_confirmation.py" evaluate \
        --actor-protocol "$STAGE1_ACTOR_PROTOCOL" --evaluation-protocol "$STAGE1_EVALUATION_PROTOCOL" \
        --capture-root "$RUN_DIR/stage1" --actor-result "$RUN_DIR/stage1/results/actor_results.json" \
        --output "$RUN_DIR/stage1/results/evaluation_report.json" \
        > "$RUN_DIR/stage1/results/evaluator.log" 2>&1
    STAGE1_EVALUATOR_RC=$?
fi

STAGE2_ACTOR_FAILURES=0
STAGE2_EVALUATOR_FAILURES=0
if [[ "$STAGE1_EVALUATOR_RC" -eq 0 ]]; then
    while IFS='|' read -r session_id stage1_session_id; do
        session_dir="$RUN_DIR/stage2/$session_id"
        mkdir -p "$session_dir"
        "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/run_stage2_tp_active.py" \
            --protocol "$PROTOCOL" --phase1-actor-result "$RUN_DIR/stage1/results/actor_results.json" \
            --stage1-session-id "$stage1_session_id" --output-dir "$session_dir" \
            --assets-data-dir "$SONOGYM_ASSETS_DATA_DIR" --headless --enable_cameras --device cuda:0 \
            > "$session_dir/actor.log" 2>&1
        actor_rc=$?
        if [[ "$actor_rc" -eq 0 ]]; then
            "$PYTHON" - "$session_dir/summary.json" "$session_dir/results/actor_result.json" <<'PY'
import json
import sys
from pathlib import Path
summary_path, actor_path = map(Path, sys.argv[1:])
if not summary_path.is_file() or not actor_path.is_file():
    raise SystemExit(2)
if json.loads(summary_path.read_text()).get("status") != "COMPLETED":
    raise SystemExit(2)
PY
            actor_rc=$?
        fi
        if [[ "$actor_rc" -eq 0 ]]; then
            "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/evaluate_stage2_tp_active.py" \
                --protocol "$PROTOCOL" --run-dir "$session_dir" --geometry-root "$GEOMETRY_ROOT" \
                --output "$session_dir/results/evaluation_report.json" \
                > "$session_dir/evaluator.log" 2>&1
            evaluator_rc=$?
            [[ "$evaluator_rc" -eq 0 ]] || STAGE2_EVALUATOR_FAILURES=$((STAGE2_EVALUATOR_FAILURES + 1))
        else
            evaluator_rc=99
            STAGE2_ACTOR_FAILURES=$((STAGE2_ACTOR_FAILURES + 1))
        fi
        {
            echo "actor_exit_code=$actor_rc"
            echo "evaluator_exit_code=$evaluator_rc"
        } > "$session_dir/application-exit-codes.env"
    done < <(
        "$PYTHON" -c 'import sys,yaml; [print(x["session_id"] + "|" + x["stage1_session_id"]) for x in yaml.safe_load(open(sys.argv[1]))["execution"]["sessions"]]' "$PROTOCOL"
    )
else
    STAGE2_ACTOR_FAILURES=4
fi

if [[ "$STAGE2_ACTOR_FAILURES" -eq 0 && "$STAGE2_EVALUATOR_FAILURES" -eq 0 ]]; then
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/evaluate_s0010_mbb_poc.py" \
        --protocol "$PROTOCOL" --stage1-evaluation "$RUN_DIR/stage1/results/evaluation_report.json" \
        --run-dir "$RUN_DIR" --output "$RUN_DIR/results/end_to_end_report.json" \
        > "$RUN_DIR/results/aggregate.log" 2>&1
    AGGREGATE_RC=$?
else
    AGGREGATE_RC=99
fi
set -e

kill "$TELEMETRY_PID" 2>/dev/null || true
wait "$TELEMETRY_PID" 2>/dev/null || true
nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv > "$RUN_DIR/metrics/gpu-after.csv"
sstat --jobs="${SLURM_JOB_ID}.batch" --format=JobID,AveCPU,MaxRSS,MaxVMSize -P \
    > "$RUN_DIR/metrics/slurm.csv" || true
{
    echo "tp_active_test_exit_code=$TP_TEST_RC"
    echo "active_regression_test_exit_code=$ACTIVE_TEST_RC"
    echo "mbb_contract_test_exit_code=$MBB_TEST_RC"
    echo "stage1_capture_exit_code=$STAGE1_CAPTURE_RC"
    echo "stage1_actor_exit_code=$STAGE1_ACTOR_RC"
    echo "stage1_evaluator_exit_code=$STAGE1_EVALUATOR_RC"
    echo "stage2_actor_failure_count=$STAGE2_ACTOR_FAILURES"
    echo "stage2_evaluator_failure_count=$STAGE2_EVALUATOR_FAILURES"
    echo "aggregate_exit_code=$AGGREGATE_RC"
} > "$RUN_DIR/context/application-exit-codes.env"
capture_run_end "$RUN_DIR"
du -sb "$RUN_DIR" > "$RUN_DIR/metrics/storage-bytes.txt"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"

if [[ "$TP_TEST_RC" -ne 0 || "$ACTIVE_TEST_RC" -ne 0 || "$MBB_TEST_RC" -ne 0 ]]; then exit 2; fi
if [[ "$STAGE1_CAPTURE_RC" -ne 0 ]]; then exit "$STAGE1_CAPTURE_RC"; fi
if [[ "$STAGE1_ACTOR_RC" -ne 0 ]]; then exit "$STAGE1_ACTOR_RC"; fi
if [[ "$STAGE1_EVALUATOR_RC" -ne 0 ]]; then exit "$STAGE1_EVALUATOR_RC"; fi
if [[ "$STAGE2_ACTOR_FAILURES" -ne 0 || "$STAGE2_EVALUATOR_FAILURES" -ne 0 ]]; then exit 2; fi
exit "$AGGREGATE_RC"

