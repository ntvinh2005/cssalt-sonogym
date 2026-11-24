#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit command}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common.sh"

export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/evaluation"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/sonogym-matplotlib-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_mbb_single_anatomy_poc_protocol_v6.yaml"
STAGE1_SOURCE="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_mbb_single_anatomy_poc_v5r2/52505715/stage1/results"
GEOMETRY_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_right_sap_tp_correction/51779234"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_mbb_single_anatomy_poc_v6/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"/{context,metrics,results,review,stage1/results,stage2}
capture_run_context "$RUN_DIR"
cp "$PROTOCOL" "${PROTOCOL%.yaml}.sha256" "$RUN_DIR/context/"
(cd "$RUN_DIR/context" && sha256sum -c s0010_mbb_single_anatomy_poc_protocol_v6.sha256)

(cd "$STAGE1_SOURCE" && sha256sum -c actor_results.sha256 evaluation_report.sha256)
cp "$STAGE1_SOURCE/actor_results.json" "$STAGE1_SOURCE/actor_results.sha256" \
   "$STAGE1_SOURCE/evaluation_report.json" "$STAGE1_SOURCE/evaluation_report.sha256" \
   "$RUN_DIR/stage1/results/"

nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv > "$RUN_DIR/metrics/gpu-before.csv"
nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv --loop=2 > "$RUN_DIR/metrics/gpu-telemetry.csv" &
TELEMETRY_PID=$!

set +e
"$PYTHON" -m unittest -q \
    tests.test_stage2_tp_active \
    tests.test_sonoanatomy_active_scan \
    tests.test_sonoanatomy_multiview \
    tests.test_sonoanatomy_segmentation \
    tests.test_stage2_sap_tp_evaluator \
    tests.test_incremental_landmark_map \
    tests.test_s0010_mbb_stage2_review_package \
    > "$RUN_DIR/results/tests.log" 2>&1
TEST_RC=$?

STAGE2_ACTOR_FAILURES=0
STAGE2_EVALUATOR_FAILURES=0
if [[ "$TEST_RC" -eq 0 ]]; then
    while IFS='|' read -r session_id stage1_session_id; do
        session_dir="$RUN_DIR/stage2/$session_id"
        mkdir -p "$session_dir"
        "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/run_stage2_tp_active.py" \
            --protocol "$PROTOCOL" \
            --phase1-actor-result "$RUN_DIR/stage1/results/actor_results.json" \
            --stage1-session-id "$stage1_session_id" \
            --output-dir "$session_dir" \
            --assets-data-dir "$SONOGYM_ASSETS_DATA_DIR" \
            --headless --enable_cameras --device cuda:0 \
            > "$session_dir/actor.log" 2>&1
        actor_rc=$?
        if [[ "$actor_rc" -eq 0 ]]; then
            "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/evaluate_stage2_tp_active.py" \
                --protocol "$PROTOCOL" --run-dir "$session_dir" \
                --geometry-root "$GEOMETRY_ROOT" \
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

AGGREGATE_RC=99
if [[ "$STAGE2_ACTOR_FAILURES" -eq 0 && "$STAGE2_EVALUATOR_FAILURES" -eq 0 ]]; then
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/evaluate_s0010_mbb_poc.py" \
        --protocol "$PROTOCOL" \
        --stage1-evaluation "$RUN_DIR/stage1/results/evaluation_report.json" \
        --run-dir "$RUN_DIR" \
        --output "$RUN_DIR/results/end_to_end_report.json" \
        > "$RUN_DIR/results/aggregate.log" 2>&1
    AGGREGATE_RC=$?
fi

REVIEW_RC=99
if [[ "$STAGE2_EVALUATOR_FAILURES" -eq 0 ]]; then
    REVIEW_SESSION=$("$PYTHON" - "$RUN_DIR/stage2" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
ranked = []
for actor_path in root.glob("*/results/actor_result.json"):
    evaluator_path = actor_path.with_name("evaluation_report.json")
    if not evaluator_path.is_file():
        continue
    actor = json.loads(actor_path.read_text())
    evaluator = json.loads(evaluator_path.read_text())
    landmark_count = sum(
        row["status"] == "CONFIRMED"
        for row in (actor.get("incremental_landmark_map") or {}).get("landmarks", {}).values()
    )
    ranked.append((
        int(evaluator.get("integrated_stage2_session_success", False)),
        landmark_count,
        actor_path.parent.parent,
    ))
if ranked:
    print(max(ranked, key=lambda row: (row[0], row[1], str(row[2])))[2])
PY
    )
    if [[ -n "$REVIEW_SESSION" ]]; then
        "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/render_s0010_incremental_landmark_map_demo.py" \
            --session-dir "$REVIEW_SESSION" \
            --source-protocol "$PROTOCOL" \
            --map-protocol "$PROTOCOL" \
            --output-dir "$RUN_DIR/review/actual_execution" \
            --fps 5 \
            > "$RUN_DIR/review/render.log" 2>&1
        REVIEW_RC=$?
    fi
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
    echo "stage2_actor_failure_count=$STAGE2_ACTOR_FAILURES"
    echo "stage2_evaluator_failure_count=$STAGE2_EVALUATOR_FAILURES"
    echo "aggregate_exit_code=$AGGREGATE_RC"
    echo "review_exit_code=$REVIEW_RC"
} > "$RUN_DIR/context/application-exit-codes.env"
capture_run_end "$RUN_DIR"
du -sb "$RUN_DIR" > "$RUN_DIR/metrics/storage-bytes.txt"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"

if [[ "$TEST_RC" -ne 0 ]]; then exit 2; fi
if [[ "$STAGE2_ACTOR_FAILURES" -ne 0 || "$STAGE2_EVALUATOR_FAILURES" -ne 0 ]]; then exit 2; fi
if [[ "$AGGREGATE_RC" -ne 0 || "$REVIEW_RC" -ne 0 ]]; then exit 2; fi
exit 0
