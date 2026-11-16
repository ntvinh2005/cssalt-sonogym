#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit command}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common_cpu.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/stage2_temporal_replay_protocol_v1.yaml"
PROTOCOL_SEAL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/stage2_temporal_replay_protocol_v1.sha256"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/stage2_temporal_replay_v1/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR/results" "$RUN_DIR/metrics"
capture_cpu_run_context "$RUN_DIR"
cp "$PROTOCOL" "$PROTOCOL_SEAL" "$RUN_DIR/context/"
(
    cd "$RUN_DIR/context"
    sha256sum -c stage2_temporal_replay_protocol_v1.sha256
)

set +e
/usr/bin/time -v -o "$RUN_DIR/metrics/time-tests.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/tests/test_stage2_temporal_decision.py" \
    > "$RUN_DIR/tests-temporal.log" 2>&1
TEMPORAL_TEST_RC=$?
if [[ "$TEMPORAL_TEST_RC" -eq 0 ]]; then
    /usr/bin/time -v -o "$RUN_DIR/metrics/time-stage2-regression.txt" \
        "$PYTHON" "$SONOGYM_PROJECT_DIR/tests/test_two_stage_stage2.py" \
        > "$RUN_DIR/tests-stage2-regression.log" 2>&1
    REGRESSION_TEST_RC=$?
else
    REGRESSION_TEST_RC=99
fi
if [[ "$TEMPORAL_TEST_RC" -eq 0 && "$REGRESSION_TEST_RC" -eq 0 ]]; then
    /usr/bin/time -v -o "$RUN_DIR/metrics/time-replay.txt" \
        "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/replay_stage2_temporal.py" \
            --protocol "$PROTOCOL" --output-dir "$RUN_DIR/results" \
            > "$RUN_DIR/stage2-temporal-replay.log" 2>&1
    REPLAY_RC=$?
else
    REPLAY_RC=99
fi
set -e

{
    echo "temporal_test_exit_code=$TEMPORAL_TEST_RC"
    echo "stage2_regression_exit_code=$REGRESSION_TEST_RC"
    echo "replay_exit_code=$REPLAY_RC"
} > "$RUN_DIR/context/application-exit-codes.env"
sstat --jobs="${SLURM_JOB_ID}.batch" --format=JobID,AveCPU,MaxRSS,MaxVMSize -P \
    > "$RUN_DIR/metrics/slurm.csv" || true
capture_cpu_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"

if [[ "$TEMPORAL_TEST_RC" -ne 0 || "$REGRESSION_TEST_RC" -ne 0 ]]; then
    exit 2
fi
exit "$REPLAY_RC"
