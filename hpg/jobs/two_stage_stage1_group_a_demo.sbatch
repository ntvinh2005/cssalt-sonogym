#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit helper}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck disable=SC1090
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

SOURCE_RUN="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/stage1_group_a_v7/51209243"
CAPTURE_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/g0_g1_v2/50908807"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/stage1_group_a_demo/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR/artifacts" "$RUN_DIR/metrics"
capture_cpu_run_context "$RUN_DIR"

set +e
/usr/bin/time -v -o "$RUN_DIR/metrics/time-stage1-group-a-demo.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/render_stage1_group_a_demo.py" \
        --protocol "$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/stage1_group_a_protocol_v7.yaml" \
        --capture-root "$CAPTURE_ROOT" \
        --bundle "$SOURCE_RUN/results/stage1_group_a.joblib" \
        --bundle-manifest "$SOURCE_RUN/results/bundle_manifest.json" \
        --evaluation-report "$SOURCE_RUN/results/evaluation_report.json" \
        --output-dir "$RUN_DIR/artifacts" \
        > "$RUN_DIR/stage1-group-a-demo.log" 2>&1
APPLICATION_RC=$?
set -e
echo "$APPLICATION_RC" > "$RUN_DIR/context/application-exit-code.txt"
sstat --jobs="${SLURM_JOB_ID}.batch" --format=JobID,AveCPU,MaxRSS,MaxVMSize -P \
    > "$RUN_DIR/metrics/slurm.csv" || true
capture_cpu_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
exit "$APPLICATION_RC"
