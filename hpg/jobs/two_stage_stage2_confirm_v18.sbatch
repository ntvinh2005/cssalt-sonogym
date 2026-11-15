#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit command}"
: "${STAGE2_CONFIRMATION_CAPTURE_JOB_ID:?STAGE2_CONFIRMATION_CAPTURE_JOB_ID was not exported by the submit command}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common_cpu.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

[[ "$STAGE2_CONFIRMATION_CAPTURE_JOB_ID" =~ ^[0-9]+$ ]] || {
    echo "Invalid STAGE2_CONFIRMATION_CAPTURE_JOB_ID: $STAGE2_CONFIRMATION_CAPTURE_JOB_ID" >&2
    exit 2
}
PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/stage2_confirmation_protocol_v18.yaml"
PROTOCOL_SEAL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/stage2_confirmation_protocol_v18.sha256"
CAPTURE_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/stage2_confirmation_v18/$STAGE2_CONFIRMATION_CAPTURE_JOB_ID"
REFERENCE_CAPTURE_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/stage2_dataset_v14/51377943"
MODEL_BUNDLE="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/stage2_development_v15/51380423/results/stage2_models.joblib"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/stage2_confirmation_evaluation_v18/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR/results" "$RUN_DIR/metrics"
capture_cpu_run_context "$RUN_DIR"
cp "$PROTOCOL" "$PROTOCOL_SEAL" "$RUN_DIR/context/"
(
    cd "$RUN_DIR/context"
    sha256sum -c stage2_confirmation_protocol_v18.sha256
)

set +e
/usr/bin/time -v -o "$RUN_DIR/metrics/time-stage2-confirmation.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/evaluate_stage2_confirmation.py" \
        --protocol "$PROTOCOL" --capture-root "$CAPTURE_ROOT" \
        --reference-capture-root "$REFERENCE_CAPTURE_ROOT" \
        --model-bundle "$MODEL_BUNDLE" --output-dir "$RUN_DIR/results" \
        > "$RUN_DIR/stage2-confirmation.log" 2>&1
APPLICATION_RC=$?
set -e
echo "$APPLICATION_RC" > "$RUN_DIR/context/application-exit-code.txt"
sstat --jobs="${SLURM_JOB_ID}.batch" --format=JobID,AveCPU,MaxRSS,MaxVMSize -P \
    > "$RUN_DIR/metrics/slurm.csv" || true
capture_cpu_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
exit "$APPLICATION_RC"

