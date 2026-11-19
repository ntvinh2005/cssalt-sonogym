#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit command}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common_cpu.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

SOURCE_RUN="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_dataset_v10/51831979"
TRAINING_RUN="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_simulator_proxy_exploratory_v1/51836756/training"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_proxy_l4_failure_audit_v1/$SLURM_JOB_ID"
SOURCE_ZIP="$SOURCE_RUN/s0010_tp_surface_core_review_v3_51831979.zip"
OUTPUT="$RUN_DIR/audit"

mkdir -p "$RUN_DIR"/{context,metrics} "$OUTPUT"
capture_cpu_run_context "$RUN_DIR"
sha256sum "$SOURCE_ZIP" "$TRAINING_RUN/tp_surface_core_simulator_proxy_extratrees.joblib" \
    > "$RUN_DIR/context/frozen-inputs.sha256"

/usr/bin/time -v -o "$RUN_DIR/metrics/time-tests.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/tests/test_tp_surface_core_proxy_failure_audit.py" \
    > "$RUN_DIR/tests-failure-audit.log" 2>&1
/usr/bin/time -v -o "$RUN_DIR/metrics/time-proxy-regression.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/tests/test_tp_surface_core_proxy_training.py" \
    > "$RUN_DIR/tests-proxy-regression.log" 2>&1
/usr/bin/time -v -o "$RUN_DIR/metrics/time-audit.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/audit_tp_surface_core_proxy_failures.py" \
        --dataset-dir "$SOURCE_RUN/dataset" \
        --training-dir "$TRAINING_RUN" \
        --source-zip "$SOURCE_ZIP" \
        --output-dir "$OUTPUT" \
        --surface-slab-tolerance-mm 1.5 \
        > "$RUN_DIR/audit.log" 2>&1

"$PYTHON" - "$OUTPUT/audit_result.json" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
if result["l4_false_positive_count"] != 14 or result["l4_false_negative_count"] != 5:
    raise SystemExit("frozen L4 baseline was not reproduced")
if result["fit_called"]:
    raise SystemExit("failure audit must not fit a model")
PY

ZIP="$RUN_DIR/s0010_tp_surface_core_proxy_l4_failure_audit_v1_${SLURM_JOB_ID}.zip"
(cd "$OUTPUT" && find . -type f ! -name checksums.sha256 -print0 | sort -z | xargs -0 sha256sum > checksums.sha256)
(cd "$OUTPUT" && zip -qr "$ZIP" .)
sha256sum "$ZIP" > "$ZIP.sha256"
capture_cpu_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
