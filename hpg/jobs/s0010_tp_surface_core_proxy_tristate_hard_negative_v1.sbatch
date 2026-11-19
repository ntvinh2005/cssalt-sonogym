#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit command}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common_cpu.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

TRAINING_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_simulator_proxy_exploratory_v1/51836756/training"
FAILURE_AUDIT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_proxy_l4_failure_audit_v1/51936904/audit/l4_false_positive_audit.csv"
PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_tp_surface_core_proxy_tristate_hard_negative_v1.yaml"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_proxy_tristate_hard_negative_v1/$SLURM_JOB_ID"
OUTPUT="$RUN_DIR/preparation"

mkdir -p "$RUN_DIR"/{context,metrics} "$OUTPUT"
capture_cpu_run_context "$RUN_DIR"
sha256sum "$PROTOCOL" "$TRAINING_DIR/tp_surface_core_simulator_proxy_extratrees.joblib" \
    "$TRAINING_DIR/proxy_label_audit.csv" "$TRAINING_DIR/proxy_sample_manifest.csv" \
    "$FAILURE_AUDIT" > "$RUN_DIR/context/frozen-inputs.sha256"

/usr/bin/time -v -o "$RUN_DIR/metrics/time-tests.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/tests/test_tp_surface_core_proxy_tristate_hard_negatives.py" \
    > "$RUN_DIR/tests-tristate.log" 2>&1
/usr/bin/time -v -o "$RUN_DIR/metrics/time-proxy-regression.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/tests/test_tp_surface_core_proxy_training.py" \
    > "$RUN_DIR/tests-proxy-regression.log" 2>&1
/usr/bin/time -v -o "$RUN_DIR/metrics/time-preparation.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/prepare_tp_surface_core_proxy_tristate_hard_negatives.py" \
        --training-dir "$TRAINING_DIR" \
        --failure-audit-csv "$FAILURE_AUDIT" \
        --protocol "$PROTOCOL" \
        --output-dir "$OUTPUT" \
        > "$RUN_DIR/preparation.log" 2>&1

"$PYTHON" - "$OUTPUT/result.json" "$OUTPUT/l4_hard_negative_selected.csv" <<'PY'
import csv
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
rows = list(csv.DictReader(open(sys.argv[2], newline="", encoding="utf-8")))
if any(row["split"] != "train" for row in rows):
    raise SystemExit("a non-training row entered the hard-negative manifest")
if len({row["frame_sha256"] for row in rows}) != len(rows):
    raise SystemExit("hard-negative manifest contains correlated same-frame rows")
if result["future_l4_v2_fit_run"]:
    raise SystemExit("preparation job must not fit L4 v2")
PY

capture_cpu_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
