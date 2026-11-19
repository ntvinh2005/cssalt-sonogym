#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit command}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common_cpu.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

TRAINING_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_simulator_proxy_exploratory_v1/51836756/training"
DATASET_MANIFEST="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_dataset_v10/51831979/dataset/candidate_sample_manifest.csv"
HARD_NEGATIVE_MANIFEST="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_proxy_tristate_hard_negative_v1/51956725/preparation/l4_hard_negative_selected.csv"
PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_tp_surface_core_proxy_l4_existing_data_ablation_v1.yaml"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_proxy_l4_existing_data_ablation_v1/$SLURM_JOB_ID"
OUTPUT="$RUN_DIR/ablation"

mkdir -p "$RUN_DIR"/{context,metrics} "$OUTPUT"
capture_cpu_run_context "$RUN_DIR"
sha256sum "$PROTOCOL" "$DATASET_MANIFEST" "$HARD_NEGATIVE_MANIFEST" \
    "$TRAINING_DIR/tp_surface_core_simulator_proxy_extratrees.joblib" \
    "$TRAINING_DIR/proxy_label_audit.csv" "$TRAINING_DIR/proxy_sample_manifest.csv" \
    > "$RUN_DIR/context/frozen-inputs.sha256"

/usr/bin/time -v -o "$RUN_DIR/metrics/time-tests.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/tests/test_tp_surface_core_l4_existing_data_ablation.py" \
    > "$RUN_DIR/tests-ablation.log" 2>&1
/usr/bin/time -v -o "$RUN_DIR/metrics/time-proxy-regression.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/tests/test_tp_surface_core_proxy_training.py" \
    > "$RUN_DIR/tests-proxy-regression.log" 2>&1
/usr/bin/time -v -o "$RUN_DIR/metrics/time-ablation.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/run_tp_surface_core_l4_existing_data_ablation.py" \
        --training-dir "$TRAINING_DIR" \
        --dataset-manifest "$DATASET_MANIFEST" \
        --hard-negative-manifest "$HARD_NEGATIVE_MANIFEST" \
        --protocol "$PROTOCOL" \
        --output-dir "$OUTPUT" \
        > "$RUN_DIR/ablation.log" 2>&1

"$PYTHON" - "$OUTPUT/result.json" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
if result["development_used_for_selection"]:
    raise SystemExit("development data influenced candidate selection")
if result["l5_modified"] or result["new_capture_run"]:
    raise SystemExit("ablation changed the L5 control or launched capture")
PY

capture_cpu_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
