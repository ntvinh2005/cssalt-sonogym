#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit command}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common.sh"

RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_dataset_v6/51811811"
PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_protocol_v6.yaml"
GEOMETRY_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_right_sap_tp_correction/51779234"
DATASET_DIR="$RUN_DIR/dataset_final"
PREFLIGHT_DIR="$RUN_DIR/preflight_final"
OUTPUT_ZIP="$RUN_DIR/s0010_tp_surface_core_review_51811811_final.zip"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

[[ ! -e "$DATASET_DIR" && ! -e "$PREFLIGHT_DIR" && ! -e "$OUTPUT_ZIP" ]] || {
    echo "Refusing to overwrite existing finalization artifacts" >&2
    exit 2
}

cd "$SONOGYM_PROJECT_DIR"
"$PYTHON" tests/test_tp_surface_core_training.py
"$PYTHON" tests/test_tp_surface_core_capture_contract.py
"$PYTHON" -m py_compile \
    workflows/two_stage_navigation/prepare_tp_surface_core_dataset.py \
    workflows/two_stage_navigation/package_tp_surface_core_review.py

"$PYTHON" workflows/two_stage_navigation/prepare_tp_surface_core_dataset.py \
    --protocol "$PROTOCOL" --capture-root "$RUN_DIR" --geometry-root "$GEOMETRY_ROOT" \
    --output-dir "$DATASET_DIR" > "$RUN_DIR/results/dataset-finalization.log" 2>&1

set +e
"$PYTHON" workflows/two_stage_navigation/train_tp_surface_core.py \
    --dataset-dir "$DATASET_DIR" --output-dir "$PREFLIGHT_DIR" --no-fit \
    > "$RUN_DIR/results/training-preflight-final.log" 2>&1
PREFLIGHT_RC=$?
set -e
[[ "$PREFLIGHT_RC" -eq 3 ]] || {
    echo "Expected fail-closed no-fit result, got $PREFLIGHT_RC" >&2
    exit 3
}

"$PYTHON" workflows/two_stage_navigation/package_tp_surface_core_review.py \
    --run-dir "$RUN_DIR" --dataset-dir "$DATASET_DIR" --preflight-dir "$PREFLIGHT_DIR" \
    --geometry-root "$GEOMETRY_ROOT" --output-zip "$OUTPUT_ZIP" \
    > "$RUN_DIR/results/review-package-final.log" 2>&1

STAGING="${OUTPUT_ZIP%.zip}"
(cd "$STAGING" && sha256sum -c checksums.sha256)
sha256sum "$OUTPUT_ZIP" > "$OUTPUT_ZIP.sha256"
