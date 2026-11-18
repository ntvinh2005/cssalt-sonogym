#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit command}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common.sh"

PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_protocol_v9.yaml"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_review_v2/$SLURM_JOB_ID"
V6_RUN="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_dataset_v6/51811811"
V7_RUN="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_dataset_v7/51815016"
V8_RUN="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_dataset_v8/51817719"
V9_RUN="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_dataset_v9/51818619"
GEOMETRY_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_right_sap_tp_correction/51779234"
PYTHON="$SONOGYM_ENV_PREFIX/bin/python"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg

mkdir -p "$RUN_DIR"/{results,metrics,context,sessions}
capture_run_context "$RUN_DIR"
cp "$PROTOCOL" "$RUN_DIR/context/"

(
    cd "$SONOGYM_PROJECT_DIR"
    "$PYTHON" tests/test_tp_surface_core_training.py
) > "$RUN_DIR/results/focused-tests.log" 2>&1

/usr/bin/time -v -o "$RUN_DIR/metrics/time-dataset.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/prepare_tp_surface_core_dataset.py" \
    --protocol "$PROTOCOL" \
    --capture-root "$V6_RUN" --capture-root "$V7_RUN" \
    --capture-root "$V8_RUN" --capture-root "$V9_RUN" \
    --geometry-root "$GEOMETRY_ROOT" \
    --prior-dataset-dir "$V9_RUN/dataset" \
    --output-dir "$RUN_DIR/dataset" \
    > "$RUN_DIR/results/dataset-preparation.log" 2>&1

set +e
/usr/bin/time -v -o "$RUN_DIR/metrics/time-preflight.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/train_tp_surface_core.py" \
    --dataset-dir "$RUN_DIR/dataset" --output-dir "$RUN_DIR/preflight" --no-fit \
    > "$RUN_DIR/results/training-preflight.log" 2>&1
PREFLIGHT_RC=$?
set -e
[[ "$PREFLIGHT_RC" -eq 3 ]] || {
    echo "Expected fail-closed review-pending result, got $PREFLIGHT_RC" >&2
    exit 4
}

ln -s "$V8_RUN/sessions/s0010tpscv8_smoke" "$RUN_DIR/sessions/s0010tpscv8_smoke"
ZIP="$RUN_DIR/s0010_tp_surface_core_review_selection_v2_${SLURM_JOB_ID}.zip"
/usr/bin/time -v -o "$RUN_DIR/metrics/time-package.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/package_tp_surface_core_review.py" \
    --run-dir "$RUN_DIR" --dataset-dir "$RUN_DIR/dataset" \
    --preflight-dir "$RUN_DIR/preflight" --geometry-root "$GEOMETRY_ROOT" \
    --smoke-session-id s0010tpscv8_smoke --output-zip "$ZIP" \
    > "$RUN_DIR/results/review-package.log" 2>&1

STAGING="${ZIP%.zip}"
(cd "$STAGING" && sha256sum -c checksums.sha256)
sha256sum "$ZIP" > "$ZIP.sha256"
du -sb "$RUN_DIR" > "$RUN_DIR/metrics/storage-bytes.txt"
capture_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
