#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit command}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common_cpu.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_simulator_proxy_exploratory_v1/$SLURM_JOB_ID"
SOURCE_RUN="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_dataset_v10/51831979"
SOURCE_ZIP="$SOURCE_RUN/s0010_tp_surface_core_review_v3_51831979.zip"
DATASET_DIR="$SOURCE_RUN/dataset"
CONTRACT="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_tp_surface_core_simulator_proxy_exploratory_v1_labels.csv"
CONTRACT_SEAL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_tp_surface_core_simulator_proxy_exploratory_v1_labels.sha256"
LAUNCHER="$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/train_tp_surface_core_proxy.py"

mkdir -p "$RUN_DIR"/{context,metrics,preflight,training}
capture_cpu_run_context "$RUN_DIR"
cp "$CONTRACT" "$CONTRACT_SEAL" "$RUN_DIR/context/"
(
    cd "$RUN_DIR/context"
    sha256sum -c "$(basename "$CONTRACT_SEAL")"
)
sha256sum "$SOURCE_ZIP" > "$RUN_DIR/context/source-zip.sha256"
sha256sum "$DATASET_DIR/candidate_sample_manifest.csv" > "$RUN_DIR/context/source-manifest.sha256"

/usr/bin/time -v -o "$RUN_DIR/metrics/time-tests.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/tests/test_tp_surface_core_proxy_training.py" \
    > "$RUN_DIR/tests-proxy.log" 2>&1
/usr/bin/time -v -o "$RUN_DIR/metrics/time-regression.txt" \
    "$PYTHON" "$SONOGYM_PROJECT_DIR/tests/test_tp_surface_core_training.py" \
    > "$RUN_DIR/tests-existing-training.log" 2>&1

/usr/bin/time -v -o "$RUN_DIR/metrics/time-preflight.txt" \
    "$PYTHON" "$LAUNCHER" \
        --dataset-dir "$DATASET_DIR" \
        --source-zip "$SOURCE_ZIP" \
        --label-contract "$CONTRACT" \
        --output-dir "$RUN_DIR/preflight" \
        --no-fit > "$RUN_DIR/preflight.log" 2>&1

"$PYTHON" - "$RUN_DIR/preflight/result.json" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
if result["status"] != "PROXY_TRAINING_READY" or result["fit_called"]:
    raise SystemExit("proxy no-fit did not pass fail-closed readiness")
PY

/usr/bin/time -v -o "$RUN_DIR/metrics/time-training.txt" \
    "$PYTHON" "$LAUNCHER" \
        --dataset-dir "$DATASET_DIR" \
        --source-zip "$SOURCE_ZIP" \
        --label-contract "$CONTRACT" \
        --output-dir "$RUN_DIR/training" > "$RUN_DIR/training.log" 2>&1

"$PYTHON" - "$RUN_DIR/training/result.json" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
if result["status"] != "EXPLORATORY_PROXY_TRAINING_COMPLETED":
    raise SystemExit("proxy training did not complete")
if not result["checkpoint_saved"] or not result["checkpoint_reload_verified"]:
    raise SystemExit("proxy checkpoint persistence check failed")
PY

find "$RUN_DIR" -type f -not -name checksums.sha256 -print0 \
    | sort -z | xargs -0 sha256sum > "$RUN_DIR/checksums.sha256"
date --iso-8601=seconds > "$RUN_DIR/context/end-time.txt"
