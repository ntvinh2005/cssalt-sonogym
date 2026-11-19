#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit command}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common.sh"

PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_protocol_v15.yaml"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_proxy_l4_coverage_repair_v3/$SLURM_JOB_ID"
OLD_TRAINING="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_simulator_proxy_exploratory_v1/51836756/training"
OLD_HARD="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_proxy_tristate_hard_negative_v1/51956725/preparation/l4_hard_negative_selected.csv"
GEOMETRY_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_right_sap_tp_correction/51779234"
PLANNER_ROOT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_tp_surface_core_proxy_l4_waypoint_plan_v1/52006787"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$RUN_DIR"/{sessions,dataset,training,metrics,context,results}
capture_run_context "$RUN_DIR"
cp "$PROTOCOL" "${PROTOCOL%.yaml}.sha256" "$RUN_DIR/context/"
cp "$PLANNER_ROOT/waypoint_plan.json" "$RUN_DIR/context/"
(cd "$RUN_DIR/context" && sha256sum -c s0010_tp_surface_core_dataset_protocol_v15.sha256)
echo "496781bd01761f59dce837716de3c3b2c000d20f302cf9ab6edb155ff1abc78e  $RUN_DIR/context/waypoint_plan.json" | sha256sum -c -

"$PYTHON" "$SONOGYM_PROJECT_DIR/tests/test_tp_surface_core_l4_waypoint_planner.py" > "$RUN_DIR/results/tests-planner.log" 2>&1
"$PYTHON" "$SONOGYM_PROJECT_DIR/tests/test_tp_surface_core_l4_coverage_repair.py" > "$RUN_DIR/results/tests-coverage-repair.log" 2>&1
"$PYTHON" "$SONOGYM_PROJECT_DIR/tests/test_tp_surface_core_targeted_supplement.py" > "$RUN_DIR/results/tests-targeted.log" 2>&1
"$PYTHON" "$SONOGYM_PROJECT_DIR/tests/test_acquisition_integrity.py" > "$RUN_DIR/results/tests-acquisition.log" 2>&1
"$PYTHON" "$SONOGYM_PROJECT_DIR/tests/test_ultrasound_plane_geometry.py" > "$RUN_DIR/results/tests-plane.log" 2>&1

capture_session() {
    local session_id="$1"
    local session_dir="$RUN_DIR/sessions/$session_id"
    mkdir -p "$session_dir"
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/collect_stage2_dataset.py" \
        --protocol "$PROTOCOL" --session-id "$session_id" --output-dir "$session_dir" \
        --assets-data-dir "$SONOGYM_ASSETS_DATA_DIR" --headless --device cuda:0 \
        > "$session_dir/launcher.log" 2>&1
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/verify_exact_pose_capture.py" \
        --protocol "$PROTOCOL" --session-dir "$session_dir" --assets-data-dir "$SONOGYM_ASSETS_DATA_DIR" \
        --output "$session_dir/verification_report.json" > "$session_dir/verifier.log" 2>&1
}

capture_session s0010tpscv15_train_targeted_00
capture_session s0010tpscv15_train_targeted_01
capture_session s0010tpscv15_cal_targeted_00
capture_session s0010tpscv15_eval_targeted_00

"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/prepare_tp_surface_core_dataset.py" \
    --protocol "$PROTOCOL" --capture-root "$RUN_DIR" --geometry-root "$GEOMETRY_ROOT" \
    --output-dir "$RUN_DIR/dataset" > "$RUN_DIR/results/dataset.log" 2>&1

"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/train_tp_surface_core_l4_coverage_repair.py" \
    --old-training-dir "$OLD_TRAINING" --old-hard-negatives "$OLD_HARD" \
    --new-dataset-manifest "$RUN_DIR/dataset/candidate_sample_manifest.csv" \
    --protocol "$PROTOCOL" --output-dir "$RUN_DIR/training" \
    > "$RUN_DIR/results/training.log" 2>&1

capture_run_end "$RUN_DIR"
du -sb "$RUN_DIR" > "$RUN_DIR/metrics/storage-bytes.txt"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
