#!/usr/bin/env bash
#SBATCH --job-name=sg_sap_corr_cpu
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept-b
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/%x_%j.out

set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck source=common_cpu.sh
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

RUN="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_correspondence_cpu/$SLURM_JOB_ID"
ARCHIVE="$RUN/dataset_archive"
LOGISTIC="$RUN/logistic"
EXTRA_TREES="$RUN/extra_trees"
SELECTED="$RUN/selected"
EXTERNAL="$RUN/opened_calibration_evaluation"
PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_sap_existing_capture_development_protocol_v3.yaml"
PREDICTION="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_semantic_review_v3/52078067/raw_prediction/s0010.nii.gz"
SEGMENTATION="$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
LABELS="$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml"
NEW_CALIBRATION="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_validation_resume_v1/52182100/analysis_v2/dataset_calibration"
RESUME_PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_sap_validation_resume_protocol_v1.yaml"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$SONOGYM_CACHE_DIR/matplotlib"
mkdir -p "$ARCHIVE" "$LOGISTIC" "$EXTRA_TREES" "$EXTERNAL" "$MPLCONFIGDIR"
capture_cpu_run_context "$RUN"
cp "$PROTOCOL" "${PROTOCOL%.yaml}.sha256" "$RESUME_PROTOCOL" "${RESUME_PROTOCOL%.yaml}.sha256" "$RUN/context/"

set +e
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/prepare_s0010_sap_dataset.py" \
    --protocol "$PROTOCOL" --run-root "$RUN" --prediction "$PREDICTION" \
    --segmentation "$SEGMENTATION" --label-conversion "$LABELS" \
    --phase development --require-all-proposals-clear-for-negative-image \
    --minimum-distinct-translation-mm 2.0 \
    --minimum-distinct-orientation-deg 1.0 \
    --output-dir "$ARCHIVE" > "$ARCHIVE/stdout.json" 2> "$ARCHIVE/stderr.log"
BUILDER_RC=$?
echo "$BUILDER_RC" > "$ARCHIVE/exit-code.txt"
if [[ "$BUILDER_RC" -eq 1 ]]; then
    exit "$BUILDER_RC"
fi

for estimator in logistic extra_trees; do
    output="$RUN/$estimator"
    "$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/validate_s0010_sap_correspondence.py" \
        --dataset-dir "$ARCHIVE" --protocol "$PROTOCOL" --estimator "$estimator" \
        --output-dir "$output" > "$output/stdout.json" 2> "$output/stderr.log"
    echo "$?" > "$output/exit-code.txt"
done
set -e

"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/select_s0010_sap_correspondence_checkpoint.py" \
    --candidate "$LOGISTIC/correspondence_validation.json" \
    --candidate "$EXTRA_TREES/correspondence_validation.json" \
    --output-dir "$SELECTED" > "$RUN/selection-stdout.json"

set +e
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/evaluate_s0010_sap_correspondence.py" \
    --dataset-dir "$NEW_CALIBRATION" --protocol "$RESUME_PROTOCOL" \
    --checkpoint-dir "$SELECTED" --split calibration --output-dir "$EXTERNAL" \
    > "$EXTERNAL/stdout.json" 2> "$EXTERNAL/stderr.log"
echo "$?" > "$EXTERNAL/exit-code.txt"
set -e
capture_cpu_run_end "$RUN"
find "$RUN" -type f -print | sort > "$RUN/files.txt"
