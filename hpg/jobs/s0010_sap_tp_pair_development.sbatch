#!/usr/bin/env bash
#SBATCH --job-name=sg_sap_tp_pair
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept-b
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/%x_%j.out

set -euo pipefail
: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck disable=SC1090
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

PROTOCOL_STEM="${SAP_TP_PAIR_PROTOCOL_STEM:-s0010_sap_tp_pair_development_v1}"
PROTOCOL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/$PROTOCOL_STEM.yaml"
PROTOCOL_SEAL="$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/$PROTOCOL_STEM.sha256"
SOURCE_RUN="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_l4mb_sap_tp_signal_v1/52300444"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/$PROTOCOL_STEM/$SLURM_JOB_ID"
PREDICTION="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_semantic_review_v3/52078067/raw_prediction/s0010.nii.gz"
SEGMENTATION="$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
LABEL_CONVERSION="$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml"
export PYTHONPATH="$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/evaluation:$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$SONOGYM_CACHE_DIR/matplotlib"

mkdir -p "$RUN_DIR"/{analysis,context} "$MPLCONFIGDIR"
capture_cpu_run_context "$RUN_DIR"
cp "$PROTOCOL" "$PROTOCOL_SEAL" "$RUN_DIR/context/"
(cd "$RUN_DIR/context" && sha256sum -c "$(basename "$PROTOCOL_SEAL")")

"$PYTHON" -m unittest -q \
  tests.test_ultrasound_sap_tp_configuration \
  tests.test_s0010_sap_tp_signal \
  > "$RUN_DIR/analysis/tests.log" 2>&1

"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/run_s0010_sap_tp_pair_development.py" \
  --protocol "$PROTOCOL" \
  --run-root "$SOURCE_RUN" \
  --prediction "$PREDICTION" \
  --segmentation "$SEGMENTATION" \
  --label-conversion "$LABEL_CONVERSION" \
  --output-dir "$RUN_DIR/analysis/result" \
  > "$RUN_DIR/analysis/run.log" 2>&1

capture_cpu_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
