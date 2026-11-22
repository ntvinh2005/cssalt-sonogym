#!/usr/bin/env bash
#SBATCH --job-name=sg_pair_hn_ablate
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept-b
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/%x_%j.out

set -euo pipefail
: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck disable=SC1090
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_tp_pair_hard_negative_ablation/$SLURM_JOB_ID"
PREDICTION="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_semantic_review_v3/52078067/raw_prediction/s0010.nii.gz"
SEGMENTATION="$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
LABEL_CONVERSION="$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml"
export PYTHONPATH="$SONOGYM_PROJECT_DIR/source/spinal_surgery:$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/evaluation:$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

mkdir -p "$RUN_DIR"
capture_cpu_run_context "$RUN_DIR"
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/run_s0010_sap_tp_pair_hard_negative_ablation.py" \
  --training-protocol "$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_l4mb_sap_tp_signal_v1.yaml" \
  --training-run "$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_l4mb_sap_tp_signal_v1/52300444" \
  --development-protocol "$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_sap_tp_pair_confirmation_v1r3.yaml" \
  --development-run "$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_tp_pair_confirmation_v1r3/52308098" \
  --prediction "$PREDICTION" --segmentation "$SEGMENTATION" \
  --label-conversion "$LABEL_CONVERSION" --output-dir "$RUN_DIR/result" \
  > "$RUN_DIR/stdout.log" 2> "$RUN_DIR/stderr.log"
capture_cpu_run_end "$RUN_DIR"
