#!/usr/bin/env bash
#SBATCH --job-name=sg_l4graph_prep
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
: "${L4_VIEWPOINT_JOB_ID:=52355104}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
# shellcheck disable=SC1090
source "$SONOGYM_PROJECT_DIR/hpg/jobs/common_cpu.sh"

RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_l4_landmark_graph_development_prepare/$SLURM_JOB_ID"
PREDICTION="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_semantic_review_v3/52078067/raw_prediction/s0010.nii.gz"
SEGMENTATION="$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
LABEL_CONVERSION="$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml"
EXACT="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_l4mb_broad_probe_search/$L4_VIEWPOINT_JOB_ID/search/exact_candidates.csv"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery:$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/evaluation:$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation:$SONOGYM_PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

mkdir -p "$RUN_DIR"
capture_cpu_run_context "$RUN_DIR"
"$PYTHON" -m unittest -q tests.test_two_stage_contracts \
  tests.test_acquisition_integrity tests.test_ultrasound_spatial_map \
  tests.test_s0010_landmark_graph_confirmation \
  > "$RUN_DIR/tests.log" 2>&1
"$PYTHON" "$SONOGYM_PROJECT_DIR/workflows/two_stage_navigation/prepare_s0010_l4_landmark_graph_development.py" \
  --base-protocol "$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_sap_tp_public_6dof_confirmation_v1.yaml" \
  --exclude-protocol "$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_l4mb_sap_tp_signal_v1.yaml" \
  --negative-protocol "$SONOGYM_PROJECT_DIR/experiments/unity_two_stage_transfer/s0010_l4mb_target_observability_v1.yaml" \
  --exact-candidates "$EXACT" --prediction "$PREDICTION" \
  --segmentation "$SEGMENTATION" --label-conversion "$LABEL_CONVERSION" \
  --patient-surface-dir "$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010" \
  --implementation-commit "$(git -C "$SONOGYM_PROJECT_DIR" rev-parse HEAD)" \
  --output-dir "$RUN_DIR/prepared" > "$RUN_DIR/stdout.log" 2> "$RUN_DIR/stderr.log"
capture_cpu_run_end "$RUN_DIR"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
