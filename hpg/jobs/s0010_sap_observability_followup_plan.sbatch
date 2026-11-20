#!/usr/bin/env bash
#SBATCH --job-name=sg_sap_plan
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/%x_%j.out

set -euo pipefail
source /blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl-sap-plan"

PROJECT="$SONOGYM_PROJECT_DIR"
PYTHON="$SONOGYM_ENV_PREFIX/bin/python"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_observability_followup_plan/$SLURM_JOB_ID"
REFERENCE="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_mbb_single_anatomy_poc_v3/52039283/stage2"
PREDICTION="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_semantic_review_v3/52078067/raw_prediction/s0010.nii.gz"
SEGMENTATION="$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
LABEL_CONVERSION="$PROJECT/source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml"
mkdir -p "$RUN_DIR"/{context,results,logs,metrics} "$MPLCONFIGDIR"
git -C "$PROJECT" rev-parse HEAD > "$RUN_DIR/context/git-commit.txt"
git -C "$PROJECT" status --short > "$RUN_DIR/context/git-status.txt"
scontrol show job "$SLURM_JOB_ID" > "$RUN_DIR/context/slurm-job.txt"

cd "$PROJECT"
PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$PROJECT/source/spinal_surgery:$PROJECT/workflows/two_stage_navigation" \
  "$PYTHON" -m unittest -v tests.test_s0010_sap_observability_followup \
  > "$RUN_DIR/logs/tests.log" 2>&1
/usr/bin/time -v -o "$RUN_DIR/metrics/time-plan.txt" \
  "$PYTHON" workflows/two_stage_navigation/plan_s0010_sap_observability_followup.py \
  --reference-root "$REFERENCE" \
  --prediction "$PREDICTION" \
  --segmentation "$SEGMENTATION" \
  --label-conversion "$LABEL_CONVERSION" \
  --output-root "$RUN_DIR/results/plan" \
  > "$RUN_DIR/logs/plan.log" 2>&1
