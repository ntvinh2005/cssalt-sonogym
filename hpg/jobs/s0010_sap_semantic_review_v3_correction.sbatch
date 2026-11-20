#!/usr/bin/env bash
#SBATCH --job-name=sg_sap_v3_fix
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
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl-sap-v3-fix"

PROJECT="$SONOGYM_PROJECT_DIR"
PYTHON="$SONOGYM_RUNTIME_DIR/envs/nnunet-v1-spine-subregions/bin/python"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_semantic_review_v3_correction/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"/{context,review_packages,logs,metrics}
git -C "$PROJECT" rev-parse HEAD > "$RUN_DIR/context/git-commit.txt"
git -C "$PROJECT" status --short > "$RUN_DIR/context/git-status.txt"
scontrol show job "$SLURM_JOB_ID" > "$RUN_DIR/context/slurm-job.txt"

cd "$PROJECT"
"$PYTHON" tests/test_s0010_sap_semantic_review_v3.py > "$RUN_DIR/logs/tests-v3.log" 2>&1
"$PYTHON" tests/test_s0010_sap_semantic_review_v3_correction.py > "$RUN_DIR/logs/tests-correction.log" 2>&1
/usr/bin/time -v -o "$RUN_DIR/metrics/time-build.txt" \
  "$PYTHON" workflows/two_stage_navigation/correct_s0010_sap_semantic_review_v3.py \
  --ct "$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset/s0010/ct.nii.gz" \
  --segmentation "$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz" \
  --prediction "$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_semantic_review_v3/52078067/raw_prediction/s0010.nii.gz" \
  --label-conversion "$PROJECT/source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml" \
  --v2-package "$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_static_review_v2/manual_20261120T0120/review_packages/s0010_right_l4l5_sap_review_v2" \
  --original-v3-zip "$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_semantic_review_v3/52078067/review_packages/s0010_right_l4l5_sap_semantic_review_v3.zip" \
  --output-root "$RUN_DIR/review_packages" > "$RUN_DIR/logs/build.log" 2>&1

PACKAGE="$RUN_DIR/review_packages/s0010_right_l4l5_sap_semantic_review_v3_correction"
(cd "$PACKAGE" && sha256sum -c checksums.sha256)
unzip -t "$RUN_DIR/review_packages/s0010_right_l4l5_sap_semantic_review_v3_correction.zip" > "$RUN_DIR/logs/zip-test.log"
sha256sum "$RUN_DIR/review_packages/s0010_right_l4l5_sap_semantic_review_v3_correction.zip" > "$RUN_DIR/review_packages/s0010_right_l4l5_sap_semantic_review_v3_correction.zip.sha256"
