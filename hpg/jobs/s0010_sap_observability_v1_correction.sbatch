#!/usr/bin/env bash
#SBATCH --job-name=sg_sap_obs_fix
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
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl-sap-observability-fix"

PROJECT="$SONOGYM_PROJECT_DIR"
PYTHON="$SONOGYM_ENV_PREFIX/bin/python"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_observability_v1_correction/$SLURM_JOB_ID"
PACKAGE="$RUN_DIR/review_packages/s0010_right_l4l5_sap_observability_v1_correction"
ZIP="$RUN_DIR/review_packages/s0010_right_l4l5_sap_observability_v1_correction.zip"
SOURCE_RUN="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_observability_v1r1/52092083"
PREDICTION="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_semantic_review_v3/52078067/raw_prediction/s0010.nii.gz"
SEGMENTATION="$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
LABEL_CONVERSION="$PROJECT/source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml"
POLYLINES="$PROJECT/experiments/unity_two_stage_transfer/s0010_sap_observability_cortical_polylines_v1.csv"

mkdir -p "$RUN_DIR"/{context,review_packages,logs,metrics} "$MPLCONFIGDIR"
git -C "$PROJECT" rev-parse HEAD > "$RUN_DIR/context/git-commit.txt"
git -C "$PROJECT" status --short > "$RUN_DIR/context/git-status.txt"
git -C "$PROJECT" diff --binary > "$RUN_DIR/context/uncommitted.patch"
scontrol show job "$SLURM_JOB_ID" > "$RUN_DIR/context/slurm-job.txt"

cd "$PROJECT"
PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$PROJECT/source/spinal_surgery:$PROJECT/workflows/two_stage_navigation" \
  "$PYTHON" -m unittest -v \
  tests.test_s0010_sap_observability \
  tests.test_s0010_sap_observability_correction \
  tests.test_s0010_sap_observability_contract \
  tests.test_ultrasound_plane_geometry \
  tests.test_acquisition_integrity \
  > "$RUN_DIR/logs/tests.log" 2>&1

/usr/bin/time -v -o "$RUN_DIR/metrics/time-build.txt" \
  "$PYTHON" workflows/two_stage_navigation/correct_s0010_sap_observability_v1.py \
  --run-root "$SOURCE_RUN" \
  --prediction "$PREDICTION" \
  --segmentation "$SEGMENTATION" \
  --label-conversion "$LABEL_CONVERSION" \
  --polylines "$POLYLINES" \
  --output-root "$PACKAGE" \
  --zip "$ZIP" \
  --test-log "$RUN_DIR/logs/tests.log" \
  > "$RUN_DIR/logs/build.log" 2>&1

(cd "$PACKAGE" && sha256sum -c checksums.sha256)
unzip -t "$ZIP" > "$RUN_DIR/logs/zip-test.log"
sha256sum "$ZIP" > "$ZIP.sha256"
du -sb "$RUN_DIR" > "$RUN_DIR/metrics/storage-bytes.txt"
