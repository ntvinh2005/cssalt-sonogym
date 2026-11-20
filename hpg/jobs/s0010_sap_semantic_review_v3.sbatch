#!/usr/bin/env bash
#SBATCH --job-name=sg_sap_sem_v3
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept-b
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --time=00:30:00
#SBATCH --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/%x_%j.out

set -euo pipefail

SONOGYM_CONFIG_FILE="${SONOGYM_CONFIG_FILE:-/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"

[[ "$SLURM_JOB_ACCOUNT" == "anest-dept" ]] || { echo "Unexpected account" >&2; exit 2; }
[[ "$SLURM_CPUS_PER_TASK" -le 4 ]] || { echo "CPU budget exceeded" >&2; exit 2; }
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "No GPU allocated" >&2; exit 2; }
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#GPU_NAMES[@]}" -eq 1 && "${GPU_NAMES[0]}" == *L4* ]] || {
    echo "Expected one NVIDIA L4; saw ${GPU_NAMES[*]}" >&2
    exit 2
}

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl-sap-v3"
export nnUNet_raw_data_base="$SONOGYM_RUNTIME_DIR/external/Spine_Subregions/V1/nnUNet_raw"
export nnUNet_preprocessed="$SONOGYM_RUNTIME_DIR/external/Spine_Subregions/V1/nnUNet_preprocessed"
export RESULTS_FOLDER="$SONOGYM_RUNTIME_DIR/external/Spine_Subregions/V1/nnUNet_results"

PROJECT="$SONOGYM_PROJECT_DIR"
PYTHON="$SONOGYM_RUNTIME_DIR/envs/nnunet-v1-spine-subregions/bin/python"
PREDICT="$SONOGYM_RUNTIME_DIR/envs/nnunet-v1-spine-subregions/bin/nnUNet_predict"
PROTOCOL="$PROJECT/experiments/unity_two_stage_transfer/s0010_sap_semantic_review_v3_protocol.yaml"
SEAL="${PROTOCOL%.yaml}.sha256"
MODEL_ARCHIVE="$SONOGYM_RUNTIME_DIR/external/Spine_Subregions/V1/nnUNetTrainerV2__nnUNetPlansv2.1.zip"
CT="$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset/s0010/ct.nii.gz"
SEGMENTATION="$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
V2_PACKAGE="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_static_review_v2/manual_20261120T0120/review_packages/s0010_right_l4l5_sap_review_v2"
ARCHIVED_US="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_mbb_single_anatomy_poc_v3/52039283"
RUN_DIR="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_semantic_review_v3/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"/{context,input,raw_prediction,preflight,review_packages,metrics,logs}

date --iso-8601=seconds > "$RUN_DIR/context/start-time.txt"
git -C "$PROJECT" rev-parse HEAD > "$RUN_DIR/context/git-commit.txt"
git -C "$PROJECT" status --short > "$RUN_DIR/context/git-status.txt"
git -C "$PROJECT" diff --binary > "$RUN_DIR/context/uncommitted.patch"
scontrol show job "$SLURM_JOB_ID" > "$RUN_DIR/context/slurm-job.txt"
cp "$PROTOCOL" "$SEAL" "$RUN_DIR/context/"
(cd "$RUN_DIR/context" && sha256sum -c "$(basename "$SEAL")")
"$PYTHON" -m pip freeze > "$RUN_DIR/context/pip-freeze.txt"
nvidia-smi --query-gpu=timestamp,name,uuid,driver_version,memory.total --format=csv > "$RUN_DIR/context/gpu.txt"

nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv --loop=2 \
    > "$RUN_DIR/metrics/gpu-telemetry.csv" &
TELEMETRY_PID=$!
trap 'kill "$TELEMETRY_PID" 2>/dev/null || true' EXIT

cd "$PROJECT"
"$PYTHON" tests/test_s0010_sap_semantic_review_v3.py > "$RUN_DIR/logs/tests.log" 2>&1
"$PYTHON" workflows/two_stage_navigation/build_s0010_sap_semantic_review_v3.py preflight \
    --ct "$CT" --segmentation "$SEGMENTATION" --model-archive "$MODEL_ARCHIVE" \
    --expected-model-sha256 384b6ead3dd4df2b34d3fc6dc1d72c845dab76807b237a488c91d56758c06938 \
    --output-dir "$RUN_DIR/preflight" > "$RUN_DIR/logs/preflight.log" 2>&1

ln -s "$CT" "$RUN_DIR/input/s0010_0000.nii.gz"
INFERENCE_COMMAND="$PREDICT -i $RUN_DIR/input -o $RUN_DIR/raw_prediction -t 005 -m 3d_fullres -tr nnUNetTrainerV2 -p nnUNetPlansv2.1 -f 0 --num_threads_preprocessing 2 --num_threads_nifti_save 1 --overwrite_existing -chk model_final_checkpoint"
/usr/bin/time -v -o "$RUN_DIR/metrics/time-inference.txt" \
    "$PREDICT" -i "$RUN_DIR/input" -o "$RUN_DIR/raw_prediction" \
    -t 005 -m 3d_fullres -tr nnUNetTrainerV2 -p nnUNetPlansv2.1 -f 0 \
    --num_threads_preprocessing 2 --num_threads_nifti_save 1 \
    --overwrite_existing -chk model_final_checkpoint \
    > "$RUN_DIR/logs/inference.log" 2>&1

[[ -s "$RUN_DIR/raw_prediction/s0010.nii.gz" ]] || { echo "Missing prediction" >&2; exit 3; }
GPU_NAME="${GPU_NAMES[0]}"
ELAPSED_SECONDS=$(awk '/Elapsed \(wall clock\) time/{value=$NF} END{split(value,a,":"); if(length(a)==3) print a[1]*3600+a[2]*60+a[3]; else print a[1]*60+a[2]}' "$RUN_DIR/metrics/time-inference.txt")
/usr/bin/time -v -o "$RUN_DIR/metrics/time-build.txt" \
    "$PYTHON" workflows/two_stage_navigation/build_s0010_sap_semantic_review_v3.py build \
    --ct "$CT" --segmentation "$SEGMENTATION" \
    --prediction "$RUN_DIR/raw_prediction/s0010.nii.gz" \
    --model-archive "$MODEL_ARCHIVE" \
    --expected-model-sha256 384b6ead3dd4df2b34d3fc6dc1d72c845dab76807b237a488c91d56758c06938 \
    --v2-package "$V2_PACKAGE" --archived-ultrasound-root "$ARCHIVED_US" \
    --output-root "$RUN_DIR/review_packages" --job-id "$SLURM_JOB_ID" \
    --inference-command "$INFERENCE_COMMAND" --elapsed-seconds "$ELAPSED_SECONDS" \
    --gpu-name "$GPU_NAME" --cpus "$SLURM_CPUS_PER_TASK" \
    > "$RUN_DIR/logs/build.log" 2>&1

PACKAGE="$RUN_DIR/review_packages/s0010_right_l4l5_sap_semantic_review_v3"
(cd "$PACKAGE" && sha256sum -c checksums.sha256)
unzip -t "$RUN_DIR/review_packages/s0010_right_l4l5_sap_semantic_review_v3.zip" \
    > "$RUN_DIR/logs/zip-test.log"
sha256sum "$RUN_DIR/review_packages/s0010_right_l4l5_sap_semantic_review_v3.zip" \
    > "$RUN_DIR/review_packages/s0010_right_l4l5_sap_semantic_review_v3.zip.sha256"

kill "$TELEMETRY_PID" 2>/dev/null || true
wait "$TELEMETRY_PID" 2>/dev/null || true
trap - EXIT
date --iso-8601=seconds > "$RUN_DIR/context/end-time.txt"
sstat --jobs="${SLURM_JOB_ID}.batch" --format=JobID,AveCPU,MaxRSS,MaxVMSize -P > "$RUN_DIR/metrics/slurm.csv" || true
du -sb "$RUN_DIR" > "$RUN_DIR/metrics/storage-bytes.txt"
find "$RUN_DIR" -type f -print | sort > "$RUN_DIR/files.txt"
