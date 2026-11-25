#!/usr/bin/env bash
#SBATCH --job-name=sg_robust_review
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=30G
#SBATCH --time=00:18:00

set -euo pipefail
checkout="$PWD"
[[ "$(squeue -A anest-dept --states=RUNNING,COMPLETING,CONFIGURING -h -o '%C' | awk '{s+=$1} END {print s+0}')" -le 4 ]]
source "$checkout/hpg/jobs/common.sh"
SONOGYM_PROJECT_DIR="$checkout"
export SONOGYM_GUIDANCE_CONFIG="$checkout/config/guidance_baseline.yaml"
export PYTHONPATH="$checkout/source/spinal_surgery:$SONOGYM_RUNTIME_DIR/envs/stage1-python:$checkout/source/spinal_surgery/spinal_surgery/evaluation:$checkout/workflows/two_stage_navigation:$checkout"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/sonogym-robust-$SLURM_JOB_ID"
[[ -z "$(git status --short)" && "$(git rev-parse HEAD)" == "$ROBUST_REVIEW_COMMIT" ]]
profile=experiments/unity_two_stage_transfer/s0010_simple_landmark_robustness_v1.yaml
conversion=source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml
destination="$ROBUST_OUTPUT/review-$SLURM_JOB_ID"
capture_run_context "$destination"
cp "$profile" "$destination/context"
sha256sum "$profile" "$ROBUST_PROTOCOL" "$conversion" > "$destination/context/configs.sha256"
"$PYTHON" -m unittest -q tests.test_surface_landmark_mapping tests.test_simple_landmark_mapping tests.test_acquisition_integrity > "$destination/context/tests.log" 2>&1
base="$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer"
original_data="$base/s0010_sonoanatomy_dataset_v5/52497655/dataset/dataset.npz"
original_manifest="$base/s0010_sonoanatomy_segmentation_v9/52498821/training/l5_split_manifest.csv"
phase () {
    local name="$1"
    shift
    printf '%s\tSTARTED\t%s\n' "$name" "$(date --iso-8601=seconds)" >> "$destination/attempts.tsv"
    if "$@" > "$destination/$name.log" 2>&1; then
        printf '%s\tCOMPLETED\t%s\n' "$name" "$(date --iso-8601=seconds)" >> "$destination/attempts.tsv"
    else
        printf '%s\tFAILED\t%s\n' "$name" "$(date --iso-8601=seconds)" >> "$destination/attempts.tsv"
        return 1
    fi
}
evaluation=("$PYTHON" workflows/two_stage_navigation/evaluate_robust_mapping.py
    --protocol "$ROBUST_PROTOCOL" --profile "$profile" --reference "$ROBUST_REFERENCE" --label-conversion "$conversion")
phase mapping "${evaluation[@]}" --acquisition "$ROBUST_ACQUISITION" --extended-tp --output "$destination/mapping" || true
if phase speckle "$PYTHON" workflows/two_stage_navigation/render_robust_speckle.py \
    --profile "$profile" --protocol "$ROBUST_PROTOCOL" --reference "$ROBUST_REFERENCE" \
    --acquisition "$ROBUST_ACQUISITION" --label-conversion "$conversion" \
    --output "$destination/E3-dataset" --headless --device cuda:0; then
    phase training "$PYTHON" workflows/two_stage_navigation/finetune_robust_l5.py \
        --profile "$profile" --protocol "$ROBUST_PROTOCOL" --dataset "$destination/E3-dataset" \
        --original-dataset "$original_data" --original-manifest "$original_manifest" \
        --output "$destination/E3-training" || true
fi
if [[ -s "$destination/E3-training/training_result.json" ]]; then
    phase trained_mapping "${evaluation[@]}" --fine-tuned "$destination/E3-training" --output "$destination/E3-mapping" || true
fi
demo="$ROBUST_ACQUISITION/E1_2026121700"
if [[ -s "$demo/evaluation.json" && -d "$demo/public/robot" ]]; then
    phase video "$PYTHON" workflows/two_stage_navigation/render_simple_sap_demo.py \
        --session "$demo" --protocol "$ROBUST_PROTOCOL" --output "$destination/demo" || true
fi
capture_run_end "$destination"
! rg -q 'FAILED' "$destination/attempts.tsv"
