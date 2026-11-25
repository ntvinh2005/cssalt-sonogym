#!/usr/bin/env bash
#SBATCH --job-name=sg_robust
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=30G
#SBATCH --time=00:22:00

set -euo pipefail
checkout="$PWD"
[[ "$(squeue -A anest-dept --states=RUNNING,COMPLETING,CONFIGURING -h -o '%C' | awk '{s+=$1} END {print s+0}')" -le 4 ]]
source "$checkout/hpg/jobs/common.sh"
SONOGYM_PROJECT_DIR="$checkout"
export SONOGYM_GUIDANCE_CONFIG="$checkout/config/guidance_baseline.yaml"
export PYTHONPATH="$checkout/source/spinal_surgery:$SONOGYM_RUNTIME_DIR/envs/stage1-python:$checkout/source/spinal_surgery/spinal_surgery/evaluation:$checkout"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/sonogym-robust-$SLURM_JOB_ID"
[[ -z "$(git status --short)" && "$(git rev-parse HEAD)" == "$ROBUST_ACQUISITION_COMMIT" ]]
profile=experiments/unity_two_stage_transfer/s0010_simple_landmark_robustness_v1.yaml
baseline=experiments/unity_two_stage_transfer/s0010_simple_landmark_independent_v1.yaml
(cd "$(dirname "$baseline")" && sha256sum -c "$(basename "${baseline%.yaml}.sha256")")
destination="$ROBUST_OUTPUT/acquisition-$SLURM_JOB_ID"
capture_run_context "$destination"
cp "$profile" "$destination/context"
sha256sum "$profile" "$ROBUST_PROTOCOL" "$ROBUST_STAGE1" > "$destination/context/configs.sha256"
"$PYTHON" -m unittest -q tests.test_simple_landmark_mapping tests.test_acquisition_integrity > "$destination/context/tests.log" 2>&1
label_conversion=source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml
segmentation="$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
for entry in E1:00:2026121700 E2:00:2026121700 E1:02:2026121702 E2:02:2026121702; do
    IFS=: read -r arm index seed <<< "$entry"
    run="$destination/${arm}_$seed"
    profile_path="$ROBUST_REFERENCE/context/session_$index.yaml"
    printf '%s\t%s\tSTARTED\t%s\n' "$arm" "$seed" "$(date --iso-8601=seconds)" >> "$destination/attempts.tsv"
    if "$PYTHON" workflows/two_stage_navigation/capture_simple_landmark_sweep.py \
        --protocol "$ROBUST_PROTOCOL" --revision "$profile_path" --robustness-arm "$arm" --robustness-profile "$profile" \
        --stage1-session cf7_00 --stage1-actor "$ROBUST_STAGE1" --seed "$seed" --output "$run" \
        --headless --device cuda:0 --record-robot --enable_cameras --evaluator-segmentation "$segmentation" \
        > "$destination/actor-${arm}-$seed.log" 2>&1; then
        if (cd "$run" && sha256sum actor_result.json > actor_result.json.sha256) \
            && "$PYTHON" workflows/two_stage_navigation/evaluate_simple_landmark_sweep.py \
                --session "$run" --protocol "$ROBUST_PROTOCOL" --label-conversion "$label_conversion" \
                --oracle-diagnostic "$run/oracle_diagnostic.json" > "$destination/evaluator-${arm}-$seed.log" 2>&1; then
            outcome=EVALUATED
        else
            outcome=EVALUATION_FAILED
        fi
    else
        outcome=CAPTURE_FAILED
    fi
    printf '%s\t%s\t%s\t%s\n' "$arm" "$seed" "$outcome" "$(date --iso-8601=seconds)" >> "$destination/attempts.tsv"
done
capture_run_end "$destination"
! rg -q 'FAILED' "$destination/attempts.tsv"
