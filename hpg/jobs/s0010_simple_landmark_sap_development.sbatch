#!/usr/bin/env bash
#SBATCH --job-name=sg_sap_sweep
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=30G
#SBATCH --time=00:13:00

set -euo pipefail
: "${SONOGYM_CONFIG_FILE:?}"
: "${SIMPLE_MAPPING_RUN_ROOT:?}"
: "${SIMPLE_MAPPING_PROTOCOL:?}"
: "${SIMPLE_MAPPING_OUTPUT_ROOT:?}"
experiment_checkout="$PWD"
running_cpus="$(squeue -A anest-dept --states=RUNNING,COMPLETING,CONFIGURING -h -o '%C' | awk '{sum+=$1} END{print sum+0}')"
[[ "$running_cpus" -le 4 ]]
source "$experiment_checkout/hpg/jobs/common.sh"
SONOGYM_PROJECT_DIR="$experiment_checkout"
SONOGYM_GUIDANCE_CONFIG="$experiment_checkout/config/guidance_baseline.yaml"
export SONOGYM_GUIDANCE_CONFIG
export PYTHONPATH="$experiment_checkout/source/spinal_surgery:$SONOGYM_RUNTIME_DIR/envs/stage1-python:$experiment_checkout/source/spinal_surgery/spinal_surgery/evaluation:$experiment_checkout"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/sonogym-simple-sap-$SLURM_JOB_ID"

destination="$SIMPLE_MAPPING_OUTPUT_ROOT/$SLURM_JOB_ID"
revision="${SIMPLE_MAPPING_SAP_REVISION:-experiments/unity_two_stage_transfer/s0010_simple_landmark_sap_development_v1.yaml}"
(cd "$(dirname "$revision")" && sha256sum -c "$(basename "${revision%.yaml}.sha256")")
capture_run_context "$destination"
cp "$revision" "${revision%.yaml}.sha256" "$destination/context/"
"$PYTHON" -m unittest -q tests.test_simple_landmark_mapping tests.test_acquisition_integrity \
  > "$destination/context/tests.log" 2>&1
source_commands="$destination/context/seeds.tsv"
"$PYTHON" -c 'import sys, yaml; from pathlib import Path; p=yaml.safe_load(Path(sys.argv[1]).read_text()); [print(seed, p["stage1_session"], int(seed == p["demo_seed"]), sep="\t") for seed in p["seeds"]]' "$revision" > "$source_commands"
while IFS=$'\t' read -r seed reference camera; do
    run="$destination/fresh_s0010_$seed"
    camera_args=()
    if [[ "$camera" -eq 1 ]]; then
        camera_args=(--record-robot --enable_cameras)
    fi
    printf '%s\t%s\n' "$seed" "STARTED" >> "$destination/attempts.tsv"
    if "$PYTHON" workflows/two_stage_navigation/capture_simple_landmark_sweep.py \
        --protocol "$SIMPLE_MAPPING_PROTOCOL" --seed "$seed" --stage1-session "$reference" \
        --stage1-actor "$SIMPLE_MAPPING_RUN_ROOT/stage1/results/actor_results.json" \
        --output "$run" --revision "$revision" --headless --device cuda:0 \
        --evaluator-segmentation "$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz" \
        "${camera_args[@]}" > "$destination/actor-$seed.log" 2>&1; then
        printf '%s\t%s\n' "$seed" "CAPTURE_COMPLETED" >> "$destination/attempts.tsv"
        if "$PYTHON" workflows/two_stage_navigation/evaluate_simple_landmark_sweep.py \
            --session "$run" --protocol "$SIMPLE_MAPPING_PROTOCOL" \
            --label-conversion source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml \
            --oracle-diagnostic "$run/oracle_diagnostic.json" > "$destination/evaluator-$seed.log" 2>&1; then
            printf '%s\t%s\n' "$seed" "EVALUATED" >> "$destination/attempts.tsv"
        else
            printf '%s\t%s\n' "$seed" "EVALUATION_FAILED" >> "$destination/attempts.tsv"
        fi
    else
        printf '%s\t%s\n' "$seed" "CAPTURE_FAILED" >> "$destination/attempts.tsv"
    fi
done < "$source_commands"
capture_run_end "$destination"
! rg -q 'FAILED' "$destination/attempts.tsv"
