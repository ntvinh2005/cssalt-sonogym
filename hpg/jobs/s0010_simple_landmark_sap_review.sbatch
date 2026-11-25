#!/usr/bin/env bash
#SBATCH --job-name=sg_sap_review
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept-b
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:10:00

set -euo pipefail
: "${SONOGYM_CONFIG_FILE:?}"
: "${SIMPLE_MAPPING_PROTOCOL:?}"
: "${SIMPLE_MAPPING_ORIGINAL_SWEEP:?}"
: "${SIMPLE_MAPPING_OLD_VIDEO:?}"
: "${SIMPLE_MAPPING_OUTPUT_ROOT:?}"
: "${SIMPLE_MAPPING_FRESH_JOB:?}"
source "$SONOGYM_CONFIG_FILE"
[[ "$SLURM_JOB_ACCOUNT" == anest-dept && "${SLURM_GPUS_ON_NODE:-0}" -eq 0 ]]
running_cpus="$(squeue -A anest-dept --states=RUNNING,COMPLETING,CONFIGURING -h -o '%C' | awk '{sum+=$1} END{print sum+0}')"
[[ "$running_cpus" -le 4 ]]
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$PWD/source/spinal_surgery/spinal_surgery/evaluation:$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg MPLCONFIGDIR="${TMPDIR:-/tmp}/sonogym-simple-sap-review-$SLURM_JOB_ID"

python="$SONOGYM_ENV_PREFIX/bin/python"
job="$SIMPLE_MAPPING_OUTPUT_ROOT/$SIMPLE_MAPPING_FRESH_JOB"
repair_job="$SIMPLE_MAPPING_OUTPUT_ROOT/${SIMPLE_MAPPING_REPAIR_JOB:-none}"
repair="$repair_job/fresh_s0010_2026120500"
review="$SIMPLE_MAPPING_OUTPUT_ROOT/review"
mkdir "$review"
git rev-parse HEAD > "$review/render_commit.txt"
cp "$SIMPLE_MAPPING_OUTPUT_ROOT/sap_evidence.json" "$review/"
cp "$job/attempts.tsv" "$review/"
cp "$job/context"/{s0010_simple_landmark_sap_development_v1.yaml,s0010_simple_landmark_sap_development_v1.sha256,git-commit.txt,tests.log} "$review/"
sessions=()
for seed in 2026120500 2026120501 2026120502; do
    session="$job/fresh_s0010_$seed"
    if [[ -f "$session/actor_result.json" ]]; then
        mkdir "$review/$seed"
        cp "$session/actor_result.json" "$review/$seed/"
    fi
    if [[ -f "$session/evaluation.json" ]]; then
        cp "$session/evaluation.json" "$review/$seed/"
        if [[ -f "$session/oracle_diagnostic.json" ]]; then
            cp "$session/oracle_diagnostic.json" "$review/$seed/"
            sessions+=("$session")
            video_parameter=()
            if [[ "$seed" -ne 2026120500 || -f "$repair/oracle_diagnostic.json" ]]; then
                video_parameter=(--static-only)
            fi
            "$python" workflows/two_stage_navigation/render_simple_sap_demo.py \
              --session "$session" --protocol "$SIMPLE_MAPPING_PROTOCOL" \
              --output "$review/$seed/visualization" "${video_parameter[@]}"
        fi
    fi
done
if [[ -f "$repair_job/attempts.tsv" ]]; then
    mkdir "$review/matched-repair"
    cp "$repair_job/attempts.tsv" "$repair_job/context"/{s0010_simple_landmark_sap_development_v2.yaml,s0010_simple_landmark_sap_development_v2.sha256,git-commit.txt,tests.log} "$review/matched-repair/"
    for file in actor_result.json evaluation.json oracle_diagnostic.json; do
        if [[ -f "$repair/$file" ]]; then
            cp "$repair/$file" "$review/matched-repair/"
        fi
    done
    if [[ -f "$repair/oracle_diagnostic.json" ]]; then
        sessions+=("$repair")
        "$python" workflows/two_stage_navigation/render_simple_sap_demo.py \
          --session "$repair" --protocol "$SIMPLE_MAPPING_PROTOCOL" \
          --output "$review/matched-repair/visualization"
    fi
fi
"$python" workflows/two_stage_navigation/render_simple_sap_evidence.py \
  --original "$SIMPLE_MAPPING_ORIGINAL_SWEEP" --diagnosis "$review/sap_evidence.json" \
  --protocol "$SIMPLE_MAPPING_PROTOCOL" --output "$review/diagnostic_examples" --revised "${sessions[@]}"
if [[ ! -s "$review/matched-repair/visualization/revised_systematic_sweep.mp4" \
      && ! -s "$review/2026120500/visualization/revised_systematic_sweep.mp4" ]]; then
    cp "$SIMPLE_MAPPING_OLD_VIDEO" "$review/offline_recorded_data_fallback.mp4"
fi
(
    cd "$review"
    archive="$SIMPLE_MAPPING_OUTPUT_ROOT/simple-landmark-sap-review.tar.gz"
    tar -czf "$archive" .
    tar -tzf "$archive" > "$SIMPLE_MAPPING_OUTPUT_ROOT/review-files.txt"
)
(
    cd "$SIMPLE_MAPPING_OUTPUT_ROOT"
    sha256sum simple-landmark-sap-review.tar.gz > simple-landmark-sap-review.tar.gz.sha256
)
