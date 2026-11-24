#!/usr/bin/env bash
#SBATCH --job-name=sg_simple_render
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
: "${SIMPLE_MAPPING_OFFLINE_DIR:?}"
source "$SONOGYM_CONFIG_FILE"
[[ "$SLURM_JOB_ACCOUNT" == anest-dept && "${SLURM_GPUS_ON_NODE:-0}" -eq 0 ]]
running_cpus="$(squeue -A anest-dept --states=RUNNING,COMPLETING,CONFIGURING -h -o '%C' | awk '{sum+=$1} END{print sum+0}')"
[[ "$running_cpus" -le 4 ]]

export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$PWD/source/spinal_surgery/spinal_surgery/evaluation:$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg MPLCONFIGDIR="${TMPDIR:-/tmp}/sonogym-simple-render-$SLURM_JOB_ID"
offline="$SIMPLE_MAPPING_OFFLINE_DIR"
python="$SONOGYM_ENV_PREFIX/bin/python"
review="${SIMPLE_MAPPING_REVIEW_ROOT:-$offline}"
mkdir -p "$review"
fresh_args=()
fresh_files=()
review_diagnostics=()
if [[ -n "${SIMPLE_MAPPING_FRESH_SESSION:-}" && -f "$SIMPLE_MAPPING_FRESH_SESSION/actor_result.json" ]]; then
    fresh_args=(--fresh-session "$SIMPLE_MAPPING_FRESH_SESSION")
    fresh_files=(-C "$(dirname "$SIMPLE_MAPPING_FRESH_SESSION")"
      "$(basename "$SIMPLE_MAPPING_FRESH_SESSION")/actor_result.json"
      "$(basename "$SIMPLE_MAPPING_FRESH_SESSION")/waypoints.json")
    if [[ -f "$SIMPLE_MAPPING_FRESH_SESSION/evaluation.json" ]]; then
        fresh_files+=("$(basename "$SIMPLE_MAPPING_FRESH_SESSION")/evaluation.json")
        if [[ -n "${SIMPLE_MAPPING_PROTOCOL:-}" ]]; then
            "$python" workflows/two_stage_navigation/evaluate_simple_landmark_sweep.py \
              --session "$SIMPLE_MAPPING_FRESH_SESSION" --protocol "$SIMPLE_MAPPING_PROTOCOL" \
              --label-conversion source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml \
              --output "$review/fresh_sweep_rescore.json" \
              --oracle-diagnostic "$review/fresh_oracle_diagnostic.json"
            review_diagnostics=(fresh_sweep_rescore.json fresh_oracle_diagnostic.json)
        fi
    fi
fi
"$python" workflows/two_stage_navigation/render_simple_landmark_mapping.py \
  --results "$offline/results" --output "$review/visualizations" "${fresh_args[@]}"
git rev-parse HEAD > "$review/visualizations/git-commit.txt"
(
    cd "$review"
    tar -czf simple-landmark-mapping-review.tar.gz \
      -C "$offline" results slurm-job.txt tests.log git-commit.txt pip-freeze.txt \
      start-time.txt end-time.txt -C "$review" visualizations "${review_diagnostics[@]}" "${fresh_files[@]}"
    sha256sum simple-landmark-mapping-review.tar.gz > simple-landmark-mapping-review.tar.gz.sha256
    tar -tzf simple-landmark-mapping-review.tar.gz > review-files.txt
)
