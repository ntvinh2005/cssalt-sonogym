#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit helper}"
# shellcheck disable=SC1090
set -a
source "$SONOGYM_CONFIG_FILE"
set +a

[[ "$SLURM_JOB_ACCOUNT" == "anest-dept" ]] || {
    echo "Refusing unexpected account: $SLURM_JOB_ACCOUNT" >&2
    exit 2
}
[[ "${SLURM_CPUS_PER_TASK:-0}" -eq 4 ]] || {
    echo "Stage 1 requires exactly four CPUs" >&2
    exit 2
}
[[ "${SLURM_GPUS_ON_NODE:-0}" -eq 0 ]] || {
    echo "Stage 1 training is CPU-only" >&2
    exit 2
}
[[ "$PWD" == /blue/* ]] || cd "$SONOGYM_PROJECT_DIR"
[[ "$PWD" == /blue/* ]] || {
    echo "Workloads must execute from Blue storage; got $PWD" >&2
    exit 2
}

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export PATH="$SONOGYM_ENV_PREFIX/bin:$PATH"
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/evaluation"

PYTHON="$SONOGYM_ENV_PREFIX/bin/python"
[[ -x "$PYTHON" ]] || { echo "Python environment is missing: $PYTHON" >&2; exit 2; }
[[ -d "$SONOGYM_RUNTIME_DIR/envs/stage1-python/sklearn" ]] || {
    echo "Pinned Stage 1 Python packages are missing" >&2
    exit 2
}

mkdir -p "$SONOGYM_OUTPUT_DIR/slurm" "$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer"
echo "host=$(hostname) job_id=$SLURM_JOB_ID account=$SLURM_JOB_ACCOUNT pwd=$PWD"
echo "cpu_only=true cpus=$SLURM_CPUS_PER_TASK"
"$PYTHON" --version

capture_cpu_run_context() {
    local run_dir="$1"
    mkdir -p "$run_dir/context"
    date --iso-8601=seconds > "$run_dir/context/start-time.txt"
    git -C "$SONOGYM_PROJECT_DIR" rev-parse HEAD > "$run_dir/context/git-commit.txt"
    git -C "$SONOGYM_PROJECT_DIR" status --short > "$run_dir/context/git-status.txt"
    git -C "$SONOGYM_PROJECT_DIR" diff --binary > "$run_dir/context/uncommitted.patch"
    cp "$SONOGYM_CONFIG_FILE" "$run_dir/context/machine-config.env"
    scontrol show job "$SLURM_JOB_ID" > "$run_dir/context/slurm-job.txt"
    "$PYTHON" -c "import joblib,numpy,scipy,sklearn; print('numpy='+numpy.__version__); print('scipy='+scipy.__version__); print('scikit-learn='+sklearn.__version__); print('joblib='+joblib.__version__)" \
        > "$run_dir/context/stage1-versions.txt"
    {
        echo "job_id=$SLURM_JOB_ID"
        echo "account=$SLURM_JOB_ACCOUNT"
        echo "cpus=$SLURM_CPUS_PER_TASK"
        echo "memory=$SLURM_MEM_PER_NODE"
        echo "gpu_count=0"
        echo "python=$("$PYTHON" --version 2>&1)"
    } > "$run_dir/context/run-context.env"
}

capture_cpu_run_end() {
    local run_dir="$1"
    date --iso-8601=seconds > "$run_dir/context/end-time.txt"
}
