#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit helper}"
# shellcheck disable=SC1090
set -a
source "$SONOGYM_CONFIG_FILE"
set +a

[[ "$SLURM_JOB_ACCOUNT" == "anest-dept" ]] || { echo "Refusing unexpected account: $SLURM_JOB_ACCOUNT" >&2; exit 2; }
[[ "${SLURM_CPUS_PER_TASK:-0}" -eq 4 ]] || { echo "Initial jobs require exactly four CPUs" >&2; exit 2; }
[[ "${SLURM_GPUS_ON_NODE:-0}" -eq 1 ]] || { echo "Initial jobs require exactly one GPU" >&2; exit 2; }
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "Slurm did not assign a visible GPU" >&2; exit 2; }
[[ "$PWD" == /blue/* ]] || cd "$SONOGYM_PROJECT_DIR"
[[ "$PWD" == /blue/* ]] || { echo "Workloads must execute from Blue storage; got $PWD" >&2; exit 2; }

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1
export WANDB_MODE="${SONOGYM_WANDB_MODE:-disabled}"
export WANDB_SILENT=true
export MPLBACKEND=Agg
export PYVISTA_OFF_SCREEN=true
export QT_QPA_PLATFORM=offscreen
export XDG_CACHE_HOME="$SONOGYM_CACHE_DIR/xdg"
export PIP_CACHE_DIR="$SONOGYM_CACHE_DIR/pip"
export OMNI_USER_DIR="$SONOGYM_CACHE_DIR/omniverse"

mkdir -p "$SONOGYM_OUTPUT_DIR/slurm" "$SONOGYM_OUTPUT_DIR/smoke" \
    "$SONOGYM_OUTPUT_DIR/ppo" "$SONOGYM_CACHE_DIR/xdg" \
    "$SONOGYM_CACHE_DIR/pip" "$SONOGYM_CACHE_DIR/omniverse"

PYTHON="$SONOGYM_ENV_PREFIX/bin/python"
[[ -x "$PYTHON" ]] || { echo "Python environment is missing: $PYTHON" >&2; exit 2; }
export PATH="$SONOGYM_ENV_PREFIX/bin:$PATH"

echo "host=$(hostname) job_id=$SLURM_JOB_ID account=$SLURM_JOB_ACCOUNT pwd=$PWD"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"$PYTHON" --version
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#GPU_NAMES[@]}" -eq 1 && "${GPU_NAMES[0]}" == *L4* ]] || {
    echo "The initial pinned workflow requires exactly one L4; saw: ${GPU_NAMES[*]}" >&2
    exit 2
}

capture_run_context() {
    local run_dir="$1"
    mkdir -p "$run_dir/context"
    date --iso-8601=seconds > "$run_dir/context/start-time.txt"
    git -C "$SONOGYM_PROJECT_DIR" rev-parse HEAD > "$run_dir/context/git-commit.txt"
    git -C "$SONOGYM_PROJECT_DIR" status --short > "$run_dir/context/git-status.txt"
    git -C "$SONOGYM_PROJECT_DIR" diff --binary > "$run_dir/context/uncommitted.patch"
    git -C "$SONOGYM_RUNTIME_DIR/src/IsaacLab" rev-parse HEAD \
        > "$run_dir/context/isaaclab-commit.txt"
    cp "$SONOGYM_CONFIG_FILE" "$run_dir/context/machine-config.env"
    cp "$SONOGYM_GUIDANCE_CONFIG" "$run_dir/context/guidance-config.yaml"
    cp "$SONOGYM_RUNTIME_DIR/records/preflight.json" "$run_dir/context/preflight.json"
    cp "$SONOGYM_RUNTIME_DIR/records/pip-freeze.txt" "$run_dir/context/pip-freeze.txt"
    scontrol show job "$SLURM_JOB_ID" > "$run_dir/context/slurm-job.txt"
    {
        echo "job_id=$SLURM_JOB_ID"
        echo "account=$SLURM_JOB_ACCOUNT"
        echo "cpus=$SLURM_CPUS_PER_TASK"
        echo "memory=$SLURM_MEM_PER_NODE"
        echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
        echo "python=$("$PYTHON" --version 2>&1)"
    } > "$run_dir/context/run-context.env"
}

capture_run_end() {
    local run_dir="$1"
    date --iso-8601=seconds > "$run_dir/context/end-time.txt"
}
