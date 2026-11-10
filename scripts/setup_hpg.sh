#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 --config PATH [--inspect-only | --install]" >&2
    exit 2
}

CONFIG=""
MODE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --inspect-only) MODE=inspect; shift ;;
        --install) MODE=install; shift ;;
        *) usage ;;
    esac
done
[[ -n "$CONFIG" && -n "$MODE" ]] || usage

set -a
# shellcheck disable=SC1090
source "$CONFIG"
set +a

[[ "$SONOGYM_PROJECT_DIR" == /blue/* ]] || { echo "Project must be on Blue storage" >&2; exit 2; }
[[ "$SONOGYM_RUNTIME_DIR" == /blue/* ]] || { echo "Runtime must be on Blue storage" >&2; exit 2; }
[[ "${SLURM_CPUS:-0}" -le 4 ]] || { echo "CPU budget exceeds four cores" >&2; exit 2; }

LDD_VERSION_OUTPUT="$(ldd --version 2>&1)"
LDD_FIRST_LINE="${LDD_VERSION_OUTPUT%%$'\n'*}"

mkdir -p "$SONOGYM_RUNTIME_DIR/records"
RECORD="$SONOGYM_RUNTIME_DIR/records/host-inspection.txt"
{
    date --iso-8601=seconds
    hostname
    uname -a
    echo "$LDD_FIRST_LINE"
    echo "node=$([[ -n ${SLURM_JOB_ID:-} ]] && echo compute/development || echo login)"
    echo "pwd=$(pwd)"
    df -h "$SONOGYM_PROJECT_DIR" "$SONOGYM_RUNTIME_DIR"
    echo "--- conda/mamba modules ---"
    module -t spider conda 2>&1 || true
    module -t spider mamba 2>&1 || true
    echo "--- apptainer modules ---"
    module -t spider apptainer 2>&1 || true
    echo "--- account associations ---"
    sacctmgr -n -P show assoc user="$USER" format=User,Account,Partition,QOS,DefaultQOS 2>&1 || true
    echo "--- qos ---"
    sacctmgr -n -P show qos format=Name,MaxWall,MaxTRESPerUser,GrpTRES 2>&1 || true
    echo "--- gpu scheduler metadata ---"
    sinfo -N -o '%N|%P|%G|%f|%t' 2>&1 || true
    echo "--- git identity ---"
    git config --global --get user.name || true
    git config --global --get user.email || true
    git -C "$SONOGYM_PROJECT_DIR" remote -v || true
} | tee "$RECORD"

if [[ "$MODE" == inspect ]]; then
    exit 0
fi

[[ -n ${SLURM_JOB_ID:-} ]] || {
    echo "Run installation in a CPU development allocation, not on a login node." >&2
    echo "Example: srun --account=anest-dept --cpus-per-task=4 --mem=32gb --time=04:00:00 --pty bash -i" >&2
    exit 2
}
[[ "${SLURM_JOB_ACCOUNT:-}" == "anest-dept" ]] || { echo "Unexpected Slurm account" >&2; exit 2; }

glibc_version="${LDD_FIRST_LINE##* }"
if [[ "$(printf '%s\n' 2.34 "$glibc_version" | sort -V | head -n 1)" != "2.34" ]]; then
    echo "GLIBC $glibc_version is below the Isaac Sim pip minimum 2.34; stop and select the Apptainer route." >&2
    exit 3
fi

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export MAKEFLAGS=-j4 CMAKE_BUILD_PARALLEL_LEVEL=4
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="$SONOGYM_CACHE_DIR/pip"
mkdir -p "$SONOGYM_CACHE_DIR/pip" "$SONOGYM_RUNTIME_DIR/src" "$(dirname "$SONOGYM_ENV_PREFIX")"

CONDA_EXE="${SONOGYM_CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$CONDA_EXE" ]]; then
    module load conda
    CONDA_EXE="$(command -v conda)"
fi

if [[ ! -x "$SONOGYM_ENV_PREFIX/bin/python" ]]; then
    "$CONDA_EXE" create --yes --prefix "$SONOGYM_ENV_PREFIX" python=3.10 pip
fi
PYTHON="$SONOGYM_ENV_PREFIX/bin/python"
"$PYTHON" -m pip install --upgrade \
    pip==25.0.1 setuptools==75.8.0 wheel==0.45.1
"$PYTHON" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
"$PYTHON" -m pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

ISAACLAB_DIR="$SONOGYM_RUNTIME_DIR/src/IsaacLab"
if [[ ! -d "$ISAACLAB_DIR/.git" ]]; then
    git clone --branch v2.1.0 https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
fi
git -C "$ISAACLAB_DIR" fetch --tags origin
git -C "$ISAACLAB_DIR" checkout --detach 21f7136325136ca3f6ca4e0a8125edffe5c24f7e

export PIP_CONSTRAINT="$SONOGYM_PROJECT_DIR/environment/constraints.txt"
(cd "$ISAACLAB_DIR" && "$PYTHON" -m pip install -e source/isaaclab -e source/isaaclab_assets -e source/isaaclab_tasks -e source/isaaclab_mimic -e 'source/isaaclab_rl[skrl]')
"$PYTHON" -m pip install \
    scipy==1.15.2 psutil==6.1.1 pyvista==0.44.2 ruamel.yaml==0.18.10 \
    pydicom==3.0.1 nibabel==5.3.2 monai==1.4.0 wandb==0.19.8 \
    opencv-python-headless==4.11.0.86
"$PYTHON" -m pip install --editable "$SONOGYM_PROJECT_DIR/source/spinal_surgery" --no-deps
"$PYTHON" "$SONOGYM_PROJECT_DIR/scripts/download_assets.py" --runtime-dir "$SONOGYM_RUNTIME_DIR"
"$PYTHON" -m pip check

"$CONDA_EXE" list --prefix "$SONOGYM_ENV_PREFIX" --explicit > "$SONOGYM_RUNTIME_DIR/records/conda-explicit.txt"
"$PYTHON" -m pip freeze --all > "$SONOGYM_RUNTIME_DIR/records/pip-freeze.txt"
git -C "$ISAACLAB_DIR" rev-parse HEAD > "$SONOGYM_RUNTIME_DIR/records/isaaclab-commit.txt"
git -C "$SONOGYM_PROJECT_DIR" rev-parse HEAD > "$SONOGYM_RUNTIME_DIR/records/sonogym-commit.txt"
"$PYTHON" "$SONOGYM_PROJECT_DIR/scripts/preflight.py" \
    --config "$CONFIG" --json-output "$SONOGYM_RUNTIME_DIR/records/preflight.json"

echo "CPU-safe installation and preflight completed. Simulator/GPU checks remain NOT_RUN_NO_GPU."
