#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit command}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
PYTHON="$SONOGYM_ENV_PREFIX/bin/python"
[[ -x "$PYTHON" ]] || { echo "Python environment is missing: $PYTHON" >&2; exit 2; }
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg

cd "$SONOGYM_PROJECT_DIR"
"$PYTHON" tests/test_acquisition_integrity.py
"$PYTHON" tests/test_ultrasound_plane_geometry.py
"$PYTHON" tests/test_two_stage_contracts.py
"$PYTHON" tests/test_tp_surface_core_training.py
"$PYTHON" tests/test_tp_surface_core_capture_contract.py
"$PYTHON" -m py_compile \
    workflows/two_stage_navigation/collect_stage2_dataset.py \
    workflows/two_stage_navigation/verify_exact_pose_capture.py \
    workflows/two_stage_navigation/prepare_tp_surface_core_dataset.py \
    workflows/two_stage_navigation/package_tp_surface_core_review.py
bash -n hpg/jobs/s0010_tp_surface_core_dataset_v7.sbatch
bash -n hpg/jobs/s0010_tp_surface_core_dataset_v8.sbatch
