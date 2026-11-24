#!/usr/bin/env bash
set -euo pipefail

: "${SONOGYM_CONFIG_FILE:?SONOGYM_CONFIG_FILE was not exported by the submit command}"
# shellcheck disable=SC1090
source "$SONOGYM_CONFIG_FILE"
export POC_PROTOCOL_VERSION=v6r9
exec bash "$SONOGYM_PROJECT_DIR/hpg/jobs/s0010_mbb_single_anatomy_poc_v6r3.sbatch"
