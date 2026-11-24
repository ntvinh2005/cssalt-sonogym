#!/usr/bin/env bash
set -euo pipefail

export POC_PROTOCOL_VERSION=v6r4
exec bash "$(dirname "$0")/s0010_mbb_single_anatomy_poc_v6r3.sbatch"
