#!/usr/bin/env bash
#SBATCH --job-name=sg_mbb_oblique
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=30G
#SBATCH --time=00:15:00

set -euo pipefail
checkout="$PWD"
[[ "$(squeue -A anest-dept --states=RUNNING,COMPLETING,CONFIGURING -h -o '%C' | awk '{s+=$1} END {print s+0}')" -le 4 ]]
source "$checkout/hpg/jobs/common.sh"
SONOGYM_PROJECT_DIR="$checkout"
export SONOGYM_GUIDANCE_CONFIG="$checkout/config/guidance_baseline.yaml"
export PYTHONPATH="$checkout/source/spinal_surgery:$SONOGYM_RUNTIME_DIR/envs/stage1-python:$checkout/source/spinal_surgery/spinal_surgery/evaluation:$checkout"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/sonogym-mbb-oblique-$SLURM_JOB_ID"

study="${1:?Output study directory}"
manifest="${2:?Sealed reconstruction manifest}"
stage1="${3:?Sealed Stage 1 actor output}"
[[ "$(git rev-parse HEAD)" == "${MBB_CAPTURE_COMMIT:?Expected execution revision}" ]]
survey=experiments/unity_two_stage_transfer/s0010_mbb_oblique_development_v1.json
profile=experiments/unity_two_stage_transfer/s0010_simple_landmark_robustness_v1.yaml
destination="$study/capture-$SLURM_JOB_ID"
capture_run_context "$destination"
context="$destination/context"
"$PYTHON" - "$manifest" "$survey" "$profile" "$context" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys
import yaml

manifest, survey, profile, context = map(Path, sys.argv[1:])
spec = json.loads(manifest.read_text())
study = json.loads(survey.read_text())
assert study["seed"] == 2026121700 and study["seed"] not in spec["reserved_seeds"]
source_protocol = Path(spec["protocol"]["path"])
assert sha256(source_protocol.read_bytes()).hexdigest() == spec["protocol"]["sha256"]
base = yaml.safe_load(source_protocol.read_text())
base["models"]["sonoanatomy"].update(spec["models"])
density = yaml.safe_load(profile.read_text())
density["capture_seeds"] = [study["seed"]]
(context/"profile.yaml").write_text(yaml.safe_dump(density))
protocol = context/"protocol.yaml"
protocol.write_text(yaml.safe_dump(base, sort_keys=False))
digest = sha256(protocol.read_bytes()).hexdigest()
protocol.with_suffix(".sha256").write_text(f"{digest}  {protocol.name}\n")
revision = context/"revision.yaml"
revision.write_text(yaml.safe_dump({
    "scope": study["anatomy"], "source_protocol_sha256":digest,
    "stage1_session":study["stage1_session"], "seeds":[study["seed"]],
    "demo_seed":study["seed"], "acquisition":"outer_transverse_midpoints_near_l4",
    "sap_level_assignment":"projected_position_with_stage1_reference_bands",
    "tp_evidence_on_added_sap_waypoints":True,
}))
revision.with_suffix(".sha256").write_text(f"{sha256(revision.read_bytes()).hexdigest()}  {revision.name}\n")
PY
cp "$survey" "${survey%.json}.sha256" "$context/"
sha256sum "$stage1" "$manifest" "$context/protocol.yaml" "$context/profile.yaml" "$context/revision.yaml" > "$context/input.sha256"
segmentation="$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
"$PYTHON" workflows/two_stage_navigation/capture_simple_landmark_sweep.py \
  --protocol "$context/protocol.yaml" --revision "$context/revision.yaml" \
  --stage1-actor "$stage1" --stage1-session cf7_00 --seed 2026121700 \
  --robustness-arm E1-M --robustness-profile "$context/profile.yaml" --additional-sweep "$survey" \
  --headless --device cuda:0 --enable_cameras --record-robot --evaluator-segmentation "$segmentation" \
  --output "$destination/session"
(cd "$destination/session" && sha256sum actor_result.json > actor_result.json.sha256)
capture_run_end "$destination"
