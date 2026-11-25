#!/usr/bin/env bash
#SBATCH --job-name=sg_ind_maps
#SBATCH --account=anest-dept
#SBATCH --qos=anest-dept
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=30G
#SBATCH --time=00:18:00

set -euo pipefail
checkout="$PWD"
[[ "$(squeue -A anest-dept --states=RUNNING,COMPLETING,CONFIGURING -h -o '%C' | awk '{s+=$1} END{print s+0}')" -le 4 ]]
source "$checkout/hpg/jobs/common.sh"
SONOGYM_PROJECT_DIR="$checkout"
SONOGYM_GUIDANCE_CONFIG="$checkout/config/guidance_baseline.yaml"
export SONOGYM_GUIDANCE_CONFIG
export PYTHONPATH="$checkout/source/spinal_surgery:$SONOGYM_RUNTIME_DIR/envs/stage1-python:$checkout/source/spinal_surgery/spinal_surgery/evaluation:$checkout"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/sonogym-independent-$SLURM_JOB_ID"

[[ -f "$SIMPLE_MAPPING_OUTPUT_ROOT/replay-$SIMPLE_MAPPING_REPLAY_JOB/revised_sap.json" ]]
[[ -z "$(git status --short)" ]]
[[ "$(git rev-parse HEAD)" == "$SIMPLE_MAPPING_EXECUTION_COMMIT" ]]
protocol=experiments/unity_two_stage_transfer/s0010_simple_landmark_independent_v1.yaml
development=experiments/unity_two_stage_transfer/s0010_simple_landmark_sap_development_v2.yaml
(cd "$(dirname "$protocol")" && sha256sum -c "$(basename "${protocol%.yaml}.sha256")")
destination="$SIMPLE_MAPPING_OUTPUT_ROOT/$SLURM_JOB_ID"
capture_run_context "$destination"
cp "$protocol" "${protocol%.yaml}.sha256" "$destination/context"
"$PYTHON" -m unittest -q tests.test_simple_landmark_mapping tests.test_acquisition_integrity \
    > "$destination/context/tests.log" 2>&1
stage1="$SIMPLE_MAPPING_RUN_ROOT/stage1/results/actor_results.json"
label_conversion=source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml
segmentation="$SONOGYM_ASSETS_DATA_DIR/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
"$PYTHON" - "$protocol" "$development" "$stage1" "$SIMPLE_MAPPING_PROTOCOL" "$label_conversion" "$segmentation" "$destination/context" <<'PY'
import hashlib
from pathlib import Path
import subprocess
import sys
import yaml

plan, dev, stage1, source, conversion, segmentation, output = map(Path, sys.argv[1:])
confirmation = yaml.safe_load(plan.read_text())
definition = yaml.safe_load(source.read_text())
matches = (
    (dev, confirmation["development_profile_sha256"]),
    (stage1, confirmation["stage1"]["sha256"]),
    (source, confirmation["source_protocol_sha256"]),
    (conversion, confirmation["evaluation"]["label_conversion_sha256"]),
    (segmentation, confirmation["evaluation"]["source_segmentation_sha256"]),
    (Path(definition["evaluator"]["subregion_prediction_path"]),
     confirmation["evaluation"]["subregion_prediction_sha256"]),
    *((Path(definition["models"]["sonoanatomy"][level]["path"]),
       confirmation["checkpoints"][level]["sha256"]) for level in ("L4", "L5")),
)
for path, seal in matches:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != seal:
        raise RuntimeError(f"input hash mismatch: {path}")
(output / "inputs.sha256").write_text("".join(f"{seal}  {file}\n" for file, seal in matches))
subprocess.run(["git", "diff", "--quiet", confirmation["implementation_revision"], "HEAD", "--",
    "config/", "source/spinal_surgery/", "workflows/two_stage_navigation/capture_simple_landmark_sweep.py",
    "workflows/two_stage_navigation/evaluate_simple_landmark_sweep.py",
    "workflows/two_stage_navigation/run_simple_landmark_mapping.py",
    "workflows/two_stage_navigation/stage2_sim.py"], check=True)
profile = yaml.safe_load(dev.read_text())
if profile["stage1_session"] != confirmation["stage1"]["session"] or profile["tp_evidence_on_added_sap_waypoints"]:
    raise RuntimeError("frozen acquisition settings differ")
for index, seed in enumerate(confirmation["seeds"]):
    execution = dict(profile, seeds=[seed], demo_seed=seed)
    recipe = output / f"session_{index:02d}.yaml"
    recipe.write_text(yaml.safe_dump(execution, sort_keys=False))
    recipe.with_suffix(".sha256").write_text(f"{hashlib.sha256(recipe.read_bytes()).hexdigest()}  {recipe.name}\n")
(output / "seeds.tsv").write_text("".join(f"{i:02d}\t{seed}\n" for i, seed in enumerate(confirmation["seeds"])))
(output / "waypoints_expected.sha256").write_text(confirmation["acquisition"]["ordered_waypoint_sha256"]+"  waypoints.json\n")
PY

while IFS=$'\t' read -r index seed; do
    run="$destination/session_$index"
    printf '%s\t%s\tSTARTED\t%s\n' "$index" "$seed" "$(date --iso-8601=seconds)" >>"$destination/attempts.tsv"
    if "$PYTHON" workflows/two_stage_navigation/capture_simple_landmark_sweep.py \
        --protocol "$SIMPLE_MAPPING_PROTOCOL" --revision "$destination/context/session_$index.yaml" \
        --stage1-session cf7_00 --stage1-actor "$stage1" --seed "$seed" --output "$run" \
        --headless --device cuda:0 --record-robot --enable_cameras --evaluator-segmentation "$segmentation" \
        >"$destination/actor-$index.log" 2>&1; then
        if (cd "$run" && sha256sum actor_result.json > actor_result.json.sha256 \
                && sha256sum -c "$destination/context/waypoints_expected.sha256") \
            && "$PYTHON" workflows/two_stage_navigation/evaluate_simple_landmark_sweep.py \
                --session "$run" --protocol "$SIMPLE_MAPPING_PROTOCOL" --label-conversion "$label_conversion" \
                --oracle-diagnostic "$run/oracle_diagnostic.json" >"$destination/evaluator-$index.log" 2>&1 \
            && (cd "$run" && sha256sum -c actor_result.json.sha256); then
            outcome=EVALUATED
        else
            outcome=EVALUATION_OR_INTEGRITY_FAILED
        fi
    else
        outcome=CAPTURE_FAILED
    fi
    printf '%s\t%s\t%s\t%s\n' "$index" "$seed" "$outcome" "$(date --iso-8601=seconds)" >>"$destination/attempts.tsv"
done <"$destination/context/seeds.tsv"
capture_run_end "$destination"
! rg -q 'FAILED' "$destination/attempts.tsv"
