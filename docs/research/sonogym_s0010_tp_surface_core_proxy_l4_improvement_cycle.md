# s0010 L4 TP surface-core proxy improvement cycle

This record compares the frozen simulator-proxy v1 checkpoint with bounded L4
one-vs-rest proxy classifier experiments on fixed patient `s0010`. It covers
image-only TP_SURFACE_CORE proxy recognition; whole-TP, SAP, MBB-target,
patient-generalization, active-acquisition, and Stage 3 claims are out of scope.

## Result

**IMPROVEMENT_DEMONSTRATED: protocol v16 reduced sealed-development L4 false
positives from 5 / 15 predicted-positive images to 1 / 12 while increasing true
positives from 10 / 14 to 11 / 14 at the unchanged 0.50 threshold.**

| step | what was measured | criterion | measured | verdict |
|---|---|---|---|---|
| 1 | existing-data candidate | retain at least 11 TP and reduce FP | 10 TP, 0 FP | FAIL |
| 2 | first targeted supplement | calibration pass and sufficient sealed evidence | calibration passed; sealed evidence insufficient | FAIL |
| 3 | protocol v16 capture | split coverage and all replay/acquisition gates | 192 / 192 frames passed; every split passed coverage | PASS |
| 4 | protocol v16 candidate | retain at least 90% of baseline TP, strictly reduce FP, do not increase ambiguous activations | TP 10 to 11; FP 5 to 1; ambiguous activations 3 to 2 | PASS |

The first two negative results remain immutable. The protocol v16 result supports
researcher review of active-acquisition integration but does not itself remove
`ACTIVE_ACQUISITION_INTEGRATION_HOLD`.

## 1. Existing-data ablation — can existing images alone repair L4 errors?

`workflows/two_stage_navigation/run_tp_surface_core_l4_existing_data_ablation.py`,
run by the CPU Slurm wrapper, fitted four bounded ExtraTrees alternatives and
selected one using calibration only.

**Criterion.** The selected candidate had to retain at least 11 of the frozen
checkpoint's 12 opened-development true-positive images while reducing false
positives; the criterion was frozen in
`s0010_tp_surface_core_proxy_l4_existing_data_ablation_v1.yaml`.

### 1.1 Method

- Training images supplied fitting and hard-negative mining.
- Calibration selected `l4_ovr_hn10` without development access.
- Previously opened development images measured the selected candidate once.
- L5 remained the frozen control.

### 1.2 Setup

| what | value | spec |
|---|---|---|
| run | job `51961297` | 4 CPU, 16 GB, 44 s |
| frozen checkpoint | SHA-256 `338351fdee5a8444dee5a6ec250d828b8f1eedbc3f76e79495aa1aafbdeb2f8b` | frozen v1 |
| protocol | `experiments/unity_two_stage_transfer/s0010_tp_surface_core_proxy_l4_existing_data_ablation_v1.yaml` | §1 |
| threshold | 0.50 | unchanged |

```bash
sbatch --account=anest-dept --cpus-per-task=4 --mem=16G \
  hpg/jobs/s0010_tp_surface_core_proxy_l4_existing_data_ablation_v1.sbatch
```

| | value |
|---|---:|
| frozen opened-development true positives | 12 |
| frozen opened-development false positives | 7 |
| selected candidate true positives | 10 |
| selected candidate false positives | 0 |

### 1.3 Results

| opened development | TP | FP | FN | precision | recall |
|---|---:|---:|---:|---:|---:|
| frozen v1 | 12 | 7 | 5 | 0.6316 | 0.7059 |
| `l4_ovr_hn10` | 10 | 0 | 7 | 1.0000 | 0.5882 |

**Verdict: FAIL**, because 10 retained true positives were below the required 11.

### 1.4 Reading it

- Existing hard negatives removed the observed false positives.
- The same candidate lost two true-positive images.
- This opened-development result was not independent confirmation.
- Frozen L5 remained TP=16, FP=4, FN=3.

### 1.5 Files

| file | contents |
|---|---|
| `experiments/unity_two_stage_transfer/s0010_tp_surface_core_proxy_l4_existing_data_ablation_v1.yaml` | frozen design |
| runtime job `51961297` | fit, calibration, and opened-development evidence |

## 2. First targeted supplement — does a small new capture restore recall?

`workflows/two_stage_navigation/train_tp_surface_core_l4_targeted_supplement.py`,
run by the GPU Slurm wrapper, fitted one L4 one-vs-rest proxy classifier using
new synchronized train observations.

**Criterion.** Calibration had to reduce false positives without losing true
positives; sealed development also required at least three frozen-baseline true
positives and two frozen-baseline false positives before an improvement claim.

### 2.1 Method

- Forty-eight train, 24 calibration, and 24 development observations were captured.
- Every observation passed native-map replay, full-pose replay, freshness,
  viewpoint-diversity, and serialized-array checks.
- The 0.50 threshold and 774-value image-only feature contract were unchanged.
- Development features were inaccessible until calibration completed.

### 2.2 Setup

| what | value | spec |
|---|---|---|
| run | job `51992814` | 1 NVIDIA L4, 4 CPU, 226 s |
| execution commit | `dc4d26d13b194ad84182ae3ccbf5a9e42bd9d250` | clean capture revision |
| protocol | `s0010_tp_surface_core_dataset_protocol_v12.yaml` | SHA-256 `a699216f29cec1b4ef543b1dbc18c5bc056d320440add93a8fae9ab63859668e` |
| candidate checkpoint | SHA-256 `ab0ca634796c1af9e4b2824999e54f2096b68142854a51ff29516f6b79b1afa9` | reload verified |

```bash
sbatch --account=anest-dept --partition=hpg-turin --cpus-per-task=4 \
  --mem=30G --time=00:30:00 --gres=gpu:l4:1 \
  hpg/jobs/s0010_tp_surface_core_proxy_l4_targeted_supplement_v2.sbatch
```

| | value |
|---|---:|
| attempted observations | 96 |
| dropped | 0 |
| **final** | **96** |

### 2.3 Results

| split and model | TP | FP | FN | precision | recall | ambiguous activations |
|---|---:|---:|---:|---:|---:|---:|
| calibration, frozen v1 | 2 | 2 | 0 | 0.5000 | 1.0000 | 1 |
| calibration, candidate | 2 | 1 | 0 | 0.6667 | 1.0000 | 0 |
| sealed development, frozen v1 | 1 | 1 | 2 | 0.5000 | 0.3333 | 2 |
| sealed development, candidate | 1 | 1 | 2 | 0.5000 | 0.3333 | 0 |

**Verdict: FAIL**, because the sealed split contained only one baseline TP and one
baseline FP and the candidate did not improve primary TP/FP/FN counts.

### 2.4 Reading it

- Calibration passed its decision gate.
- The candidate removed ambiguous activations but not the sealed image-level FP.
- Only five new train images satisfied the complete L4 proxy-positive contract.
- The result motivated increased positive and confuser coverage, not another margin search.

### 2.5 Files

| file | contents |
|---|---|
| runtime job `51992814` | 96 synchronized observations and fitted candidate |
| `experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_protocol_v12.yaml` | frozen protocol |

## 3. Protocol v16 acquisition — is the new evidence sufficient and valid?

`workflows/two_stage_navigation/plan_tp_surface_core_l4_waypoints.py` and
`collect_stage2_dataset.py`, run by one CPU planner job and one GPU capture job,
constructed four split-isolated waypoint banks and acquired their observations.

**Criterion.** Each 48-frame session had to contain 48 distinct frame hashes and
pass all five acquisition/replay gates. Train/calibration/development required at
least 16/8/8 PROXY_POSITIVE images and 24/12/12 PROXY_NEGATIVE images.

### 3.1 Method

- Eight prior L4 proxy-positive acquisitions supplied the positive anchors.
- Five frozen L4 false triggers and seven competing L5 acquisitions supplied confusers.
- Each split received 24 positive-targeted, 10 false-trigger-targeted, and 14
  competing-L5-targeted waypoints.
- Privileged geometry selected collection poses but never entered model features.

### 3.2 Setup

| what | value | spec |
|---|---|---|
| planner run | job `52006787` | 4 CPU; `CPU_PREFLIGHT_PASS` |
| planner output | `waypoint_plan.json` | SHA-256 `496781bd01761f59dce837716de3c3b2c000d20f302cf9ab6edb155ff1abc78e` |
| CPU gate | job `52009562` | 43 / 43 tests passed |
| capture run | job `52009585` | 1 NVIDIA L4, 4 CPU, 30 GB, 6 min 55 s |
| execution commit | `3be07395f320e282eda35200a5a85f95e5fe2766` | clean worktree |
| protocol | `s0010_tp_surface_core_dataset_protocol_v16.yaml` | SHA-256 `b8b899d787bfb7f23cf5c5099d71e7d61c44edb2c25c5b1457fd6cddca28b287` |

```bash
python workflows/two_stage_navigation/plan_tp_surface_core_l4_waypoints.py \
  --reference-run /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_tp_surface_core_proxy_l4_targeted_supplement_v1/51992814 \
  --output /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_tp_surface_core_proxy_l4_waypoint_plan_v1/52006787/waypoint_plan.json

sbatch --parsable --job-name=sg_l4_covrep_v4 --partition=hpg-turin \
  --account=anest-dept --qos=anest-dept --cpus-per-task=4 --mem=30G \
  --time=00:30:00 --gres=gpu:l4:1 \
  --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env \
  hpg/jobs/s0010_tp_surface_core_proxy_l4_coverage_repair_v4.sbatch
```

| | value |
|---|---:|
| attempted observations | 192 |
| dropped by replay/acquisition gates | 0 |
| **final** | **192** |

### 3.3 Results

| split | images | PROXY_POSITIVE | PROXY_NEGATIVE | PROXY_AMBIGUOUS | coverage verdict |
|---|---:|---:|---:|---:|---|
| train | 96 | 31 | 60 | 5 | PASS |
| calibration | 48 | 16 | 25 | 7 | PASS |
| sealed development | 48 | 14 | 28 | 6 | PASS |

**Verdict: PASS**, because 192 / 192 observations passed all acquisition/replay
gates and every split exceeded both frozen coverage requirements.

### 3.4 Reading it

- Every session contained 48 distinct ultrasound hashes.
- Serialized coordinate-hash mismatch count was zero in every session.
- Full-pose raw-label agreement and bone-mask IoU passed per frame.
- The run consumed 179,857,146 bytes, below the 3 GB limit.

### 3.5 Files

| file | contents |
|---|---|
| `experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_protocol_v16.yaml` | sealed acquisition, coverage, and evaluation rules |
| runtime job `52006787/waypoint_plan.json` | waypoint banks and anchor provenance |
| runtime job `52009585/sessions/*/verification_report.json` | per-frame replay and acquisition gates |
| runtime job `52009585/dataset/` | split manifests and extracted patches |

## 4. Protocol v16 training and evaluation — does L4 improve without changing L5?

`workflows/two_stage_navigation/train_tp_surface_core_l4_coverage_repair.py`,
run inside job `52009585`, fitted two 240-tree L4 one-vs-rest proxy classifier
variants, selected by calibration, and evaluated once on sealed development.

**Criterion.** Calibration required TP not below baseline, FP not above baseline,
and ambiguous activations not above baseline. Sealed development required at least
eight baseline TP and three baseline FP, candidate TP at least 90% of baseline,
strict FP reduction, and no increase in ambiguous activations.

### 4.1 Method

- `coverage_balanced` used positive weight 1.0.
- `coverage_positive_1p5` used positive weight 1.5.
- Both used 240 trees, balanced class weights, seed `20261123`, four CPU workers,
  and fixed threshold 0.50.
- Calibration selected `coverage_balanced` by deterministic tie-breaking.
- Development feature rows loaded before calibration: 0.
- Frozen L5 inference was evaluated without fitting or checkpoint modification.

### 4.2 Setup

| what | value | spec |
|---|---|---|
| run | job `52009585` | §3.2 |
| fit rows | 534 from 408 unique source images | train only |
| new train evidence | 31 positive; 12 frozen false-trigger; 18 competing L5 | PROXY_AMBIGUOUS excluded |
| selected checkpoint | SHA-256 `6188c7713cac2f21485d030ac79f4e26ac9c4f0f179008aee57a6d891fe9e8c0` | reload verified |
| threshold | 0.50 | unchanged |

```bash
python workflows/two_stage_navigation/train_tp_surface_core_l4_coverage_repair.py \
  --old-training-dir /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_tp_surface_core_simulator_proxy_exploratory_v1/51836756/training \
  --old-hard-negatives /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_tp_surface_core_proxy_tristate_hard_negative_v1/51956725/preparation/l4_hard_negative_selected.csv \
  --new-dataset-manifest /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_tp_surface_core_proxy_l4_coverage_repair_v4/52009585/dataset/candidate_sample_manifest.csv \
  --protocol experiments/unity_two_stage_transfer/s0010_tp_surface_core_dataset_protocol_v16.yaml \
  --output-dir /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_tp_surface_core_proxy_l4_coverage_repair_v4/52009585/training
```

| | value |
|---|---:|
| attempted fit rows | 534 |
| dropped from fit because split was not train | enforced before construction |
| **final fit rows** | **534** |

### 4.3 Results

| split and model | TP | FP | FN | precision | recall | ambiguous activations | clean-negative detections |
|---|---:|---:|---:|---:|---:|---:|---:|
| calibration, frozen v1 | 12 | 5 | 4 | 0.7059 | 0.7500 | 4 | 8 |
| calibration, candidate | 13 | 2 | 3 | 0.8667 | 0.8125 | 3 | 2 |
| sealed development, frozen v1 | 10 | 5 | 4 | 0.6667 | 0.7143 | 3 | 7 |
| sealed development, candidate | 11 | 1 | 3 | 0.9167 | 0.7857 | 2 | 1 |

| frozen L5 control | TP | FP | FN | precision | recall |
|---|---:|---:|---:|---:|---:|
| calibration | 7 | 0 | 2 | 1.0000 | 0.7778 |
| sealed development | 6 | 1 | 1 | 0.8571 | 0.8571 |

**Verdict: PASS — IMPROVEMENT_DEMONSTRATED**, because the sealed evidence gate
was sufficient, candidate TP increased from 10 to 11, FP decreased strictly from
5 to 1, and ambiguous activations decreased from 3 to 2.

### 4.4 Reading it

- Sealed L4 precision increased from 10 / 15 (66.67%) to 11 / 12 (91.67%).
- Sealed L4 recall increased from 10 / 14 (71.43%) to 11 / 14 (78.57%).
- Clean-negative detections decreased from seven to one.
- The L5 checkpoint and decision rule were unchanged.
- This is fixed-anatomy SIMULATOR_GEOMETRY_PROXY evidence, not anatomical or clinical validation.
- Active-acquisition integration still requires a separate researcher review.

### 4.5 Files

| file | contents |
|---|---|
| runtime job `52009585/training/result.json` | frozen calibration and sealed-development metrics |
| runtime job `52009585/training/l4_coverage_repair_candidate.joblib` | reload-verified selected checkpoint |
| runtime job `52009585/training/sealed_evaluation_image_decisions.csv` | per-image decisions |
| `experiments/unity_two_stage_transfer/s0010_tp_surface_core_proxy_l4_improvement_cycle_result.json` | complete machine-readable cycle result |
