# s0010 TP surface-core simulator proxy exploratory pilot

This report records the isolated `SIMULATOR_GEOMETRY_PROXY` experiment on fixed anatomy `s0010`.
It does not modify the verified-label protocol and does not measure anatomical recognition, an MBB
target, patient generalization, or Stage 3 control.

## Result

**The bounded CPU pilot completed reproducibly, but whole-image proposal precision—especially L4 at 12/26 (46.15%)—does not support a reliable anatomical detector claim.**

| step | what was measured | criterion | measured | verdict |
|---|---|---|---|---|
| 1 | input and split integrity | all proxy contract gates pass before fit | 5,469/5,469 rows audited; no blockers; no cross-split exact duplicate | pass |
| 2 | separated fit/calibration/development execution | train-only fit; calibration-only threshold selection; checkpoint reload exact | 463/151/139 samples; checkpoint reload verified | pass |
| 3 | proxy prediction | exploratory; no absolute performance threshold | patch 125/139 (89.93%); full-image L4 precision 12/26 (46.15%), L5 16/20 (80.00%) | exploratory mixed result |

The engineering path is complete for this experiment.  The result only establishes performance
against simulator geometry proxy labels; the existing verified-label status remains
`ENGINEERING_READY_LABEL_REVIEW_PENDING`.

## 1. No-fit integrity — whether proxy fitting may start

`workflows/two_stage_navigation/train_tp_surface_core_proxy.py`, run by Slurm job `51836756`.
It validates the sealed package, every source proposal, proxy conditions, source images, splits,
and production features before fitting is permitted.

**Criterion.** The source ZIP, manifest, geometry and label-contract hashes must match; all source
frame hashes and 774-value features must resolve; smoke must be excluded; competing cores and all
uncertainty overlap must be zero; the positive projected-support rule must pass; every class must
have at least 40/10/10 unique train/calibration/development source images; no exact image hash may
cross splits; and no-fit must record `fit_called=false`.

### 1.0 Quantities

No symbols are introduced.  `SIMULATOR_GEOMETRY_PROXY`, `TP_SURFACE_CORE`, and the experiment are
defined in [`docs/notation.md`](../notation.md).

### 1.1 Method

- The original v3 manifest is immutable; the launcher writes a separate proxy audit and manifest.
- Every one of 5,469 source rows is checked against its source array and recorded frame hash.
- Privileged geometry selects offline proxy labels only; the 774 model inputs contain clean image features only.
- Exactly one deterministic proposal per class and source image is admitted to prevent correlated patches from inflating coverage.
- `opened_development_smoke` is excluded before sample selection.

### 1.2 Setup

| what | value | spec |
|---|---|---|
| run | `51836756`; commit `4d0500ef80f35966b9ea5c6965e5aae3a39550b7` | §1 |
| source ZIP | `s0010_tp_surface_core_review_v3_51831979.zip`; SHA-256 `dd24c07541bf8e5d6ffe07c7904ec104996a3bb11d39f421f90e4c75fbcc423e` | §1 |
| source manifest | SHA-256 `5a60e34e5a2abd0032a2f40f689844aa570194fa2a83bd94353b5a1c6d17a426` | §1 |
| label contract | SHA-256 `e33a17cf0e9618a1a6b56d3dafb2e747c0653be86f92059f566fb15250ce9adc` | §1 |
| resources | `anest-dept`; 4 CPU; 16 GB; 0 GPU | §1 |

```bash
python workflows/two_stage_navigation/train_tp_surface_core_proxy.py \
  --dataset-dir "$SOURCE_RUN/dataset" \
  --source-zip "$SOURCE_RUN/s0010_tp_surface_core_review_v3_51831979.zip" \
  --label-contract experiments/unity_two_stage_transfer/s0010_tp_surface_core_simulator_proxy_exploratory_v1_labels.csv \
  --output-dir "$RUN_DIR/preflight" --no-fit
```

| | value |
|---|---:|
| attempted source proposals | 5,469 |
| proxy-condition candidate proposals | 3,388 |
| dropped, additional proposals from the same class/source image or proxy conditions not met | 2,635 |
| **final unique class/source-image samples** | **753** |

### 1.3 Results

| class | train | calibration | development evaluation |
|---|---:|---:|---:|
| background / other structure | 347 | 110 | 103 |
| L4 RIGHT TP surface core | 55 | 12 | 17 |
| L5 RIGHT TP surface core | 61 | 29 | 19 |

No exact image hash crossed split boundaries, `blocking_conditions=[]`, and no-fit recorded
`fit_called=false`.

**Verdict: pass, because all nine unique-image counts exceed 40/10/10 and every integrity gate passed.**

### 1.4 Reading it

- Passing no-fit authorizes this proxy experiment only.
- Geometric intersection is not acoustic visibility or anatomical approval.
- Randomized starts on one patient do not provide patient-level generalization.

### 1.5 Files

| file | contents |
|---|---|
| `preflight/result.json` | fail-closed result and split IDs |
| `preflight/proxy_label_audit.csv` | all 5,469 source rows and proxy decisions |
| `preflight/proxy_sample_manifest.csv` | 753 selected samples |

## 2. CPU fitting — whether splits and checkpoint persistence remain valid

`workflows/two_stage_navigation/train_tp_surface_core_proxy.py`, run without `--no-fit` only after
§1 returned `PROXY_TRAINING_READY`.

**Criterion.** Only `train` enters the 240-tree ExtraTrees `fit()` call; only calibration chooses
the L4/L5 operating thresholds; development remains read-only; the saved checkpoint must reload
without changing development probabilities.

### 2.1 Method

- The fit partition has 463 samples: background 347, L4 55, L5 61.
- The calibration partition has 151 samples and selects a separate L4/L5 threshold from 0.50--0.95 at 0.05 increments, requiring precision at least 0.90 when achievable and maximizing recall.
- The development partition has 139 samples and is opened only after threshold selection.

### 2.2 Setup

| what | value | spec |
|---|---|---|
| estimator | 240-tree `ExtraTreesClassifier`; balanced class weights; seed `20261117`; 4 workers | §2 |
| fit/calibration/development | 463/151/139 samples | §2 |
| environment | Python 3.10.22; NumPy 1.26.4; scikit-learn 1.5.2; joblib 1.4.2 | §2 |

```bash
sbatch --account=anest-dept --qos=anest-dept-b --partition=hpg-default \
  --cpus-per-task=4 --mem=16G --time=00:30:00 \
  --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env \
  hpg/jobs/s0010_tp_surface_core_simulator_proxy_exploratory_v1.sbatch
```

Nothing was dropped after §1.

### 2.3 Results

| check | result |
|---|---|
| L4 calibration threshold | 0.50; precision 1.0; recall 7/12 (58.33%) |
| L5 calibration threshold | 0.50; precision 1.0; recall 23/29 (79.31%) |
| checkpoint | saved; SHA-256 `338351fdee5a8444dee5a6ec250d828b8f1eedbc3f76e79495aa1aafbdeb2f8b` |
| reload | development probabilities exactly unchanged |
| tests | proxy 4/4; existing TP training 11/11 |
| runtime | job 27 s; fit command 7.02 s; peak RSS 267,976 kB |

**Verdict: pass, because fit/calibration/development IDs are disjoint by split and checkpoint reload is exact.**

### 2.4 Reading it

- Threshold 0.50 is calibration-selected under the declared grid, not post-hoc development tuning.
- The checkpoint must not replace the verified-label model path.
- CPU completion is an engineering result, not evidence of MBB readiness.

### 2.5 Files

| file | contents |
|---|---|
| `training/tp_surface_core_simulator_proxy_extratrees.joblib` | fitted checkpoint and frozen thresholds |
| `training/result.json` | split IDs, calibration grid and metrics |
| `metrics/time-training.txt` | elapsed time and peak memory |

## 3. Development evaluation — what the proxy model predicts

`workflows/two_stage_navigation/train_tp_surface_core_proxy.py` evaluates both the privileged-
selected patch unit and every actor-visible proposal in each development image.

**Criterion.** Exploratory—no absolute performance threshold.  Patch results are compared with
full-image proposal results to expose the effect of removing privileged proposal selection.

### 3.1 Method

- Patch evaluation uses one geometry-selected proxy patch per class/source image.
- Whole-image evaluation scores every actor-visible proposal with clean-image features, applies the frozen calibration thresholds, and aggregates accepted proposal classes per source image.
- Proxy truth is formed only from the sealed geometry conditions after actor predictions are fixed.

### 3.2 Setup

| what | value | spec |
|---|---|---|
| development patches | 139 selected patches | §3 |
| development images | 103 unique source images | §3 |
| anatomy | fixed patient `s0010` | §3 |
| inspected status | opened development; not an untouched confirmation set | §3 |

The invocation is the §2 command.  Nothing was dropped after §1.

### 3.3 Results

| unit / class | TP | FP | FN | precision | recall |
|---|---:|---:|---:|---:|---:|
| patch / background | 103 | 14 | 0 | 88.03% | 100.00% |
| patch / L4 | 9 | 0 | 8 | 100.00% | 52.94% |
| patch / L5 | 13 | 0 | 6 | 100.00% | 68.42% |
| whole image / background | 52 | 8 | 15 | 86.67% | 77.61% |
| whole image / L4 | 12 | 14 | 5 | 46.15% | 70.59% |
| whole image / L5 | 16 | 4 | 3 | 80.00% | 84.21% |

Patch accuracy is 125/139 (89.93%); all 14 patch errors are positive proxy patches abstained to
background (8 L4, 6 L5).  Whole-image L4 produces 14 false positive images.

**Verdict: exploratory mixed result; proxy morphology signal is measurable, but whole-image L4 precision is insufficient for a reliable detector claim.**

### 3.4 Reading it

- Patch precision benefits from privileged offline selection of the proxy patch and is not full-image localization performance.
- Whole-image results are the relevant current diagnostic for actor-visible proposal use.
- The model has not learned clinician-approved anatomy, whole TP, SAP, medial branches, an MBB target, or needle control.

### 3.5 Files

| file | contents |
|---|---|
| `training/development_patch_failures.csv` | 14 patch-level errors with probabilities |
| `training/development_full_image_proposals.csv` | per-image proxy truth and accepted proposal classes |
| `experiments/unity_two_stage_transfer/s0010_tp_surface_core_simulator_proxy_exploratory_v1_result.json` | concise immutable result |
