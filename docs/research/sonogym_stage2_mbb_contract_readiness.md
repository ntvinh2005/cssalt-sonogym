# SonoGym MBB-oriented Stage 2 contract readiness

This record audits `s0010` landmark ground truth and verifies the isolated fail-closed contracts
for unilateral L4-L5 facet-related localization. It does not train a detector, run Isaac Sim,
localize a nerve, construct an injection target, or change historical Stage 2 results.

## Result

**The actor/evaluator contracts are implemented and regression-safe, but semantic localization is `MBB_STAGE2_BLOCKED_GROUND_TRUTH` because L4/L5 SAP is unresolved, TP is only provisional, and ultrasound SAP/TP observability is unverified.**

| step | what was measured | criterion | measured | verdict |
|---|---|---|---|---|
| annotation readiness | L4/L5 SAP and TP, both sides | every requested semantic landmark is `READY` before detector fitting | SAP 0 / 4 ready; TP 0 / 4 ready | FAIL |
| actor contract | taxonomy, causal evidence, uncertainty, target-region fail closure | all focused tests and legacy regressions pass; manifest reproduces byte-for-byte | 19 / 19 focused; 61 / 61 regressions; exact manifest match | PASS |

The code is ready to accept independently reviewed annotations; current assets are not ready to
supervise a reliable SAP/TP detector or an MBB surrogate target-region model.

## 1. Annotation audit — can current s0010 data supervise the requested landmarks?

`workflows/two_stage_navigation/audit_stage2_mbb_readiness.py`, run by Slurm job `51697175`.
It converts the accepted annotation v2 record into the task-specific L4-L5 readiness matrix.

**Criterion.** User-approved ground-truth gate: each required L4/L5 SAP and TP annotation must be
independently supportable (`READY`), and actor-visible ultrasound observability must be validated,
before semantic model training or target-region localization.

### 1.0 Quantities

| symbol | definition | source |
|---|---|---|
| MBB-oriented Stage 2 | unilateral, multi-view L4-L5 landmark-localization method | [`docs/notation.md`](../notation.md#mbb-oriented-stage-2) |
| MBB landmark evidence | actor-side landmark hypothesis with causal frame, pose, and uncertainty provenance | [`docs/notation.md`](../notation.md#mbb-landmark-evidence) |
| MBB target-region interface | branch-specific fail-closed surrogate-region boundary | [`docs/notation.md`](../notation.md#mbb-target-region-interface) |

### 1.1 Method

- Configure laterality explicitly; there is no default side.
- Map L4-L5 to two distinct bony sites: L3 medial branch with L4 SAP/TP, and L4 medial branch with
  L5 SAP/TP.
- Keep the L5 dorsal ramus in a separate L5-S1 taxonomy because its sacral-ala/S1-SAP relationship
  differs from L1-L4 medial branches.
- Read annotation status, provenance, uncertainty, and evidence without promoting mesh quantiles.
- Treat nerve location and target-region construction as missing when no reviewed mapping exists.

The branch mapping follows primary lumbar dorsal-ramus anatomy and the established dual-level facet
innervation relationship: [Bogduk and Long (1979)](https://pubmed.ncbi.nlm.nih.gov/156249/),
[Bogduk et al. (1982)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1167925/), and the explicit L4-L5
example in [Gofeld et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC2684949/). These sources define
the taxonomy; they do not supply patient-specific s0010 coordinates.

### 1.2 Setup

| what | value | spec |
|---|---|---|
| run | Slurm `51697175`; code `e01979de3481299043da34dda718600ed6e22d27` | `anest-dept`, 4 CPU, 16 GB, 0 GPU |
| input | `experiments/unity_two_stage_transfer/s0010_anatomical_annotations_v2.json` | SHA-256 `25df75aadb83b6d1faf39cfd8b376fb8976a08dbd834e302bad8f38e9de023c3` |
| patient | `s0010` | one fixed simulated anatomy |
| software | Python 3.10.22; NumPy 1.26.4; SciPy 1.15.2; scikit-learn 1.5.2 | pinned SonoGym runtime |
| execution | 2026-11-16 23:00:07--23:00:15 EST; scheduler elapsed 13 s | node `c0706a-s12` |

```bash
sbatch --parsable --account=anest-dept --qos=anest-dept --partition=hpg-default \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:15:00 \
  --job-name=sg-s2-mbb-contracts \
  --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/sg-s2-mbb-contracts-%j.out \
  --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env \
  hpg/jobs/two_stage_stage2_mbb_contracts.sbatch

# Resolved audit command inside the job
$SONOGYM_ENV_PREFIX/bin/python \
  workflows/two_stage_navigation/audit_stage2_mbb_readiness.py \
  --repository-root "$SONOGYM_PROJECT_DIR" \
  --source experiments/unity_two_stage_transfer/s0010_anatomical_annotations_v2.json \
  --output "$RUN_DIR/results/stage2_mbb_landmark_readiness_v1.json"
```

| | value |
|---|---:|
| required semantic landmark/side records | 8 |
| dropped | 0 |
| **final audited records** | **8** |

### 1.3 Results

| site | landmark | left | right | evidence and blocker |
|---|---|---|---|---|
| L3 medial branch-related L4 site | L4 SAP | `UNRESOLVED` | `UNRESOLVED` | mesh quantile is not semantic SAP truth; no dedicated annotation |
| L3 medial branch-related L4 site | L4 TP | `PROVISIONAL` | `PROVISIONAL` | low-confidence quantile region; 6.0 mm stated uncertainty; needs review |
| L4 medial branch-related L5 site | L5 SAP | `UNRESOLVED` | `UNRESOLVED` | mesh quantile is not semantic SAP truth; no dedicated annotation |
| L4 medial branch-related L5 site | L5 TP | `PROVISIONAL` | `PROVISIONAL` | low-confidence quantile region; 6.0 mm stated uncertainty; needs review |
| both sites | medial-branch nerve reference | `MISSING` | `MISSING` | no nerve segmentation or reviewed patient-specific nerve course |
| both sites | ultrasound observability | `UNRESOLVED` | `UNRESOLVED` | raw-label occupancy does not prove SAP/TP discriminability |
| both sites | surrogate target region | `UNRESOLVED` | `UNRESOLVED` | SAP and branch-specific mapping are not validated |

The 1.5 mm isotropic CT/segmentation grid and <=1.5 mm major mesh/segmentation p95 agreement support
major vertebral geometry, not semantic fine-process boundaries. Raw labels 25--31 all become acoustic
bone class 13, so the renderer does not carry vertebral or process identity as an actor-visible class.

**Verdict: FAIL**, because 0 / 8 required semantic landmark/side records are `READY` and semantic
ultrasound observability has not been established.

### 1.4 Reading it

- Current data cannot responsibly supervise a SAP/TP localization model.
- TP geometry is useful as a review candidate, not as training truth.
- No exact MBB target region can be estimated from current ground truth.
- A future output may be a bony-landmark-derived surrogate region, never a confirmed nerve point.
- Fixed `s0010` evidence cannot establish patient generalization or clinical accuracy.

### 1.5 Files

| file | contents |
|---|---|
| `experiments/unity_two_stage_transfer/stage2_mbb_landmark_readiness_v1.json` | machine-readable bilateral readiness matrix |
| `experiments/unity_two_stage_transfer/stage2_mbb_landmark_readiness_v1.json.sha256` | manifest integrity seal |
| `experiments/unity_two_stage_transfer/s0010_anatomical_annotations_v2.json` | accepted development-only source annotations |

## 2. Contract verification — does the new path fail closed without changing old Stage 2?

`stage2_mbb_landmarks.py`, run by `tests/test_stage2_mbb_landmarks.py` and legacy regression tests.
It defines taxonomy, causal multi-view evidence, uncertainty requirements, decision output, and an
unimplemented target-region boundary.

**Criterion.** All 19 focused tests, 61 legacy annotation/Stage 1/Stage 2 tests, the audit, and the
byte-for-byte manifest comparison must exit zero. Historical Stage 2 files and verdicts must remain
unchanged.

### 2.1 Method

- Reject CT, segmentation, target/evaluator coordinates, future frames, and cross-reset evidence.
- Count neither duplicate hashes nor distinct hashes at one pose as independent physical viewpoints.
- Reject wrong-level/wrong-side landmarks, absent SAP/TP, unvalidated transforms, uncalibrated
  uncertainty, and contradictory hypotheses.
- Preserve multiple landmark-specific views instead of one maximum-quality image.
- Permit `LANDMARKS_LOCALIZED` only after required landmarks, distinct views, calibration, and
  validated anatomical consistency; keep `target_region_localized=false` without a validated map.
- Reject vertebral centroids as MBB target substitutes.

### 2.2 Setup

| what | value | spec |
|---|---|---|
| focused suite | `tests/test_stage2_mbb_landmarks.py` | 19 tests |
| regressions | annotation 7; Stage 1 20; frozen Stage 2 34 | 61 tests |
| CPU peak RSS | 95,232 KiB | maximum `/usr/bin/time -v` value |
| output footprint | 96 KiB | external job directory |

```bash
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$SONOGYM_PROJECT_DIR/source/spinal_surgery/spinal_surgery/evaluation"
$SONOGYM_ENV_PREFIX/bin/python tests/test_stage2_mbb_landmarks.py
$SONOGYM_ENV_PREFIX/bin/python tests/test_anatomical_ground_truth.py
$SONOGYM_ENV_PREFIX/bin/python tests/test_two_stage_stage1.py
$SONOGYM_ENV_PREFIX/bin/python tests/test_two_stage_stage2.py
```

Nothing was dropped.

### 2.3 Results

| check | result |
|---|---:|
| MBB contract tests | 19 / 19 pass |
| anatomical ground-truth regressions | 7 / 7 pass |
| Stage 1 regressions | 20 / 20 pass |
| frozen Stage 2 regressions | 34 / 34 pass |
| readiness audit exit | 0 |
| manifest byte comparison | exact, exit 0 |
| Slurm job | `COMPLETED`, exit `0:0` |

**Verdict: PASS**, because all 80 / 80 tests and both reproducibility checks passed without GPU,
simulator execution, model fitting, or a change to `CONFIRMATION_NO_GO`.

### 2.4 Reading it

- The implementation is an interface and safety boundary, not a landmark detector.
- Existing Stage 1 level estimation, physical coarse/fine/micro motion, relative odometry, candidate
  return, freshness checks, and historical baselines remain reusable.
- The old maximum-quality Stage 2 path remains separate and reproducible.
- Stage 3 remains out of scope.

### 2.5 Files

| file | contents |
|---|---|
| `source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/stage2_mbb_landmarks.py` | isolated taxonomy, evidence, memory, sufficiency, and fail-closed result contracts |
| `tests/test_stage2_mbb_landmarks.py` | 19 focused CPU tests |
| `workflows/two_stage_navigation/audit_stage2_mbb_readiness.py` | deterministic readiness audit |
| `hpg/jobs/two_stage_stage2_mbb_contracts.sbatch` | bounded CPU verification launcher |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_mbb_contracts/51697175/` | logs, timings, context, and reproduced manifest |
