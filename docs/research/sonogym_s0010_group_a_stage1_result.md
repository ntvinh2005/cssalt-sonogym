# SonoGym s0010 Group A Stage 1 result

This report tests whether the frozen s0010 ultrasound captures support reconstruction of the ordered sacrum, S1, and L1--L5 map using [Unity Stage 1](../notation.md#unity-stage-1). It covers fixed-anatomy development only; Stage 2 and patient-held-out claims are out of scope.

## Result

**Group A Stage 1 failed its frozen development gate: patch morphology carried anatomical signal, but no threshold reconstructed a complete seven-structure map.**

| step | what was measured | criterion | measured | verdict |
|---|---|---|---|---|
| 1 | session-disjoint patch recognition and spatial reconstruction | precision and recall at least 0.90; complete-map rate at least 0.80; median error at most 10 mm; every structure at least 0.80; zero wrong stops | selected threshold: precision 1.00, recall 0.179, complete maps 0 / 4; highest-recall threshold: precision 0.722, recall 0.464, complete maps 0 / 4 | FAIL |

The failure occurred before the reused evaluation split was opened. No actor bundle was released and Stage 2 was not started.

## 1. Group A Stage 1 — can seven ordered structures be reconstructed from ultrasound and relative motion?

workflows/two_stage_navigation/train_stage1_group_a.py, run by a bounded CPU Slurm job, fits the existing primary interface model plus a seven-class ExtraTrees model and aggregates class probabilities in the Home-relative spatial map.

**Criterion.** The frozen protocol requires event precision and recall at least 0.90, complete-map session rate at least 0.80, median localization error at most 10 mm, every structure successful in at least 0.80 of sessions, and zero wrong actor stops. These thresholds reuse the approved Stage 1 event and localization requirements.

### 1.1 Method

- Input consists of clean ultrasound frames and executed Home-relative motion from capture job 50908807.
- Privileged raw labels are joined by session, frame sequence, and frame hash for offline supervision only.
- The primary ExtraTrees model predicts whether a tracked patch contains Group A anatomy.
- A separate ExtraTrees model predicts one of seven anatomical classes for supervised observability diagnosis.
- Per-class image scores are the minimum of primary and class probability.
- Spatial evidence requires forward and reverse support, 10 mm event separation, and a 0.10 top-event score margin.
- One global coordinate offset is calibrated across all structures; per-structure offsets are forbidden.
- The reused evaluation split remains closed unless development passes.

### 1.2 Setup

| what | value | spec |
|---|---|---|
| code | dec3596214374ec950056227bc9dcfa60737724f | clean worktree recorded by job |
| protocol | sonogym-unity-two-stage-stage1-group-a-v2 | SHA-256 b5ab90b09ae49d4237bc58680680b542b485d7f5e2aa7ea008fc2ae75102238b |
| capture | job 50908807 | 3,376 frames; 16 sessions; patient s0010 |
| training split | td_00--td_07 | 8 complete sessions |
| calibration split | cd_00--cd_03 | 4 complete sessions |
| evaluation split | ua_00--ua_03 | not opened |
| model | ExtraTreesClassifier | 240 trees; depth 18; minimum leaf 3; balanced classes; seed 20261112 |
| resources | job 51116088 | anest-dept; 4 CPU; 16 GB; 0 GPU; 30-minute bound |
| software | Python 3.10.22; NumPy 1.26.4; SciPy 1.15.2; scikit-learn 1.5.2; joblib 1.4.2 | pinned Stage 1 environment |

~~~bash
sbatch \
  --account=anest-dept \
  --qos=anest-dept \
  --partition=hpg-default \
  --cpus-per-task=4 \
  --mem=16gb \
  --time=00:30:00 \
  --job-name=sg-stage1-group-a \
  --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env \
  hpg/jobs/two_stage_stage1_group_a.sbatch
# Resolved job: 51116088; start 2026-11-12T23:28:05-05:00; end 2026-11-12T23:32:38-05:00.
~~~

| | value |
|---|---|
| available sessions | 16 |
| training | 8 |
| calibration | 4 |
| not opened, evaluation gate | 4 |
| **development final** | **12** |

Job 51115154 was a pre-fit validation failure caused by rounded coordinate transcription; protocol v2 records the correction, and no model fitting occurred in that job.

### 1.3 Results

| setting | anatomy | precision | recall | successful events | wrong stops | abstentions |
|---|---|---:|---:|---:|---:|---:|
| calibration patch classifier | macro across seven classes | 0.857 | 0.844 | not an event metric | not comparable | not comparable |
| selected spatial threshold 0.85 | all structures | 1.000 | 0.179 | 5 / 28 (17.9%) | 0 | 23 |
| highest-recall frozen threshold 0.60 | all structures | 0.722 | 0.464 | 13 / 28 (46.4%) | 5 | 10 |

| structure | patch precision | patch recall | selected-threshold success | highest-recall-threshold success |
|---|---:|---:|---:|---:|
| sacrum | 0.971 | 0.936 | 1 / 4 (25%) | 0 / 4 (0%) |
| S1 | 0.861 | 0.961 | 4 / 4 (100%) | 2 / 4 (50%) |
| L5 | 0.833 | 0.945 | 0 / 4 (0%) | 1 / 4 (25%) |
| L4 | 0.868 | 0.847 | 0 / 4 (0%) | 3 / 4 (75%) |
| L3 | 0.897 | 0.728 | 0 / 4 (0%) | 4 / 4 (100%) |
| L2 | 0.771 | 0.862 | 0 / 4 (0%) | 3 / 4 (75%) |
| L1 | 0.799 | 0.629 | 0 / 4 (0%) | 0 / 4 (0%) |

| resource | measured |
|---|---:|
| elapsed | 4 min 37 s |
| maximum resident memory | 870,956 KiB |
| CPU time reported by Slurm batch step | 4 min 39 s |
| output growth | 1.6 MiB |
| GPU | 0 |

**Verdict: FAIL**, because no frozen threshold produced any complete map, recall remained below 0.90, and the highest-recall threshold produced five wrong stops.

### 1.4 Reading it

- The primary detector was strong: precision 0.975 and recall 0.977.
- Seven-class patch accuracy was 0.876, so clean ultrasound contains some level-related development signal.
- Patch scores did not yield a reliable spatial map.
- Sacrum, S1, and L5 often produced multiple accepted events and triggered score-margin abstention.
- L1 coordinates were consistently displaced by approximately 15.5--20.5 mm under the single global offset.
- L3 was the only structure successful in 4 / 4 sessions at the highest-recall threshold, but that setting failed the overall precision and wrong-stop requirements.
- The failure does not invalidate the earlier single-L4 pilot; it shows that the same construction does not extend directly to all seven Group A structures.
- The next design decision is whether to revise spatial evidence and geometry using only training/calibration sessions or collect new session-held-out data. Reopening the current evaluation split is not justified.

### 1.5 Files

| file | contents |
|---|---|
| [result manifest](../../experiments/unity_two_stage_transfer/stage1_group_a_result.json) | concise metrics, resources, hashes, and failure analysis |
| [result digest](../../experiments/unity_two_stage_transfer/stage1_group_a_result.sha256) | manifest SHA-256 |
| [protocol](../../experiments/unity_two_stage_transfer/stage1_group_a_protocol.yaml) | frozen splits, model, gates, and information boundary |
| [trainer](../../workflows/two_stage_navigation/train_stage1_group_a.py) | supervised fit, spatial aggregation, and fail-closed evaluation |
| external development report | /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage1_group_a/51116088/results/development_report.json |
| external threshold sweep | /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage1_group_a/51116088/results/calibration_thresholds.json |
