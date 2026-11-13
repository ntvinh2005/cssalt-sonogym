# SonoGym s0010 Group A Stage 1 result

This report tests whether [Unity Stage 1](../notation.md#unity-stage-1) can reconstruct the ordered sacrum, S1, and L1--L5 map from SonoGym ultrasound and executed relative motion. It reports fixed-anatomy development evidence from existing patient `s0010` captures; locked confirmation, patient generalization, and Stage 2 remain out of scope.

## Result

**DEVELOPMENT PASS: protocol v7 reconstructed all 28 / 28 (100%) expected landmarks in 4 / 4 (100%) reused development-evaluation sessions, with no abstention or wrong stop.**

| step | what was measured | criterion | measured | verdict |
|---|---|---|---|---|
| 1 | session-separated seven-landmark spatial reconstruction | 28 / 28 correct; 4 / 4 complete maps; median error at most 10 mm; zero wrong stops | 28 / 28 correct; 4 / 4 complete; median 3.125 mm; p90 5.825 mm; zero wrong stops | DEVELOPMENT PASS |

The result establishes a working fixed-`s0010` development reconstruction and demo. It is not locked confirmation because `ua_00`--`ua_03` had been inspected during earlier failed revisions.

## 1. Seven-landmark reconstruction — can image evidence and relative motion recover the complete ordered map?

`workflows/two_stage_navigation/train_stage1_group_a.py`, run by `hpg/jobs/two_stage_stage1_group_a_v7.sbatch`, fits the supervised patch models, accumulates Home-relative evidence, and selects one complete anatomically ordered path.

**Criterion.** Protocol v7 requires event precision 1.0, event recall 1.0, 4 / 4 complete maps, every class successful in 4 / 4 sessions, median absolute localization error at most 10 mm, and zero wrong actor stops.

### 1.0 Quantities

| quantity | definition | source |
|---|---|---|
| absolute localization error | absolute difference in millimetres between an actor-emitted Home-relative landmark coordinate and its evaluator-only reference | protocol v7 evaluation section |
| complete map | one accepted event for each of `sacrum, S1, L5, L4, L3, L2, L1`, in that order and at least 10 mm apart | protocol v7 ordered-map section |

### 1.1 Method

- Clean ultrasound patches and executed Home-relative motion are actor-visible inputs.
- Raw anatomical identities supervise fitting and score final results offline; they are forbidden at inference.
- A 240-tree ExtraTrees primary model detects Group A anatomy and a 240-tree ExtraTrees model scores the seven anatomical classes.
- Per-class evidence is the minimum of the primary probability and the corresponding class probability.
- A landmark event requires forward and reverse support within 5 mm and an image score of at least 0.30.
- The complete path maximizes summed log image score minus `0.05` per millimetre of deviation from calibration-session adjacent separations.
- Per-class coordinate offsets and adjacent separations are fitted only on `cd_00`--`cd_03`.
- Ground-truth coordinates are opened only for offline calibration or evaluation and are absent from the actor interface.

### 1.2 Setup

| what | value | spec |
|---|---|---|
| code | `4db234fc4a7f20c5216f792b235c3b5c097981ca` | clean worktree recorded by job 51209243 |
| protocol | `sonogym-unity-two-stage-stage1-group-a-v7` | SHA-256 `f2c5498f8909e914ffe08565ad3693760d9c5b3bb9fd0b1eafbe56cd4c473fcb` |
| capture | job 50908807 | 3,376 frames; 16 sessions; patient `s0010` |
| model | ExtraTreesClassifier | 240 trees; depth 18; minimum leaf 3; balanced; seed 20261112 |
| selected map settings | score 0.30; separation 10 mm; support radius 5 mm; spacing penalty 0.05/mm | protocol v7 |
| resources | job 51209243 | `anest-dept`; 4 CPU; 16 GB RAM; 0 GPU; 30-minute bound |
| software | Python 3.10.22; NumPy 1.26.4; SciPy 1.15.2; scikit-learn 1.5.2; joblib 1.4.2 | recorded job context |

~~~bash
sbatch --parsable \
  --account=anest-dept --qos=anest-dept --partition=hpg-default \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:30:00 \
  --job-name=sg-stage1-map-v7 \
  --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/sg-stage1-map-v7-%j.out \
  --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env \
  /blue/lampotan/share/vinhnguyen1/cssalt-sonogym/hpg/jobs/two_stage_stage1_group_a_v7.sbatch
# Resolved job 51209243; 2026-11-13T14:58:28-05:00 to 2026-11-13T15:03:14-05:00.
~~~

| | value |
|---|---:|
| training sessions | 8 |
| calibration sessions | 4 |
| reused development-evaluation sessions | 4 |
| total frames | 3,376 |
| dropped sessions | 0 |
| **final evaluated landmarks** | **28** |

### 1.3 Results

| setting | complete maps | correct landmarks | wrong stops | abstentions | median error | p90 error |
|---|---:|---:|---:|---:|---:|---:|
| v7 calibration, `cd_00`--`cd_03` | 4 / 4 (100%) | 28 / 28 (100%) | 0 | 0 | 2.875 mm | 6.500 mm |
| v7 reused development evaluation, `ua_00`--`ua_03` | 4 / 4 (100%) | 28 / 28 (100%) | 0 | 0 | 3.125 mm | 5.825 mm |

| landmark | successful sessions | median error | maximum error |
|---|---:|---:|---:|
| sacrum | 4 / 4 (100%) | 3.50 mm | 4.50 mm |
| S1 | 4 / 4 (100%) | 3.25 mm | 4.75 mm |
| L5 | 4 / 4 (100%) | 6.00 mm | 9.00 mm |
| L4 | 4 / 4 (100%) | 3.75 mm | 4.50 mm |
| L3 | 4 / 4 (100%) | 2.00 mm | 5.25 mm |
| L2 | 4 / 4 (100%) | 3.75 mm | 5.75 mm |
| L1 | 4 / 4 (100%) | 2.00 mm | 2.75 mm |

| resource | measured |
|---|---:|
| elapsed | 4 min 50 s |
| maximum resident memory | 928,044 KiB |
| output size | 73,294,125 bytes |
| GPU | 0 |

**Verdict: DEVELOPMENT PASS**, because every frozen v7 development gate passed; this is not a locked or patient-held-out pass.

### 1.4 Reading it

- Actor inference uses ultrasound pixels, executed motion, calibration-derived offsets, and calibration-derived adjacent separations.
- Actor inference does not use raw labels, segmentation, patient identity, vertebra meshes, reference coordinates, goal pose, target distance, or evaluator score.
- The final development result followed preserved failures: job 51116088 reached 5 / 28, jobs 51206842 and 51208322 reached 26 / 28, and job 51207828 failed technically before evaluation.
- The passing threshold lies in a measured stable development region: thresholds 0.30--0.40 and spacing penalties 0.03--0.10 produced 28 / 28 in the stored sensitivity analysis.
- The 71.6 MiB checkpoint and result bundle remain on Blue Storage and are excluded from Git.
- A 14.07-second demo from lexicographically first session `ua_00` shows the map remain abstaining until bidirectional evidence yields all seven landmarks at frame 206.
- Fixed-anatomy success does not establish patient-held-out recognition or Stage 2 best-view acquisition.

### 1.5 Files

| file | contents |
|---|---|
| [result manifest](../../experiments/unity_two_stage_transfer/stage1_group_a_result.json) | concise metrics, hashes, resources, and limitations |
| [result digest](../../experiments/unity_two_stage_transfer/stage1_group_a_result.sha256) | result-manifest SHA-256 |
| [protocol v7](../../experiments/unity_two_stage_transfer/stage1_group_a_protocol_v7.yaml) | selected development design and gates |
| [trainer](../../workflows/two_stage_navigation/train_stage1_group_a.py) | fit, calibration, spatial aggregation, and evaluation |
| [demo renderer](../../workflows/two_stage_navigation/render_stage1_group_a_demo.py) | actor-only reconstruction visualization with evaluator-only overlays |
| external run | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage1_group_a_v7/51209243` |
| external demo | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage1_group_a_demo/51210068` |
