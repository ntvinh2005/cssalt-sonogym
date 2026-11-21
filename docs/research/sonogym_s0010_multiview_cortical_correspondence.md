# s0010 multi-view cortical correspondence

This report tests whether actor-visible ultrasound interfaces and executed relative motion improve RIGHT L4/L5 SAP candidate ranking on fixed anatomy `s0010`. It covers opened development evidence only; no new capture, independent confirmation, active-acquisition integration, or clinical claim is in scope.

## Result

**Outcome C: the tested multi-view cortical correspondence does not supply reliable SAP-specific ranking on calibration, so evaluation remained unopened for the learned ranker and no GPU capture is justified.**

| step | what was measured | criterion | measured | verdict |
|---|---|---|---|---|
| 1 | existing image-only baseline | reference only | L5 full-image top-1 positive in 1 / 2 sessions; patch F1 0.818 | reference |
| 2 | fixed correspondence aggregation | calibration gate | 0 / 5 L5 positive sessions accepted correctly | fail |
| 3 | interface similarity by label pair | distribution separation | positive-positive and negative-negative distributions overlap | fail |
| 4 | broader fixed correspondence aggregation | calibration gate | 1 / 5 L5 positive sessions accepted correctly; 2 wrong selections | fail |
| 5 | Learned multi-view ranking | calibration gate | 1 / 5 L5 positive sessions accepted correctly; evaluation unopened | fail |
| 6 | recorded-behavior video | artifact and information-boundary checks | 133 frames, three calibration outcomes, separate evaluator labels | pass |

The repeated failure on calibration shows that cross-view interface resemblance is not SAP identity in this renderer. The next method should rank a stable visible bone configuration or motion-conditioned localization evidence, not add another threshold or single-frame SAP classifier to the same correspondence signal.

## 1. Existing baseline — what problem remains after the image representation experiment

`workflows/two_stage_navigation/run_s0010_ultrasound_representation_experiment.py`, run by `hpg/jobs/s0010_ultrasound_representation.sbatch`. It compares the frozen patch checkpoint with image-only interface representations.

**Criterion.** No absolute threshold; this is the frozen reference against which the new session-level methods are compared.

### 1.1 Method

- Fit on training sessions -> no evaluation image enters fitting.
- Select the image representation and threshold on calibration -> opened evaluation is read after selection.
- Report ambiguous images separately -> they are not hard negatives.

### 1.2 Setup

| what | value | spec |
|---|---|---|
| run | job `52203168`, source commit `b6888d87c7602190217b3d482bcf2a15c50741ad` | §1 |
| inputs | `s0010_sap_validation_resume_v2/52199322/analysis_v1` | §1 |
| frozen patch checkpoint | SHA-256 `c173aa9812a2df25114f99604b7330ed06b435c3a77edc7e2901f2b5dd3896e5` | §1 |

```bash
sbatch --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env hpg/jobs/s0010_ultrasound_representation.sbatch
# The job runs run_s0010_ultrasound_representation_experiment.py with the frozen development and evaluation manifests recorded in result.json.
```

| | value |
|---|---:|
| acquired evaluation images | 80 |
| clear L5 images | 44 |
| ambiguous L5 images | 36 |
| **final session rankings** | **2** |

### 1.3 Results

| L5 reference | TP | FP | FN | TN | precision | recall | F1 | top-1 positive sessions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen patch | 9 | 0 | 4 | 31 | 1.000 | 0.692 | 0.818 | not reported |
| selected full image | 13 | 0 | 0 | 31 | 1.000 | 1.000 | 1.000 | 1 / 2 (50%) |

**Verdict: reference established**, because image-level classification improved while session top-1 ranking remained wrong in 1 / 2 sessions.

### 1.4 Reading it

- The 17 / 36 ambiguous activations prevent promotion of the selected full-image result.
- Top-3 contained a positive in 2 / 2 sessions, motivating session-level ranking rather than more single-frame fitting.
- These evaluation sessions were already opened and are not independent confirmation.

### 1.5 Files

| file | contents |
|---|---|
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_ultrasound_representation/52203168/result.json` | frozen reference metrics and rankings |
| `docs/research/sonogym_s0010_ultrasound_representation.md` | preceding representation experiment |

## 2. Fixed aggregation — does strict repeated-view support identify SAP candidates

`workflows/two_stage_navigation/run_s0010_stage2_multiview_evidence.py`, run by `hpg/jobs/s0010_stage2_multiview_evidence.sbatch`. It combines the image-only score with cortical-interface correspondence from distinct nearby views.

**Criterion.** Calibration must accept at least 4 / 5 L5 positive sessions, select no wrong frame in a positive session, and accept no session without a positive.

### 2.1 Method

- Reconstruct cumulative relative pose from executed actor motion -> no exact target pose enters the actor features.
- Exclude inadmissible frames from evidence but retain their executed motion -> odometry remains consistent.
- Require distinct physical viewpoints and repeated interface support -> cached or repeated frames cannot vote.

### 2.2 Setup

| what | value | spec |
|---|---|---|
| run | job `52232636`, commit `c1b46e1` | §2 |
| protocol | `s0010_stage2_multiview_evidence_protocol_v1r1.yaml`, SHA-256 `f9965538e7d51d895cf3ff68b4ae27451975bb415f1e12293ebfb4eab1c2f596` | §2 |
| resources | 4 CPU, 16 GB, 0 GPU, 1:52 | §2 |

```bash
sbatch --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env hpg/jobs/s0010_stage2_multiview_evidence.sbatch
# The job runs run_s0010_stage2_multiview_evidence.py with protocol v1r1 and the frozen development/evaluation manifests.
```

| | value |
|---|---:|
| L5 calibration sessions | 11 |
| L5 positive calibration sessions | 5 |
| excluded inadmissible frames | 20 |
| **accepted positive sessions** | **0** |

### 2.3 Results

| level | accepted positive sessions | wrong accepted positive sessions | top-1 positive sessions | top-3 positive sessions |
|---|---:|---:|---:|---:|
| L4 diagnostic | 0 / 4 | 0 | 0 / 4 | 0 / 4 |
| L5 | 0 / 5 | 0 | 0 / 5 | 0 / 5 |

**Verdict: fail**, because strict interface support produced no eligible accepted L5 candidate.

### 2.4 Reading it

- The strict settings were technically valid but removed all usable repeated-view evidence.
- The result does not imply SAP absence; it rejects this aggregation setting.

### 2.5 Files

| file | contents |
|---|---|
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_stage2_multiview_evidence/52232636/result.json` | calibration and opened-development results |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_stage2_multiview_evidence/52232636/frozen_selection.json` | settings fixed before evaluation |

## 3. Calibration diagnostic — is interface similarity specific to positive anatomy

`workflows/two_stage_navigation/diagnose_s0010_multiview_correspondence.py`, run by `hpg/jobs/s0010_multiview_correspondence_diagnostic.sbatch`. It measures interface similarity by proxy-label pair on calibration only.

**Criterion.** No absolute threshold; compare positive-positive correspondence with negative-negative and ambiguous pair distributions.

### 3.1 Method

- Use only admissible calibration frames -> evaluation remains unopened.
- Group within-session pairs by proxy status -> report overlap without converting ambiguity into negatives.
- Count frames with at least two physically distinct supporting views -> establish usable support coverage.

### 3.2 Setup

| what | value | spec |
|---|---|---|
| run | job `52233164`, commit `2b14955` | §3 |
| inputs | calibration portion of `dataset_development_merged` | §3 |
| resources | 4 CPU, 12 GB, 0 GPU, 1:37 | §3 |

```bash
sbatch --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env,MULTIVIEW_RUN_ID=52232636 hpg/jobs/s0010_multiview_correspondence_diagnostic.sbatch
# The job runs diagnose_s0010_multiview_correspondence.py on calibration only.
```

| | value |
|---|---:|
| admissible frames per level | 406 |
| L5 positive-positive pairs | 7 |
| L5 negative-negative pairs | 145 |
| **frames with at least two supports at 0.25** | **65** |

### 3.3 Results

| L5 pair | median | q75 | q90 | maximum |
|---|---:|---:|---:|---:|
| positive-positive | 0.357 | 0.383 | 0.423 | 0.475 |
| negative-negative | 0.245 | 0.327 | 0.404 | 0.651 |
| ambiguous-positive | 0.230 | 0.304 | 0.338 | 0.418 |

**Verdict: fail as semantic evidence**, because positive-positive correspondence overlaps negative-negative correspondence and the highest negative-negative value exceeds every positive-positive value.

### 3.4 Reading it

- The detector can associate similar cortical morphology across movement.
- Similar morphology occurs in non-SAP bone, so correspondence alone is not semantic identity.
- Seven positive-positive pairs are insufficient for a strong learned pair model.

### 3.5 Files

| file | contents |
|---|---|
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_stage2_multiview_evidence/52232636/calibration_correspondence_diagnostic.json` | pair distributions and support coverage |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_stage2_multiview_evidence/52232636/calibration_correspondence_diagnostic.csv` | calibration pair records |

## 4. Broader fixed aggregation — does increased support recover session ranking

`workflows/two_stage_navigation/run_s0010_stage2_multiview_evidence.py`, run by `hpg/jobs/s0010_stage2_multiview_evidence.sbatch`. It applies the predeclared broader correspondence candidates after §3.

**Criterion.** The same calibration gate as §2; evaluation results cannot override a failed calibration gate.

### 4.1 Method

- Lower the minimum correspondence only through the sealed protocol -> no evaluation-driven threshold change.
- Select settings on calibration -> preserve the chosen setting before evaluation.
- Require calibration pass in the final development gate -> the later fix prevents a false pass.

### 4.2 Setup

| what | value | spec |
|---|---|---|
| run | job `52233930`, commit `bb0ee33` | §4 |
| protocol | `s0010_stage2_multiview_evidence_protocol_v2.yaml`, SHA-256 `78c6b3f8c652976b6fe0af06ab3411cefc6cf1cd3cac32f18a1649e6da70faab` | §4 |
| resources | 4 CPU, 16 GB, 0 GPU, 1:54 | §4 |

```bash
sbatch --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env hpg/jobs/s0010_stage2_multiview_evidence.sbatch
# The job runs run_s0010_stage2_multiview_evidence.py with protocol v2.
```

| | value |
|---|---:|
| L5 positive calibration sessions | 5 |
| accepted positive calibration sessions | 1 |
| wrong accepted positive sessions | 2 |
| **top-1 positive calibration sessions** | **1** |

### 4.3 Results

| split | top-1 positive sessions | accepted positive sessions | wrong accepted positive sessions |
|---|---:|---:|---:|
| calibration | 1 / 5 (20%) | 1 / 5 (20%) | 2 |
| opened development | 2 / 2 (100%) | 2 / 2 (100%) | 0 |

**Verdict: fail**, because calibration failed; the historical `l5_development_gate_pass=true` field was an implementation defect and is invalidated by commit `431a7d8`.

### 4.4 Reading it

- The two opened-development sessions cannot rescue a method selected from failed calibration.
- The regression test now requires `calibration_gate_pass=true` before any development pass.
- Historical artifacts remain unchanged; the correction is explicit rather than retroactive.

### 4.5 Files

| file | contents |
|---|---|
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_stage2_multiview_evidence/52233930/result.json` | frozen historical output including the invalid pass field |
| `tests/test_s0010_stage2_multiview_evidence.py` | fail-closed regression test |

## 5. Learned multi-view ranking — can training combine image and motion evidence better

`workflows/two_stage_navigation/run_s0010_stage2_multiview_ranker.py`, run by `hpg/jobs/s0010_stage2_multiview_ranker.sbatch`. It fits the approved Learned multi-view ranking from session-out-of-fold actor-visible scores and correspondence features.

**Criterion.** Calibration must accept at least 4 / 5 L5 positive sessions, select no wrong frame in a positive session, and accept no session without a positive; evaluation opens only after a calibration pass.

### 5.1 Method

- Generate base scores with five session-disjoint folds -> the ranker never sees an in-fold base score.
- Exclude ambiguous and unsupported frames from hard training labels -> uncertainty is preserved.
- Fit a class-balanced logistic model -> test a minimal learned combination before considering a larger architecture.
- Freeze threshold and top-1 margin on calibration -> evaluation remains unopened if the gate fails.

### 5.2 Setup

| what | value | spec |
|---|---|---|
| run | job `52234900`, commit `431a7d8` | §5 |
| protocol | `s0010_stage2_multiview_ranker_protocol_v1.yaml`, SHA-256 `1393aa7a82d268895fd1ed05d406325377afecf1344b51aa79b0869a130662cb` | §5 |
| resources | 4 CPU, 16 GB, 0 GPU, 2:44 | §5 |

```bash
sbatch --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env hpg/jobs/s0010_stage2_multiview_ranker.sbatch
# The job fits 240-tree topology base models in five session-out-of-fold partitions, then fits one logistic ranker per level.
```

| | L4 | L5 |
|---|---:|---:|
| session-out-of-fold clear images | 102 | 93 |
| positive | 10 | 21 |
| negative | 92 | 72 |
| excluded ambiguous | 494 | 331 |
| excluded unsupported clear | 259 | 431 |

### 5.3 Results

| level | positive sessions | top-1 positive | top-3 positive | accepted positive | wrong accepted | ambiguous activations |
|---|---:|---:|---:|---:|---:|---:|
| L4 diagnostic | 4 | 1 | 1 | 1 | 1 | 18 |
| L5 | 5 | 1 | 3 | 1 | 1 | 11 |

**Verdict: fail**, because L5 accepted only 1 / 5 positive sessions and selected one wrong frame; evaluation was not opened.

### 5.4 Reading it

- The selected L5 threshold was 0.80 with a 0.05 top-1 margin.
- The L5 ranker assigned near-one scores to ambiguous candidates in multiple sessions.
- The low count of session-out-of-fold positives limits a more flexible learned model.
- A larger model would fit the same non-specific correspondence evidence and is not justified by this result.

### 5.5 Files

| file | contents |
|---|---|
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_stage2_multiview_ranker/52234900/result.json` | calibration failure and unopened-evaluation verdict |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_stage2_multiview_ranker/52234900/frozen_selection.json` | all calibration settings and training accounting |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_stage2_multiview_ranker/52234900/s0010_l5_multiview_ranker.joblib` | failed exploratory checkpoint, SHA-256 `322bb358d8dd9e7d6fa7a6dbe41e177ed8a492ef1e65aff6d5918650c484ce99` |

## 6. Recorded-behavior demo — what does the ranker actually follow

`workflows/two_stage_navigation/render_s0010_multiview_ranker_demo.py`, run by `hpg/jobs/s0010_stage2_multiview_demo.sbatch`. It renders clean ultrasound, detected interfaces, strongest supporting view, executed-motion history, and a separate evaluator-only label.

**Criterion.** The MP4 must decode, contain one correct selection, one wrong selection, and one positive-session abstention, and must not place privileged geometry in actor panels.

### 6.1 Method

- Render actual calibration arrays and model outputs -> no motion or prediction is fabricated.
- Show interface tracks and the strongest supporting view -> expose what supplies the rank score.
- Put the proxy label in the evaluator-only title -> actor/evaluator separation remains visible.

### 6.2 Setup

| what | value | spec |
|---|---|---|
| run | job `52235846`, commit `aa8a0a6` | §6 |
| checkpoint | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_stage2_multiview_ranker/52234900/s0010_l5_multiview_ranker.joblib` | §6 |
| resources | 4 CPU, 16 GB, 0 GPU, 1:26 | §6 |

```bash
sbatch --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env,MULTIVIEW_RANKER_JOB_ID=52234900 hpg/jobs/s0010_stage2_multiview_demo.sbatch
# The job renders L5 calibration behavior at 3 fps into an H.264 1280x720 MP4.
```

| | value |
|---|---:|
| selected sessions | 3 |
| rendered frames | 133 |
| duration | 44.33 s |
| **video resolution** | **1280 x 720** |

### 6.3 Results

| session | outcome |
|---|---|
| `s0010tpscv10_cal_l5_00` | accepted positive |
| `s0010sapval_calibration_03` | accepted ambiguous frame |
| `s0010sapval_calibration_00` | positive present, abstained |

**Verdict: pass as an engineering demonstration**, because the video reproduces all three outcomes and keeps evaluator labels outside the actor panels; it does not change the scientific fail in §5.

### 6.4 Reading it

- Cortical contours are detected in both successful and failed sessions.
- High cross-view similarity can reinforce an ambiguous bone interface.
- Executed motion adds viewpoint identity but does not by itself name the anatomy.

### 6.5 Files

| file | contents |
|---|---|
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_stage2_multiview_demo/52235846/s0010_l5_multiview_correspondence.mp4` | real calibration behavior, SHA-256 `e5ddb2814d5ddac193fce647d336e1c72248d145929586df1e7f715d6989cecb` |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_stage2_multiview_demo/52235846/calibration_sessions.csv` | per-session outcomes |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_stage2_multiview_demo/52235846/calibration_frames.csv` | per-frame scores, support, labels, and selection |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_stage2_multiview_demo/52235846/demo_result.json` | video provenance and information-boundary declaration |
