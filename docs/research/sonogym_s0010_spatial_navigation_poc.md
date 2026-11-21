# s0010 Motion-compensated spatial cortical mapping and navigation

This experiment tests an actor-side 2.5D map on existing fixed-`s0010` observations, then delivers the strongest verified navigation workflow when the new map fails its frozen gate. The new evidence is opened-development replay; the navigation demonstration replays a previously executed closed-loop run and makes no SAP, MBB-intervention, clinical, or patient-generalization claim.

## Result

**The new map is `SPATIAL_MAPPING_NOT_SUPPORTED` for the declared two-target gate; the highest verified milestone is `NAVIGATION_L5_ONLY_PASS`, supported by the historical closed-loop job `52039283` and an explicitly labelled offline replay rather than fresh independent execution.**

| step | what was measured | criterion | measured | verdict |
|---|---|---|---|---|
| 1 | Motion-compensated spatial cortical mapping | every L4/L5 target/session selection record clears all frozen geometric gates | L5 passed 2/2; L4 passed 0/2 | fail |
| 2 | historical Stage 1 to RIGHT L5 TP navigation | Stage 1 4/4 and L5 selected, returned, and freshly confirmed in 4/4 sessions | Stage 1 4/4; L5 4/4 | pass, historical replay |

The L5 map records show physical consistency beyond shuffled motion and appearance-only matching, but the predeclared method-selection rule required both target levels. The failed gate prevented Phase B, prevented the authorized GPU job, and triggered the predetermined L5 fallback.

## 1. Motion-compensated spatial cortical mapping

`source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/ultrasound_spatial_map.py`, run by `hpg/jobs/s0010_spatial_cortical_mapping.sbatch`. It transforms image-derived cortical samples with public calibration and executed reduced pose, associates mutual nearest samples across distinct views, and opens native coordinates only in the evaluator.

**Criterion.** Each L4/L5 target/session record must contain at least 50 associations, median native separation at most 5 mm, p90 at most 10 mm, at most 20% associations over 10 mm, and a correct-motion median no more than 75% of shuffled motion. These limits were sealed in protocol SHA-256 `7b56ae7a0f5f42cc05b1d6054c9b5cb15a3f9e842d34695f1faa8fc6df6ad93f` before job `52246037`; the final-comparison sessions remain unopened when no hypothesis passes every selection record.

### 1.0 Quantities

| symbol | definition | source |
|---|---|---|
| Motion-compensated spatial cortical mapping | actor-side 2.5D method used in this step | [`docs/notation.md`](../notation.md#motion-compensated-spatial-cortical-mapping) |

### 1.1 Method

For image column offset $u$, executed surface pose $(x,z,\theta)$, public image resolution $r_i$, and public label resolution $r_l$, the actor stores

$$
\left(r_lx+r_i u\cos\theta,\ r_lz+r_i u\sin\theta,\ r_i d\right),
$$

where $d$ is the clean-image cortical-candidate row. The surface-only control sets the third coordinate to zero.

- distinct frame hash and pose -> eligible independent support
- duplicate hash or sub-threshold pose change -> no support
- mutual-nearest distance at most 4 mm -> actor association
- native pixel-to-volume distance -> evaluator-only measurement
- shuffled executed poses -> motion control
- row and ridge matching without motion -> appearance-only control

### 1.2 Setup

| what | value | spec |
|---|---|---|
| run | job `52246037`, commit `ef853f797ea2f9faf6f3abc86fffc0eb8722ee09` | §1 |
| input | `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_mbb_single_anatomy_poc_v3/52039283` | §1 |
| selection sessions | `mbbpoc_v3_00`, `mbbpoc_v3_01` | §1 |
| final-comparison sessions | `mbbpoc_v3_02`, `mbbpoc_v3_03`; unopened by fail-closed runner | §1 |
| hypotheses | 2.5D with depth; surface-only without depth | §1 |
| resources | account `anest-dept`; 4 CPU; 16 GB; 0 GPU; 37 s | §1 |

```bash
SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env \
  sbatch hpg/jobs/s0010_spatial_cortical_mapping.sbatch
# python workflows/two_stage_navigation/run_s0010_spatial_cortical_mapping.py \
#   --protocol experiments/unity_two_stage_transfer/s0010_spatial_cortical_mapping_protocol_v1.yaml \
#   --run-root /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_mbb_single_anatomy_poc_v3/52039283 \
#   --output-dir /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_spatial_cortical_mapping/52246037/analysis \
#   --source-commit ef853f797ea2f9faf6f3abc86fffc0eb8722ee09
```

| | value |
|---|---:|
| attempted target/session records per hypothesis | 4 |
| dropped | 0 |
| **final** | **4** |

### 1.3 Results

| hypothesis | session | target | associations | median / p90, mm | fraction over 10 mm | correct / shuffled median | gate |
|---|---|---|---:|---:|---:|---:|---|
| 2.5D | `mbbpoc_v3_00` | L4 | 3,119 | 2.188 / 11.945 | 0.221 | 0.216 | fail |
| 2.5D | `mbbpoc_v3_00` | L5 | 2,853 | 2.571 / 8.376 | 0.087 | 0.226 | pass |
| 2.5D | `mbbpoc_v3_01` | L4 | 2,766 | 2.467 / 10.228 | 0.119 | 0.280 | fail |
| 2.5D | `mbbpoc_v3_01` | L5 | 3,096 | 2.154 / 6.551 | 0.045 | 0.201 | pass |
| surface-only | `mbbpoc_v3_00` | L4 | 5,094 | 4.873 / 19.472 | 0.355 | 0.386 | fail |
| surface-only | `mbbpoc_v3_00` | L5 | 4,941 | 4.451 / 14.873 | 0.261 | 0.341 | fail |
| surface-only | `mbbpoc_v3_01` | L4 | 4,933 | 5.466 / 15.158 | 0.314 | 0.541 | fail |
| surface-only | `mbbpoc_v3_01` | L5 | 4,900 | 3.471 / 11.541 | 0.168 | 0.248 | fail |

**Verdict: fail (`SPATIAL_MAPPING_NOT_SUPPORTED`)**, because neither hypothesis passed all four target/session records; Phase B and the GPU eligibility gate remained false.

### 1.4 Reading it

- Motion compensation reduced the median evaluator separation relative to shuffled motion in every record.
- 2.5D depth retained physically useful information that the surface-only control lost.
- L5 passed every selection-session gate under 2.5D mapping.
- L4 retained a long error tail despite low median error.
- Final-comparison sessions were not read by the corrected run.
- No threshold, hypothesis, or gate was changed after the result.

### 1.5 Files

| file | contents |
|---|---|
| `experiments/unity_two_stage_transfer/s0010_spatial_cortical_mapping_protocol_v1.yaml` | sealed hypotheses and gates |
| runtime job `52246037/analysis/result.json` | fail-closed aggregate result |
| runtime job `52246037/analysis/mapping_metrics.csv` | per-target/session measurements |
| runtime job `52246037/tests.log` | 14 focused regression tests |
| `tests/test_ultrasound_spatial_map.py` | geometry, freshness, and leakage checks |

## 2. Historical RIGHT L5 TP closed-loop navigation replay

`workflows/two_stage_navigation/build_s0010_spatial_navigation_demo.py`, run by `hpg/jobs/s0010_spatial_navigation_demo.sbatch`. It combines actual saved ultrasound, actor cortical candidates, the accumulated 2.5D map, recorded robot video, and a separately labelled privileged evaluator panel.

**Criterion.** The fallback passes only if the source closed-loop run has complete Stage 1 maps in 4/4 sessions and independently evaluator-confirmed RIGHT L5 TP selection, physical return, and fresh confirmation in 4/4 sessions. This is the historical V3 criterion; replay does not create new independent evidence.

### 2.1 Method

- select one historical L5 success and one historical L4 failure -> show both verified capability and limitation
- replay actual frame identities, ultrasound arrays, executed poses, and recorded robot video -> no fabricated motion
- show actor-visible and privileged panels separately -> preserve the information boundary
- label the artifact as offline replay -> no fresh-execution claim

### 2.2 Setup

| what | value | spec |
|---|---|---|
| source execution | job `52039283`, commit `79d3f35560d36addf5185af55993a449118c6043` | §2 |
| source protocol | SHA-256 `9ed7680856e64e54270d33f36867021732de7de8da77821fcaf951d4f4e65a55` | §2 |
| demo job | `52246624`, CPU-only | §2 |
| input | runtime job `52039283/stage2` | §2 |
| video mode | offline replay of historical closed-loop execution | §2 |

```bash
sbatch --dependency=afterok:52246037 \
  --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env,SPATIAL_ANALYSIS_JOB_ID=52246037 \
  hpg/jobs/s0010_spatial_navigation_demo.sbatch
# python workflows/two_stage_navigation/build_s0010_spatial_navigation_demo.py \
#   --analysis /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_spatial_cortical_mapping/52246037/analysis \
#   --run-root /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_mbb_single_anatomy_poc_v3/52039283 \
#   --protocol experiments/unity_two_stage_transfer/s0010_spatial_cortical_mapping_protocol_v1.yaml \
#   --output /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_spatial_navigation_demo/52246624/s0010_spatial_navigation_replay.mp4
```

| | value |
|---|---:|
| historical Stage 1 sessions | 4 |
| historical L5 target attempts | 4 |
| dropped | 0 |
| **final** | **4** |

### 2.3 Results

| source measurement | result |
|---|---:|
| Stage 1 complete maps | 4 / 4 |
| correct Stage 1 landmark events | 28 / 28 |
| Stage 1 median localization error | 3.0 mm |
| RIGHT L5 TP target success | 4 / 4 |
| RIGHT L4 TP target success | 2 / 4 |
| mean / maximum return position error | 0.239 / 0.250 mm |
| mean / maximum return orientation error | 0.027 / 0.141 degrees |
| fresh proxy-positive / ambiguous confirmations | 17 / 7 of 24 |

**Verdict: pass (`NAVIGATION_L5_ONLY_PASS`)**, because the historical closed-loop run completed Stage 1 and RIGHT L5 selection, physical return, and fresh confirmation in all four declared sessions; L4 and full two-target navigation did not pass.

### 2.4 Reading it

- The video is evidence visualization, not a new simulator run.
- Panel 1 uses clean ultrasound and actor cortical candidates.
- Panel 2 uses only executed reduced pose and public calibration.
- Panel 3 uses actual recorded robot/probe video.
- Panel 4 is privileged evaluator evidence and is not an actor input.
- `FULL_MBB_INTERVENTION = BLOCKED`; no validated MBB target-region or needle/injection contract exists.

### 2.5 Files

| file | contents |
|---|---|
| runtime job `52246624/s0010_spatial_navigation_replay.mp4` | four-panel success/failure replay |
| runtime job `52246624/s0010_spatial_navigation_replay.json` | video provenance and scenario identities |
| runtime job `52039283/results/end_to_end_report.json` | historical integrated measurements |
| `experiments/unity_two_stage_transfer/s0010_mbb_single_anatomy_poc_v3_result.json` | concise preserved historical result |

## 3. RIGHT L5 spatial target-view localizer follow-up

This follow-up isolates the RIGHT L5 TP surface-core task that had passed the earlier association gate. It audits a calibrated actor-side image transform, evaluates cortical candidates against physical bone surfaces, and makes the previously missing target-view decision from the accumulated map.

**Result.** `SPATIAL_GEOMETRY_PARTIAL`; `L5_SPATIAL_LOCALIZER_FAILED`; `STAGE2_INTEGRATION_BLOCKED`. The support hypothesis accepted one correct view and safely abstained once, but did not improve over the single-frame or historical Stage 2 baselines and did not distinguish the shuffled-motion control. The fail-closed runner did not open `mbbpoc_v3_02` or `mbbpoc_v3_03`.

### 3.1 Calibrated geometry audit

**Criterion.** The full-pose public-calibration transform must reproduce the saved native coordinate map with median and p90 error at most 0.001 mm and maximum error at most 0.01 mm. A full spatial-geometry verdict additionally requires synchronized 6DoF pose in the actor-visible historical record and at least 50% of image-derived cortical samples within 3 mm of a vertebral surface in every evaluated session.

**Method.** The actor transform maps ultrasound pixels through probe-local calibration and synchronized forward-kinematics pose. Native pixel-to-volume coordinates and vertebral masks are opened only by the evaluator. The recorded V3 actor input lacks synchronized 6DoF, so the localizer itself retains the historical reduced `(x,z,angle)` representation.

**Setup.** Protocol SHA-256 `cce72e9945799aa0968f4dcbf66989c918246f3de7f4f2b8a7b38139d48ff186`; implementation commit `792dfffb9383e0f86821e69aa631336b81c64068`; CPU job `52253724`; two opened-development selection sessions; 4 CPU, 16 GB requested, 0 GPU, 21 s job elapsed and 10.33 s measured analysis runtime. Slurm reported only 3 MB for the batch shell rather than a credible Python-process peak, so observed peak memory is unavailable rather than inferred from that incomplete counter.

**Results.**

| session | full-pose median / p90 / max error, mm | reduced 2.5D median / p90 horizontal error, mm | cortical samples within 3 mm of bone | target/non-target association crossing |
|---|---:|---:|---:|---:|
| `mbbpoc_v3_00` | 0.0000217 / 0.0000346 / 0.0000742 | 5.169 / 11.428 | 59.24% | 3.44% |
| `mbbpoc_v3_01` | 0.0000221 / 0.0000353 / 0.0000760 | 5.053 / 8.033 | 47.37% | 2.16% |

**Reading it.** The calibrated transform is numerically correct when full 6DoF is supplied, across depth and orientation bins. This does not validate the historical actor map: those public records contain only reduced pose, which produces about 5 mm median horizontal error, and one session misses the bone-surface inlier gate. The correct verdict is therefore `SPATIAL_GEOMETRY_PARTIAL`, not calibrated 3D reconstruction.

**Files.** Runtime `52253724/analysis/geometry_validation.json`; `ultrasound_spatial_map.py`; `tests/test_ultrasound_spatial_map.py`.

### 3.2 L5 target localization

**Criterion.** A hypothesis must correctly accept both selection sessions, make no wrong acceptance, place a positive in the top three in both sessions, keep mean proposal-center-to-core error at most 5 mm, and use at least three distinct supporting frames. Only then may the two final-comparison sessions be opened.

**Method.** Two frozen hypotheses rank candidate views from the existing target probability and distinct-view cortical support: direct support, and support multiplied by a Stage 1 landmark-relative prior. The decision can abstain on low score or insufficient margin. Controls are maximum single-frame probability, historical Stage 2 selection, and deliberately shuffled executed motion.

**Setup.** `mbbpoc_v3_00` and `mbbpoc_v3_01` are method-selection evidence. `mbbpoc_v3_02` and `mbbpoc_v3_03` are reserved for one final comparison and remain unopened because neither hypothesis passed selection.

**Results.**

| hypothesis | correct accepted | wrong accepted | abstained | top-3 contains positive | gate |
|---|---:|---:|---:|---:|---|
| support | 1/2 | 0/2 | 1/2 | 2/2 | fail |
| landmark-relative | 0/2 | 0/2 | 2/2 | 2/2 | fail |

For direct support, session `00` selected frame 73, a proxy-positive proposal 1.223 mm from the target core supported by 12 distinct frames. Session `01` abstained because the two leading eligible views were not separated by the frozen score margin, although its top three were all proxy-positive. Maximum single-frame probability and historical Stage 2 selection were proxy-positive in 2/2 sessions. Shuffled motion reproduced the direct-support outcome pattern: one proxy-positive acceptance and one abstention.

**Reading it.** Spatial support can rank valid candidates, but these data do not show that it adds target identity or robustness: it underperforms the baselines and does not separate correct from shuffled motion. This is a scientific localizer failure under the frozen gate, not an engineering failure. No Stage 2 integration interface is promoted and no new closed-loop success is claimed.

**Files.** Runtime `52253724/analysis/result.json` (SHA-256 `7e83bf5be4852c74478d93f3d4ef765e4a6daebb459c2e04de635d92229cfccd`); `session_results.csv`; `frame_rankings.csv` (SHA-256 `280e6381d734f7e930a3ea68d28da16ebe1ce38ff3ab4ad463109d26c29d5539`); `localizer_summary.png`; job `tests.log` (19/19 pass).
