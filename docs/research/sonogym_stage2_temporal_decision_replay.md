# SonoGym Stage 2 temporal and multi-view decision replay

This report evaluates an isolated temporal/multi-view actor contract on opened fixed-`s0010`
captures. It covers CPU-only causal replay and implementation readiness; it does not cover a new
confirmation, physical return-to-best, fresh acquisition, or complete Unity-equivalent Stage 2.

## Result

**EXPLORATORY_NO_GO: causal replay and evidence-independence checks pass, but the declared temporal/multi-view rule accepts only 1 / 144 trajectories (0.6944444444444444%), so eliminating false accepts by 143 / 144 abstentions is not a useful Stage 2 repair.**

| step | what was measured | criterion | measured | verdict |
|---|---|---|---|---|
| 1 | replay identity, ordering, motion and source boundaries | D0: reconstruct every in-scope source without privileged actor input | 18 sessions; 144 trajectories; 1,350 frames reconstructed | PASS |
| 2 | causal actor and evidence-independence implementation | D1/D2/D4: all focused and Stage 2 regression tests pass | 19 / 19 focused; 34 / 34 regression | PASS |
| 3 | declared temporal/multi-view policy on opened data | D3: fewer false accepts than Baseline B, at least half its 103 accepts, no source regression, accepts in at least three sources | 0 false accepts, but only 1 accept; accepted in one source | EXPLORATORY_NO_GO |
| 4 | return-to-best and fresh confirmation availability | actual recorded return and post-return fresh frames | absent from all four capture sources | NOT_TESTABLE |
| 5 | physical closed-loop completion | D5/D6 require separately approved GPU execution | no GPU or Isaac Sim session launched | NOT_RUN |

The failure is not a retroactive change to V18. `V16`, `V17`, and `V18` remain
`CONFIRMATION_NO_GO`; `FULL_STAGE2_CLOSED_LOOP` remains `NOT_RUN`.

## 1. Source and replay audit — can the opened records support causal temporal analysis?

`workflows/two_stage_navigation/replay_stage2_temporal.py`, run by
`hpg/jobs/two_stage_stage2_temporal_replay.sbatch`. It verifies capture identities, recomputes the
frozen model outputs, joins evaluator records only after actor decisions, and reconstructs relative
pose by summing actually executed motion inside each trajectory.

**Criterion.** D0 passes only when frame identity, public/privileged one-to-one joins, reset
boundaries, monotonic causal order, frozen predictions, and executed relative motion are
reconstructable. A source that fails is excluded rather than repaired by privileged pose.

### 1.0 Quantities

| symbol | definition | source |
|---|---|---|
| `p_valid` | frozen V15 image-validity probability | frozen ExtraTrees classifier |
| `q_hat` | frozen V15 continuous image-quality prediction | frozen ExtraTrees regressor |
| `r_t` | actor-relative `(x_mm, z_mm, angle_rad)` after frame `t` | cumulative executed deltas inside one trajectory |
| coverage | accepted trajectories divided by all replayed trajectories | actor decisions |
| accepted precision | image-valid accepted candidates divided by all accepted candidates | evaluator join after selection |

### 1.1 Method

- Public actor records supply image, hash, session/trajectory identity, sequence, executed deltas,
  action feedback, contact/workspace state, and technical image validity.
- Privileged records supply image-validity and geometric labels only after the selected candidate is
  frozen.
- `r_t` restarts at zero at every trajectory boundary; evidence never crosses a reset.
- Equal `producer_sequence` is retained for historical single-frame baselines but cannot support
  temporal or multi-view evidence because simulator time did not advance.
- Decreasing producer sequence, invalid joins, invalid hashes, or forbidden actor fields fail closed.
- Exact hashes define duplicate images. A 20 x 15 normalized-image RMSE at most 0.01 is reported as
  a near-duplicate diagnostic only and never becomes actor evidence.

### 1.2 Setup

| what | value | spec |
|---|---|---|
| run | Slurm `51518820`; source `ef7f5dba9b5c44097351d3ce110637d6859816f1` | CPU replay v1 |
| protocol | `stage2_temporal_replay_protocol_v1.yaml`; SHA-256 `f4e9cd385b4f1a83814c7b0ac2f5675ebd02dd3fd1511e4e830ea528f0a3fa28` | fixed before comparative replay |
| model | V15 `stage2_models.joblib`; SHA-256 `7d3b113ba33b1b80448264a0d2e7f2c2cd5561b26021707648c4eef43d586672` | unchanged classifier/regressor |
| V14 | job `51377943`; capture-summary SHA-256 `5b1498c458f632cd255244b6750659e4cdb6af53b19884741707e4ed3b2fe285` | development/opened |
| V16 | job `51387027`; capture-summary SHA-256 `6b2802f77b29d7a47708916b563bbe03ffafba9690c1833c2c1d46dc79bf252a` | confirmation opened for diagnosis |
| V17 | job `51400282`; capture-summary SHA-256 `4df533ad249b16c70156d4068a255390063e0948cd6dadceec3dae91ee79246e` | confirmation opened for diagnosis |
| V18 | job `51402759`; capture-summary SHA-256 `5937bd65da7df1df723877f8e371e3b6d7a18194d5b3a5e02073e5fcb6e2b863` | confirmation opened for diagnosis |
| environment | Python 3.10.22; NumPy 1.26.4; SciPy 1.15.2; scikit-learn 1.5.2; joblib 1.4.2 | pinned SonoGym CPU environment |
| resources | `anest-dept`; `hpg-default`; 4 CPUs; 16 GB; 0 GPUs | bounded CPU-only |

```bash
sbatch --parsable --account=anest-dept --qos=anest-dept --partition=hpg-default \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:30:00 \
  --job-name=sg-s2-temporal \
  --output=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/slurm/sg-s2-temporal-%j.out \
  --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env \
  hpg/jobs/two_stage_stage2_temporal_replay.sbatch

PYTHONPATH=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/envs/stage1-python:/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/source/spinal_surgery/spinal_surgery/evaluation \
/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/envs/sonogym/bin/python \
  workflows/two_stage_navigation/replay_stage2_temporal.py \
  --protocol experiments/unity_two_stage_transfer/stage2_temporal_replay_protocol_v1.yaml \
  --output-dir /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/stage2_temporal_replay_v1/51518820/results
```

| | value |
|---|---:|
| attempted sources | 4 |
| dropped sources | 0 |
| attempted sessions | 18 |
| dropped sessions | 0 |
| attempted trajectories | 144 |
| dropped trajectories | 0 |
| **final frames** | **1,350** |

### 1.3 Results

| opened source | sessions | trajectories | frames | exact hashes | non-advancing producer frames | action steps | clamped transition frames |
|---|---:|---:|---:|---:|---:|---:|---:|
| V14 development | 6 | 48 | 450 | 197 | 253 | 5,709 | 108 |
| V16 confirmation | 4 | 32 | 300 | 125 | 175 | 3,799 | 69 |
| V17 confirmation | 4 | 32 | 300 | 136 | 164 | 3,953 | 67 |
| V18 confirmation | 4 | 32 | 300 | 126 | 174 | 3,898 | 71 |
| **total** | **18** | **144** | **1,350** | **584 source-wise** | **766** | **17,359** | **315** |

All 1,350 transition frames were accepted by the recorder; no rejected transition frame entered
the replay. No source records physical return-to-best or same-pose reacquisition after return.

**Verdict: PASS**, because all four sources satisfy D0 and no source, session, trajectory, or frame
was dropped.

### 1.4 Reading it

- Causal temporal replay is directly supported.
- Within-trajectory multi-view replay is directly supported for viewpoints that actually occurred.
- File order alone was not treated as continuity; trajectory identity and sequence zero define resets.
- 766 / 1,350 frames did not advance producer sequence and cannot be independent support.
- Fresh confirmation, physical return-to-best, and live coarse/fine/micro control are not testable.
- All captures contain one patient anatomy, `s0010`.

### 1.5 Files

| file | contents |
|---|---|
| [protocol](../../experiments/unity_two_stage_transfer/stage2_temporal_replay_protocol_v1.yaml) | frozen data, policy, gates, and resource contract |
| [protocol seal](../../experiments/unity_two_stage_transfer/stage2_temporal_replay_protocol_v1.sha256) | protocol SHA-256 |
| external `replay_report.json` | full audit and metrics |
| external `frame_scores.csv` | reproducible frozen scores and evaluator-only diagnostics |

## 2. Decision implementation — is evidence causal, distinct, deterministic, and leakage-safe?

`stage2_temporal_decision.py`, exercised by `tests/test_stage2_temporal_decision.py`. It implements
pure candidate memory, temporal/multi-view evidence, deterministic selection, and fail-closed
abstention without changing the existing deployed Stage 2 helpers.

**Criterion.** D1, D2, and D4 pass only if all 19 focused cases and all 34 existing Stage 2
regression tests pass, privileged fields are rejected, and repeated images or invalid movement do
not add evidence.

### 2.1 Method

The predeclared candidate contract is:

```text
SEARCH
  -> threshold-accepted, physically admissible view
  -> CANDIDATE_FOUND
  -> collect only causal, producer-advanced observations in the same trajectory
  -> GATHER_EVIDENCE
  -> require 3 distinct hashes within 4.5 mm and 6 degrees
  -> for multi-view, require every support pair to differ by >=1 mm or >=1 degree
  -> rank qualifying candidates by q_hat, then p_valid, then earliest sequence
  -> SELECT_BEST -> ACCEPT
  -> otherwise ABSTAIN
```

- `RETURN_BEST` and `FRESH_CONFIRM` exist as states but replay cannot enter them.
- The historical V18 threshold remains exactly `0.5666666666666667`.
- Quality consistency permits support no more than 0.08 below the candidate prediction.
- Quality ranks threshold-valid candidates; it cannot override failed image validity.
- `fresh_confirmation_hashes_are_distinct` closes the duplicate-confirmation integrity gap in an
  isolated helper; the deployed `fresh_confirmation_passes` behavior was not silently changed.

### 2.2 Setup

| what | value | spec |
|---|---|---|
| pure module | `source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/stage2_temporal_decision.py` | no simulator imports |
| focused tests | `tests/test_stage2_temporal_decision.py` | synthetic integrity/causality cases only |
| regression tests | `tests/test_two_stage_stage2.py` | existing Stage 2 suite |
| fixed parameters | 3 hashes; 4.5 mm / 6 degrees neighborhood; 1 mm or 1 degree view separation; 0.08 quality drop | protocol v1 |

```bash
PYTHONPATH=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/envs/stage1-python:/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/source/spinal_surgery/spinal_surgery/evaluation \
/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/envs/sonogym/bin/python \
  tests/test_stage2_temporal_decision.py
PYTHONPATH=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/envs/stage1-python:/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/source/spinal_surgery/spinal_surgery/evaluation \
/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/envs/sonogym/bin/python \
  tests/test_two_stage_stage2.py
```

| | value |
|---|---:|
| focused cases attempted | 19 |
| focused cases dropped | 0 |
| regression cases attempted | 34 |
| regression cases dropped | 0 |
| **final cases** | **53** |

### 2.3 Results

| test family | passed | failed | verdict |
|---|---:|---:|---|
| duplicate/hash/pose independence | 6 | 0 | PASS |
| reset, trajectory, causal-prefix and leakage boundaries | 5 | 0 | PASS |
| invalid action, missing data and abstention | 5 | 0 | PASS |
| deterministic ordering and fresh-hash contract | 3 | 0 | PASS |
| existing Stage 2 regression | 34 | 0 | PASS |

**Verdict: PASS**, because 53 / 53 tests pass and all three implementation gates D1, D2, and D4
are met.

### 2.4 Reading it

- Three copies of one false-positive image contribute one hash, not three votes.
- Different hashes at one pose do not become distinct viewpoints.
- Invalid, clamped, rejected, stale, cross-reset, future, or cross-trajectory evidence is unusable.
- Forbidden evaluator fields in actor input raise an exception.
- Empty or insufficient evidence returns `ABSTAIN`, not forced selection.
- These are engineering correctness tests, not evidence of scientific performance.

### 2.5 Files

| file | contents |
|---|---|
| [decision module](../../source/spinal_surgery/spinal_surgery/evaluation/two_stage_navigation/stage2_temporal_decision.py) | pure actor state and evidence logic |
| [focused tests](../../tests/test_stage2_temporal_decision.py) | 19 synthetic and regression guards |
| [CPU job](../../hpg/jobs/two_stage_stage2_temporal_replay.sbatch) | tests, replay, telemetry, and failure propagation |

## 3. Opened-data replay — does temporal/multi-view evidence improve useful decisions?

`replay_stage2_temporal.py`, run under the configuration in §1.2. It compares five predeclared
policies without fitting, calibrating, sweeping, or changing either frozen model.

**Criterion.** D3 is exploratory because V14 and V16--V18 are opened. Candidate D is promising
only if it has fewer false accepts than Baseline B, retains at least half of Baseline B's 103
accepts, has no source-level false-accept regression, and accepts in at least three sources.

### 3.1 Method

| policy | actor rule |
|---|---|
| A, single-frame | first physically admissible frame at or above historical threshold |
| B, quality-ranked | maximum `q_hat` among physically admissible threshold-accepted frames |
| C, temporal | B plus three distinct causal hashes in the declared local neighborhood |
| D, multi-view | C plus three pairwise physically separated viewpoints |
| quality-only diagnostic | maximum `q_hat` without STOP authority |

The actor consumes a trajectory sequentially. The evaluator reads the chosen frame only after the
decision. Baseline A decides at its first accepted observation; B--D select after the last actually
recorded observation. No frame or transition is synthesized.

### 3.2 Setup

| what | value | spec |
|---|---|---|
| replay job | `51518820` | source and command in §1.2 |
| historical threshold | `0.5666666666666667` | V18; unchanged |
| trajectory population | 144 / 144 | all four opened sources |
| evaluation labels | image validity primary; 10 mm / 10 degree geometry secondary | joined after selection |
| tuning | none | one predeclared parameter set |

The sample accounting is identical to §1.2: 144 trajectories attempted, zero dropped, 144 final.

### 3.3 Results

| policy | accepted | false accepted | accepted precision | coverage | abstained | median decision frames | median support hashes / views |
|---|---:|---:|---:|---:|---:|---:|---:|
| A single-frame | 103 / 144 | 3 / 103 | 0.970873786407767 | 103 / 144 (71.52777777777777%) | 41 / 144 | 3 | 1 / 1 |
| B quality-ranked + validity | 103 / 144 | 2 / 103 | 0.9805825242718447 | 103 / 144 (71.52777777777777%) | 41 / 144 | 10 | 1 / 1 |
| C temporal | 1 / 144 | 0 / 1 | 1.0 | 1 / 144 (0.6944444444444444%) | 143 / 144 | 10 | 3 / 3 |
| D multi-view | 1 / 144 | 0 / 1 | 1.0 | 1 / 144 (0.6944444444444444%) | 143 / 144 | 10 | 3 / 3 |
| quality-only diagnostic | 144 / 144 | 37 / 144 | 0.7430555555555556 | 144 / 144 (100%) | 0 / 144 | 10 | 1 / 1 |

| D3 condition for Candidate D | measured | verdict |
|---|---|---|
| fewer false accepts than B | 0 versus 2 | PASS |
| at least half of B's 103 accepts | 1 versus required at least 52 | FAIL |
| no higher false accepts in any source | true | PASS |
| accepts in at least three sources | one source, V14 only | FAIL |

Candidate D's only acceptance was V14 trajectory `s2v13cal_00_seg04`. It had three distinct
hashes/viewpoints and was image-valid and geometrically valid. C and D accepted 0 / 96 trajectories
from opened V16--V18.

Frame-level context across all opened sources was TP 726, FP 21, FN 52, TN 551: precision
0.9718875502008032 and recall 0.9331619537275064. Quality Spearman was 0.8776610063439115 and
MSE 0.0066272829147851365. These pooled opened-data numbers do not supersede any confirmation.

**Verdict: EXPLORATORY_NO_GO**, because Candidate D passes two of four declared conditions and
obtains zero false accepts by abstaining on 143 / 144 trajectories.

### 3.4 Reading it

- Quality-guided ranking removed one false selection relative to A at unchanged 103 / 144 coverage.
- Quality ranking is useful but not uniformly corrective: on V18 it selected one false candidate in
  `s2v18cf_01`, while A selected none at trajectory level.
- Quality-only ranking is unsafe: 37 / 144 selected candidates were image-invalid.
- The existing grid trajectories seldom contain three fresh, nearby, threshold-valid, physically
  separated observations around one candidate.
- Zero false accepts for C/D is uninformative at 1 / 144 coverage.
- No parameter was relaxed after this outcome; the negative result is retained.

### 3.5 Files

| file | contents |
|---|---|
| [result manifest](../../experiments/unity_two_stage_transfer/stage2_temporal_replay_v1_result.json) | jobs, metrics, gates, hashes, and boundaries |
| [result digest](../../experiments/unity_two_stage_transfer/stage2_temporal_replay_v1_result.sha256) | result-manifest SHA-256 |
| external `trajectory_results.jsonl` | every policy decision for every trajectory |
| external `policy_comparison.png` | acceptance, false acceptance, and abstention comparison |
| external `representative_examples.png` | preselected false accept, false reject, and duplicate examples |

## 4. Historical failure and current limits — what is demonstrated versus hypothesized?

The immutable V16--V18 result manifests and the replay in §3 are compared here. No code is run in
this step.

**Criterion.** Historical verdicts and thresholds must remain unchanged; a cause is called
demonstrated only when the recorded data directly support it. Otherwise it remains a hypothesis.

### 4.1 Method

- V16 tested threshold 0.6833333333 and failed recall.
- V17 tested threshold 0.40 and failed precision.
- V18 tested threshold 0.5666666666666667 and narrowly failed frame-level precision.
- The replay tests a different actor contract on the opened data; it cannot overwrite these
  confirmation verdicts.

### 4.2 Setup

| confirmation | immutable decision | relevant measurement |
|---|---|---|
| V16 | `CONFIRMATION_NO_GO` | precision 1.00; recall 0.6824 |
| V17 | `CONFIRMATION_NO_GO` | precision 0.8791; recall 0.9697 |
| V18 | `CONFIRMATION_NO_GO` | precision 0.949685534591195; recall 0.893491124260355 |

No samples were added or dropped; this step reads the three immutable results.

### 4.3 Results

| explanation | evidence status | finding |
|---|---|---|
| threshold calibration instability | DEMONSTRATED | V16 over-conservative, V17 over-permissive, V18 still below its precision gate |
| session dependence | DEMONSTRATED | 7 / 8 V18 frame-level false positives came from `s2v18cf_01` |
| duplicate/repeated evidence | DEMONSTRATED DATA LIMIT | 1,350 frames contain only 584 source-wise exact hashes; 766 frames did not advance producer sequence |
| repeated evidence caused every V18 error | NOT DEMONSTRATED | unique-hash precision was exactly 0.95, but the frozen gate was frame-level |
| spatial corroboration will repair Stage 2 | NOT DEMONSTRATED | the declared rule had only 1 / 144 coverage |
| quality can rank candidates | SUPPORTED EXPLORATORILY | B reduced 3 false accepts to 2 at unchanged coverage |
| quality alone is a safe STOP rule | CONTRADICTED | 37 / 144 quality-only selections were image-invalid |

**Verdict: PASS for preservation and diagnosis**, because all old verdicts remain immutable and
demonstrated facts are separated from hypotheses. This is not a Stage 2 scientific pass.

### 4.4 Reading it

- V18 precision is below 0.95 and is not rounded upward.
- Separate frozen protocols V16--V18 were followed; their failure is not a protocol violation.
- The strategic error was continuing single-frame threshold work after instability became evident.
- Image validity and full 10 mm / 10 degree geometry remain separate targets.
- The opened captures cannot establish return repeatability or fresh confirmation.
- Fixed `s0010` results cannot establish patient-held-out generalization.

### 4.5 Files

| file | contents |
|---|---|
| [V16 result](../../experiments/unity_two_stage_transfer/stage2_confirmation_v16_result.json) | immutable conservative-threshold failure |
| [V17 result](../../experiments/unity_two_stage_transfer/stage2_confirmation_v17_result.json) | immutable permissive-threshold failure |
| [V18 result](../../experiments/unity_two_stage_transfer/stage2_confirmation_v18_result.json) | immutable final single-frame NO_GO |

## 5. Future closed-loop protocol — what evidence is still required?

This step specifies the minimum next development experiment; it does not submit a job or run Isaac
Sim.

**Criterion.** D5 and D6 remain `NOT_RUN` until an approved protocol physically executes candidate
neighborhood acquisition, return-to-best, fresh confirmation, and independent evaluation without
privileged actor leakage.

### 5.1 Method

The smallest defensible next experiment is a development-only acquisition-contract pilot, not
V19 and not confirmation:

1. Start from a Phase 1 L4 handoff or a separately declared engineering start prior.
2. Execute a bounded coarse candidate visit using the existing valid SonoGym action interface.
3. Around each promising candidate, physically acquire three producer-advanced views separated by
   at least the predeclared motion offsets; record rejected/clamped motion.
4. Rank only validity-gated candidates by the unchanged quality regressor.
5. Physically return to the remembered candidate and verify executed return error.
6. Acquire two new same-pose confirmation images with increasing sequences and distinct hashes.
7. Freeze actor output, then join image-validity and geometric evaluator fields.
8. Return `ABSTAIN` on insufficient diversity, invalid motion, stale frames, failed return, or
   inconsistent fresh confirmation.

### 5.2 Setup

| proposed item | bound |
|---|---|
| scientific status | development-only contract feasibility; not independent confirmation |
| anatomy | fixed `s0010` |
| controller | deterministic coarse/fine/micro skeleton using unchanged V15 models |
| resource request | one L4, at most 4 CPUs, at most 30 GB, at most 30 minutes |
| concurrency | one job; zero concurrent project GPU jobs |
| episode bound | four starts; at most 300 action steps each |
| early stop | any stale-frame, transform, leakage, return-verification, or resource failure |

```text
NOT RUN — no sbatch command is authorized by this report.
```

| | value |
|---|---:|
| attempted GPU jobs | 0 |
| completed GPU jobs | 0 |
| **scientific episodes** | **0** |

### 5.3 Results

| gate | result |
|---|---|
| D5 fresh physical confirmation | NOT_RUN |
| D6 full Unity-equivalent Stage 2 | NOT_RUN |

**Verdict: NOT_RUN**, because existing logs contain neither return-to-best execution nor fresh
post-return observations, and this task authorizes no GPU work.

### 5.4 Reading it

- The current replay implementation is ready to supply deterministic actor logic for a reviewed
  engineering pilot.
- The current evidence rule is not ready to freeze as a scientific confirmation policy because its
  observed coverage is 1 / 144.
- The next experiment must first show that the simulator can physically produce the required local
  evidence and fresh reacquisition records.
- A successful development pilot would justify a separate policy-selection review and only then a
  new independently frozen protocol; it would not be named V19.
- The exact next approval is authorization to implement and run one bounded development-only
  physical acquisition-contract pilot under the §5.2 resource limit.

### 5.5 Files

| file | contents |
|---|---|
| [notation](../notation.md) | temporal evidence, distinct viewpoint, and Stage 2 state definitions |
| [current report](sonogym_stage2_temporal_decision_replay.md) | CPU replay evidence and proposed boundary |

## Appendix A. Technical execution record

| defect | effect and disposition |
|---|---|
| job `51515957` required strictly increasing `producer_sequence` | stopped before comparative metrics; records with valid zero-action acquisition can retain sequence; fixed to nondecreasing order while excluding non-advancing records from evidence |
| technical amendment | all scientific parameters, datasets, models, thresholds, gates, and policy variants remained unchanged; protocol re-sealed before job `51518820` |

Job `51518820` completed `0:0` in 21 seconds. Replay itself used 11.98 seconds and 352,400 KiB
maximum resident memory. The run tree contains 29 files and occupies 1.4 MiB. Checksums for
`replay_report.json`, `trajectory_results.jsonl`, and `frame_scores.csv` all verify.
