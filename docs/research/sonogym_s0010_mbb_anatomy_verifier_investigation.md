# s0010 MBB anatomy and L4 verifier investigation

This investigation audits RIGHT L4/L5 SAP source geometry and tests an image-only L4 TP boundary verifier against the frozen V3 failure. It does not define an MBB injection target, modify L5, or rerun the closed-loop actor.

## Result

| Branch | Verdict | Main result |
|---|---|---|
| Anatomy ground truth | **`SAP_CANDIDATES_NEED_REVIEW`** | Whole-vertebra meshes, labels and exact-pose projection exist, but no separate SAP/facet semantics exist and every SAP candidate vertex remains uncertain. |
| L4 verifier | **`VERIFIER_INCONCLUSIVE`** | Calibration precision improved, but recall collapsed on opened development and no unopened session-separated holdout exists. |
| MBB next-phase readiness | **Not ready** | Branch-specific bony surrogate geometry and a robust L4 verifier are both unresolved; no needle/injection simulator should be built from the current evidence. |

The terms and verdict gates are defined once in [project notation](../notation.md).

## A. Executive summary

The geometry-to-ultrasound pipeline is not the SAP blocker. Across 336 V3 frames from four sessions, exact-pose reconstruction remained below the frozen coordinate-error gate, and the unresolved SAP candidates intersected recorded imaging planes in 133 L4 and 135 L5 frames. The blocker is semantic anatomy: `s0010` has complete L4/L5 vertebral meshes and vertebral labels, but no separately named or labelled SAP, facet surface, mammillary process, accessory process, medial-branch nerve, or approved SAP boundary.

The L4 failure is primarily candidate ranking, not search coverage. In failed sessions `mbbpoc_v3_02` and `mbbpoc_v3_03`, the search contained respectively 11 and 9 actor-eligible proxy-positive frames. Session 02 selected a boundary-ambiguous view despite those alternatives; session 03 selected a positive search frame, but two of three later confirmations were ambiguous. The tested verifier reduces false and ambiguous selections but rejects too many positives, so it is not ready for active acquisition.

## B. Anatomy audit — `s0010_sap_geometry_audit_v1`

**Criterion.** Mark SAP geometry ready only if RIGHT L4/L5 SAP surfaces can be reproduced from explicit semantic geometry or a supported, reviewable boundary. Exact scan-plane intersection alone is insufficient.

### Quantities

| Quantity | Result |
|---|---:|
| Separate SAP/facet semantic objects or labels | 0 |
| L4 SAP original / proposed semantic / uncertain vertices | 2,198 / 0 / 2,198 |
| L5 SAP original / proposed semantic / uncertain vertices | 2,422 / 0 / 2,422 |
| Exact-pose sessions / frames | 4 / 336 |
| Maximum coordinate disagreement | `4.0528e-05` volume index |
| Frozen coordinate gate | `0.001` volume index |
| L4 / L5 frames with candidate-plane intersection | 133 / 135 |
| Semantic training labels created | 0 |
| MBB surrogate regions emitted | 0 |

### Method

The audit inventories the actual patient directory, L4/L5 STL files, combined label map, URDF and USD hierarchy. It preserves the existing candidate and uncertainty masks, reconstructs the recorded exact ultrasound planes, counts unique projected pixels rather than duplicate STL vertex entries, and exports privileged review figures separately from clean actor-visible ultrasound.

The anatomy mapping used for future review is branch-specific: the L4-L5 facet is associated with the L3 and L4 medial branches, and lumbar medial branches traverse bony relationships near the SAP/TP region. These references support the taxonomy, not a patient-specific nerve coordinate or needle endpoint: [lumbar medial branch anatomy review](https://pmc.ncbi.nlm.nih.gov/articles/PMC7362874/), [facet-joint intervention review](https://pmc.ncbi.nlm.nih.gov/articles/PMC11372964/), and [cadaveric anatomic study](https://pubmed.ncbi.nlm.nih.gov/26717379/).

### Reproducible setup

```bash
source .sonogym.env
export SONOGYM_CONFIG_FILE="$PWD/.sonogym.env"
sbatch hpg/jobs/s0010_sap_geometry_audit_v1.sbatch
```

- Job: `52053307`; source revision: `097fda3321ea941c4a945a5b034de42af59e3c06`.
- Resources: four CPUs, zero GPUs, 8 GB requested RAM, 15 seconds elapsed, 448,908 KB peak RSS.
- Tests: 3/3 audit tests passed in the job.

### Sample accounting

All 336 acquired V3 frames were retained for transform checking. Candidate-plane intersection was evaluated for both unresolved SAP candidates, producing 672 candidate-frame records. No intersection record was converted into acoustic visibility, semantic identity, a training label, a nerve location, or an injection target.

### Results

**`SAP_CANDIDATES_NEED_REVIEW`.** The coordinate transforms and projection code are sufficient for future annotation generation, but the source assets do not provide a reproducible semantic SAP boundary. The URDF has one combined link, the USD and filenames expose no process/facet object, and segmentation labels 28 and 27 represent the whole L4 and L5 vertebrae. The existing SAP candidates are therefore review geometry only.

The proposed MBB surrogate remains `UNAVAILABLE`. The project can show the expected bony context for the L3 medial-branch-related L4 site and the L4 medial-branch-related L5 site, but it cannot defensibly infer an actual nerve course or needle endpoint from those meshes.

### Reading it

The verified overlays show where a provisional candidate intersects the ultrasound plane. They do not establish that the corresponding cortical interface is acoustically visible or distinguishable as SAP. The most direct remedy is expert review of the exact versioned L4/L5 SAP surface boundary and TP-root relationship, followed by candidate-specific ultrasound observability review using the already verified transforms.

### Files

- Code: `workflows/two_stage_navigation/audit_s0010_sap_geometry.py`.
- Tests: `tests/test_s0010_sap_geometry_audit.py`.
- Small result: `experiments/unity_two_stage_transfer/s0010_sap_geometry_audit_v1_result.json`.
- Runtime artifacts: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_sap_geometry_audit_v1/52053307/audit/`.

## C. L4 verifier investigation — `s0010_l4_boundary_verifier_v1`

**Criterion.** A verifier improvement requires a frozen image-only rule to improve target-view selection on unopened session-separated observations without violating recall, false-positive, abstention, L5-control or information-boundary gates. Calibration and opened V3 diagnostics cannot establish `VERIFIER_IMPROVED`.

### Quantities

| Data | Baseline TP/FP/FN | Verifier TP/FP/FN | Precision | Recall | Selected ambiguous |
|---|---:|---:|---:|---:|---:|
| Calibration, 158 images | 19/4/9 | 18/1/10 | 0.8261 → 0.9474 | 0.6786 → 0.6429 | 10 → 1 |
| Opened development, 151 images | 25/4/6 | 18/0/13 | 0.8621 → 1.0000 | 0.8065 → 0.5806 | 12 → 0 |

The verifier fit used 129 rows from 106 unique training images: 86 `PROXY_POSITIVE`, 36 `PROXY_AMBIGUOUS`, and only 7 clear `PROXY_NEGATIVE` rows that had passed the frozen primary gate. The calibration-selected verifier threshold was 0.65. Geometry labels were used only for offline supervision and evaluation; runtime features are the existing clean-ultrasound patch features.

### Method

The three-state 240-tree ExtraTrees verifier is fit on training sessions only. Calibration selects one threshold from the sealed candidate list; development and V3 are materialized only afterward and are explicitly marked as opened diagnostics. A candidate must pass both the frozen primary L4 threshold and the verifier positive-class rule. L5 is unchanged.

### Reproducible setup

```bash
source .sonogym.env
export SONOGYM_CONFIG_FILE="$PWD/.sonogym.env"
sbatch hpg/jobs/s0010_l4_boundary_verifier_v1.sbatch
```

- Final job: `52054026`; source revision: `5a51de34e8efe4b24fe1b959d05287fd1c30c6db`.
- Resources: four CPUs, zero GPUs, 16 GB requested RAM, 46 seconds elapsed, 330,260 KB peak RSS.
- Tests: 12/12 verifier tests passed in the job; checkpoint save/reload matched exactly.
- Checkpoint SHA-256: `5255574078695b8f3f2bdff03e44ec8889f8652cb28b4eec79feb72b3e31b02a`.

Recovery history is preserved. Job `52050577` exposed missing-proposal handling; jobs `52051440` and `52052620` exposed reporting and replay-gate defects. Only job `52054026` is the final scientific output. A duplicate submission `52051442` was cancelled before allocation and produced no scientific evidence.

### Sample accounting

Session and exact-image-hash overlap are rejected across split roles. One proposal per image and proxy state is retained for fitting. Ambiguous rows remain an explicit class and never become clear negatives. The opened-development and V3 records were not used for fitting, threshold selection, or model choice.

### Results

**`VERIFIER_INCONCLUSIVE`.** Calibration supports the hypothesis that an image-only verifier can suppress ambiguous selections: false positives fell 4→1 and top-1 selection accuracy when a positive was available rose 0.5357→0.6429. However, opened-development recall fell from 0.8065 to 0.5806, and the L4-focused session `s0010tpscv9_dev_l4_00` had zero selected true positives. This is a demonstrated robustness problem, not merely missing confirmation.

The V3 reconstruction localizes two different failure mechanisms:

- `mbbpoc_v3_02`: 12 geometry-positive search frames existed, 11 passed the frozen primary gate, but the actor selected an ambiguous frame. The verifier's hindsight selection is positive, demonstrating a candidate-ranking failure.
- `mbbpoc_v3_03`: 11 geometry-positive frames existed, 9 passed the primary gate, and the actor initially selected a positive frame; two of three fresh confirmations were ambiguous. The verifier's hindsight selection is still ambiguous, so the tested representation does not resolve this session.

None of the four historically selected L4 views passes the combined verifier gate. Direct integration would therefore abstain on 4/4 historical selections. Alternative return-to-best and fresh confirmation were not physically executed, so hindsight selections are not active-acquisition successes. L5 remains the unchanged 4/4 V3 control.

### Reading it

Search coverage was sufficient in both failed L4 sessions. The dominant demonstrated issue is ranking/confirmation reliability under boundary ambiguity. The tested verifier reduces false selection by becoming too conservative; only seven clear-negative primary-accepted training examples and strong session variation limit the present evidence. A new GPU holdout was not justified because the candidate first failed the opened-development robustness check.

### Files

- Code: `workflows/two_stage_navigation/run_s0010_l4_boundary_verifier.py`.
- Protocol: `experiments/unity_two_stage_transfer/s0010_l4_boundary_verifier_v1.yaml`.
- Tests: `tests/test_s0010_l4_boundary_verifier.py`.
- Small result: `experiments/unity_two_stage_transfer/s0010_l4_boundary_verifier_v1_result.json`.
- Runtime artifacts: `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_l4_boundary_verifier_v1/52054026/verifier/`.

## D. Scientific assessment

Supported claims:

- The native exact-pose mapping is accurate enough to project versioned geometry into stored V3 ultrasound frames.
- The simulator has whole L4/L5 anatomy but no explicit semantic SAP source.
- V3 L4 failures were not caused by a lack of proxy-positive search observations.
- The present verifier suppresses false and ambiguous selections at the cost of unacceptable opened-development recall.
- Runtime inference remains image-only; no privileged geometry field was added to the actor.

Unsupported claims:

- That the blue SAP candidates are correct SAP ground truth or acoustically identifiable.
- That TP/SAP geometry identifies an actual medial-branch nerve or injection endpoint.
- That the verifier improves independent-session active acquisition.
- That a simulated MBB intervention can now be evaluated.

## E. Next-phase readiness

The project is **not ready** to implement branch-specific MBB surrogate targets or a dedicated needle/injection simulator. The minimum anatomy gate is an independently reviewed, versioned RIGHT L4/L5 SAP surface definition plus an explicit bony-surrogate construction for the L3- and L4-medial-branch-related sites. This review may approve, revise, or reject the current candidates; it must not infer nerve coordinates from vertebral centroids.

The L4 verifier must remain outside active acquisition. Its next defensible experiment is not a full MBB rerun: first add session-diverse clear-negative and boundary-ambiguous training evidence, freeze a rule that does not reproduce the opened-development recall collapse, and only then evaluate it once on newly seeded, unopened Stage-2-only sessions with physical return and fresh confirmation. No further GPU job was launched in this investigation.

## F. Reproducibility

Final integrated regression command:

```bash
source .sonogym.env
export PYTHONNOUSERSITE=1
export PYTHONPATH="$SONOGYM_RUNTIME_DIR/envs/stage1-python:$PWD/source/spinal_surgery/spinal_surgery/evaluation"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m unittest -v \
  tests.test_s0010_sap_geometry_audit \
  tests.test_ultrasound_plane_geometry \
  tests.test_anatomical_ground_truth \
  tests.test_s0010_right_landmark_correction \
  tests.test_stage2_mbb_landmarks \
  tests.test_stage2_tp_active \
  tests.test_s0010_l4_boundary_verifier
```

Result: 70/70 tests passed. Historical V3, V16, V17 and V18 verdicts and artifacts were not modified. No GPU, Isaac Sim, new ultrasound capture, full MBB rerun, needle action or injection action was executed.
