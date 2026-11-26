# Final Stage 2 test of anatomically constrained rigid registration

## 1. Protocol and provenance — Which observations and which shape enter each arm?

`workflows.mbb_final_registration.prepare` reserves independent geometric cases using the existing SonoGym local assets. `workflows.mbb_final_registration.predict` registers the fixed mean and seals every proposal.

**Criterion.** Keep the previous population mean, raw reconstruction, calibration and U-Net inputs fixed. Evaluate exactly A0, A1, A2 and A3. Reserve new external cases before opening the development measurements.

### 1.1 Method

- A0 calls the earlier rigid mean code with no shape modes and the original confidence and per-view weighting. Historical vertex correspondence, starts and nominal admission rule are preserved. Its subset runs are saved as diagnostics without changing the baseline admission.
- A1 uses TP and SAP with anatomically aligned starts. A3 uses the same TP/SAP inputs, point-to-triangle optimizer, semantic compatibility and confidence weights with the previous initialization policy. A2 uses A1 starts and admits compatible OTHER_BONE with a fixed class share selected on development. These three fits all optimize a proper rotation and translation.
- Retain the strongest confidence per anatomical class, 1 mm voxel and acquisition group. Reject isolated or competing anchor groups and implausible separation. Anchor centers are medians of per-view medians. Derive a local plane normal only when nearby points support it. Mean-shape compatibility regions come from the previous training-side positional regions. These regions represent anatomical approximations and have no subject-specific coordinates.
- Align the measured TP–SAP direction with the mean direction. A public-orientation candidate, rotations around the aligned direction and an optional surface-normal candidate preserve orientation alternatives. Align the midpoint of the two anchor pairs by translation. The two centers leave rotation about their connecting axis underdetermined.
- Fit by robust closest-triangle correspondences within compatible mean regions. Weight the two anchors equally after normalizing confidence by view within each class. A2 normalizes OTHER_BONE independently and admits it only near the observed anchors and a compatible model region. A0 retains its historical nearest-vertex fitting.
- A1–A3 enforce the 5 mm observed P95 rule for all included observations and both anchor classes. They reject solutions with greater than 2 mm hidden-posterior disagreement among similar-objective alternatives or after the predefined view-subset refits. The periodic refit may be skipped when both anchors lose support. The strongest-view deletion is mandatory for admission. These perturbations measure sensitivity.
- Reopen the six historical external broad-mask scans only for development. They are geometric simulations from partial CT surfaces. Their archived inputs have only OTHER_BONE labels, so A1–A3 derive left TP/SAP proxies from observed coordinate bands alone. A0 consumes the original input. Attribution to initialization on synthetic geometry uses A1 versus A3. The full comparison is repeated with the pre-existing TP/SAP/OTHER_BONE labels on s0010.
- Reserve previously unscored local TotalSegmentator-derived L5 masks after the unchanged earlier geometry QC. The prior receipt excludes the nine candidate source IDs from every prior stage. The source capture manifests show no recorded Stage 2 use for these candidates. The morphology screen checks similar-extents candidates for a near-identical match with the earlier cohort. Source IDs and morphology establish the available separation. The release has no cross-visit patient linkage key.

### 1.2 Setup

| Setting | Value |
|---|---|
| Base revision | 24ca351a9207d31c8bb91cc4feb5c206a749b2c6 |
| Mean source SHA-256 | c5a9a568960cbc4b48154e7ef344c37ec2244ec974aecf8fb40cf78a6de70f6b |
| Protocol | experiments/unity_two_stage_transfer/mbb_final_registration_v1.json and its SHA-256 |
| Frame | Public Stage 1 basis and robot-base millimetres on s0010; fixed RAS orientation in the geometric benchmark |
| s0010 | Four distinct 102-waypoint scans and a 126-waypoint supplement of 2026121700 |
| Gate selection | All-case hidden coverage within 1.5 mm, then lower forward P95; exact ties favor A1 |
| Reserve job | 52840262 |
| Development job | 52841048 |
| Final job | 52841056 |

Run from the isolated worktree with access to the archived model and the local SonoGym assets. Use a new study artifact directory to reproduce.

```bash
project=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym
runtime="$project-runtime"
study="$project/artifacts/mbb-final-anatomical-registration-014812"
previous="$project/artifacts/mbb-statistical-shape-model-003239"
shape_previous="$project/artifacts/mbb-ultrasound-shape-completion-231044"
cd "$study/checkout"
protocol=experiments/unity_two_stage_transfer/mbb_final_registration_v1.json
prep=$(sbatch --parsable --cpus-per-task=2 --time=00:15:00 --export=ALL,VTK_SMP_MAX_THREADS=2 \
  --output="$study/reserve-%j.log" hpg/jobs/s0010_mbb_observed_surface.sbatch "$runtime" \
  -m workflows.mbb_final_registration.prepare --previous "$previous" --runtime "$runtime" \
  --output "$study" --protocol "$protocol")
dev=$(sbatch --parsable --dependency="afterok:$prep" --kill-on-invalid-dep=yes \
  --output="$study/development-%j.log" hpg/jobs/mbb_final_registration.sbatch \
  "$runtime" "$study" development "$previous" "$shape_previous")
sbatch --dependency="afterok:$dev" --kill-on-invalid-dep=yes \
  --output="$study/final-%j.log" hpg/jobs/mbb_final_registration.sbatch \
  "$runtime" "$study" final "$previous" "$shape_previous"
```

### 1.3 Results

| Cohort | Accounting |
|---|---|
| External development | Six previously opened broad-mask cases, with byte-identical inputs and references |
| Local external candidates | Nine candidates reserved before tuning |
| Excluded by prior geometry QC | Three. Two outside physical volume limits and one outside complete L5 extents |
| Final external set | Six source IDs, with no near-identical prior-cohort geometry in the predeclared screen |
| s0010 transfer | Ten acquisition–perception inputs from the prior campaign, five Oracle and five U-Net |

On the six previously opened geometric cases, A0 reproduced every historical nominal pose,
P95 residual and disposition. The frozen selection rule chose A2 with OTHER_BONE weight
0.30. At 1.5 mm its all-case hidden coverage was 10.30%, versus 8.74% for A1, 8.25% for
the smaller A2 weight, 1.22% for A3, and 9.65% for the replayed A0.
A2 at 0.30 returned one fit and A1 returned two. The paired development result fixed all
remaining settings before either final evaluator read its reference.

### 1.4 Interpretation

- Both final prediction sets are sealed before their reference geometry is opened by the separate scoring process.
- The geometric benchmark evaluates shape reconstruction under synthetic occlusion. The s0010 cohort evaluates actual recorded ultrasound on a repeatedly studied anatomy.

### 1.5 Files

| File | Contents |
|---|---|
| external_reservation.json | Mask QC, historical-overlap checks and sealed final external input and reference manifests |
| freeze.json | Selected arm, A2 weight and development, protocol and model hashes |
| external-confirmation/predictions/seals.json | Final external actor seal |
| transfer-predictions/seals.json | Final ultrasound actor seal |

## 2. Evaluation — Does better fitting yield accurate and stable hidden geometry?

**Criterion.** Apply G1–G6 at their frozen thresholds. A1 versus A3 isolates initialization. A2 versus A1 tests the incremental inclusion of compatible OTHER_BONE. A0 reproduces the historical nominal result. The MBB reference status remains `TARGET_LOCALIZATION_NOT_EVALUABLE`.

### 2.1 Method

- Retain observed coverage for abstentions. For external evaluation, compare both finite surfaces within the right posterior mask, with 1.0, 1.5 and 2.0 mm coverage tolerances. Report each directed P95 and the maximum. Evaluate the selected arm against A0 with pooled all-case 1.5 mm coverage and forward P95.
- For s0010, use the frozen 9.5 mm ball around the archived diagnostic cortical-path centroid. Clip the CT and predicted faces separately to this neighborhood. Use true CT face quads from the local segmentation to define the reverse reference patch. This scoring differs from the previous campaign's reverse distance to all L5 faces. The diagnostic-path points remain evaluator-only.
- Compute the conservative bidirectional P95 per accepted case, then take the median across cases. Leave unavailable distances explicit when no fit is admitted. The consistency gate uses different 102-waypoint recordings. Report the shared 102-frame prefix pair separately and preserve comparisons with the 126-frame supplement in the measured output.
- An accepted fit contributes to pooled coverage. Store explanatory geometry and distances from rejected proposals separately. Keep observed points, inferred geometry and unresolved proposals distinct. Historical A0 geometry, status and observed P95 are checked against saved proposals, separately from updated local evaluation.
- G6 is verified by the frozen mean SHA-256, original input receipts, predictor dependencies, and sealed outputs generated before either evaluator opens the grading geometry or native transform.

### 2.2 Setup

The preceding invocations generate both the prediction and evaluation artifacts. Generate the four figures from sealed results using the SonoGym Python environment.

```bash
export PYTHONPATH="$PWD:$PWD/source/spinal_surgery/spinal_surgery/evaluation:$PWD/workflows/two_stage_navigation:$PWD/workflows/perception_s0010:$runtime/envs/stage1-python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VTK_SMP_MAX_THREADS=1 MPLBACKEND=Agg
"$runtime/envs/sonogym/bin/python" -m unittest discover -s tests -p 'test_mbb*.py'
"$runtime/envs/sonogym/bin/python" -m workflows.mbb_final_registration.review \
  --study "$study" --output "$study/figures"
```

### 2.3 Results

Pending sealed evaluation.

### 2.4 Interpretation

Pending sealed evaluation.

### 2.5 Files

| File | Contents |
|---|---|
| development/results.json | Model selection and exact historical A0 replay check |
| external-confirmation/results.json | Untouched external geometry comparison and every abstention |
| transfer/results.json | All s0010 scores, pairwise consistency and G1–G6 |
| transfer/per-session.csv | Forty arm-session rows with observed and evaluator scores |
| transfer/clouds | Separate observed, inferred, unresolved and evaluator-only PLY layers |
| figures | Four diagnostics for initialization, comparison, local error and ambiguity |
