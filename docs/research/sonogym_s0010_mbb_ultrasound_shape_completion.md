# Ultrasound-only local cortical shape completion on s0010

This study measures recovery of deliberately withheld cortical interfaces and replays the same completion methods on the missing right L5 SAP–TP transition. The right L4 medial branch motivates the region of interest. The intended bone reference remains anatomically unresolved.

## 1. Methods and holdout — Can nearby ultrasound observations recover a known missing surface?

The frozen protocol is `experiments/unity_two_stage_transfer/s0010_mbb_shape_completion_v1.json`. The inputs are previously saved cortical clouds and public executed-pose ledgers. The output keeps observed points, inferred samples and unresolved diagnostic proposals in separate layers.

**Criterion.** Evaluate recovery within 1.5 mm of withheld, ultrasound-supported Oracle points and report outliers beyond 3 mm. The tolerances describe reconstruction on s0010. Compute distances in both directions. Keep abstentions in the coverage denominator. Require each synthetic negative control to abstain.

### 1.1 Method

- The curator chooses one site per session and class for right L5 TP, right L5 SAP and L5-routed other bone. Sites lie on a public Stage 1-relative 1.5 mm grid. Eligible sites have enough withheld samples at all three gap sizes, with the 9 mm hole surrounded by at least three distinct viewpoints in at least four of eight angular sectors. Rank eligible centers by exterior support. The 12 sites are selected favorable observation neighborhoods. Masks from the same site are nested.
- Remove each projected 2, 5 and 9 mm disk from all input depths, frames and semantic classes. A separate process receives the remaining points and public two-dimensional mask. Both arms use the same masks and withheld Oracle evaluation points. Secondary metrics compare each arm with its own withheld samples.
- Within a 10 mm radius, fit a local height field to the matching semantic class. Each view supplies a median per spatial cell, then the view medians receive equal weight. Reject competing layers, steep or inconsistent height fields, and inadequate view or spatial support. Both fits share a convex-hull constraint, 4.75 mm maximum planar support distance and local slope bound. Evaluate height on a 0.5 mm grid. Require a supported disk center and valid nominal samples on at least 70% of the grid.
- Compare Delaunay piecewise-linear interpolation and a regularized thin-plate radial basis fit. The regularized arm uses a 4 mm coordinate scale and smoothing 0.10. The no-completion arm retains the original observations outside the disk.
- Stress each fit by omitting the three most sampled viewpoint groups individually, scaling the neighborhood radius by 0.85 and 1.15, rotating the chart by ±4°, and applying view-coherent 1 mm translation and 0.5° rotation stresses. The regularized fit also varies smoothing by factors of 0.25 and 4. Require all three view removals to retain support, at least five finite variants on at least 80% of valid nominal samples, and a maximum ensemble range of 3 mm.
- Open withheld data and full L5 CT faces after verifying actor prediction and input hashes. Compare inferred points to exposed L5 CT voxel faces when at least 80% of withheld Oracle points lie within 1.5 mm of those faces. The CT label volume has 1.5 mm spacing. Per-case JSON includes directed means, medians, P95s and outlier counts for the applicable comparisons.

### 1.2 Setup

| Setting | Value |
|---|---|
| Anatomy and Stage 1 | s0010, existing actor Stage 1 cf7_00 |
| Data | Four existing 102-frame acquisitions, seeds 2026121700, 2026121702, 2026121900, 2026121901 |
| Perception | Cached Oracle-visible interfaces, cached original L4 and T4 L5 U-Net outputs |
| Checkpoints | Original L4 SHA-256 c6b7ecc9bc245a88ccc0a5db31dbfd919b57407988dde5ea8211dd7eea2c1b2a, T4 L5 SHA-256 1efc323e9c379bcf657eb59fb793cfa30b98e618033a3abc1966a4f3cc69bf47 |
| Calibration | Public executed robot-base poses and calibrated projection, 0.5 mm image pixels |
| Input base revision | 71978a20c98de8a1a108279926dcdafeeab573ee |
| Benchmark jobs | 52815369 predicts, 52815734 evaluates. Both request 2 CPUs, 8 GiB and 0 GPUs. |
| Prior study | artifacts/mbb-end-to-end-210237, immutable reconstruction and reference receipts |
| Study worktree | artifacts/mbb-ultrasound-shape-completion-231044/checkout |

Run from the experimental worktree. Use a new output folder for any reproduction.

```bash
project=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym
runtime=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime
study="$project/artifacts/mbb-ultrasound-shape-completion-231044"
cd "$study/checkout"
job="$PWD/hpg/jobs/s0010_mbb_shape_completion.sbatch"
sbatch --output="$study/reproduce-A-%j.log" "$job" "$runtime" "$study/reproduce-A" A all
```

| Accounting | Count |
|---|---:|
| Site strata and selected sites | 12/12 |
| Mask conditions | 36 per perception arm |
| Withheld Oracle points for the 2, 5 and 9 mm masks | 81, 348, 1,001 per arm |
| Benchmark requests including synthetic controls | 77 |
| CT correspondence eligible | 72/72 arm-mask cases |

### 1.3 Results

Distances are withheld Oracle sample to estimate, median and P95 in millimetres, conditional on returning a fit. Observed-only rows use the surviving local points in all cases. Coverage counts all sites and uses the union of observation and accepted inference. Abstention is per method and includes rejected partial fits.

| Perception | Gap (mm) | Method | Median | P95 | Coverage within 1.5 mm | Abstained |
|---|---:|---|---:|---:|---:|---:|
| Oracle-visible | 2 | Observed only | 0.71 | 1.32 | 80/81 (98.8%) | N/A |
| Oracle-visible | 2 | Linear | 0.37 | 0.85 | 81/81 (100%) | 4/12 |
| Oracle-visible | 2 | Regularized | 0.37 | 0.92 | 81/81 (100%) | 4/12 |
| Oracle-visible | 5 | Observed only | 1.52 | 2.59 | 169/348 (48.6%) | N/A |
| Oracle-visible | 5 | Linear | 0.28 | 0.73 | 222/348 (63.8%) | 7/12 |
| Oracle-visible | 5 | Regularized | 0.38 | 1.13 | 242/348 (69.5%) | 5/12 |
| Oracle-visible | 9 | Observed only | 2.08 | 4.29 | 298/1,001 (29.8%) | N/A |
| Oracle-visible | 9 | Linear | 0.33 | 1.46 | 496/1,001 (49.6%) | 7/12 |
| Oracle-visible | 9 | Regularized | 0.51 | 1.63 | 402/1,001 (40.2%) | 9/12 |
| U-Net | 2 | Observed only | 1.12 | 5.60 | 47/81 (58.0%) | N/A |
| U-Net | 2 | Linear | 0.57 | 0.93 | 49/81 (60.5%) | 6/12 |
| U-Net | 2 | Regularized | 0.65 | 1.00 | 49/81 (60.5%) | 6/12 |
| U-Net | 5 | Observed only | 2.13 | 6.28 | 104/348 (29.9%) | N/A |
| U-Net | 5 | Linear | 0.44 | 0.90 | 136/348 (39.1%) | 9/12 |
| U-Net | 5 | Regularized | 0.51 | 1.05 | 145/348 (41.7%) | 8/12 |
| U-Net | 9 | Observed only | 2.79 | 8.16 | 231/1,001 (23.1%) | N/A |
| U-Net | 9 | Linear | 0.52 | 1.28 | 272/1,001 (27.2%) | 11/12 |
| U-Net | 9 | Regularized | 0.41 | 0.99 | 273/1,001 (27.3%) | 11/12 |

Both completion methods abstained on all five negative controls. At 5 mm, the inferred-to-Oracle median and P95 of the regularized fit were 0.88 and 2.42 mm for Oracle input and 0.81 and 1.70 mm for learned input. The corresponding inferred-to-CT voxel-face distances, at 1.5 mm CT label spacing, were 0.60 and 1.11 mm with Oracle input and 0.40 and 0.89 mm with learned input, conditional on different subsets of accepted patches. These conditional CT values do not rank the two perceptions.

### 1.4 Interpretation

- Regularization recovered the most withheld samples at 5 mm in both arms. At 9 mm, linear interpolation recovered more samples with Oracle input. Both learned fits accepted one of 12 sites.
- The Oracle linear 9 mm fit had a 4.02 mm reverse P95 and 1.46 mm forward P95. Sampling gaps and reconstruction error can both increase the reverse distance.
- At 5 mm the regularized stress ranges enclosed the heights of 87/152 matched Oracle-input points and 56/91 U-Net-input points. At 9 mm these counts were 66/153 and 17/62. The ranges have insufficient empirical coverage for use as confidence intervals.
- Four sites in each arm repeatedly lacked a supported height field at all sizes. The maximum ensemble disagreement reached 4.83 mm with Oracle input, among rejected fits.

### 1.5 Files

| File | Contents |
|---|---|
| benchmark/benchmark_results.json | Exact counts and per-case forward, reverse, CT and uncertainty metrics |
| benchmark/curator_receipt.json | All-depth masked point counts and held-out hashes |
| benchmark/inputs and benchmark/predictions | Observations, nominal fits, alternatives and seals |
| figures/benchmark-summary.png | Coverage, conditional error and abstentions |
| figures/controlled-patches-{oracle,learned}.png | Observed, withheld, inferred, alternative and CT layers |


## 2. Actual SAP–TP gap — Do the same fits recover the missing diagnostic cortex?

Use the frozen methods from Section 1. Derive a local chart from nearby observed, right-side L5 TP and SAP flanks. Additional L5-routed bone may support the chart. The inferred grid lies between the flanks. Candidate query locations must be at least 0.9 mm in the chart from the observed points. The predictor uses the mean of the two flank normals to establish the depth axis and evaluates alternative 3 and 5 mm flank windows around the nominal 4 mm window. The same surface support and uncertainty gates apply. An ancillary observed-only quadratic fit reports supported flank normals and curvature without changing decisions.

**Criterion.** Report whether a completion survives the frozen geometric and view-removal tests. Measure whether the union of observations and accepted inference increases coverage of the seven evaluator-only diagnostic CT trace samples within 1.5 mm. Evaluate accepted patches against all exposed L5 faces and compare patches across sessions in evaluator-only CT coordinates. The original CT trace probes a plausible intervening cortical region. Its geometry supplies no anatomical target acceptance label.

### 2.1 Method

- Replay the four 102-frame scans and the previously captured 126-frame scan. The 126-frame recording shares its first 102 image hashes and executed poses with 2026121700. Keep the 24 oblique acquisitions as the sole paired increment.
- Choose the closest laterally ordered L5 semantic flanks within a 2–12 mm distance range. Require supported local planes with at least two view groups each and no more than 82° of disagreement in the oriented normals. Keep the selected flank frame numbers and the local frame support.
- Limit other-bone evidence to L5-routed samples near the supported semantic flanks. Preserve the original observed interfaces, including excluded uncertain points, in the output cloud. Retain nominal samples from rejected fits separately when available.
- Verify actor seals before the scorer reads the independently archived 1.5 mm CT segmentation, the diagnostic face centers or native simulator poses. Use the latter only to express evaluator geometry in each session's robot-base frame.
- Measure the diagnostic trace and broader local cortex separately. The broader denominator comprises the 271 CT face centers within 9.5 mm of the trace centroid. Coverage on that set is geometric and does not establish anatomic identity. Cross-session distances use the CT coordinate alignment after prediction sealing.

### 2.2 Setup

| Setting | Value |
|---|---|
| Replay job | 52815764, 2 CPUs requested, 0 GPUs |
| Existing supplemented acquisition | 52801423, development seed 2026121700, 102+24 valid images |
| Prediction and evaluation separation | Actor request receipts, per-case prediction seals, post-seal evaluator receipt |
| Reserved seeds | 2026121800–2026121803, unused |
| Benchmarks and tests | 34 mbb-focused unittest cases passed, including all-depth holdout, repeated-view abstention, transformed anisotropic CT face distances and prediction-seal checks |

```bash
a=$(sbatch --parsable --output="$study/reproduce-A-%j.log" \
  "$job" "$runtime" "$study/reproduce-A" A all)
sbatch --dependency="afterok:$a" --kill-on-invalid-dep=yes \
  --output="$study/reproduce-B-%j.log" "$job" "$runtime" "$study/reproduce-B" B all

cd "$study/checkout"
export PYTHONPATH="$PWD:$PWD/source/spinal_surgery/spinal_surgery/evaluation:$PWD/workflows/two_stage_navigation:$PWD/workflows/perception_s0010:$runtime/envs/stage1-python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
protocol=experiments/unity_two_stage_transfer/s0010_mbb_shape_completion_v1.json
"$runtime/envs/sonogym/bin/python" -m workflows.mbb_shape_completion.score \
  --protocol "$protocol" --phase B --output "$study/reproduce-B" --compare-only
"$runtime/envs/sonogym/bin/python" -m workflows.mbb_shape_completion.review \
  --protocol "$protocol" --benchmark "$study/reproduce-A" --actual "$study/reproduce-B" \
  --output "$study/reproduce-figures"
"$runtime/envs/sonogym/bin/python" -m unittest discover -s tests -p 'test_mbb*.py'
```

All ten arm-session pairs were scored. Across phases A and B, the Slurm jobs completed in 1m55s, 14s and 20s respectively, using 54.256 CPU seconds in total. The analysis used at most two Slurm CPUs and zero GPUs at any one time. No new ultrasound was acquired or generated.

### 2.3 Results

The diagnostic trace columns show observed then completed samples within 1.5 mm. Completion counts and error use the regularized fit unless another method is specified.

| Recording | Input | Trace observed → completed | Broader local cortex observed → completed | Completion and nearest full L5 cortex error | Target |
|---|---|---:|---:|---|---|
| 2026121700, 102 | Oracle | 1/7 → 1/7 | 48/271 → 48/271 | Height field P90 residual 3.27 mm exceeds 3 mm | N/E |
| 2026121700, 102 | U-Net | 0/7 → 0/7 | 36/271 → 48/271 | 68 points, median/P95 0.83/1.31 mm, range 2.35 mm. Linear abstained at 3.09 mm range. | N/E |
| 2026121702, 102 | Oracle | 0/7 → 0/7 | 43/271 → 43/271 | Flank normal separation 107.45° | N/E |
| 2026121702, 102 | U-Net | 0/7 → 0/7 | 41/271 → 64/271 | 108 points, 0.77/1.20 mm, range 1.74 mm. Linear returned 109 points, 0.87/1.41 mm, range 1.57 mm and 58/271 local coverage. | N/E |
| 2026121900, 102 | Oracle | 0/7 → 0/7 | 53/271 → 53/271 | Flank orientation unsupported | N/E |
| 2026121900, 102 | U-Net | 0/7 → 0/7 | 46/271 → 46/271 | Only 59/112 nominal grid samples supported, below 70% | N/E |
| 2026121901, 102 | Oracle | 0/7 → 0/7 | 46/271 → 46/271 | Flank orientation unsupported | N/E |
| 2026121901, 102 | U-Net | 0/7 → 0/7 | 31/271 → 31/271 | Flank orientation unsupported | N/E |
| 2026121700, 126 | Oracle | 1/7 → 1/7 | 59/271 → 59/271 | Flank normal separation 90.95° | N/E |
| 2026121700, 126 | U-Net | 0/7 → 0/7 | 48/271 → 48/271 | Competing depth layers in 18/81 cells, above 15% | N/E |

The two admitted regularized U-Net patches use 17 and 18 independent viewpoint groups and span 11.57 and 12.32 mm in maximum 3D point separation. Their centers lie 0.86 mm apart after evaluator alignment, and their bidirectional P95 distances are 1.44 and 2.32 mm. Their closest samples still lie 3.64 and 3.03 mm from the diagnostic trace. The linear patch in 2026121702 is at least 3.43 mm away. None improves the diagnostic coverage.

### 2.4 Interpretation

- The learned fits recover parts of a nearby L5 surface under the tested assumptions. Their close CT distances and alignment across two acquisitions do not establish recovery of the missing SAP–TP transition.
- With Oracle-visible inputs, two recordings exceeded the 82° separation bound, two lacked supported flank orientations, and one exceeded the height-field residual bound. These failures limit the proposed representation independently of segmentation.
- In the matched comparison, 24 extra oblique frames raised observed local cortex coverage from 48/271 to 59/271 with Oracle and from 36/271 to 48/271 with U-Net. Diagnostic trace coverage stayed unchanged. The learned fit was rejected after the extra views exposed conflicting layers in its chart.

### 2.5 Files

| File | Contents |
|---|---|
| actual/actual_gap_results.json and .csv | All ten arm-session outcomes, CT distances, view supports and rejection stages |
| actual/actual_accepted_surface_comparison.json | Post-seal cross-session fit comparison and proximity to the diagnostic trace |
| actual/clouds | Complete measured clouds, separate inferred and unresolved samples, evaluator CT and trace PLY files |
| figures/actual-gap.png | Same-camera view of all five recordings and both perception arms |
| figures/observed_flank_geometry.json | Observed-only orientation and curvature diagnostics |
| figures/paired_acquisition_check.json | Public-pose and image-hash parity for the 102-frame prefix |

## 3. Target identifiability — Does the completed shape constrain the provisional reference?

**Criterion.** An independently defensible anatomical bone-reference region must exist before a target can be scored. Preserve the archived alternatives and abstain from defining a reference from fitted surfaces.

### 3.1 Method

Inspect the sealed anatomical reference audit and carry its status forward. The audit preserves competing cortical hypotheses on the L5 TP-root and lateral SAP-neck sides. The measured trace crosses a lamina-labelled region. This campaign scores geometric recovery of that trace separately from potential target localization.

### 3.2 Setup

The sealed reference receipt is `reference/checksums.sha256` from the prior end-to-end study. Its SHA-256 is d5400e53660e29ac373492f25ff56169c2a9725ed05b0571d171a729ad5987a2. The reference file has status `NOT_EVALUABLE` and no accepted region. All CT access occurs on the evaluator side after actor sealing.

### 3.3 Results

**TARGET_LOCALIZATION_NOT_EVALUABLE.** No candidate target coordinate or area was emitted. The archive's alternatives remain 5.30 mm apart at the nearest centers. The nearby admitted fits do not recover the intervening 7.68 mm diagnostic cortex trace. Anatomical identifiability is unestablished on these data.

### 3.4 Interpretation

- The generic shape models are useful for some well-observed neighborhoods. Their transfer to the tested SAP–TP transition is unestablished.
- Generic interpolation failing on this transition does not establish that a learned prior is necessary, or that it would be sufficient. An anatomical reference review and a test of a frozen multi-anatomy prior on anatomies excluded from training would address different uncertainties. The inference input for such a test would remain ultrasound-derived observations.

### 3.5 Files

| File | Contents |
|---|---|
| reference/reference.json | Anatomical identity, alternative hypotheses and unresolved status |
| reference/external_cortex_review.png | Existing cross-sectional anatomy and separately marked cortical trace |
| s0010-mbb-ultrasound-shape-completion-review.tar.gz | Portable review bundle with file hashes, predictions, code patch and run provenance |
