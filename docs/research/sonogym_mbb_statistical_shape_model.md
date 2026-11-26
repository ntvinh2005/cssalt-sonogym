# L5 statistical shape completion from partial observations

The population model is learned from external TotalSegmentator L5 masks. The s0010 transfer fit receives the archived ultrasound-derived interface points and public pose information.

## 1. Population and correspondence — Do the training surfaces support a common template?

`workflows.mbb_ssm.acquire` extracts individual L5 masks from the official archive. `workflows.mbb_ssm.prepare` checks geometry and partitions the eligible IDs. `workflows.mbb_ssm.train` builds the registered population.

**Criterion.** A mask must contain a connected, complete and watertight L5. Correspondences must meet the frozen bidirectional global and posterior distance, edge-distortion and face-orientation gates. At least eight training meshes must pass registration. The inference model uses physical millimetres and preserves anatomical size variation.

### 1.1 Method

- Verify HTTP range and ZIP entry integrity, then hash every training mask. The first retrieval uses a fixed hash order from the source index. The second uses a fixed hash order within abdominal–pelvic, whole-body and related scan categories after measuring mask availability. The supplement excludes all earlier downloads.
- Group by the release's pseudonymous `image_id`. The release metadata does not contain a cross-visit linkage key. Preserve the original partitions when adding the supplemental masks. Reject exact duplicate shapes on the common physical grid. Exclude all 12 SonoGym asset IDs, including s0010. Perform an independent near-identical geometry screen against the final training meshes after sealing the s0010 fits.
- Extract the largest connected external surface from each accepted mask on a 1.25 mm RAS grid. Use rigid alignment without scaling. Choose the training template closest to the training median extents, then use normal-compatible surface correspondences and Laplacian-regularized displacement on the mesh graph. Reverse geometric correspondence contributes a bounded update. PCA uses thin SVD of the training-only aligned vertex vectors.
- Benady et al. used rigid and nonrigid correspondence followed by PCA and Gaussian-process reconstruction in a CT geometry benchmark. Here the statistical fit uses PCA and a quadratic coefficient penalty. The method comparison uses the author abstract and the repository record. The linked manuscript PDF returned HTTP 403 during retrieval. [Publication and author abstract](https://cris.tau.ac.il/en/publications/statistical-shape-modeling-and-gaussian-process-based-reconstruct/) · [DOI](https://doi.org/10.1007/s11548-026-03757-2)

### 1.2 Setup

| Setting | Value |
|---|---|
| Data source | Wasserthal et al., TotalSegmentator CT v2.0.1, doi:10.5281/zenodo.10047292, CC BY 4.0 |
| Scope | Binary L5 masks only. The prior uses neither full patient images nor simulator CT. |
| Data version verification | [Official Zenodo record and license](https://zenodo.org/records/10047292/export/json) |
| Worktree base | 846e1d1c0fbc58dc84724e3cd4c1191c01e721e8 |
| Protocol | experiments/unity_two_stage_transfer/mbb_ssm_v1.json and SHA-256 |
| runtime | /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/envs/sonogym |
| jobs | 52833413 initial mask QC, 52833617 correspondence preflight, 52834740 expanded mask QC, 52834756 expanded training preflight, 52836793 guarded correspondence |

Use the experimental worktree and a new artifact root when reproducing the experiment. The official archive source and candidate selection salt are arguments in the acquisition command.

```bash
project=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym
runtime="$project-runtime"
study="$project/artifacts/mbb-statistical-shape-model-003239"
cd "$study/checkout"
protocol=experiments/unity_two_stage_transfer/mbb_ssm_v1.json
python_exe="$runtime/envs/sonogym/bin/python"
wrapper=hpg/jobs/s0010_mbb_observed_surface.sbatch
find "$runtime/assets/data/HumanModels/selected_dataset" -mindepth 1 -maxdepth 1 \
  -type d -printf '%f\n' | sort > "$study/local-subject-ids.txt"
"$python_exe" -m workflows.mbb_ssm.acquire --output "$study/external-ct" \
  --exclude "$study/local-subject-ids.txt" --count 56 --salt ssml5-pilot-v1
"$python_exe" -m workflows.mbb_ssm.acquire --output "$study/external-ct-expanded" \
  --exclude "$study/local-subject-ids.txt" --count 38 --salt ssml5-pilot-v1-body-supplement \
  --initial-manifest "$study/external-ct/acquisition.json" --body-coverage
first=$(sbatch --parsable --cpus-per-task=2 --time=00:20:00 \
  --export=ALL,VTK_SMP_MAX_THREADS=2 --output="$study/prepare-%j.log" "$wrapper" "$runtime" \
  -m workflows.mbb_ssm.prepare --protocol "$protocol" --manifest "$study/external-ct/acquisition.json" \
  --output "$study/cohort")
expanded=$(sbatch --parsable --dependency="afterok:$first" --cpus-per-task=2 --time=00:25:00 \
  --export=ALL,VTK_SMP_MAX_THREADS=2 --output="$study/expanded-prepare-%j.log" "$wrapper" "$runtime" \
  -m workflows.mbb_ssm.prepare --protocol "$protocol" --manifest "$study/external-ct-expanded/acquisition.json" \
  --output "$study/cohort-expanded" --prior-split "$study/cohort/subjects.json")
prior=$(sbatch --parsable --dependency="afterok:$expanded" --cpus-per-task=2 --time=00:45:00 \
  --export=ALL,VTK_SMP_MAX_THREADS=2 --output="$study/expanded-train-%j.log" "$wrapper" "$runtime" \
  -m workflows.mbb_ssm.train --protocol "$protocol" --cohort "$study/cohort-expanded" \
  --output "$study/prior-guarded")
```

### 1.3 Results

| Accounting | Count |
|---|---:|
| SonoGym local L5 IDs, all excluded from the external prior | 12 |
| Official ZIP members named L5 masks | 1,228 |
| Downloaded candidates | 56 initial + 38 supplemental |
| Accepted at mask QC | 41 |
| Training, development, held-out mask QC | 29 / 6 / 6 |
| Preflight correspondence gate, initial training only | 6/12, below minimum |
| Preflight correspondence gate, expanded training | 4/29, below minimum |
| Final guarded training correspondences | 8/29 |
| Median subject-level global and posterior bidirectional P95 among included training shapes | 2.29 and 3.18 mm |
| Worst included posterior P95, extreme edge fraction and reversed-face fraction | 3.64 mm, 0.96%, 0% |
| Available PCA modes and template size | 7 modes, 14,238 vertices |

### 1.4 Interpretation

- The official count includes empty, truncated and unusable masks. Counting entries alone overstates the eligible shape population.
- The dataset provider uses subject identifiers in its release description. The downloadable table exposes one image ID per row and has no cross-visit patient keys. Subject-level alias independence is qualified accordingly.
- The supplement improves mask availability by conditioning on scan type, which affects population representativeness.
- The expanded preflight produced low geometric residuals with many template triangles folded by the update. This failed the original quality gates. Before either development or held-out scoring, the registration was repaired by raising graph smoothness, backtracking updates that invert or shrink the reference-oriented triangle area below 2%, and allowing 48 iterations. All original QC limits remained fixed. The initial configuration is archived alongside the repaired protocol.

### 1.5 Files

| File | Contents |
|---|---|
| external-ct*/acquisition.json | Private acquisition provenance and per-entry SHA-256 |
| cohort-expanded/subjects.json | Private split membership, exclusions and mask QC receipt |
| prior-guarded/correspondence.json | Training correspondence geometry and strain checks |
| prior-guarded/model.npz and model_receipt.json | Frozen physical template, vertex modes, variance and training manifest |


## 2. External geometric benchmark — Do PCA modes improve held-out completion?

`workflows.mbb_ssm.benchmark` creates public synthetic observations and disjoint withheld geometry from each external anatomy. `workflows.mbb_ssm.fit` reads the observations, public frame and frozen population template.

**Criterion.** Select the mode count and prior weight on the six external development IDs by the median of subject-level forward P95 hidden-surface distance in the broad posterior mask. A rejected fit contributes the frozen 20 mm loss. Compare the frozen PCA fit against a rigid mean and the three existing local baselines on the six held-out IDs.

### 2.1 Method

- Simulate three geometric first-visible sample families at 0° and ±25°. Replace angularly overlapping points with a single set and assign synthetic view groups in 2 mm lateral bands. Mask projected disks of diameter 2, 5 and 9 mm at the same site, deleting all depths across all families. The broad mask clears right posterior observations beyond the 58th percentile of lateral position. These observations approximate scanner geometry and carry no simulated acoustics.
- Fit the population shape with orientation initialized from the public RAS axes. Alternate robust rigid matching and bounded standard-deviation coefficients. Normalize total influence per view. Named TP and SAP observations have permissive positional windows on the population mesh. Unlabelled bone uses the broader posterior window.
- Use five initializations around the public orientation and observed centroid, then keep the lowest robust objective. Correspondence during fitting is to compatible model vertices. Triangle-distance evaluation is separate. Test alternative initializations, weakly constrained mode combinations, whole-view removal and 1 mm/0.5° perturbations. Report disagreement as a sensitivity diagnostic. An observed-residual, coefficient, mesh or alternative-shape gate may reject a fit. Preserve rejected proposals separately.
- Compare P0 observation only, P1 linear completion, P2 regularized local completion, P3 rigid population mean, and P4 PCA with rigid pose. Score all subjects at 1, 1.5 and 3 mm. For point-sample coverage use original observations plus accepted inferred points in the masked region. For full-shape error score each direction against the closest point from the 12 nearest triangle centers. Report posterior and full mesh errors separately. Conditional errors describe admitted fits.

### 2.2 Setup

```bash
previous="$project/artifacts/mbb-ultrasound-shape-completion-231044"
export PYTHONPATH="$PWD:$PWD/source/spinal_surgery/spinal_surgery/evaluation:$PWD/workflows/two_stage_navigation:$PWD/workflows/perception_s0010:$runtime/envs/stage1-python"
"$python_exe" -m workflows.mbb_ssm.transfer prepare --previous "$previous" \
  --protocol "$protocol" --output "$study/transfer-inputs"
development=$(sbatch --parsable --dependency="afterok:$prior" --kill-on-invalid-dep=yes \
  --time=00:25:00 --output="$study/development-%j.log" hpg/jobs/mbb_ssm.sbatch "$runtime" "$study" development "$previous")
heldout=$(sbatch --parsable --dependency="afterok:$development" --kill-on-invalid-dep=yes \
  --time=00:30:00 --output="$study/heldout-%j.log" hpg/jobs/mbb_ssm.sbatch "$runtime" "$study" heldout "$previous")
transfer=$(sbatch --parsable --dependency="afterok:$development" --kill-on-invalid-dep=yes \
  --time=00:20:00 --output="$study/transfer-%j.log" hpg/jobs/mbb_ssm.sbatch "$runtime" "$study" transfer "$previous")
sbatch --dependency="afterok:$heldout:$transfer" --kill-on-invalid-dep=yes \
  --output="$study/review-%j.log" hpg/jobs/mbb_ssm.sbatch "$runtime" "$study" review "$previous"
sbatch --dependency="afterany:$heldout:$transfer" --time=00:10:00 --cpus-per-task=2 \
  --output="$study/init-diagnostic-%j.log" "$wrapper" "$runtime" \
  -m workflows.mbb_ssm.benchmark diagnose-initialization --protocol "$protocol" \
  --model "$study/prior-guarded/model.npz" --development "$study/development" \
  --output "$study/development/lateral_initialization_diagnostic.json"
"$python_exe" -m unittest discover -s tests -p 'test_mbb*.py'
```

### 2.3 Results

The development job `52836801` evaluated nine configurations with 2, 4 and 7 modes and three prior weights. Every configuration reached the 20 mm median penalty. The frozen tie rule selected **2 modes, 50.28% retained variance, and a prior weight of 1.5 mm²**. At that setting 0/6 development fits passed the observed-residual gate. The most permissive alternatives passed 1/6.

The held-out job `52836802` scored six external IDs at each gap. Coverage below is percent of withheld samples within 1.5 mm of observed or admitted inferred geometry. The 2, 5, 9 mm and broad sets contain 9, 92, 314 and 19,806 withheld samples, respectively.

| Gap | P0 observed | P1 linear | P2 regularized | P3 rigid mean | P4 PCA | P3/P4 admitted |
|---|---:|---:|---:|---:|---:|---|
| 2 mm | 100% | 100% | 100% | 100% | 100% | 0/6 and 0/6 |
| 5 mm | 60.9% | 60.9% | 60.9% | 60.9% | 60.9% | 0/6 and 0/6 |
| 9 mm | 33.4% | 33.4% | 33.4% | 33.4% | 33.4% | 0/6 and 0/6 |
| Broad posterior | 1.2% | 1.2% | 1.2% | 9.4% | 9.3% | 2/6 and 2/6 |

All-case broad hidden median/P95 were **17.52/35.53 mm** for P0–P2, **11.60/33.98 mm** for P3 and **11.61/33.98 mm** for P4. On the two admitted P4 cases, forward and reverse mesh-sample P95 were **13.14 and 5.86 mm**. The broad P4 posterior error P95 was **27.12 mm** over all six cases, with observations retained for abstentions.

| Private held-out ID | Broad P3/P4 observed P95 (mm) | Broad P3/P4 hidden sample to proposed mesh P95 (mm) | P3/P4 result |
|---|---:|---:|---|
| s1297 | 14.18 / 14.18 | 25.04 / 25.01 | Both abstain |
| s0712 | 6.57 / 6.57 | 16.56 / 16.56 | Both abstain |
| s0649 | 4.85 / 4.87 | 13.40 / 13.53 | Both admitted |
| s0011 | 8.30 / 8.29 | 7.51 / 7.46 | Both abstain |
| s1098 | 7.99 / 7.94 | 15.26 / 15.27 | Both abstain |
| s0790 | 4.12 / 4.13 | 5.91 / 5.99 | Both admitted |

### 2.4 Interpretation

- PCA adds no measurable benefit over the rigid mean in this pilot. The prior adds partial geometric coverage only for broad holes, with large residual errors.
- The local baselines reject 5/24 cases for competing depth layers and 19/24 for unsupported height fields. External ray families supply mixed geometry and no hand-labelled SAP/TP subregions. The small-hole benchmark therefore measures a different input regime from the prior s0010 artificial-gap benchmark.
- P3 and P4 reject 22/24 because the observed geometry already disagrees with the fitted mesh by more than the 5 mm P95 limit. Four of six broad cases abstain. Two optimizations passing that observed gate still produce hidden P95 of approximately 6 and 14 mm.
- Three of six held-out L5 shapes exceed at least one training extent limit. One of those three fits passes the observed gate. Two in-envelope shapes also fail. The extent check is a size diagnostic.
- In a development-only replay, adding ±8 and ±16 mm lateral starts reduces the median hidden P95 of P4 proposals from 19.80 to 12.86 mm, while observed P95 changes from 6.70 to 6.86 mm and 0/6 fits pass. This diagnoses initialization sensitivity without rescuing the primary result. The same replay reduces the rigid mean proposal's hidden P95 from 19.72 to 14.30 mm. Job 52837525 runs this diagnostic without modifying either frozen evaluation.
- Across the six broad fits, P4 stress displacement envelopes enclosed the nearest fitting error for 13,508/19,806 withheld samples, 68.2%, including rejected proposals. The largest per-case spread P95 was 33.07 mm. This diagnostic neither specifies a confidence level nor controls anatomical error.

### 2.5 Files

| File | Contents |
|---|---|
| development/selection_experiment.json and freeze.json | Validation grid and selected immutable prior settings |
| heldout/inputs, reference and predictions | Matched held-out cohort, withheld geometry and prediction hashes |
| heldout/heldout_results.json | Per-case and aggregate errors, coverage, abstentions and uncertainty diagnostics |
| figures/population-modes.png | Example anatomy correspondence and PCA variation at a common physical scale |
| figures/heldout-{metrics,reconstruction,error}.png | Held-out measurement, geometry and error review |
| figures/heldout_size_and_uncertainty.json | Training size envelope and withheld prediction sensitivity |
| development/lateral_initialization_diagnostic.json | Additional lateral initializations in development only |

## 3. s0010 transfer — Does the inferred L5 surface recover the diagnostic cortex?

**Criterion.** Apply the frozen external model to each of the five recorded scans with Oracle-visible and U-Net observations. Score the local 9.5 mm cortical region and contact with the seven archived diagnostic cortical-path samples within 1.5 mm. The anatomical bone reference of interest remains unresolved for the right L4 medial branch.

### 3.1 Method

- Reuse Stage 1 cf7_00, the original L4 and T4 L5 checkpoints, the public pose, and unchanged reconstructed cortical interfaces. Select L5-routed points and the original frame groups. Keep the raw observations separate from model surfaces. The old local-completion result is the paired P0–P2 baseline.
- Verify all predictions and perturbation receipts before opening s0010 CT or any simulator transformation used for scoring. Apply the transform only to compare geometry in the evaluator. Compare s0010 with all included training meshes for near-identical geometry offline.
- Measure both directed local distances, wider posterior and global surface error, forward coverage of previously unobserved local cortex, and the diagnostic cortical-path count. A discovery of a point near the path cannot establish an anatomically validated MBB region.

### 3.2 Setup

| Setting | Value |
|---|---|
| Five recorded scans | Seeds 2026121700, 2026121702, 2026121900, 2026121901 at 102 frames each, plus the 126-frame supplement of 2026121700 |
| CT reference | The SHA-verified previous s0010 segmentation and frozen diagnostic cortical path |
| Accounting | Peak 4 allocated CPU cores, 0 GPUs, 847.868 Slurm CPU seconds including failed correspondence preflights. Forty local MBB-focused tests pass. |
| Perception arms | Oracle-visible and cached U-Net, 10 arm-session pairs |
| Reserved seeds | 2026121800–2026121803 excluded |

Run the separate scoring of previously sealed stress predictions from the same worktree after transfer scoring.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VTK_SMP_MAX_THREADS=1
"$python_exe" -m workflows.mbb_ssm.transfer score-stress --previous "$previous" \
  --inputs "$study/transfer-inputs" --predictions "$study/transfer-predictions" \
  --transfer "$study/transfer" --output "$study/transfer/sealed_stress_scoring.json" \
  --protocol "$protocol"
```

### 3.3 Results

Transfer job `52836803` verifies all ten observed-only prediction receipts before opening reference geometry. The nearest candidate in the training-geometry duplicate screen had bidirectional median/P95 of 0.87/2.88 mm. It did not pass the prespecified near-identical screen. No matching pseudonymous ID or near-identical training geometry was found.

| Recording | Oracle previous trace P0–P2 | Oracle trace hits P3 / P4 | Oracle local two-way P95 for P4 (mm) | U-Net P4 observed P95 (mm) and disposition |
|---|---:|---:|---:|---|
| 2026121700, 102 frames | 1/7 | 5/7 / 5/7 | 6.40 / 5.72 | 6.74, abstained |
| 2026121702, 102 frames | 0/7 | 6/7 / 6/7 | 4.10 / 5.09 | 6.85, abstained |
| 2026121900, 102 frames | 0/7 | 3/7 / 3/7 | 3.93 / 4.59 | 6.60, abstained |
| 2026121901, 102 frames | 0/7 | 6/7 / 6/7 | 4.18 / 4.93 | 6.01, abstained |
| 2026121700, 126 frames | 1/7 | 3/7 / 3/7 | 5.93 / 5.55 | 5.61, abstained |

Both population fits admit 5/5 Oracle fits and 0/5 learned fits. The median observed P95 for P4 is 3.45 mm with Oracle and 6.60 mm with U-Net. The nominal Oracle trace count rises from 2/35 to 23/35 for both P3 and P4, with the 102-frame prefix intentionally counted in both matched recordings. Median subject-level P95 for P4 is 4.18 mm from local CT to reconstructed surfaces, 5.09 mm in the reverse direction, and 4.21 mm from the previously unobserved local subset to the reconstruction. Median nominal-fitting runtime is 1.12 s for Oracle and 1.10 s for learned. These times exclude the stress replays and evaluation.

### 3.4 Interpretation

- The population prior places surfaces closer to the diagnostic trace using Oracle observations. Equal coverage from P3 and P4 locates the measurable effect in the rigid population mean within this comparison.
- The median local corresponding-vertex P95 between admitted Oracle PCA scans is 13.96 mm after evaluator alignment. The 1 mm and 0.5° pose stress shifts the hidden posterior by up to 20.36 mm P95. Removing all OTHER_BONE samples reaches 39.01 mm P95. These refits use one initialization, so their disagreement includes changes in optimization basin. A nominal fit provides insufficient reproducibility for the actual missing region.
- The 126-frame learned fit produced a rejected proposal close to four trace samples. It exceeded the observed-residual limit.
- In the sealed semantic-support diagnostic, both population approaches pass the fit gate in all five U-Net trials after retaining TP/SAP alone. The mean and PCA reach 20/35 and 21/35 diagnostic-path samples. The median CT-to-PCA local P95 remains 5.30 mm, with 17.08–29.11 mm posterior disagreement from the corresponding nominal proposals. This removes correctly and incorrectly labelled OTHER_BONE together and uses single-start refits. It identifies sensitivity to the support choice, without isolating a false-positive cause or changing the primary outcome.
- The intended provisional bone reference remains anatomically unresolved. `TARGET_LOCALIZATION_NOT_EVALUABLE` applies to the right L4 medial branch target region.

### 3.5 Files

| File | Contents |
|---|---|
| transfer-predictions/seals.json | Prediction and prior receipts for ultrasound-only inference |
| transfer/transfer_results.json | Both directed errors, diagnostic trace and cross-session disagreement |
| transfer/sealed_stress_scoring.json | CT diagnostics for previously sealed pose and semantic-support fits |
| transfer/s0010_identity_audit.json | Evaluator-only geometry duplicate screen |
| transfer/clouds | Separate observations, inferred meshes, proposals, CT bounds and diagnostic trace in robot-base mm |
| figures/s0010-transfer.png | Five-session, two-perception review |
