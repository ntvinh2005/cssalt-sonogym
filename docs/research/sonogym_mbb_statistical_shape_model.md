# L5 statistical shape completion from partial observations

The population model is learned from external TotalSegmentator L5 masks. The s0010 transfer fit receives the archived ultrasound-derived interface points and public pose information.

## 1. Population and correspondence — Do the training surfaces support a common template?

`workflows.mbb_ssm.acquire` extracts individual L5 masks from the official archive. `workflows.mbb_ssm.prepare` checks geometry and partitions the eligible IDs. `workflows.mbb_ssm.train` builds the registered population.

**Criterion.** A mask must contain a connected, complete and watertight L5. Correspondences must meet the frozen bidirectional global and posterior distance, edge-distortion and face-orientation gates. At least eight training meshes must pass registration. The inference model uses physical millimetres and preserves anatomical size variation.

### 1.1 Method

- Verify HTTP range and ZIP entry integrity, then hash every training mask. The first retrieval uses a fixed hash order from the source index. The second uses a fixed hash order within abdominal–pelvic, whole-body and related scan categories after measuring mask availability. The supplement excludes all earlier downloads.
- Group by the release's pseudonymous `image_id`. The release metadata does not contain a cross-visit linkage key. Preserve the original partitions when adding the supplemental masks. Reject exact duplicate shapes on the common physical grid. Exclude all 12 SonoGym asset IDs, including s0010. Perform an independent near-identical geometry screen against the final training meshes after sealing the s0010 fits.
- Extract the largest connected external surface from each accepted mask on a 1.25 mm RAS grid. Use rigid alignment without scaling. Choose the training template closest to the training median extents, then use normal-compatible surface correspondences and Laplacian-regularized displacement on the mesh graph. Reverse geometric correspondence contributes a bounded update. PCA uses thin SVD of the training-only aligned vertex vectors.
- Benady et al. used rigid and nonrigid correspondence followed by PCA and Gaussian-process reconstruction in a CT geometry benchmark. Here the statistical fit uses PCA and a quadratic coefficient penalty. [Publication and author abstract](https://cris.tau.ac.il/en/publications/statistical-shape-modeling-and-gaussian-process-based-reconstruct/) · [DOI](https://doi.org/10.1007/s11548-026-03757-2)

### 1.2 Setup

| Setting | Value |
|---|---|
| Data source | Wasserthal et al., TotalSegmentator CT v2.0.1, doi:10.5281/zenodo.10047292, CC BY 4.0 |
| Scope | Binary L5 masks only. The prior uses neither full patient images nor simulator CT. |
| Data version verification | [Official Zenodo record and license](https://zenodo.org/records/10047292/export/json) |
| Worktree base | 846e1d1c0fbc58dc84724e3cd4c1191c01e721e8 |
| Protocol | experiments/unity_two_stage_transfer/mbb_ssm_v1.json and SHA-256 |
| runtime | /blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/envs/sonogym |
| jobs | 52833413 initial mask QC, 52833617 correspondence preflight, 52834740 expanded mask QC, 52834756 expanded training |

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
  --output "$study/prior-expanded")
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

### 1.4 Interpretation

- The official count includes empty, truncated and unusable masks. Counting entries alone overstates the eligible shape population.
- The dataset provider uses subject identifiers in its release description. The downloadable table exposes one image ID per row and has no cross-visit patient keys. Subject-level alias independence is qualified accordingly.
- The supplement improves mask availability by conditioning on scan type, which affects population representativeness.

### 1.5 Files

| File | Contents |
|---|---|
| external-ct*/acquisition.json | Private acquisition provenance and per-entry SHA-256 |
| cohort-expanded/subjects.json | Private split membership, exclusions and mask QC receipt |
| prior-expanded/correspondence.json | Training correspondence geometry and strain checks |
| prior-expanded/model.npz and model_receipt.json | Frozen physical template, vertex modes, variance and training manifest |
