# s0010 SAP semantic-boundary correction V2

This records the CPU-only source-geometry audit, adjacent-level facet review and static package build for provisional RIGHT L4/L5 SAP annotations. It does not approve semantic anatomy, define an MBB target or change actor inputs.

## Result

**Both structures remain `SOURCE_SEMANTICS_INSUFFICIENT`: paired surfaces narrow the V1 candidates, but the sources do not label SAP, IAP, lamina or pars separately and the archived ultrasound does not independently establish SAP cortical visibility.**

| step | what was measured | criterion | measured | verdict |
|---|---|---|---|---|
| 1 | source and derived surface integrity | explain every zero-area source face; derived surface has zero degenerate faces | every source zero-area face has a repeated physical corner; derived L3/L4/L5 zero-area faces `0 / 17,008`, `0 / 16,692`, `0 / 16,756` | pass |
| 2 | evidence-based SAP partition | partition every V1 candidate location without unsupported negative anatomy | L4 `55` core, `172` uncertain, `0` likely non-SAP; L5 `77`, `177`, `0` | pass for review construction; fail for semantic acceptance |
| 3 | self-contained static review package | all references, checksums and ZIP members valid | `57 / 57` references, `69 / 69` checksums, ZIP test pass | pass |

The paired-surface subsets are review proposals, not semantic labels. Lumbar facet anatomy supports reviewing the SAP of the lower vertebra against the IAP of the vertebra above, but the source labels cannot distinguish those processes from adjacent posterior elements ([facet-joint anatomy](https://pmc.ncbi.nlm.nih.gov/articles/PMC11595282/), [facet biomechanics](https://pmc.ncbi.nlm.nih.gov/articles/PMC7002062/)).

## 1. Geometry audit — why the source STL contains degenerate faces

`workflows/two_stage_navigation/build_s0010_sap_static_review_v2.py`, run by the bounded CPU command below. It compares raw STL records with unique physical locations and reconstructs exposed voxel faces from the original vertebral labels.

**Criterion.** Every reported zero-area source face must have a measured cause, each separately reconstructed surface must have zero zero-area faces, and source-to-derived coordinate disagreement must equal the half-voxel-corner offset expected from the two surface conventions.

### 1.0 Quantities

| symbol | definition | source |
|---|---|---|
| `PROPOSED_SAP_SURFACE_CORE` | [project notation](../notation.md#proposed_sap_surface_core) | approved task |
| `BOUNDARY_UNCERTAIN` | [project notation](../notation.md#boundary_uncertain) | approved task |
| `LIKELY_NON_SAP_NEIGHBOR` | [project notation](../notation.md#likely_non_sap_neighbor) | approved task |

### 1.1 Method

- Face area is computed from the cross product of its two edge vectors.
- Physical vertices are deduplicated by exact segmentation-index coordinates.
- A separate surface is reconstructed from every exposed face of each labelled voxel; source STL files are unchanged.
- Nearest-location correspondence records the source/derived relationship without claiming preserved face indices.

### 1.2 Setup

| what | value | spec |
|---|---|---|
| run / checkpoint | `manual_20261120T0120`, commit `7a91337e640757d5ded7a75d42e71b84d3974f8c` | §1 |
| inputs | `vertebrae_L3.stl`, `vertebrae_L4.stl`, `vertebrae_L5.stl`, `combined_label_map.nii.gz` under the configured s0010 asset root | §1 |
| filters | source face area `> 0`; exact coordinate deduplication | §1 |

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  $SONOGYM_ENV_PREFIX/bin/python \
  workflows/two_stage_navigation/build_s0010_sap_static_review_v2.py \
  --asset-root "$SONOGYM_ASSETS_DATA_DIR/HumanModels" \
  --geometry-root "$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_right_sap_tp_correction/51779234" \
  --v3-root "$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_mbb_single_anatomy_poc_v3/52039283" \
  --audit-root "$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_geometry_audit_v1/52053307/audit" \
  --output-root "$SONOGYM_OUTPUT_DIR/unity_two_stage_transfer/s0010_sap_static_review_v2/manual_20261120T0120/review_packages" \
  --surface-slab-tolerance-mm 1.0
```

| | value |
|---|---:|
| attempted source faces, L3/L4/L5 | `17,008` / `16,696` / `16,756` |
| dropped from valid source surface, zero area | `5,986` / `6,076` / `6,776` |
| **final valid source faces** | **`11,022` / `10,620` / `9,980`** |

### 1.3 Results

| level | raw vertex records | unique locations | zero-area source faces | derived faces | zero-area derived faces |
|---|---:|---:|---:|---:|---:|
| L3 | 51,024 | 5,511 | 5,986 | 17,008 | 0 |
| L4 | 50,088 | 5,310 | 6,076 | 16,692 | 0 |
| L5 | 50,268 | 4,990 | 6,776 | 16,756 | 0 |

**Verdict: pass**, because every zero-area source face has fewer than three unique physical corners, each derived surface has zero degenerate faces, and the maximum source-to-derived distance is the expected `0.8660254037844386` index (`1.299038105676658` mm at 1.5 mm spacing).

### 1.4 Reading it

- Integer-coordinate quantization and repeated STL corner records explain the degenerate triangles.
- The derived surfaces improve geometric validity but not anatomical semantics or CT resolution.
- Exact source face correspondence is unavailable after reconstruction; nearest-location maps are packaged.

### 1.5 Files

| file | contents |
|---|---|
| `04_GEOMETRY_QUALITY_AUDIT.md` in the package | per-level audit |
| `geometry/*_source_to_derived_correspondence.csv` | explicit nearest-location maps |
| `geometry/*_v2_regions.npz` | source masks and derived surfaces |

## 2. Boundary correction — can paired facet geometry isolate a semantic SAP region

`workflows/two_stage_navigation/build_s0010_sap_static_review_v2.py`, run by the §1 entry point. It intersects valid physical locations of each original candidate with valid surface locations on the articulating vertebra above.

**Criterion.** The three regions must be disjoint and exhaust the V1 candidate; only explicitly supported locations may enter the proposed core or likely-non-SAP region. Semantic acceptance additionally requires source semantics or independent anatomical review, which are absent.

### 2.1 Method

- RIGHT L4 is paired with the RIGHT L3 opposing surface; RIGHT L5 is paired with RIGHT L4.
- Exact coincident locations supported by valid faces on both source meshes enter `PROPOSED_SAP_SURFACE_CORE`.
- Every remaining V1 location enters `BOUNDARY_UNCERTAIN`.
- `LIKELY_NON_SAP_NEIGHBOR` remains empty because no source label identifies lamina, pars, pedicle or IAP separately.

### 2.2 Setup

| what | value | spec |
|---|---|---|
| run / checkpoint | `manual_20261120T0120`, commit `7a91337e640757d5ded7a75d42e71b84d3974f8c` | §2 |
| inputs | correction job `51779234`; adjacent L3/L4/L5 source meshes | §2 |
| filters | nonzero-area source faces; exact physical-location coincidence | §2 |

```bash
# Same resolved command as §1; no separate threshold search or mask tuning.
```

| | L4 | L5 |
|---|---:|---:|
| attempted V1 unique candidate locations | 227 | 254 |
| proposed core | 55 | 77 |
| boundary uncertain | 172 | 177 |
| likely non-SAP | 0 | 0 |
| **final partition** | **227** | **254** |

### 2.3 Results

| structure | raw V1 records | unique core | unique uncertain | unique likely non-SAP | archived intersecting frames | core at least 10 pixels inside image |
|---|---:|---:|---:|---:|---:|---:|
| RIGHT L4 SAP | 2,198 | 55 | 172 | 0 | 117 | 0 |
| RIGHT L5 SAP | 2,422 | 77 | 177 | 0 | 123 | 65 |

**Verdict: fail for semantic SAP acceptance**, because paired geometry supports review proposals but no source semantic separates SAP/IAP/lamina/pars, and acoustic visibility plus semantic identity remain unconfirmed. Both structures therefore retain `SOURCE_SEMANTICS_INSUFFICIENT`.

### 2.4 Reading it

- V2 narrows the strongest paired-surface evidence instead of shrinking by a coordinate threshold.
- No archived L4 plane provides the full projected core with a 10-pixel image margin.
- L5 has better geometric coverage, but geometry-plane intersection is not semantic ultrasound recognition.
- The minimum additional evidence is a process-specific SAP/IAP/lamina/pars annotation or qualified boundary edit against the packaged coordinates, plus synchronized ultrasound views centered on the reviewed L4 region.

### 2.5 Files

| file | contents |
|---|---|
| `03_STRUCTURE_SUMMARY.csv` | V1/V2 region counts |
| `figures/3d/*_overview_sheet.png` | opaque paired-level context with region overlay |
| `figures/geometry_context/*_facet_CT.png` | enlarged paired-level CT crops |
| `figures/ultrasound/ULTRASOUND_INDEX.csv` | ten curated correspondence examples |

## 3. Package validation — is the review artifact self-contained

The build entry point validates relative references, writes checksums and tests the ZIP before reporting success.

**Criterion.** Every referenced and required file exists, every listed checksum passes, and the ZIP integrity test reports no corrupt member.

### 3.1 Method

- Resolve every manifest reference relative to the package root.
- Hash every package file except the checksum manifest itself.
- Reopen the ZIP and test every member.

### 3.2 Setup

| what | value | spec |
|---|---|---|
| run / checkpoint | `manual_20261120T0120`, commit `7a91337e640757d5ded7a75d42e71b84d3974f8c` | §3 |
| inputs | V2 package directory | §3 |
| filters | none | §3 |

```bash
sha256sum -c checksums.sha256
unzip -t s0010_right_l4l5_sap_review_v2.zip
```

Nothing was dropped.

### 3.3 Results

| check | measured |
|---|---:|
| referenced files | 57 / 57 |
| checksums | 69 / 69 |
| focused package tests | 7 / 7 |
| final geometry/V1/V2 regression tests | 15 / 15 |
| ZIP size | 7,348,338 bytes |

**Verdict: pass**, because every declared package-integrity check succeeded.

### 3.4 Reading it

- The artifact is review-ready and reproducible.
- Package integrity does not imply anatomical correctness.
- The run used one CPU thread, zero GPUs, 51.27 seconds and 696,480 KB peak RSS because the Slurm controller was down; this execution mode is recorded rather than presented as a scheduler job.

### 3.5 Files

| file | contents |
|---|---|
| `experiments/unity_two_stage_transfer/s0010_sap_static_review_v2_result.json` | concise result |
| runtime `build_result.json` | package paths and hashes |
| runtime `metrics/time-build.txt` | measured resource use |
