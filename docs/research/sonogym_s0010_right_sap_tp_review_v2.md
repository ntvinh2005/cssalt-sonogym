# SonoGym s0010 RIGHT SAP/TP expanded review v2

This record responds to the v1 review by adding adjacent CT slices and editable 3-D evidence. It
does not promote an annotation, test ultrasound observability, train a model, or construct an MBB
target.

## Result

**The expanded package is `PACKAGE_READY_FOR_INDEPENDENT_REVIEW`, while all four anatomical boundaries remain `MORE_EVIDENCE_REQUIRED` and Stage 2 remains `MBB_STAGE2_BLOCKED_GROUND_TRUTH`.**

| step | what was measured | criterion | measured | verdict |
|---|---|---|---|---|
| expanded package | source integrity, adjacent slices, editable geometry, source meshes, volume crops, ZIP | protocol v2 package gate | 28 / 28 checksums; 10 / 10 tests; ZIP valid | PASS |
| anatomical acceptance | independent per-landmark boundary decision and uncertainty | reviewer response complete | 0 / 4 accepted | PENDING |

## 1. Expanded review export — does v2 supply the missing review material?

`expand_s0010_right_landmark_review.py`, run by Slurm job `51717444`. It preserves the v1
candidates and adds editable geometry plus non-resampled CT/segmentation context.

**Criterion.** Protocol v2 requires exact source hashes/counts, four 15-slice montages, editable
CSV/NPZ/PLY for every candidate, source L4/L5 STL, affine-preserving NIfTI crops, an empty response
template, and a checksum-valid ZIP. It defines no anatomical PASS before reviewer decisions.

### 1.0 Quantities

| symbol | definition | source |
|---|---|---|
| `MORE_EVIDENCE_REQUIRED` | no boundary accepted; additional editable evidence requested | reviewer feedback on v1 |
| editable vertex mask | Boolean selection over every source STL vertex, stored with vertices/faces | protocol v2 |

### 1.1 Method

- Preserve `s0010`, L4-L5, and `RIGHT`; preserve all four v1 quantile candidates exactly.
- Render five adjacent slices per plane at offsets -6, -3, 0, +3, and +6 mm.
- Crop CT and segmentation around L4/L5 with 12-voxel margins without resampling; update the affine
  so crop-index world coordinates equal source-index world coordinates.
- Export the complete source vertex array, faces, Boolean selection mask, selected indices, CSV
  coordinates, and selected-point PLY for every candidate.
- Include the original L4/L5 STL and an empty structured reviewer response.
- Keep all products privileged, actor-ineligible, and unavailable for training.

### 1.2 Setup

| what | value | spec |
|---|---|---|
| run | Slurm `51717444`; commit `4a3570d4e62fa2c97c68ac88f77e32c0f63add9c` | CPU-only export |
| protocol | `s0010_right_sap_tp_review_protocol_v2.yaml` | SHA-256 `e6e71d901776135d5baeb13c4e2bb6e14b98d0ad3636099289cd8a79cceb229c` |
| source | annotation v2; CT; segmentation; L4/L5 STL | hashes frozen in protocol |
| resources | `anest-dept`; 4 CPU; 16 GB requested; 0 GPU | 39 s; peak RSS 756,180 KiB |
| storage | run 9,431,045 bytes; ZIP 5,786,547 bytes | Blue Storage |

```bash
sbatch --parsable --account=anest-dept --qos=anest-dept --partition=hpg-default \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:30:00 \
  --job-name=sg-right-sap-tp-v2 \
  --export=ALL,SONOGYM_CONFIG_FILE=/blue/lampotan/share/vinhnguyen1/cssalt-sonogym/.sonogym.env \
  hpg/jobs/two_stage_s0010_right_landmark_review_v2.sbatch
```

Nothing was dropped.

### 1.3 Results

| evidence | count / result |
|---|---:|
| overview figures | 4 |
| adjacent-slice figures | 4, each containing 15 slices |
| editable geometry files | 12: CSV, NPZ, PLY for each candidate |
| source meshes | 2 STL |
| cropped volumes | CT + segmentation; 1.5 mm; not resampled |
| tests | 10 / 10 pass |
| internal checksums | 28 / 28 pass |
| annotation promotion | 0 / 4 |

The enlarged slices reinforce the v1 concern rather than resolving it automatically: TP selections
include anterior-facing adjacent surface in some slices, while SAP selections retain axis-aligned
cuts and possible posterior-element contamination. These observations justify editable review, not
an automated boundary correction.

**Verdict: package PASS; anatomical decision PENDING.** The package answers the reviewer-material
deficit while preserving `actor_usable=false` and `model_training_authorized=false`.

### 1.4 Reading it

- The reviewer can now inspect and directly edit every selected boundary in 3-D.
- The original CT/STL identity is verifiable from frozen hashes; the distributed CT is a smaller
  affine-preserving L4/L5 crop.
- No ultrasound frames are included because ultrasound observability is the next experiment only
  after anatomical boundaries are accepted.
- No result permits Stage 3 or an injection-readiness claim.

### 1.5 Files

| file | contents |
|---|---|
| `experiments/unity_two_stage_transfer/s0010_right_sap_tp_review_v2_result.json` | small execution/result manifest |
| `docs/research/s0010_right_sap_tp_review_v2_instructions.md` | 3-D review workflow |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/reviewer/s0010_right_sap_tp_review_v2_51717444.zip` | expanded reviewer package |
| `/blue/lampotan/share/vinhnguyen1/cssalt-sonogym-runtime/outputs/unity_two_stage_transfer/s0010_right_sap_tp_review_v2/51717444/` | complete job evidence |
