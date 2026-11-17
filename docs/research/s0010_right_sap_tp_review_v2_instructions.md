# s0010 RIGHT SAP/TP expanded review v2

This package addresses review feedback on v1. It adds zoomed adjacent CT slices, editable geometry,
source L4/L5 meshes, and cropped CT/segmentation volumes. No annotation has been accepted or
promoted.

## Suggested workflow

1. Open `adjacent_slices/*_adjacent_slices.png` for a rapid boundary audit.
2. Load `volumes/s0010_L4_L5_ct_crop.nii.gz` and
   `volumes/s0010_L4_L5_segmentation_crop.nii.gz` together in 3D Slicer or equivalent software.
3. Load `source_meshes/vertebrae_L4.stl` and `source_meshes/vertebrae_L5.stl`.
4. For each candidate, use the corresponding files in `editable_geometry/`:
   - `.npz`: complete source vertices/faces plus Boolean selected-vertex mask.
   - `.csv`: every selected vertex in mesh-index, patient-local, and SonoGym-world coordinates.
   - `.ply`: selected candidate vertices as a point cloud.
5. Fill `review_response_template.json`.

The cropped NIfTI volumes are not resampled. `expanded_review_manifest.json` records their source
index bounds and affine-preserving crop metadata.

## Decisions

- `ACCEPT_BOUNDARY`: candidate sufficiently isolates the named simulator landmark.
- `REVISE_BOUNDARY`: landmark is identifiable, but corrected geometry must be supplied.
- `REJECT_CANDIDATE`: evidence does not support the named landmark.

Record reviewer identity, relevant qualification, uncertainty, and contamination by adjacent
lamina, pars, pedicle, vertebral body, or another process. A prose-only correction is insufficient;
return an edited mask, point set, or surface in the documented mesh-index frame.

## Scientific boundary

All contents are privileged review data. They are forbidden at actor inference. The package does
not test ultrasound observability, authorize model training, construct an MBB target region,
localize a nerve, or establish clinical validity.
