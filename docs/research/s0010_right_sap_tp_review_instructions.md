# s0010 RIGHT L4/L5 SAP/TP review instructions

This package contains privileged CT, segmentation, and mesh views for four geometry candidates.
It is an anatomical review package, not actor input and not evidence of MBB target localization.

## Requested review

Review each candidate independently:

1. `L4_RIGHT_superior_articular_process`
2. `L4_RIGHT_transverse_process`
3. `L5_RIGHT_superior_articular_process`
4. `L5_RIGHT_transverse_process`

For each candidate, record one decision:

- `ACCEPT_BOUNDARY`: the colored surface sufficiently isolates the named structure for this
  simulator-only supervised-label pilot.
- `REVISE_BOUNDARY`: the named structure is identifiable, but the colored boundary must change.
- `REJECT_CANDIDATE`: the available evidence does not support the named structure.

An accepted candidate still remains simulator-only provisional anatomy; it is not a clinically
validated landmark, nerve location, injection target, or instrument trajectory.

## Figure legend

- Gray points: complete vertebral mesh.
- Blue points: SAP geometry candidate.
- Orange points: TP geometry candidate.
- Green contour: complete raw vertebral segmentation label.
- Black plus: candidate centroid.
- CT and segmentation are evaluator-only privileged data.

The first row shows posterior, right-lateral, and superior mesh projections. The second row shows
axial, sagittal, and coronal CT slices through the candidate centroid. Colored CT points are only
surface vertices within one mesh-index unit of the displayed slice.

## Required checks

For every candidate, verify:

- Correct vertebral level: L4 or L5.
- Correct patient side: RIGHT.
- The semantic structure is isolated.
- Adjacent lamina, pars, pedicle, vertebral body, or the other process is not materially included.
- The proposed boundary is supportable in all three CT planes and mesh projections.
- Any correction is returned as an explicit point/surface region in the documented mesh-index
  coordinate frame, not only as a prose description.

Record reviewer identity, relevant qualification, decision, limitations, and estimated boundary
uncertainty. Without those fields, no candidate can be promoted to training ground truth.

## Current interpretation

- TP candidates are plausible but broad and remain `PROVISIONAL`.
- SAP candidates occupy a plausible posterior-superior region but may include adjacent posterior
  elements and remain `UNRESOLVED`.
- No medial-branch nerve segmentation exists.
- No MBB target region is constructed by this package.
