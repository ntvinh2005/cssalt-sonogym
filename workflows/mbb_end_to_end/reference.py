"""Audit the proposed right L5 landmark on the full external cortex using anatomy alone."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nibabel as nib
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree, distance

from run_s0010_l4_medial_branch_target import load_target_geometry
from run_simple_landmark_mapping import sha256, write_json
from workflows.mbb_observed_surface.audit import exposed_faces


PRIMARY_ANATOMY = [
    "https://rapm.bmj.com/content/rapm/early/2022/05/18/rapm-2022-103653.full.pdf",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11372964/",
]


def face_properties(l5, faces, axes):
    owners = np.rint(faces).astype(int)
    alternate = owners.copy()
    along_face = (np.arange(len(faces)), axes)
    alternate[along_face] = (2 * faces[along_face] - owners[along_face]).astype(int)
    inside = l5[tuple(owners.T)]
    owners[~inside] = alternate[~inside]
    if not l5[tuple(owners.T)].all():
        raise ValueError("exposed face lacks a source L5 owner")
    normal = 2 * (faces - owners)
    return owners, normal.astype(int)


def select_review_surfaces(faces, normal, owners, sap, tp, contract, spacing):
    xyz = faces * spacing
    sap_core, tp_core = (np.argwhere(mask) * spacing for mask in (contract.sap_core, contract.tp_core))
    vicinity = np.linalg.norm(spacing)
    sap_depth = np.median(np.argwhere(sap)[:, 2])
    tp_medial = np.median(np.argwhere(tp)[:, 0])
    sap_face = ((normal[:, 0] == 1) & sap[tuple(owners.T)] & (owners[:, 2] <= sap_depth)
                & (cKDTree(sap_core).query(xyz)[0] <= vicinity + 1e-9))
    tp_face = (((normal[:, 2] == 1) | (normal[:, 0] == -1)) & tp[tuple(owners.T)]
               & (owners[:, 0] <= tp_medial) & (cKDTree(tp_core).query(xyz)[0] <= vicinity + 1e-9))
    return np.flatnonzero(sap_face), np.flatnonzero(tp_face)


def shortest_cortical_path(faces, axes, spacing, starts, ends):
    if len(starts) == 0 or len(ends) == 0:
        return np.asarray([], dtype=int), None
    corners_2x = np.repeat(np.rint(2 * faces).astype(int)[:, None], 4, axis=1)
    square = np.asarray([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    for axis in range(3):
        for column, dim in enumerate(i for i in range(3) if i != axis):
            corners_2x[axes == axis, :, dim] += square[:, column]
    edges = defaultdict(list)
    for i, boundary in enumerate(corners_2x):
        for a, b in zip(boundary, np.roll(boundary, 1, axis=0)):
            edges[tuple(sorted((tuple(a), tuple(b))))].append(i)
    neighbors = {(a, b) for ids in edges.values() for a in ids for b in ids if a != b}
    pairs = np.asarray(sorted(neighbors), dtype=int).reshape(-1, 2)
    weights = np.linalg.norm((faces[pairs[:, 0]] - faces[pairs[:, 1]]) * spacing, axis=1)
    graph = csr_matrix((weights, (pairs[:, 0], pairs[:, 1])), shape=(len(faces), len(faces)))
    lengths, predecessor, source = dijkstra(
        graph, indices=starts, return_predecessors=True, min_only=True,
    )
    last = ends[np.argmin(lengths[ends])]
    if not np.isfinite(lengths[last]):
        return np.asarray([], dtype=int), None
    path = [int(last)]
    while path[-1] != source[last]:
        path.append(int(predecessor[path[-1]]))
    return np.asarray(path[::-1]), float(lengths[last])


def review_figure(out, ct, l5, sap, tp, faces, sap_ids, tp_ids, path, affine, midpoint):
    center = np.rint(midpoint.mean(0)).astype(int)
    selections = (
        (faces[sap_ids], "#ffad41", "Right-facing caudal SAP proposal face"),
        (faces[tp_ids], "#57ddbc", "Superior/medial TP proposal face"),
        (faces[path], "#fd6f9e", "Shortest connected full-cortex trace"),
    )
    figure, panels = plt.subplots(3, 3, figsize=(12, 11), layout="constrained")
    for axial, name in enumerate(("Sagittal", "Coronal", "Axial")):
        h, v = [i for i in range(3) if i != axial]
        for panel, adjustment in zip(panels[axial], (-2, 0, 2)):
            index = center[axial] + adjustment
            x = slice(center[h] - 13, center[h] + 14)
            y = slice(center[v] - 13, center[v] + 14)
            selection = [None] * 3
            selection[axial], selection[h], selection[v] = index, x, y
            selection = tuple(selection)
            extent = (x.start - .5, x.stop - .5, y.start - .5, y.stop - .5)
            panel.imshow(ct[selection].T, cmap="gray", vmin=-150, vmax=750,
                         origin="lower", extent=extent)
            for label, color in ((l5, "#dedede"), (sap, "#d87816"), (tp, "#18856d")):
                section = label[selection].T
                if section.any() and not section.all():
                    panel.contour(section, levels=[.5], colors=[color], linewidths=1,
                                  extent=extent, origin="lower")
            for xyz, color, _ in selections:
                nearby = np.abs(xyz[:, axial] - index) <= .5 + 1e-9
                panel.scatter(xyz[nearby, h], xyz[nearby, v], s=28, color=color,
                              edgecolor="black", linewidth=.35, zorder=4)
            location = center.copy()
            location[axial] = index
            position = nib.affines.apply_affine(affine, location)[axial]
            panel.set(xlim=extent[:2], ylim=extent[2:],
                      title=f"{name} · {'RAS'[axial]} {position:.1f} mm")
            panel.set_xticks([])
            panel.set_yticks([])
    legend = [Line2D([], [], color="#dedede", label="Full L5 CT segmentation")]
    legend.extend(Line2D([], [], color=color, marker="o", linestyle="none", label=label)
                  for _, color, label in selections)
    figure.legend(handles=legend, loc="outside lower center", ncol=2, frameon=False)
    figure.suptitle("s0010 anatomy-only review · exposed L5 cortical hypotheses · reference unresolved")
    figure.savefig(out, dpi=145)
    plt.close(figure)


def sealed(path, expected=None):
    value = sha256(path)
    if expected is not None and value != expected:
        raise ValueError(f"input checksum mismatch: {path}")
    return {"path": str(path.resolve()), "sha256": value}


def freeze(args):
    manifest = json.loads(args.manifest.read_text())
    policy = json.loads(args.prior_protocol.read_text())
    if policy["reference_decision"] != "C_UNRESOLVED":
        raise ValueError("unexpected decision in archived anatomy audit")
    anatomy_provenance = {
        "manifest": sealed(args.manifest, policy["source_manifest_sha256"]),
        "prior_protocol": sealed(args.prior_protocol,
                                 args.prior_protocol.with_suffix(".sha256").read_text().split()[0]),
        "prior_audit": sealed(args.prior_audit, policy["anatomy_audit_sha256"]),
        "ct": sealed(args.ct, args.ct_sha256),
    }
    historical_audit = json.loads(args.prior_audit.read_text())
    if historical_audit["input_sha256"]["ct"] != args.ct_sha256:
        raise ValueError("CT differs from the prior audit")
    target = manifest["target"]
    for key in ("prediction", "segmentation", "reference_protocol"):
        anatomy_provenance[key] = sealed(Path(target[key]["path"]), target[key]["sha256"])
    prediction, sap, tp, old = load_target_geometry(
        Path(target["prediction"]["path"]), Path(target["segmentation"]["path"]),
    )
    source, ct_source = nib.load(target["segmentation"]["path"]), nib.load(args.ct)
    affine = source.affine
    spacing = nib.affines.voxel_sizes(affine)
    if (source.shape != ct_source.shape or not np.allclose(affine, ct_source.affine)
            or nib.aff2axcodes(affine) != ("R", "A", "S")
            or not np.allclose(affine[:3, :3], np.diag(spacing))):
        raise ValueError("expected the same axis-aligned RAS grid as prior anatomy audit")
    l5 = np.asanyarray(source.dataobj) == 27
    subregions = np.asanyarray(prediction.dataobj)
    ct = np.asanyarray(ct_source.dataobj)
    faces, axes = exposed_faces(l5)
    owners, normal = face_properties(l5, faces, axes)
    starts, ends = select_review_surfaces(faces, normal, owners, sap, tp, old, spacing)
    path, length = shortest_cortical_path(faces, axes, spacing, starts, ends)
    points = nib.affines.apply_affine(affine, faces)
    owner_labels = subregions[tuple(owners[path].T)]
    nearest = distance.cdist(points[starts], points[ends]) if len(starts) and len(ends) else None
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "anatomy_hypotheses.npz",
                        hypothesis_sap_side_points_ras_mm=points[starts],
                        hypothesis_tp_side_points_ras_mm=points[ends],
                        diagnostic_cortex_connection_ras_mm=points[path],
                        diagnostic_cortex_connection_owner_labels=owner_labels)
    np.savez_compressed(args.output / "historical_proxy.npz", points_ras_mm=old.midpoint_world_mm)
    figure = args.output / "external_cortex_review.png"
    review_figure(figure, ct, l5, sap, tp, faces, starts, ends, path, affine,
                  nib.affines.apply_affine(np.linalg.inv(affine), old.midpoint_world_mm))
    if args.prior_ct_figure:
        shutil.copyfile(args.prior_ct_figure, args.output / "historical_ct_review.png")
    geometry = {
        "status": "NOT_EVALUABLE",
        "identity": policy["identity"],
        "anatomy": manifest["anatomy"],
        "coordinate_system": "NIfTI RAS mm",
        "scope": "Evaluator-only anatomy audit. Hypotheses mark hypothesis endpoint surfaces, not an accepted localization region.",
        "used_ultrasound_to_choose_reference": False,
        "accepted_region": None,
        "provenance": anatomy_provenance,
        "literature": PRIMARY_ANATOMY,
        "anatomical_rationale": "The lateral SAP neck and TP root anatomy in dissection supports reviewing both cortical sides. Mammillary and accessory boundaries and the posterior margin of the lateral neck are not labeled in this specimen.",
        "inclusion_rules": {
            "common": "External full L5 exposed-face centers within one source CT voxel diagonal (from frozen historical SAP/TP cores). Source owner must be a corresponding right-side proposal voxel.",
            "sap": "Face normal points right, owner at or below the median superior coordinate of the full right SAP proposal.",
            "tp": "Face normal points superior or medial, owner at or medial to the median rightward coordinate of the full right TP proposal.",
        },
        "voxel_spacing_mm": spacing.tolist(),
        "hypotheses": {
            "sap_side": {"points": len(starts), "array": "hypothesis_sap_side_points_ras_mm"},
            "tp_side": {"points": len(ends), "array": "hypothesis_tp_side_points_ras_mm"},
            "points_file": "anatomy_hypotheses.npz",
        },
        "minimum_hypothesis_separation_mm": float(nearest.min()) if nearest is not None else None,
        "external_cortex_trace_mm": length,
        "external_cortex_trace_source_subregion_counts": dict(Counter(str(int(i)) for i in owner_labels)),
        "historical_proxy": {
            "usage": "Geometric comparison only, no anatomical correctness scoring.",
            "path": "historical_proxy.npz", "point_count": len(old.midpoint_world_mm),
            "midpoints_in_l5": historical_audit["point sets"]["Historical midpoint"]["in_full_L5"],
            "sap_core_interior_at_least_half_voxel":
                historical_audit["point sets"]["SAP core"]["in_L5_interior_at_least_half_voxel"],
        },
        "reason": "The cortex is physically delineated, but opposing direction-and-subregion hypotheses and any shortest cortical connection do not identify a unique lateral neck–root boundary or back the missing anatomy labels. No new cortical target contour is accepted.",
    }
    write_json(args.output / "reference.json", geometry)
    files = sorted(p for p in args.output.iterdir() if p.is_file())
    (args.output / "checksums.sha256").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in files))
    for p in args.output.iterdir():
        p.chmod(0o444)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "prior-protocol", "prior-audit", "ct", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--ct-sha256", required=True)
    parser.add_argument("--prior-ct-figure", type=Path)
    freeze(parser.parse_args())


if __name__ == "__main__":
    main()
