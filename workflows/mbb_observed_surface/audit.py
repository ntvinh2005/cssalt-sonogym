"""Compare the archived midpoint reference with the full L5 CT segmentation."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from run_s0010_l4_medial_branch_target import load_target_geometry, sha256


COLORS = {"SAP": "#d87816", "TP": "#18856d", "SAP core": "#ae3a66",
          "TP core": "#1666a3", "Historical midpoint": "#c71928"}
BANDS = {"SAP band": "#9b65a0", "TP band": "#567da6"}


def exposed_faces(mask):
    indices = np.argwhere(mask)
    padded = np.pad(mask, 1)
    centers, axes = [], []
    for axis in range(3):
        for sign in (-1, 1):
            offset = np.eye(3, dtype=int)[axis] * sign
            adjacent = padded[tuple((indices + 1 + offset).T)]
            centers.extend(indices[~adjacent] + offset / 2)
            axes.extend([axis] * int((~adjacent).sum()))
    return np.asarray(centers), np.asarray(axes)


def distance_to_faces(indices, centers, axes, spacing):
    distance = np.abs(indices[:, None] - centers)
    half_width = .5 * (np.arange(3) != axes[:, None])
    return np.linalg.norm(np.maximum(distance - half_width, 0) * spacing, axis=2).min(axis=1)


def sample(array, points, order=0):
    return ndimage.map_coordinates(array, points.T, order=order, mode="constant", cval=0)


def draw_ct(output, ct, l5, labeled, points, affine):
    center = np.rint(points["Historical midpoint"].mean(0)).astype(int)
    extent = 15
    fig, panels = plt.subplots(3, 3, figsize=(12, 11), layout="constrained")
    for axis, name in enumerate(("Sagittal", "Coronal", "Axial")):
        h, v = [i for i in range(3) if i != axis]
        for panel, offset in zip(panels[axis], (-2, 0, 2)):
            index = center[axis] + offset
            x = slice(center[h] - extent, center[h] + extent + 1)
            y = slice(center[v] - extent, center[v] + extent + 1)
            slc = [None] * 3
            slc[axis], slc[h], slc[v] = index, x, y
            slc = tuple(slc)
            grid = (x.start - .5, x.stop - .5, y.start - .5, y.stop - .5)
            panel.imshow(ct[slc].T, cmap="gray", vmin=-150, vmax=750, origin="lower", extent=grid)
            for key, mask in {"Full L5": l5, **labeled}.items():
                local = mask[slc].T
                if local.any() and not local.all():
                    panel.contour(local, levels=[.5], origin="lower", extent=grid,
                                  colors=[COLORS.get(key, "#e5e5e5")], linewidths=1.3)
            for key in ("SAP core", "TP core", "Historical midpoint"):
                coords = points[key]
                selected = np.abs(coords[:, axis] - index) <= .5 + 1e-8
                panel.scatter(coords[selected, h], coords[selected, v], c=COLORS[key],
                              marker="x" if key == "Historical midpoint" else "o", s=34)
            reference_index = center.copy()
            reference_index[axis] = index
            position = nib.affines.apply_affine(affine, reference_index)[axis]
            panel.set_title(f"{name} · {'RAS'[axis]} {position:.1f} mm")
            panel.xaxis.set_major_formatter(FuncFormatter(
                lambda x, _, dim=h: f"{x * affine[dim, dim] + affine[dim, 3]:.0f}"))
            panel.yaxis.set_major_formatter(FuncFormatter(
                lambda y, _, dim=v: f"{y * affine[dim, dim] + affine[dim, 3]:.0f}"))
            panel.set(xlabel=f"{'RAS'[h]} (mm) →", ylabel=f"{'RAS'[v]} (mm) ↑",
                      xlim=grid[:2], ylim=grid[2:])
    legend = [Line2D([0], [0], color="#777777", label="Full L5 boundary")]
    legend += [Line2D([0], [0], color=c, label=k) for k, c in COLORS.items()]
    fig.legend(handles=legend, loc="outside lower center", ncol=6, frameon=False)
    fig.suptitle("s0010 CT · archived semantically defined cores and midpoint samples")
    fig.savefig(output / "ct_review.png", dpi=160)
    plt.close(fig)


def draw_3d(output, faces, axes, masks, points, affine, pairs):
    corners = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
    polygons = np.repeat(faces[:, None], 4, axis=1)
    for axis in range(3):
        same = axes == axis
        for column, dim in enumerate(i for i in range(3) if i != axis):
            polygons[same, :, dim] += corners[:, column]
    surfaces = nib.affines.apply_affine(affine, polygons)
    world = {key: nib.affines.apply_affine(affine, p) for key, p in points.items()}
    near = np.linalg.norm(surfaces.mean(1) - world["Historical midpoint"].mean(0), axis=1) <= 21
    fig = plt.figure(figsize=(15, 7), layout="constrained")
    for panel, pooled, elev, azim in ((1, np.ones(len(faces), dtype=bool), 28, -138),
                                       (2, near, 38, -34)):
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        ax.add_collection3d(Poly3DCollection(surfaces[pooled], facecolor="#777f8a",
                                             alpha=.11 if panel == 1 else .09, linewidth=0))
        center = world["Historical midpoint"].mean(0)
        for key, xyz in world.items():
            if key in masks and panel == 2:
                xyz = xyz[np.linalg.norm(xyz - center, axis=1) < 21]
            if key in BANDS:
                if panel == 2:
                    ax.scatter(*xyz.T, facecolors="none", edgecolors=BANDS[key], s=32, linewidths=.9,
                               label=f"Historical {key}", depthshade=False)
                continue
            ax.scatter(*xyz.T, c=COLORS[key], label=key, depthshade=False,
                       alpha=.35 if key in masks else .94, s=6 if key in masks else 26,
                       marker="x" if key == "Historical midpoint" else "o")
        if panel == 2:
            for pair in pairs:
                ax.plot(*pair.T, linewidth=.7, c="#424242", alpha=.45)
        limits = surfaces[pooled].reshape(-1, 3)
        ax.auto_scale_xyz(*limits.T)
        ax.set_box_aspect(np.ptp(limits, axis=0))
        ax.view_init(elev=elev, azim=azim)
        ax.set(xlabel="R (mm)", ylabel="A (mm)", zlabel="S (mm)",
               title="Full L5 in RAS" if panel == 1 else "Local L5 cortex and nearest-core segments")
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.savefig(output / "l5_review_3d.png", dpi=160)
    plt.close(fig)


def report(manifest, ct_path, ct_sha, output):
    data = json.loads(manifest.read_text())
    assert sha256(manifest) == manifest.with_suffix(".json.sha256").read_text().split()[0]
    target = data["target"]
    for key in ("prediction", "segmentation", "reference_protocol"):
        assert sha256(Path(target[key]["path"])) == target[key]["sha256"], key
    assert sha256(ct_path) == ct_sha
    prediction, sap, tp, contract = load_target_geometry(Path(target["prediction"]["path"]),
                                                         Path(target["segmentation"]["path"]))
    seg_image = nib.load(target["segmentation"]["path"])
    ct_image = nib.load(ct_path)
    if ct_image.shape != seg_image.shape or not np.allclose(ct_image.affine, seg_image.affine):
        raise ValueError("CT and full-L5 segmentation must share the verified grid")
    affine = seg_image.affine
    spacing = nib.affines.voxel_sizes(affine)
    if nib.aff2axcodes(affine) != ("R", "A", "S") or not np.allclose(affine[:3, :3], np.diag(spacing)):
        raise ValueError("expected the audited axis-aligned RAS grid")
    l5 = np.asanyarray(seg_image.dataobj) == 27
    subregions = np.asanyarray(prediction.dataobj)
    ct = np.asanyarray(ct_image.dataobj).astype(np.float32)
    labeled = {"SAP": sap, "TP": tp}
    points = {key: np.argwhere(mask) for key, mask in
              {**labeled, "SAP core": contract.sap_core, "TP core": contract.tp_core}.items()}
    midpoint = nib.affines.apply_affine(np.linalg.inv(affine), contract.midpoint_world_mm)
    points["Historical midpoint"] = midpoint
    faces, axes = exposed_faces(l5)
    sap_core, tp_core = points["SAP core"], points["TP core"]
    pairs_voxel = np.unique(np.concatenate((
        np.stack((sap_core, tp_core[cKDTree(tp_core * spacing).query(sap_core * spacing)[1]]), axis=1),
        np.stack((sap_core[cKDTree(sap_core * spacing).query(tp_core * spacing)[1]], tp_core), axis=1),
    )), axis=0)
    sites = {}
    # Test membership in the external cortex using voxels with a six-neighbour exposed face.
    external = l5 & ~ndimage.binary_erosion(l5)
    for key, xyz in points.items():
        d = distance_to_faces(xyz, faces, axes, spacing)
        inside = sample(l5, xyz)
        role = xyz * spacing
        sites[key] = {"point_count": len(xyz), "in_full_L5": int(inside.sum()),
                      "in_L5_interior_at_least_half_voxel": int((inside & (d > spacing.min() / 2 + 1e-6)).sum()),
                      "in_external_boundary_voxel": int(sample(external, xyz).sum()),
                      "distance_to_external_faces_mm": d.tolist(),
                      "physical_span_mm": np.ptp(role, axis=0).tolist(),
                      "ct_hu_linear": sample(ct, xyz, order=1).tolist(),
                      "coordinates_ras_mm": nib.affines.apply_affine(affine, xyz).tolist()}
        if key == "Historical midpoint":
            sites[key]["source_prediction_labels"] = sample(subregions, xyz).tolist()
    segments = []
    for endpoints in pairs_voxel:
        length = float(np.linalg.norm((endpoints[1] - endpoints[0]) * spacing))
        fractions = np.linspace(.1, .9, 1 + int(np.ceil(length / .25)))
        line = endpoints[0] + fractions[:, None] * (endpoints[1] - endpoints[0])
        inside = sample(l5, line)
        classes = sample(subregions, line)
        names, counts = np.unique(classes[inside], return_counts=True)
        segments.append({"length_mm": length, "central_samples": len(line),
                         "in_full_L5": int(inside.sum()),
                         "in_L5_outside_SAP_and_TP": int((inside & ~np.isin(classes, [5, 6])).sum()),
                         "in_L5_model_label_counts": {int(k): int(n) for k, n in zip(names, counts)},
                         "endpoints_ras_mm": nib.affines.apply_affine(affine, endpoints).tolist()})
    output.mkdir(parents=True, exist_ok=False)
    draw_ct(output, ct, l5, labeled, points, affine)
    bands = {"SAP band": np.argwhere(contract.sap_uncertain & ~contract.sap_core),
             "TP band": np.argwhere(contract.tp_uncertain & ~contract.tp_core)}
    draw_3d(output, faces, axes, labeled, points | bands, affine,
            nib.affines.apply_affine(affine, pairs_voxel))
    result = {"anatomy": data["anatomy"], "method": "historical core comparison against full L5",
              "input_sha256": {"ct": ct_sha, "segmentation": target["segmentation"]["sha256"],
                               "prediction": target["prediction"]["sha256"], "input_manifest": sha256(manifest)},
              "frame": "NIfTI RAS mm", "voxel_size_mm": spacing.tolist(), "l5_voxels": int(l5.sum()),
              "minimum_surface_separation_mm": contract.minimum_surface_separation_mm,
              "point sets": sites, "central_segments": segments}
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--ct-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.manifest, args.ct, args.ct_sha256, args.output)


if __name__ == "__main__":
    main()
