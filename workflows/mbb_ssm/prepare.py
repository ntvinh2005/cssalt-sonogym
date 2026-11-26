"""Audit complete, unique L5 masks and make a private subject-level split."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import nibabel as nib
import numpy as np
import pyvista as pv
from scipy import ndimage

from workflows.mbb_ssm.geometry import Mesh, face_components, save_mesh


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_mesh(path, spacing, label=None):
    nii = nib.load(path)
    raw = np.asanyarray(nii.dataobj)
    if label is None and (raw.min() < 0 or raw.max() > 1):
        raise ValueError("Non-binary L5 mask")
    mask = raw == (1 if label is None else label)
    sites = ndimage.find_objects(mask)
    if not sites or not sites[0]:
        raise ValueError("Empty L5")
    box = sites[0]
    if any(s.start == 0 or s.stop == n for s, n in zip(box, mask.shape)):
        raise ValueError("Touches image boundary")
    cut = mask[box]
    voxel_mm = nib.affines.voxel_sizes(nii.affine)
    if voxel_mm.max() > 3.:
        raise ValueError("Voxel spacing above 3 mm")
    components, count = ndimage.label(cut)
    populations = np.bincount(components.ravel(), minlength=count + 1)
    populations[0] = 0
    fraction = populations.max() / populations.sum()
    if fraction < .98:
        raise ValueError("Disconnected L5 mask")
    cut = components == populations.argmax()
    physical_volume = cut.sum() * abs(np.linalg.det(nii.affine[:3, :3]))
    if not (8000 <= physical_volume <= 70000):
        raise ValueError("Physical volume outside 8–70 cm^3")

    start = np.array([s.start for s in box])
    end = np.array([s.stop - 1 for s in box])
    corners = np.array(np.meshgrid(*zip(start, end))).reshape(3, -1).T
    extent = nib.affines.apply_affine(nii.affine, corners)
    lo = extent.min(axis=0) - 2 * spacing
    hi = extent.max(axis=0) + 2 * spacing
    affine_inverse = np.linalg.inv(nii.affine)
    shape = tuple(np.ceil((hi - lo) / spacing).astype(int) + 1)
    samples = ndimage.affine_transform(cut.astype(np.float32), affine_inverse[:3, :3] * spacing,
                                        offset=nib.affines.apply_affine(affine_inverse, lo) - start,
                                        output_shape=shape, order=1, mode="grid-constant", cval=0.)
    volume = pv.ImageData(dimensions=shape, spacing=(spacing,) * 3, origin=lo)
    volume.point_data["mask"] = samples.ravel(order="F")
    contour = volume.contour([.5], method="flying_edges").clean().triangulate()
    points = contour.points.astype(np.float64)
    faces = contour.faces.reshape(-1, 4)[:, 1:].astype(int)
    if len(points) < 1000:
        raise ValueError("Insufficient surface vertices")
    _, vertex_labels = face_components(Mesh(points, faces))
    biggest = np.bincount(vertex_labels).argmax()
    keep = vertex_labels == biggest
    if keep.mean() < .98:
        raise ValueError("Contour components exceed 2% of surface")
    faces = (np.cumsum(keep) - 1)[faces[keep[faces].all(axis=1)]]
    points = points[keep]
    extent = np.ptp(points, axis=0)
    if np.any(extent < [44., 38., 20.]) or np.any(extent > [110., 100., 61.]):
        raise ValueError("Implausible or truncated L5 extents")
    candidate = pv.PolyData(points, np.column_stack((np.full(len(faces), 3), faces)).ravel())
    if candidate.n_open_edges:
        raise ValueError(f"Contour is not watertight ({candidate.n_open_edges} open edges)")
    candidate = candidate.compute_normals(point_normals=False, auto_orient_normals=True, consistent_normals=True)
    full = Mesh(points, candidate.faces.reshape(-1, 4)[:, 1:].astype(int))
    fingerprint = hashlib.sha256(np.round(points - points.mean(0), 3).tobytes()).hexdigest()
    return full, {"mask_spacing_mm": voxel_mm.tolist(), "volume_mm3": float(physical_volume),
                  "extent_ras_mm": extent.tolist(), "mesh_vertices": len(points),
                  "mesh_faces": len(faces), "connected_fraction": float(fraction),
                  "same_grid_shape_fingerprint": fingerprint}


def read_prior_splits(path, ids, n_eval):
    splits = json.loads(path.read_text())["split"]
    used = set().union(*map(set, splits.values()))
    if not used.issubset(ids) or sum(map(len, splits.values())) != len(used):
        raise ValueError("Initial split differs from expanded eligibility")
    remaining = [subject for subject in ids if subject not in used]
    for name in ("held_out", "development"):
        take = max(0, n_eval - len(splits[name]))
        splits[name].extend(remaining[:take])
        remaining = remaining[take:]
    splits["train"].extend(remaining)
    return splits


def run(manifest, output, protocol, prior_split=None):
    source = json.loads(manifest.read_text())
    if "s0010" not in source["excluded_ids"]:
        raise ValueError("s0010 must be excluded before preparation")
    output.mkdir(parents=True, exist_ok=False)
    store = output / "meshes"
    store.mkdir()
    info, accepted, signatures = {}, [], set()
    start = perf_counter()
    for entry in source["entries"]:
        subject = entry["subject_id"]
        if subject in source["excluded_ids"] or subject in info:
            raise ValueError(f"Unexpected duplicate or excluded subject {subject}")
        path = Path(entry["mask"])
        if digest(path) != entry["sha256"]:
            raise ValueError("Mask digest mismatch")
        try:
            mesh, quality = make_mesh(path, protocol["mesh_spacing_mm"], entry.get("label"))
            signature = quality["same_grid_shape_fingerprint"]
            if signature in signatures:
                raise ValueError("Duplicated mesh fingerprint")
        except ValueError as exc:
            info[subject] = {"accepted": False, "reason": str(exc)}
            continue
        signatures.add(signature)
        save_mesh(store / f"{subject}.npz", mesh)
        info[subject] = {**quality, "accepted": True, "mesh_sha256": digest(store / f"{subject}.npz")}
        accepted.append(subject)
    # Selection is deterministic and occurs before population registration and PCA.
    ids = sorted(accepted, key=lambda s: hashlib.sha256((protocol["split_salt"] + "/" + s).encode()).hexdigest())
    if len(ids) < 8:
        raise ValueError(f"Only {len(ids)} usable anatomies, require eight for the requested 3-way comparison")
    n_eval = max(2, round(len(ids) * .15))
    if prior_split is None:
        splits = {"train": ids[2 * n_eval:], "development": ids[n_eval:2 * n_eval], "held_out": ids[:n_eval]}
    else:
        splits = read_prior_splits(prior_split, ids, n_eval)
    hashes = {"source_manifest_sha256": digest(manifest), "protocol_name": protocol["name"]}
    (output / "subjects.json").write_text(json.dumps({"subjects": info, "split": splits,
         "excluded": source["excluded_ids"], **hashes, "seconds": perf_counter() - start}, indent=2) + "\n")
    print({name: len(group) for name, group in splits.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prior-split", type=Path)
    args = parser.parse_args()
    run(args.manifest, args.output, json.loads(args.protocol.read_text()), args.prior_split)
