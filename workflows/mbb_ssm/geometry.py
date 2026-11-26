"""Geometry for population registration and partial-surface fitting in millimetres."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


@dataclass
class Mesh:
    points: np.ndarray
    faces: np.ndarray


def load_mesh(path):
    with np.load(path, allow_pickle=False) as data:
        return Mesh(data["points"], data["faces"])


def save_mesh(path, mesh):
    np.savez_compressed(path, points=mesh.points, faces=mesh.faces)


def export_mesh(path, mesh, color=None):
    with path.open("x") as stream:
        stream.write("ply\nformat ascii 1.0\ncomment physical millimetres\n")
        stream.write(f"element vertex {len(mesh.points)}\nproperty float x\nproperty float y\nproperty float z\n")
        if color is not None:
            stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write(f"element face {len(mesh.faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        if color is None:
            np.savetxt(stream, mesh.points, fmt="%.5f")
        else:
            np.savetxt(stream, np.column_stack((mesh.points, color)), fmt="%.5f %.5f %.5f %d %d %d")
        np.savetxt(stream, np.column_stack((np.full(len(mesh.faces), 3), mesh.faces)), fmt="%d")


def edges(faces):
    return np.unique(np.sort(np.vstack((faces[:, :2], faces[:, 1:], faces[:, ::2])), axis=1), axis=0)


def face_vectors(mesh):
    triangles = mesh.points[mesh.faces]
    return np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])


def normals(mesh):
    vectors = face_vectors(mesh)
    result = np.zeros_like(mesh.points)
    for col in mesh.faces.T:
        np.add.at(result, col, vectors)
    return result / np.maximum(np.linalg.norm(result, axis=1)[:, None], 1e-10)


def face_components(mesh):
    graph = sparse.coo_matrix((np.ones(mesh.faces.size), (mesh.faces[:, 0].repeat(3), mesh.faces.ravel())),
                              shape=(len(mesh.points), len(mesh.points)))
    return connected_components(graph.tocsr(), directed=False)


def nearest_points(points, mesh, center_tree=None):
    """Closest points on nearby triangles, using triangle vertices and edges.

    This uses the nearest triangle centers to keep repeated fitting inexpensive.
    The mesh spacing and a separate nearest-vertex bound are audited.
    """
    from trimesh.triangles import closest_point

    triangles = mesh.points[mesh.faces]
    if center_tree is None:
        center_tree = cKDTree(triangles.mean(axis=1))
    k = min(12, len(triangles))
    near = center_tree.query(points, k=k)[1].reshape(len(points), k)
    candidate = closest_point(triangles[near.ravel()], np.repeat(points, k, axis=0)).reshape(len(points), k, 3)
    distance2 = np.sum((candidate - points[:, None]) ** 2, axis=2)
    best = np.argmin(distance2, axis=1)
    return candidate[np.arange(len(points)), best], near[np.arange(len(points)), best]


def rigid(source, target, weights=None):
    """Fit a proper row-vector rotation and a translation while preserving scale."""
    if weights is None:
        weights = np.ones(len(source))
    w = weights / weights.sum()
    a, b = w @ source, w @ target
    u, _, v = np.linalg.svd(((source - a) * w[:, None]).T @ (target - b))
    sign = np.eye(3)
    sign[-1, -1] = np.linalg.det(u @ v)
    rotation = u @ sign @ v
    return rotation, b - a @ rotation


def posterior(points, fraction=.6):
    return points[:, 1] <= np.quantile(points[:, 1], fraction)


def distance_metrics(values, tolerances):
    values = np.asarray(values, float)
    if not len(values):
        return {"n": 0, "median_mm": None, "mean_mm": None, "p95_mm": None,
                "coverage": {str(v): 0. for v in tolerances}}
    return {"n": len(values), "median_mm": float(np.median(values)), "mean_mm": float(np.mean(values)),
            "p95_mm": float(np.quantile(values, .95)),
            "coverage": {str(v): float(np.mean(values <= v)) for v in tolerances}}
