"""Curate matched spatial holdouts and package only actor-visible replay inputs."""

import argparse
import json
from pathlib import Path

import numpy as np

from workflows.mbb_observed_surface.localize import assign_views, distinct_poses, public_axes
from workflows.mbb_shape_completion.fit import grid, mask_disk
from workflows.mbb_shape_completion.predict import digest, dump, load_cloud, read


def checked(entry):
    path = Path(entry["path"])
    if digest(path) != entry["sha256"]:
        raise ValueError(f"modified source: {path}")
    return path


def load_surfaces(folder, rules):
    receipt_path = folder / "receipt.json"
    receipt = read(checked({"path": str(receipt_path),
                            "sha256": receipt_path.with_suffix(".json.sha256").read_text().split()[0]}))
    if not receipt["calibration_parity"]["passes"]:
        raise ValueError("source failed the public projection parity check")
    ledger = checked(receipt["actor_pose_source"])
    frames = [json.loads(line) for line in ledger.read_text().splitlines()][:len(receipt["valid_sequences"])]
    if [f["frame_sequence"] for f in frames] != receipt["valid_sequences"]:
        raise ValueError("ledger and reconstruction validity cohorts differ")
    pose_validity = ("image_valid", "contact_ok", "producer_advanced", "transition_valid")
    if any(not all(f[key] for key in pose_validity) for f in frames):
        raise ValueError("reconstruction cohort contains an invalid acquisition")
    sources = {}
    for mode in ("oracle", "learned"):
        path = checked({"path": str(folder / f"{mode}.npz"),
                        "sha256": receipt[mode]["output"]["npz"]["sha256"]})
        sources[mode] = {"path": str(path), "sha256": digest(path)}
    axes = public_axes(frames, rules)
    if axes is None:
        raise ValueError("could not recover the public Stage 1 anatomical chart")
    right, superior, error = axes
    superior = superior / np.linalg.norm(superior)
    right = right - (right @ superior) * superior
    right /= np.linalg.norm(right)
    basis = np.column_stack((right, superior, np.cross(right, superior)))
    distinct = distinct_poses(frames, rules)
    probe = np.asarray([f["actor_base_to_ee_position_m"] for f in frames]) * 1000
    pool = {}
    for mode, source in sources.items():
        data = load_cloud(Path(source["path"]))
        f = data["frame_sequence"]
        if np.any((f < 0) | (f >= len(frames))) or not np.isfinite(data["xyz_mm"]).all():
            raise ValueError("invalid point provenance")
        views, inverse = np.unique(f, return_inverse=True)
        data.update(view_group=assign_views(views, distinct)[inverse], basis=basis, probe_mm=probe)
        pool[mode] = data
    return pool, {"clouds": sources, "pose_ledger": receipt["actor_pose_source"],
                  "frame_count": len(frames), "axis_fit_residual_index": error,
                  "reconstruction_receipt_sha256": digest(receipt_path)}


def subset(data, keep):
    # Pose arrays and calibration belong to the full session, not to a cloud row.
    return {key: value if key in ("basis", "probe_mm") else value[keep]
            for key, value in data.items()}


def choose_center(data, identity, cfg):
    take = (data["identity"] == identity) & (data["level_hint"] == "L5")
    points = data["xyz_mm"][take] @ data["basis"][:, :2]
    views = data["view_group"][take]
    cell = cfg["center_grid_mm"]
    centers = np.unique(np.rint(points / cell).astype(int), axis=0) * cell
    lo, hi = cfg["screen_annulus_mm"]
    best = None
    for center in centers:
        delta = points - center
        radius = np.linalg.norm(delta, axis=1)
        if any((radius <= d / 2 + 1e-9).sum() < cfg["minimum_hidden_points_for_curator"][str(d)]
               for d in cfg["gap_diameters_mm"]):
            continue
        ring = (radius > lo) & (radius <= hi)
        n_views = len(np.unique(views[ring]))
        if n_views < cfg["minimum_largest_annulus_views"]:
            continue
        angles = np.arctan2(delta[ring, 1], delta[ring, 0])
        sectors = np.bincount((np.floor((angles + np.pi) / (np.pi / 4)).astype(int)) % 8,
                              minlength=8)
        if np.count_nonzero(sectors) < cfg["minimum_largest_annulus_sectors"]:
            continue
        rank = (np.count_nonzero(sectors), n_views, np.minimum(sectors, 4).sum())
        if best is None or rank > best[0]:
            best = (rank, center)
    return None if best is None else best[1]


def controls():
    """Geometric controls with zero access to measured anatomy or CT."""
    axis = np.arange(-10, 10.1, .75)
    uv = grid([(axis[0], axis[-1])] * 2, .75)
    surface = np.column_stack((uv, .015 * (uv ** 2).sum(axis=1)))
    stacked = np.concatenate((surface, surface + [0, 0, 6]))
    flat = np.column_stack((uv, np.zeros(len(uv))))
    inconsistent = flat.copy()
    inconsistent[:, 2] = 5 * np.sin(1.2 * uv[:, 0]) * np.cos(uv[:, 1])
    disconnected = surface[np.abs(uv[:, 0]) > 6]
    samples = {
        "oversized_gap_14mm": (surface, 14., 5, False),
        "unsupported_connection": (disconnected, 5., 5, False),
        "competing_surface_layers": (stacked, 5., 5, True),
        "inconsistent_surface": (inconsistent, 5., 5, False),
        "repeated_view": (surface, 5., 1, False),
    }
    for name, (points, diameter, n_views, conflict) in samples.items():
        n = len(points)
        data = {"xyz_mm": points, "identity": np.full(n, "RIGHT_L5_TP", dtype="U20"),
                "level_hint": np.full(n, "L5"), "basis": np.eye(3),
                "probe_mm": np.tile([0., 0., 70.], (n_views, 1)),
                "frame_sequence": np.arange(n) % n_views,
                "view_group": np.arange(n) % n_views,
                "semantic_conflict": np.r_[np.zeros(n // 2, dtype=bool), np.full(n - n // 2, conflict)]}
        # Competing nearby level labels alone cannot resolve two same-level TP
        # candidates. Keep both ambiguous hypotheses in the geometric input.
        hidden = mask_disk(points[:, :2], [0, 0], diameter / 2)
        yield name, diameter, subset(data, ~hidden)


def run(protocol, output, experiment, rules):
    sources = protocol["sources"]
    root = Path(sources["artifact_root"])
    file = "recorded_reconstruction" if experiment == "A" else "fresh_reconstruction"
    checked(sources[file])
    if experiment == "B":
        checked(sources["recorded_reconstruction"])
    forbidden = set(protocol["reserved_seeds"])
    seeds = protocol["recorded_seeds"]
    if set(seeds) & forbidden:
        raise ValueError("reserved seed overlap")
    sessions = [(str(seed), root / "reconstruction" / str(seed)) for seed in seeds]
    if experiment == "B":
        sessions.append((f"{seeds[0]}-126", root / "reconstruction-fresh" / str(seeds[0])))
    actor, withheld = output / "inputs", output / "heldout"
    actor.mkdir(parents=True, exist_ok=False)
    withheld.mkdir()
    requests, sources_read, case_table = {}, {}, {}
    mask = protocol["masks"]

    def save(key, data, definition):
        path = actor / f"{key}.npz"
        np.savez_compressed(path, **data)
        requests[key] = {**definition, "experiment": experiment, "sha256": digest(path)}

    for session, source in sessions:
        pool, sources_read[session] = load_surfaces(source, rules)
        if experiment == "B":
            for mode, data in pool.items():
                save(f"{session}-{mode}", data, {"session": session, "perception": mode})
            continue
        for identity in mask["classes"]:
            center = choose_center(pool["oracle"], identity, mask)
            if center is None:
                case_table[f"{session}-{identity}"] = {"status": "NO_CURATOR_SITE"}
                continue
            for gap in mask["gap_diameters_mm"]:
                ids = f"{session}-{identity}-{gap:g}mm"
                for mode, data in pool.items():
                    key = f"{ids}-{mode}"
                    removed = mask_disk(data["xyz_mm"] @ data["basis"][:, :2], center, gap / 2)
                    hidden = subset(data, removed)
                    truth_path = withheld / f"{key}.npz"
                    np.savez_compressed(truth_path, **hidden)
                    case_table[key] = {"heldout_sha256": digest(truth_path),
                                       "removed_all_labels": int(removed.sum())}
                    save(key, subset(data, ~removed),
                         {"session": session, "perception": mode, "identity": identity,
                          "gap_mm": gap, "center_uv_mm": center.tolist(),
                          "context_radius_mm": mask["screen_annulus_mm"][1]})
    if experiment == "A":
        for name, gap, data in controls():
            save(f"negative-{name}", data, {"perception": "synthetic", "gap_mm": gap,
                                          "identity": "RIGHT_L5_TP", "center_uv_mm": [0., 0.],
                                          "context_radius_mm": mask["screen_annulus_mm"][1]})
    dump(output / "source_receipts.json", sources_read)
    dump(output / "curator_receipt.json", case_table)
    dump(actor / "requests.json", requests)
    print(f"Prepared {len(requests)} sealed {experiment} requests.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--phase", choices=("A", "B"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view-rules", type=Path, default=Path("workflows/mbb_observed_surface/actor_rules.json"))
    args = parser.parse_args()
    if args.protocol.with_suffix(".sha256").read_text().split()[0] != digest(args.protocol):
        raise ValueError("invalid frozen protocol")
    run(read(args.protocol), args.output, args.phase, read(args.view_rules))
