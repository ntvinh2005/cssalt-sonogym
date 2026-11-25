"""Build a four-landmark map from pose-synchronized cortical observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist

from .incremental_landmark_map import LANDMARK_IDENTITIES, IncrementalLandmarkMapConfig
from .sonoanatomy_multiview import SemanticObservation
from .stage2_tp_active import group_a_target_handoff_pose
from .ultrasound_spatial_map import calibrated_image_coordinates_mm


CLASSES = ("BACKGROUND", "TP", "SAP", "OTHER_BONE")
LEVELS = ("L4", "L5")


@dataclass(frozen=True)
class Stage1Region:
    home_x: float
    l4_z: float
    l5_z: float
    ambiguous_z: tuple[float, float]

    def levels_at(self, executed_z: float) -> tuple[str, ...]:
        low, high = self.ambiguous_z
        if executed_z > high:
            return ("L4",)
        if executed_z < low:
            return ("L5",)
        return LEVELS


def stage1_region(
    actor_result: Mapping, session_id: str, protocol: Mapping
) -> Stage1Region:
    geometry = protocol["geometry"]
    resolution = geometry["label_resolution_mm"]
    home_z = geometry["home_z_index"]
    l4, l5 = (
        group_a_target_handoff_pose(
            actor_result, session_id=session_id, target_level=level,
            home_z_index=home_z, label_resolution_mm=resolution,
            stage2_angle_rad=protocol["sonoanatomy_scan"]["longitudinal_angle_rad"],
        )
        for level in LEVELS
    )
    if l4[1] <= l5[1]:
        raise ValueError("Stage 1 L4/L5 references are out of order")
    session = next(row for row in actor_result["results"] if row["session_id"] == session_id)
    intervals = {
        row["anatomy"]: row["actor_uncertainty_interval_mm"]
        for row in session["predictions"] if row["anatomy"] in LEVELS
    }
    midpoint_bounds = home_z + np.mean([intervals[level] for level in LEVELS], axis=0) / resolution
    half_spacing = protocol["sonoanatomy_scan"]["translation_step_index"] / 2.0
    return Stage1Region(
        float(l4[0]), float(l4[1]), float(l5[1]),
        (float(midpoint_bounds[0] - half_spacing), float(midpoint_bounds[1] + half_spacing)),
    )


def systematic_sweep(
    region: Stage1Region,
    scan: Mapping,
    command_minimum: Sequence[float],
    command_maximum: Sequence[float],
) -> list[tuple[str, tuple[float, float, float]]]:
    spacing = float(scan["translation_step_index"])
    lateral = region.home_x + np.arange(2.0, 20.1, spacing)
    longitudinal = float(scan["longitudinal_angle_rad"])
    transverse = float(scan["transverse_angle_rad"])
    waypoints = [
        ("longitudinal", (float(x), z, longitudinal))
        for center, z in enumerate((region.l5_z, region.l4_z))
        for x in (lateral if center == 0 else lateral[::-1])
    ]
    low, high = region.l5_z + min(scan["reference_offsets_index"]), (
        region.l4_z + max(scan["reference_offsets_index"])
    )
    cranial = np.linspace(low, high, int(np.ceil((high - low) / spacing)) + 1)
    waypoints.extend(
        ("transverse", (float(x), float(z), transverse))
        for line, x in enumerate(lateral[1::2])
        for z in (cranial if line % 2 == 0 else cranial[::-1])
    )
    poses = np.asarray([pose for _, pose in waypoints])
    if (not np.all(np.isfinite(poses)) or np.any(poses < command_minimum)
            or np.any(poses > command_maximum)):
        raise ValueError("Stage 1 systematic sweep leaves the controller workspace")
    return waypoints


def add_outer_transverse_samples(
    waypoints: Sequence[tuple[str, tuple[float, float, float]]],
    region: Stage1Region,
    reference_offsets: Sequence[float],
    minimum_separation_index: float,
) -> list[tuple[str, tuple[float, float, float]]]:
    outer_x = max(pose[0] for phase, pose in waypoints if phase == "transverse")
    low = region.l4_z + min(reference_offsets)
    high = region.l4_z + max(reference_offsets)
    result = []
    for left, right in zip(waypoints, waypoints[1:]):
        phase, (x, z, angle) = left
        next_phase, (other_x, other_z, _) = right
        result.append(left)
        midpoint = (z + other_z) / 2
        if (phase == next_phase == "transverse" and x == other_x == outer_x
                and low <= midpoint <= high
                and abs(other_z - z) / 2 > minimum_separation_index):
            result.append((phase, (x, midpoint, angle)))
    result.append(waypoints[-1])
    return result


@dataclass(frozen=True)
class CranialCoordinate:
    direction: tuple[float, float, float]
    intercept_index: float
    maximum_fit_residual_index: float
    resolution_mm: float

    def index_at(self, point_mm: Sequence[float]) -> float:
        return float(np.dot(self.direction, point_mm) / self.resolution_mm + self.intercept_index)


def transverse_cranial_coordinate(
    frames: Sequence[Mapping], resolution_mm: float, maximum_residual_index: float,
) -> CranialCoordinate | None:
    public = [frame for frame in frames if frame["image_valid"] and frame["phase"] == "transverse"]
    if len(public) < 3:
        return None
    commands = np.asarray([frame["executed_relative_pose_index"][:2] for frame in public])
    origins = np.asarray([project_image(frame, (1, 1), 1.0)[0, 0] for frame in public])
    directions, _, rank, _ = np.linalg.lstsq(commands - commands.mean(0), origins - origins.mean(0), rcond=None)
    if rank < 2 or np.linalg.norm(directions[0]) <= np.finfo(float).eps:
        return None
    lateral = directions[0] / np.linalg.norm(directions[0])
    cranial = directions[1] - np.dot(directions[1], lateral) * lateral
    magnitude = np.linalg.norm(cranial)
    if magnitude <= np.finfo(float).eps:
        return None
    cranial /= magnitude
    intercept = commands[:, 1].mean() - np.dot(origins.mean(0), cranial) / resolution_mm
    residual = float(np.max(np.abs(origins @ cranial / resolution_mm + intercept - commands[:, 1])))
    if residual > maximum_residual_index:
        return None
    return CranialCoordinate(tuple(cranial), float(intercept), residual, resolution_mm)


def sap_level_at(
    coordinate_mm: Sequence[float], region: Stage1Region, calibration: CranialCoordinate | None,
) -> tuple[float | None, tuple[str, ...]]:
    if calibration is None:
        return None, LEVELS
    index = calibration.index_at(coordinate_mm)
    margin = (region.ambiguous_z[1] - region.ambiguous_z[0]) / 2 + calibration.maximum_fit_residual_index
    if index < region.l5_z - margin:
        return index, ()
    if index <= region.l5_z + margin:
        return index, LEVELS
    if index < region.l4_z - margin:
        return index, ("L5",)
    if index <= region.l4_z + margin:
        return index, LEVELS
    return index, ("L4",)


def project_image(frame: Mapping, shape: tuple[int, int], pixel_mm: float) -> np.ndarray:
    rotation = np.asarray(frame["actor_base_to_ee_rotation_columns"])
    return calibrated_image_coordinates_mm(
        end_effector_position_m=frame["actor_base_to_ee_position_m"],
        rotation_columns=rotation, width=shape[1], depth=shape[0],
        pixel_resolution_mm=pixel_mm, image_origin_height_mm=133.0,
        home_up_axis=rotation[:, 2],
    )


def extract_observations(
    winner: np.ndarray,
    probabilities: np.ndarray | None,
    thresholds: Mapping,
    frame: Mapping,
    coordinates: np.ndarray,
) -> tuple[list[SemanticObservation], np.ndarray]:
    mask = np.zeros(winner.shape, dtype=np.uint8)
    observations = []
    position = tuple(frame["actor_base_to_ee_position_m"])
    rotation = tuple(tuple(row) for row in frame["actor_base_to_ee_rotation_columns"])
    for class_id, name in enumerate(CLASSES[1:], start=1):
        eligible = winner == class_id
        if probabilities is None:
            probability = np.ones(winner.shape)
            minimum_pixels = 1
        else:
            specification = thresholds[name]
            probability = probabilities[class_id]
            eligible &= probability >= specification["threshold"]
            minimum_pixels = specification["minimum_pixels"]
        components, count = ndimage.label(eligible, structure=np.ones((3, 3)))
        for component_id, bounds in enumerate(ndimage.find_objects(components, count), start=1):
            rows, columns = np.where(components[bounds] == component_id)
            if len(rows) < minimum_pixels:
                continue
            pixels = np.column_stack((rows + bounds[0].start, columns + bounds[1].start))
            pixel_index = tuple(pixels.T)
            mask[pixel_index] = class_id
            if name == "OTHER_BONE":
                continue
            weights = probability[pixel_index]
            center = np.average(pixels, axis=0, weights=weights)
            row, column = pixels[np.argmin(np.sum((pixels - center) ** 2, axis=1))]
            observations.append(SemanticObservation(
                semantic_class=name,
                frame_sequence=int(frame["frame_sequence"]),
                frame_sha256=str(frame["frame_sha256"]),
                viewpoint_id=str(frame["acquisition_id"]),
                viewpoint_position_m=position,
                viewpoint_rotation_columns=rotation,
                row=int(row), column=int(column),
                coordinate_mm=tuple(coordinates[row, column]),
                confidence=float(np.mean(weights)),
                pixel_count=len(rows),
            ))
    return observations, mask


def distinct_support(
    observations: Sequence[SemanticObservation], config: IncrementalLandmarkMapConfig
) -> list[SemanticObservation]:
    selected = []
    for observation in sorted(observations, key=lambda row: (-row.confidence, row.frame_sequence)):
        position = np.asarray(observation.viewpoint_position_m)
        rotation = np.asarray(observation.viewpoint_rotation_columns)
        for prior in selected:
            if (observation.frame_sequence == prior.frame_sequence
                    or observation.frame_sha256 == prior.frame_sha256):
                break
            translation = 1000 * np.linalg.norm(position - prior.viewpoint_position_m)
            relative_cosine = (np.trace(rotation.T @ prior.viewpoint_rotation_columns) - 1) / 2
            angle = np.degrees(np.arccos(np.clip(relative_cosine, -1, 1)))
            if (translation < config.minimum_view_translation_mm
                    and angle < config.minimum_view_rotation_deg):
                break
        else:
            selected.append(observation)
    return selected


def spatial_clusters(
    observations: Sequence[SemanticObservation], config: IncrementalLandmarkMapConfig
) -> list[dict]:
    if not observations:
        return []
    points = np.asarray([row.coordinate_mm for row in observations])
    groups = np.ones(len(points), dtype=int)
    if len(points) > 1:
        groups = fcluster(linkage(points, method="complete"), config.association_radius_mm,
                          criterion="distance")
    clusters = []
    for group in np.unique(groups):
        members = [observations[index] for index in np.flatnonzero(groups == group)]
        support = distinct_support(members, config)
        locations = np.asarray([row.coordinate_mm for row in support])
        medoid = locations[np.argmin(cdist(locations, locations).sum(axis=1))]
        rms = float(np.sqrt(np.mean(np.sum((locations - medoid) ** 2, axis=1))))
        clusters.append({
            "position_mm": medoid.tolist(),
            "semantic_confidence": float(np.mean([row.confidence for row in support])),
            "rms_dispersion_mm": rms,
            "distinct_viewpoints": len(support),
            "members": len(members),
            "supporting_observations": [asdict(row) for row in support],
        })
    return sorted(clusters, key=lambda row: (
        -row["distinct_viewpoints"], -row["semantic_confidence"], row["rms_dispersion_mm"],
    ))


def simple_map(
    observations: Mapping[str, Sequence[SemanticObservation]],
    ambiguities: Mapping[str, Sequence[SemanticObservation]],
    config: IncrementalLandmarkMapConfig,
) -> dict:
    result = {}
    for identity in LANDMARK_IDENTITIES:
        ambiguous_rows = ambiguities.get(identity, ())
        rows = [row for row in observations.get(identity, ())
                if row.confidence >= config.minimum_semantic_confidence]
        clusters = spatial_clusters(rows, config)
        selected = clusters[0] if clusters else None
        status = "AMBIGUOUS" if ambiguous_rows else "NOT_OBSERVED"
        if selected is not None:
            status = "CANDIDATE"
            competing = len(clusters) > 1 and (
                clusters[1]["distinct_viewpoints"] == selected["distinct_viewpoints"]
                and selected["semantic_confidence"] - clusters[1]["semantic_confidence"]
                < config.minimum_hypothesis_score_margin
            )
            if selected["distinct_viewpoints"] >= config.minimum_supporting_views:
                status = "AMBIGUOUS" if competing else "MAPPED"
        result[identity] = {
            "status": status,
            "selected": selected,
            "competing_clusters": clusters[1:],
            "ambiguous_level_observation_count": len(ambiguous_rows),
        }
    return result
