"""Associate observed cortical interfaces with complete linkage."""

from dataclasses import asdict

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import label
from scipy.spatial.distance import cdist

from .incremental_landmark_map import LANDMARK_IDENTITIES
from .simple_landmark_mapping import CLASSES, distinct_support


def observation_surfaces(identity, rows, masks, coordinates):
    level = identity.split("_")[1]
    semantic = CLASSES.index(identity.split("_")[2])
    components = {}
    result = []
    for row in rows:
        sequence = row.frame_sequence
        if sequence not in components:
            components[sequence] = label(masks[level][sequence] == semantic, np.ones((3, 3)))[0]
        membership = components[sequence]
        component = membership[row.row, row.column]
        if not component:
            raise ValueError(f"observation is outside its saved semantic mask: {identity}/{sequence}")
        points = coordinates[sequence][membership == component]
        if len(points) != row.pixel_count:
            raise ValueError(f"observation component changed: {identity}/{sequence}")
        result.append(points)
    return result


def surface_distance(first, second, quantile):
    distances = cdist(first, second)
    return float(max(np.quantile(distances.min(axis=1), quantile),
                     np.quantile(distances.min(axis=0), quantile)))


def surface_clusters(rows, surfaces, config, quantile):
    if len(rows) != len(surfaces):
        raise ValueError("each observation must have one surface")
    if not rows:
        return []
    distances = [
        surface_distance(first, second, quantile)
        for i, first in enumerate(surfaces) for second in surfaces[i + 1:]
    ]
    groups = (fcluster(linkage(distances, method="complete"), config.association_radius_mm,
                       criterion="distance") if distances else np.ones(1, dtype=int))
    clusters = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        members = [rows[i] for i in indices]
        support = distinct_support(members, config)
        positions = np.asarray([row.coordinate_mm for row in support])
        medoid = positions[np.argmin(cdist(positions, positions).sum(axis=1))]
        clusters.append({
            "position_mm": medoid.tolist(),
            "semantic_confidence": float(np.mean([row.confidence for row in support])),
            "rms_dispersion_mm": float(np.sqrt(np.mean(np.sum((positions - medoid) ** 2, axis=1)))),
            "distinct_viewpoints": len(support),
            "members": len(members),
            "supporting_observations": [asdict(row) for row in support],
            "member_observations": [asdict(row) for row in members],
            "observed_surface_mm": np.concatenate([surfaces[i] for i in indices]).tolist(),
        })
    return sorted(clusters, key=lambda row: (
        -row["distinct_viewpoints"], -row["semantic_confidence"], row["rms_dispersion_mm"],
    ))


def surface_map(observations, ambiguous, masks, coordinates, config, quantile):
    mapping = {}
    for identity in LANDMARK_IDENTITIES:
        rows = [row for row in observations.get(identity, ())
                if row.confidence >= config.minimum_semantic_confidence]
        points = observation_surfaces(identity, rows, masks, coordinates)
        clusters = surface_clusters(rows, points, config, quantile)
        selected = clusters[0] if clusters else None
        ambiguity_count = len(ambiguous.get(identity, ()))
        status = "AMBIGUOUS" if ambiguity_count else "NOT_OBSERVED"
        if selected:
            status = "CANDIDATE"
            if selected["distinct_viewpoints"] >= config.minimum_supporting_views:
                competing = len(clusters) > 1 and (
                    clusters[1]["distinct_viewpoints"] == selected["distinct_viewpoints"]
                    and selected["semantic_confidence"] - clusters[1]["semantic_confidence"]
                    < config.minimum_hypothesis_score_margin)
                status = "AMBIGUOUS" if competing else "MAPPED"
        mapping[identity] = {"selected": selected, "competing_clusters": clusters[1:],
                             "status": status, "ambiguous_level_observation_count": ambiguity_count}
    return mapping
