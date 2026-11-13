"""Leakage-controlled two-stage ultrasound navigation research components.

This package deliberately has no Isaac Sim imports. Simulator collection and motion
adaptation live in workflows so the perception, coordinate, and evidence contracts can be
tested on CPU.
"""

from .contracts import ActorFrame, FrameFreshnessGuard, PrivilegedAnnotation
from .coordinates import ExecutedMotion, HomeRelativeOdometry, PixelGeometry
from .stage1_features import PatchSpec, extract_patch_feature
from .stage1_map import (
    EvidenceObservation,
    OrderedEventPath,
    SpatialEvidenceMap,
    select_ordered_event_path,
)
from .stage1_pipeline import ActorPatchCandidate, PatchLabelConfig, reconstruct_landmark
from .stage2_features import extract_stage2_feature, normalize_ultrasound
from .stage2_models import (
    ScoredView,
    Stage2LabelConfig,
    fresh_confirmation_passes,
    select_best_view,
    supervised_targets,
)

__all__ = [
    "ActorFrame",
    "ActorPatchCandidate",
    "EvidenceObservation",
    "OrderedEventPath",
    "ExecutedMotion",
    "FrameFreshnessGuard",
    "HomeRelativeOdometry",
    "PatchSpec",
    "PatchLabelConfig",
    "PixelGeometry",
    "PrivilegedAnnotation",
    "SpatialEvidenceMap",
    "ScoredView",
    "Stage2LabelConfig",
    "select_ordered_event_path",
    "extract_patch_feature",
    "extract_stage2_feature",
    "fresh_confirmation_passes",
    "normalize_ultrasound",
    "reconstruct_landmark",
    "select_best_view",
    "supervised_targets",
]
