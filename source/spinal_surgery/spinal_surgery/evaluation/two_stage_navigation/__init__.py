"""Leakage-controlled two-stage ultrasound navigation research components.

This package deliberately has no Isaac Sim imports. Simulator collection and motion
adaptation live in workflows so the perception, coordinate, and evidence contracts can be
tested on CPU.
"""

from .contracts import ActorFrame, FrameFreshnessGuard, PrivilegedAnnotation
from .coordinates import ExecutedMotion, HomeRelativeOdometry, PixelGeometry
from .stage1_features import PatchSpec, extract_patch_feature
from .stage1_map import EvidenceObservation, SpatialEvidenceMap
from .stage1_pipeline import ActorPatchCandidate, PatchLabelConfig, reconstruct_landmark

__all__ = [
    "ActorFrame",
    "ActorPatchCandidate",
    "EvidenceObservation",
    "ExecutedMotion",
    "FrameFreshnessGuard",
    "HomeRelativeOdometry",
    "PatchSpec",
    "PatchLabelConfig",
    "PixelGeometry",
    "PrivilegedAnnotation",
    "SpatialEvidenceMap",
    "extract_patch_feature",
    "reconstruct_landmark",
]
