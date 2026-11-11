"""Leakage-controlled two-stage ultrasound navigation research components.

This package deliberately has no Isaac Sim imports. Simulator collection and motion
adaptation live in workflows so the perception, coordinate, and evidence contracts can be
tested on CPU.
"""

from .contracts import ActorFrame, FrameFreshnessGuard, PrivilegedAnnotation
from .coordinates import ExecutedMotion, HomeRelativeOdometry, PixelGeometry
from .stage1_features import PatchSpec, extract_patch_feature
from .stage1_map import EvidenceObservation, SpatialEvidenceMap

__all__ = [
    "ActorFrame",
    "EvidenceObservation",
    "ExecutedMotion",
    "FrameFreshnessGuard",
    "HomeRelativeOdometry",
    "PatchSpec",
    "PixelGeometry",
    "PrivilegedAnnotation",
    "SpatialEvidenceMap",
    "extract_patch_feature",
]
