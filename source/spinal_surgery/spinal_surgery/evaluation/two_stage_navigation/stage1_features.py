"""Stage 1 interface-centred patch features transferred from the Unity method.

The implementation matches the verified Unity feature definition. Its geometry is exposed as
configuration because the Unity numerical values are references for G0, not frozen SonoGym
hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PatchSpec:
    width_pixels: int
    depth_pixels: int
    interface_offset_pixels: int
    feature_size_pixels: int
    low_quantile: float = 0.05
    high_quantile: float = 0.95

    def __post_init__(self) -> None:
        if min(
            self.width_pixels,
            self.depth_pixels,
            self.interface_offset_pixels,
            self.feature_size_pixels,
        ) <= 0:
            raise ValueError("patch dimensions and offset must be positive")
        if self.interface_offset_pixels >= self.depth_pixels:
            raise ValueError("interface offset must lie inside the patch")
        if not 0.0 <= self.low_quantile < self.high_quantile <= 1.0:
            raise ValueError("invalid normalization quantiles")

    @property
    def feature_width(self) -> int:
        return 3 * self.feature_size_pixels**2 + 6


UNITY_REFERENCE_PATCH_SPEC = PatchSpec(
    width_pixels=48,
    depth_pixels=80,
    interface_offset_pixels=34,
    feature_size_pixels=16,
)


def _validate_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 2:
        raise ValueError(f"image must be a nontrivial 2-D array, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("image contains nonfinite values")
    return array


def bilinear_resize(image: np.ndarray, output_height: int, output_width: int) -> np.ndarray:
    """Dependency-light bilinear resize with endpoint-aligned coordinates."""
    array = _validate_image(image)
    if output_height <= 0 or output_width <= 0:
        raise ValueError("output dimensions must be positive")
    source_rows = np.arange(array.shape[0], dtype=np.float64)
    source_cols = np.arange(array.shape[1], dtype=np.float64)
    target_rows = np.linspace(0.0, array.shape[0] - 1.0, output_height)
    target_cols = np.linspace(0.0, array.shape[1] - 1.0, output_width)
    along_columns = np.vstack(
        [np.interp(target_cols, source_cols, row) for row in array]
    )
    resized = np.vstack(
        [np.interp(target_rows, source_rows, along_columns[:, column]) for column in range(output_width)]
    ).T
    return resized


def extract_interface_patch(
    image: np.ndarray,
    interface_row: float,
    interface_column: float,
    spec: PatchSpec,
) -> np.ndarray | None:
    """Extract an interface-centred source patch or return ``None`` at image boundaries."""
    array = _validate_image(image)
    if not np.isfinite(interface_row) or not np.isfinite(interface_column):
        raise ValueError("interface coordinates must be finite")
    column_start = int(round(interface_column - spec.width_pixels / 2.0))
    row_start = int(round(interface_row - spec.interface_offset_pixels))
    column_end = column_start + spec.width_pixels
    row_end = row_start + spec.depth_pixels
    if (
        row_start < 0
        or column_start < 0
        or row_end > array.shape[0]
        or column_end > array.shape[1]
    ):
        return None
    return array[row_start:row_end, column_start:column_end].copy()


def patch_feature(patch: np.ndarray, spec: PatchSpec) -> np.ndarray:
    """Return normalized intensity/gradient morphology with the Unity 6-value summary."""
    array = _validate_image(patch)
    if array.shape != (spec.depth_pixels, spec.width_pixels):
        raise ValueError(
            f"patch must have shape {(spec.depth_pixels, spec.width_pixels)}, got {array.shape}"
        )
    low, high = np.quantile(array, (spec.low_quantile, spec.high_quantile))
    normalized = np.clip((array - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    size = spec.feature_size_pixels
    small = bilinear_resize(normalized, size, size)
    gradient_x = np.gradient(small, axis=1)
    gradient_y = np.gradient(small, axis=0)
    magnitude = np.hypot(gradient_x, gradient_y)
    interface_row = int(
        np.clip(round(spec.interface_offset_pixels * size / spec.depth_pixels), 0, size - 1)
    )
    proximal = small[max(0, interface_row - 4) : interface_row]
    distal = small[interface_row + 1 : min(size, interface_row + 6)]
    summary = np.asarray(
        [
            float(np.mean(small)),
            float(np.std(small)),
            float(np.mean(proximal)) if proximal.size else 0.0,
            float(np.mean(distal)) if distal.size else 0.0,
            float(np.mean(magnitude)),
            float(np.quantile(magnitude, 0.9)),
        ],
        dtype=np.float64,
    )
    feature = np.concatenate((small.ravel(), gradient_x.ravel(), gradient_y.ravel(), summary))
    if feature.shape != (spec.feature_width,) or not np.all(np.isfinite(feature)):
        raise AssertionError("invalid patch feature output")
    return feature


def extract_patch_feature(
    image: np.ndarray,
    interface_row: float,
    interface_column: float,
    spec: PatchSpec,
) -> np.ndarray | None:
    patch = extract_interface_patch(image, interface_row, interface_column, spec)
    return None if patch is None else patch_feature(patch, spec)
