"""Image-only features for supervised best-view scoring."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom


FEATURE_WIDTH = 1536


def _resize(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError("image must be a finite two-dimensional array")
    resized = zoom(
        array,
        (shape[0] / array.shape[0], shape[1] / array.shape[1]),
        order=1,
        mode="nearest",
        prefilter=False,
    )
    if resized.shape != shape:
        raise RuntimeError(f"resize produced {resized.shape}, expected {shape}")
    return resized.astype(np.float32, copy=False)


def normalize_ultrasound(image: np.ndarray) -> np.ndarray:
    """Apply the frozen percentile normalization without changing image shape."""
    array = np.asarray(image, dtype=np.float32)
    if array.shape != (200, 150) or not np.all(np.isfinite(array)):
        raise ValueError("ultrasound must be finite with shape (200, 150)")
    low, high = np.percentile(array, (1.0, 99.0))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("ultrasound has no usable percentile range")
    return np.clip((array - low) / (high - low), 0.0, 1.0).astype(np.float32)


def extract_stage2_feature(image: np.ndarray) -> np.ndarray:
    """Return intensity and absolute-gradient features matching the Unity layout."""
    normalized = normalize_ultrasound(image)
    intensity = _resize(normalized, (32, 32))
    gradient_y, gradient_x = np.gradient(normalized)
    feature = np.concatenate(
        [
            intensity.reshape(-1),
            _resize(np.abs(gradient_x), (16, 16)).reshape(-1),
            _resize(np.abs(gradient_y), (16, 16)).reshape(-1),
        ]
    ).astype(np.float32, copy=False)
    if feature.shape != (FEATURE_WIDTH,) or not np.all(np.isfinite(feature)):
        raise RuntimeError("invalid Stage 2 feature")
    return feature
