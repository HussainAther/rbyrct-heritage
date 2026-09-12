from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def mse(reference: np.ndarray, reconstruction: np.ndarray) -> float:
    return float(np.mean((reference - reconstruction) ** 2))


def ssim(reference: np.ndarray, reconstruction: np.ndarray) -> float:
    return float(structural_similarity(reference, reconstruction, data_range=1.0))


def hidden_feature_iou(
    reference: np.ndarray,
    reconstruction: np.ndarray,
    hidden_mask: np.ndarray,
) -> float:
    """Approximate hidden-feature recovery by anomaly segmentation.

    Hidden features are defined relative to broad background material bands.
    This deliberately simple V0 metric thresholds reconstruction intensity in
    ranges associated with the high-density insert and low-density cavity/crack.
    """
    pred = (reconstruction >= 0.86) | ((reconstruction <= 0.075) & (reference > 0.0))

    # Restrict scoring to a central object region to avoid exterior background.
    yy, xx = np.indices(reference.shape)
    c = (reference.shape[0] - 1) / 2.0
    object_region = (yy - c) ** 2 + (xx - c) ** 2 <= (reference.shape[0] * 0.40) ** 2
    pred &= object_region

    union = np.logical_or(pred, hidden_mask).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(pred, hidden_mask).sum()
    return float(inter / union)
