from __future__ import annotations

import numpy as np
from skimage.transform import radon

from .reconstruction import reconstruct


def _angular_distance_deg(a: float, b: np.ndarray) -> np.ndarray:
    d = np.abs(b - a)
    return np.minimum(d, 180.0 - d)


def adaptive_angle_indices(
    full_sinogram: np.ndarray,
    all_angles: np.ndarray,
    n_select: int,
    output_size: int,
    seed_count: int = 6,
) -> np.ndarray:
    """Select angles without inspecting unmeasured projection values.

    The policy seeds the scan with uniformly separated views. Thereafter it
    reconstructs from measured views, forward-projects *that reconstruction*
    at still-unmeasured candidate angles, and scores each candidate by its
    projected gradient energy with a mild angular-diversity bonus.
    """
    n_select = int(min(max(n_select, 1), len(all_angles)))
    seed_count = int(min(seed_count, n_select))

    seed_idx = np.unique(np.linspace(0, len(all_angles) - 1, seed_count, dtype=int)).tolist()
    selected = list(seed_idx)

    while len(selected) < n_select:
        sel = np.array(sorted(selected), dtype=int)
        rec = reconstruct(full_sinogram[:, sel], all_angles[sel], output_size)

        candidates = np.array([i for i in range(len(all_angles)) if i not in selected], dtype=int)
        candidate_angles = all_angles[candidates]

        # This uses only the current reconstruction, never unmeasured data.
        predicted = radon(rec, theta=candidate_angles, circle=True)
        structural = np.mean(np.abs(np.diff(predicted, axis=0)), axis=0)
        if np.ptp(structural) > 1e-12:
            structural = (structural - structural.min()) / np.ptp(structural)
        else:
            structural = np.zeros_like(structural)

        selected_angles = all_angles[sel]
        diversity = np.array([
            np.min(_angular_distance_deg(a, selected_angles)) for a in candidate_angles
        ])
        if np.max(diversity) > 0:
            diversity = diversity / np.max(diversity)

        score = 0.20 * structural + 1.00 * diversity
        best = int(candidates[int(np.argmax(score))])
        selected.append(best)

    return np.array(sorted(selected), dtype=int)
