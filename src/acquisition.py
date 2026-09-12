from __future__ import annotations

import numpy as np
from skimage.transform import radon


def dense_angles(n_angles: int = 180) -> np.ndarray:
    return np.linspace(0.0, 180.0, n_angles, endpoint=False)


def uniform_sparse_angles(all_angles: np.ndarray, n: int) -> np.ndarray:
    idx = np.linspace(0, len(all_angles) - 1, n, dtype=int)
    return np.unique(idx)


def random_sparse_angles(all_angles: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.sort(rng.choice(len(all_angles), size=n, replace=False))


def acquire(image: np.ndarray, all_angles: np.ndarray) -> np.ndarray:
    """Compute the full synthetic sinogram once; subsets emulate acquisitions."""
    return radon(image, theta=all_angles, circle=True)
