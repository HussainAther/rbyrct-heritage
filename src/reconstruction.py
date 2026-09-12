from __future__ import annotations

import numpy as np
from skimage.transform import iradon


def reconstruct(sinogram: np.ndarray, angles_deg: np.ndarray, output_size: int) -> np.ndarray:
    rec = iradon(
        sinogram,
        theta=angles_deg,
        circle=True,
        filter_name="ramp",
        output_size=output_size,
    )
    rec = np.nan_to_num(rec, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(rec, 0.0, 1.0)
