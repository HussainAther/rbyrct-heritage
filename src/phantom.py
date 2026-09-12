from __future__ import annotations

import numpy as np
from skimage.draw import disk, ellipse, line
from scipy.ndimage import gaussian_filter


def make_heritage_phantom(size: int = 160, seed: int = 7):
    """Create a synthetic sealed heritage object and hidden-feature mask.

    The object contains an annular ceramic vessel, a hidden internal insert,
    a cavity, a crack-like feature, and smooth heterogeneous attenuation.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.float32)
    hidden = np.zeros_like(img, dtype=bool)

    cy = cx = size // 2

    # Ceramic vessel wall: outer disk minus inner cavity.
    rr, cc = disk((cy, cx), int(size * 0.39), shape=img.shape)
    img[rr, cc] = 0.70
    rr, cc = disk((cy, cx), int(size * 0.30), shape=img.shape)
    img[rr, cc] = 0.12

    # Mild heterogeneous material texture inside the vessel wall/body.
    noise = gaussian_filter(rng.normal(0, 1, img.shape), sigma=size / 25)
    noise /= max(np.max(np.abs(noise)), 1e-8)
    material = img > 0
    img[material] += (0.055 * noise[material]).astype(np.float32)

    # Hidden oval insert.
    rr, cc = ellipse(
        int(size * 0.54), int(size * 0.43),
        int(size * 0.075), int(size * 0.12),
        rotation=np.deg2rad(24), shape=img.shape,
    )
    img[rr, cc] = 1.00
    hidden[rr, cc] = True

    # Small low-density cavity.
    rr, cc = disk((int(size * 0.43), int(size * 0.59)), int(size * 0.045), shape=img.shape)
    img[rr, cc] = 0.02
    hidden[rr, cc] = True

    # Crack-like low attenuation feature through part of the wall.
    r0, c0 = int(size * 0.28), int(size * 0.67)
    r1, c1 = int(size * 0.47), int(size * 0.61)
    rr, cc = line(r0, c0, r1, c1)
    for dr in (-1, 0, 1):
        r = np.clip(rr + dr, 0, size - 1)
        img[r, cc] = 0.03
        hidden[r, cc] = True

    img = np.clip(img, 0.0, 1.0)
    return img, hidden
