import numpy as np
from scipy.ndimage import minimum_filter


def compute_dark_channel(img, patch_size):
    """
    Python translation of:

        computeDarkChannel(img, patchSize)

    MATLAB:
        minChannel = min(img, [], 3);
        se = strel('square', patchSize);
        darkCh = imerode(minChannel, se);
    """

    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError(
            "patch_size must be a positive odd integer."
        )

    # Minimum over RGB channels
    min_channel = np.min(
        img,
        axis=2
    )

    # Square grayscale erosion = local minimum
    dark_ch = minimum_filter(
        min_channel,
        size=patch_size,
        mode="nearest"
    )

    return dark_ch