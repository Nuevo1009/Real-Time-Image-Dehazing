import cv2
import numpy as np


def compute_dark_channel(
    image,
    patch_size=15
):
    """
    Compute the Dark Channel Prior.

    Input:
        image:
            RGB float32 image in [0,1]

        patch_size:
            Positive odd integer.

    Output:
        Dark channel in [0,1].

    IMPORTANT:
        The input is already RGB.

        No BGR/RGB conversion is performed.
    """

    if image is None:
        raise ValueError(
            "Input image is None."
        )

    if not isinstance(
        image,
        np.ndarray
    ):
        raise TypeError(
            "Input image must be a NumPy array."
        )

    if image.ndim != 3:
        raise ValueError(
            "Image must have shape H x W x 3."
        )

    if image.shape[2] != 3:
        raise ValueError(
            "Image must have exactly 3 channels."
        )

    if (
        patch_size <= 0
        or
        patch_size % 2 == 0
    ):
        raise ValueError(
            "patch_size must be a positive odd integer."
        )

    image = np.asarray(
        image,
        dtype=np.float32
    )

    image = np.clip(
        image,
        0.0,
        1.0
    )

    # Minimum among R, G and B
    minimum_channel = np.min(
        image,
        axis=2
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            patch_size,
            patch_size
        )
    )

    dark_channel = cv2.erode(
        minimum_channel,
        kernel
    )

    return dark_channel.astype(
        np.float32
    )