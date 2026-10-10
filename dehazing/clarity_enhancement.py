import cv2
import numpy as np


def enhance_clarity(
    image,
    clip_limit=1.5,
    tile_grid_size=(8, 8),
    strength=0.20,
    sharpen_amount=0.15
):
    """
    Optional RGB-only clarity enhancement.

    No BGR/RGB conversion.
    No LAB conversion.

    Input:
        RGB float32 [0,1]

    Output:
        RGB float32 [0,1]
    """

    image = np.asarray(
        image,
        dtype=np.float32
    )

    image = np.clip(
        image,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # RGB luminance
    # --------------------------------------------------------

    luminance = (
        0.299 * image[:, :, 0]
        +
        0.587 * image[:, :, 1]
        +
        0.114 * image[:, :, 2]
    )

    luminance = np.clip(
        luminance,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # CLAHE only on luminance
    # --------------------------------------------------------

    luminance_uint8 = (
        luminance * 255.0
    ).astype(
        np.uint8
    )

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    enhanced_luminance_uint8 = (
        clahe.apply(
            luminance_uint8
        )
    )

    enhanced_luminance = (
        enhanced_luminance_uint8.astype(
            np.float32
        )
        /
        255.0
    )

    # --------------------------------------------------------
    # Controlled blending
    # --------------------------------------------------------

    enhanced_luminance = (
        (1.0 - strength)
        *
        luminance
        +
        strength
        *
        enhanced_luminance
    )

    # --------------------------------------------------------
    # Preserve RGB ratios
    # --------------------------------------------------------

    scale = (
        enhanced_luminance
        /
        np.maximum(
            luminance,
            1e-3
        )
    )

    scale = np.clip(
        scale,
        0.75,
        1.50
    )

    result = (
        image
        *
        scale[:, :, np.newaxis]
    )

    result = np.clip(
        result,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Mild sharpening
    # --------------------------------------------------------

    if sharpen_amount > 0.0:

        blurred = cv2.GaussianBlur(
            result,
            (0, 0),
            sigmaX=1.0
        )

        result = (
            result
            +
            sharpen_amount
            *
            (
                result
                -
                blurred
            )
        )

        result = np.clip(
            result,
            0.0,
            1.0
        )

    return result.astype(
        np.float32
    )