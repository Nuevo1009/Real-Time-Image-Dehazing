import cv2
import numpy as np


def enhance_clarity(
    image,
    clip_limit=2.0,
    tile_grid_size=(8, 8),
    sharpen_amount=0.7
):
    """
    Post-processing stage for improving local contrast
    and perceived sharpness after dehazing.

    Input:
        image: RGB float image [0,1]

    Output:
        RGB float image [0,1]
    """

    # --------------------------------------------------------
    # Ensure valid range
    # --------------------------------------------------------

    image = np.clip(
        image,
        0.0,
        1.0
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Convert RGB -> uint8
    # --------------------------------------------------------

    image_uint8 = (
        image * 255.0
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # CLAHE on luminance channel
    # --------------------------------------------------------

    lab = cv2.cvtColor(
        image_uint8,
        cv2.COLOR_RGB2LAB
    )

    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    L_enhanced = clahe.apply(L)

    lab_enhanced = cv2.merge(
        [
            L_enhanced,
            A,
            B
        ]
    )

    contrast_enhanced = cv2.cvtColor(
        lab_enhanced,
        cv2.COLOR_LAB2RGB
    )

    # --------------------------------------------------------
    # Unsharp masking
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        contrast_enhanced,
        (0, 0),
        sigmaX=1.0
    )

    sharpened = cv2.addWeighted(
        contrast_enhanced,
        1.0 + sharpen_amount,
        blurred,
        -sharpen_amount,
        0
    )

    # --------------------------------------------------------
    # Back to [0,1]
    # --------------------------------------------------------

    result = (
        np.clip(
            sharpened,
            0,
            255
        )
        / 255.0
    ).astype(
        np.float32
    )

    return result