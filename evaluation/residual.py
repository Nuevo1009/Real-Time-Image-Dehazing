import numpy as np


# ============================================================
# RESIDUAL IMAGE
# ============================================================

def calculate_residual(
    output_rgb,
    ground_truth_rgb
):
    """
    Calculate absolute RGB residual.

    Residual = |Ground Truth - Dehazed|

    Input:
        RGB uint8 images

    Output:
        RGB uint8 residual
    """

    output = np.asarray(
        output_rgb,
        dtype=np.int16
    )

    ground_truth = np.asarray(
        ground_truth_rgb,
        dtype=np.int16
    )

    residual = np.abs(
        ground_truth
        -
        output
    )

    residual = np.clip(
        residual,
        0,
        255
    ).astype(
        np.uint8
    )

    return residual