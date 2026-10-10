import cv2
import numpy as np


# ============================================================
# PIXEL-WISE TRANSMISSION
# ============================================================

def pixelwise_transmission(
    image,
    atmospheric_light,
    omega=0.9375,
    gamma=2.0,
    t_min=0.1
):
    """
    Gamma-based transmission estimation.

    k =
        1 -
        (Imin / (Imin + Imax))^gamma

    ratio =
        min(
            IR/AR,
            IG/AG,
            IB/AB
        )

    t =
        1 - omega * ratio * k

    Input is RGB.
    """

    image = np.asarray(
        image,
        dtype=np.float32
    )

    atmospheric_light = np.asarray(
        atmospheric_light,
        dtype=np.float32
    )

    if image.ndim != 3:
        raise ValueError(
            "Image must have shape H x W x 3."
        )

    if image.shape[2] != 3:
        raise ValueError(
            "Image must be RGB."
        )

    # --------------------------------------------------------
    # Minimum and maximum RGB
    # --------------------------------------------------------

    min_mh = np.min(
        image,
        axis=2
    )

    max_mh = np.max(
        image,
        axis=2
    )

    # --------------------------------------------------------
    # Denominator
    # --------------------------------------------------------

    denom = (
        min_mh
        +
        max_mh
    )

    denom = np.maximum(
        denom,
        np.finfo(np.float32).eps
    )

    # --------------------------------------------------------
    # Gamma factor
    # --------------------------------------------------------

    normalized_min = (
        min_mh / denom
    )

    normalized_min = np.clip(
        normalized_min,
        0.0,
        1.0
    )

    k = (
        1.0
        -
        np.power(
            normalized_min,
            gamma
        )
    )

    # --------------------------------------------------------
    # Atmospheric light protection
    # --------------------------------------------------------

    A_safe = np.maximum(
        atmospheric_light,
        1e-3
    )

    # --------------------------------------------------------
    # RGB ratios
    # --------------------------------------------------------

    ratio_r = (
        image[:, :, 0]
        /
        A_safe[:, :, 0]
    )

    ratio_g = (
        image[:, :, 1]
        /
        A_safe[:, :, 1]
    )

    ratio_b = (
        image[:, :, 2]
        /
        A_safe[:, :, 2]
    )

    ratio = np.minimum.reduce(
        [
            ratio_r,
            ratio_g,
            ratio_b
        ]
    )

    ratio = np.clip(
        ratio,
        0.0,
        1.5
    )

    # --------------------------------------------------------
    # Transmission
    # --------------------------------------------------------

    transmission = (
        1.0
        -
        omega
        *
        ratio
        *
        k
    )

    transmission = np.clip(
        transmission,
        t_min,
        1.0
    )

    return transmission.astype(
        np.float32
    )


# ============================================================
# TRANSMISSION REFINEMENT
# ============================================================

def refine_transmission(
    transmission,
    radius=9,
    sigma_color=0.05,
    sigma_space=15
):
    """
    Refine transmission while keeping it float32.

    No RGB/BGR conversion is involved.
    """

    transmission = np.asarray(
        transmission,
        dtype=np.float32
    )

    transmission = np.clip(
        transmission,
        0.0,
        1.0
    )

    refined = cv2.bilateralFilter(
        transmission,
        d=radius,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space
    )

    return np.clip(
        refined,
        0.05,
        1.0
    ).astype(
        np.float32
    )