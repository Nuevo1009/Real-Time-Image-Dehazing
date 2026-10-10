import numpy as np

from .dark_channel import (
    compute_dark_channel
)


# ============================================================
# ATMOSPHERIC LIGHT ESTIMATION
# ============================================================

def estimate_atmospheric_light(
    image_down,
    patch_size,
    th_ref=None,
    top_fraction=0.001
):
    """
    Estimate far and near atmospheric light.

    Input:
        image_down:
            RGB float32 image [0,1]

        patch_size:
            Dark-channel patch size.

        th_ref:
            Reference threshold.

        top_fraction:
            Fraction of candidate pixels.

    Returns:
        AF
        AN
        th_current
        dark_down
    """

    if image_down.ndim != 3:
        raise ValueError(
            "image_down must have shape H x W x 3."
        )

    if image_down.shape[2] != 3:
        raise ValueError(
            "image_down must be RGB."
        )

    image_down = np.asarray(
        image_down,
        dtype=np.float32
    )

    image_down = np.clip(
        image_down,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # RGB luminance
    # --------------------------------------------------------

    gray_down = (
        0.299 * image_down[:, :, 0]
        +
        0.587 * image_down[:, :, 1]
        +
        0.114 * image_down[:, :, 2]
    )

    th_current = float(
        np.mean(gray_down)
    )

    if th_ref is None:
        th_ref = th_current

    # --------------------------------------------------------
    # Dark channel
    # --------------------------------------------------------

    dark_down = compute_dark_channel(
        image_down,
        patch_size
    )

    # --------------------------------------------------------
    # Far and near masks
    # --------------------------------------------------------

    far_mask = (
        dark_down >= th_ref
    )

    near_mask = np.logical_not(
        far_mask
    )

    # --------------------------------------------------------
    # Atmospheric light
    # --------------------------------------------------------

    AF = pick_brightest_location(
        dark_down,
        far_mask,
        image_down,
        top_fraction
    )

    AN = pick_brightest_location(
        dark_down,
        near_mask,
        image_down,
        top_fraction
    )

    return (
        AF,
        AN,
        th_current,
        dark_down
    )


# ============================================================
# PICK BRIGHTEST ATMOSPHERIC CANDIDATES
# ============================================================

def pick_brightest_location(
    dark_channel,
    mask,
    image,
    top_fraction=0.001
):
    """
    Estimate atmospheric light using several strong
    dark-channel candidates rather than one pixel.

    Median RGB value is used for robustness.
    """

    candidate_indices = np.flatnonzero(
        mask
    )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if candidate_indices.size == 0:

        candidate_indices = np.arange(
            dark_channel.size
        )

    # --------------------------------------------------------
    # Number of candidates
    # --------------------------------------------------------

    number_of_candidates = max(
        1,
        int(
            candidate_indices.size
            *
            top_fraction
        )
    )

    number_of_candidates = min(
        number_of_candidates,
        candidate_indices.size
    )

    # --------------------------------------------------------
    # Dark-channel values
    # --------------------------------------------------------

    candidate_dark = (
        dark_channel.ravel()[
            candidate_indices
        ]
    )

    # --------------------------------------------------------
    # Select strongest candidates
    # --------------------------------------------------------

    if number_of_candidates == 1:

        selected_indices = np.array([
            candidate_indices[
                np.argmax(candidate_dark)
            ]
        ])

    else:

        order = np.argpartition(
            candidate_dark,
            -number_of_candidates
        )[-number_of_candidates:]

        selected_indices = (
            candidate_indices[order]
        )

    # --------------------------------------------------------
    # RGB values
    # --------------------------------------------------------

    image_flat = image.reshape(
        -1,
        3
    )

    candidates = (
        image_flat[selected_indices]
    )

    # --------------------------------------------------------
    # Median is robust against bright sun pixels
    # --------------------------------------------------------

    atmospheric_light = np.median(
        candidates,
        axis=0
    )

    atmospheric_light = np.clip(
        atmospheric_light,
        1e-3,
        1.0
    )

    return atmospheric_light.astype(
        np.float32
    )


# ============================================================
# PIXEL-WISE ATMOSPHERIC LIGHT
# ============================================================

def pixelwise_atmospheric_light(
    image,
    AF,
    AN
):
    """
    Generate a spatially varying atmospheric-light map.

        A(x) =
            (1-eta(x))*AF
            +
            eta(x)*AN
    """

    image = np.asarray(
        image,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # RGB luminance
    # --------------------------------------------------------

    IH = (
        0.299 * image[:, :, 0]
        +
        0.587 * image[:, :, 1]
        +
        0.114 * image[:, :, 2]
    )

    Imax = float(
        np.mean(AF)
    )

    Imax_safe = max(
        Imax,
        np.finfo(np.float32).eps
    )

    # --------------------------------------------------------
    # Eta
    # --------------------------------------------------------

    eta = (
        Imax_safe - IH
    ) / Imax_safe

    eta = np.clip(
        eta,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Far region
    # --------------------------------------------------------

    far_region = (
        IH >= Imax_safe
    )

    # --------------------------------------------------------
    # Atmospheric map
    # --------------------------------------------------------

    A = np.zeros_like(
        image,
        dtype=np.float32
    )

    for c in range(3):

        Ac = (
            (1.0 - eta) * AF[c]
            +
            eta * AN[c]
        )

        Ac[far_region] = AF[c]

        A[:, :, c] = Ac

    return np.clip(
        A,
        0.0,
        1.0
    ).astype(
        np.float32
    )