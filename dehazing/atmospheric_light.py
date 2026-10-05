import numpy as np

from .dark_channel import (
    compute_dark_channel
)


def estimate_atmospheric_light(
    image_down,
    patch_size,
    th_ref=None
):
    """
    Python equivalent of estimateAtmosphericLight.m
    """

    # --------------------------------------------------------
    # AGVC
    # --------------------------------------------------------

    gray_down = np.mean(
        image_down,
        axis=2
    )

    th_current = float(
        np.mean(gray_down)
    )

    # --------------------------------------------------------
    # First frame / still image
    # --------------------------------------------------------

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
    # Far / near masks
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
        image_down
    )

    AN = pick_brightest_location(
        dark_down,
        near_mask,
        image_down
    )

    return (
        AF,
        AN,
        th_current,
        dark_down
    )


def pick_brightest_location(
    dark_channel,
    mask,
    image_down
):
    """
    Python equivalent of pickBrightestLocation().
    """

    d = dark_channel.copy()

    # MATLAB:
    # d(~mask) = -Inf;

    d[
        np.logical_not(mask)
    ] = -np.inf

    # MATLAB:
    # if all(~mask(:))
    #     d = darkCh;
    # end

    if not np.any(mask):

        d = dark_channel

    # MATLAB:
    # [~, idx] = max(d(:));

    idx = np.argmax(d)

    row, col = np.unravel_index(
        idx,
        d.shape
    )

    # MATLAB:
    # A = squeeze(imgDown(r,c,:))';

    A = image_down[
        row,
        col,
        :
    ].copy()

    return A


def pixelwise_atmospheric_light(
    image,
    AF,
    AN
):
    """
    Python equivalent of
    pixelwiseAtmosphericLight.m
    """

    # MATLAB:
    # IH = mean(Mhazy, 3);

    IH = np.mean(
        image,
        axis=2
    )

    # MATLAB:
    # Imax = mean(AF);

    Imax = float(
        np.mean(AF)
    )

    # --------------------------------------------------------
    # Eq. 10
    # eta = (Imax - IH) / Imax
    # --------------------------------------------------------

    if Imax == 0:

        Imax_safe = np.finfo(
            np.float32
        ).eps

    else:

        Imax_safe = Imax

    eta = (
        Imax_safe - IH
    ) / Imax_safe

    # MATLAB:
    # eta = min(max(eta, 0), 1);

    eta = np.clip(
        eta,
        0.0,
        1.0
    )

    # MATLAB:
    # farRegion = (IH >= Imax);

    far_region = (
        IH >= Imax
    )

    # --------------------------------------------------------
    # Create H x W x 3 atmospheric-light image
    # --------------------------------------------------------

    A = np.zeros_like(
        image,
        dtype=np.float32
    )

    for c in range(3):

        # Eq. 9
        Ac = (
            (1.0 - eta) * AF[c]
            +
            eta * AN[c]
        )

        # Far-region branch
        Ac[far_region] = AF[c]

        A[:, :, c] = Ac

    return A