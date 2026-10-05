import numpy as np


def pixelwise_transmission(
    image,
    atmospheric_light,
    omega,
    gamma,
    t_min=0.1
):
    """
    Python equivalent of:

        pixelwiseTransmission(
            Mhazy,
            A,
            omega,
            gamma,
            tMin
        )
    """

    # --------------------------------------------------------
    # MATLAB:
    #
    # minMH = min(Mhazy, [], 3);
    # maxMH = max(Mhazy, [], 3);
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
    # Eq. 13
    #
    # denom = minMH + maxMH;
    # denom(denom == 0) = eps;
    # --------------------------------------------------------

    denom = (
        min_mh
        +
        max_mh
    )

    denom = np.where(
        denom == 0,
        np.finfo(np.float32).eps,
        denom
    )

    # --------------------------------------------------------
    # k = 1 - (minMH / denom)^gamma
    # --------------------------------------------------------

    k = (
        1.0
        -
        np.power(
            min_mh / denom,
            gamma
        )
    )

    # --------------------------------------------------------
    # Guard atmospheric light
    # --------------------------------------------------------

    A_safe = np.maximum(
        atmospheric_light,
        np.finfo(np.float32).eps
    )

    # --------------------------------------------------------
    # Channel ratios
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

    # --------------------------------------------------------
    # min RGB ratio
    # --------------------------------------------------------

    ratio = np.minimum.reduce(
        [
            ratio_r,
            ratio_g,
            ratio_b
        ]
    )

    # --------------------------------------------------------
    # Eq. 12
    #
    # tm = 1 - omega * ratio * k
    # --------------------------------------------------------

    tm = (
        1.0
        -
        omega
        *
        ratio
        *
        k
    )

    # --------------------------------------------------------
    # Clamp:
    #
    # tm = min(max(tm,tMin),1)
    # --------------------------------------------------------

    tm = np.clip(
        tm,
        t_min,
        1.0
    )

    return tm