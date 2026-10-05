import numpy as np


def scene_dehazing(
    image,
    atmospheric_light,
    transmission
):
    """
    Python equivalent of sceneDehazing.m

    Eq. 6:

        J = (I - A) / t + A
    """

    # H x W x 1
    t = transmission[
        :,
        :,
        np.newaxis
    ]

    # MATLAB loops over R,G,B.
    # Broadcasting performs the same operation.

    dehazed = (
        image
        -
        atmospheric_light
    ) / t + atmospheric_light

    # MATLAB:
    # Mdehazed = min(max(Mdehazed,0),1);

    dehazed = np.clip(
        dehazed,
        0.0,
        1.0
    )

    return dehazed