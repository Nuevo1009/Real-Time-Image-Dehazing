import numpy as np


def scene_dehazing(
    image,
    atmospheric_light,
    transmission,
    t_min=0.1
):
    """
    Recover the scene using:

        J = (I - A) / t + A

    Input:
        image:
            RGB float32 [0,1]

        atmospheric_light:
            RGB atmospheric-light map

        transmission:
            H x W transmission

    Output:
        RGB float32 [0,1]
    """

    image = np.asarray(
        image,
        dtype=np.float32
    )

    atmospheric_light = np.asarray(
        atmospheric_light,
        dtype=np.float32
    )

    transmission = np.asarray(
        transmission,
        dtype=np.float32
    )

    transmission = np.maximum(
        transmission,
        t_min
    )

    t = transmission[
        :,
        :,
        np.newaxis
    ]

    # --------------------------------------------------------
    # Atmospheric scattering model
    # --------------------------------------------------------

    dehazed = (
        image
        -
        atmospheric_light
    ) / t + atmospheric_light

    dehazed = np.nan_to_num(
        dehazed,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    dehazed = np.clip(
        dehazed,
        0.0,
        1.0
    )

    return dehazed.astype(
        np.float32
    )