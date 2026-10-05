import cv2
import numpy as np


# ============================================================
# DARK CHANNEL
# ============================================================

def get_dark_channel(image, window_size=15):

    """
    Calculate the Dark Channel Prior.
    """

    image = image.astype(np.float32)

    minimum_channel = np.min(
        image,
        axis=2
    )

    kernel = cv2.getStructuringElement(

        cv2.MORPH_RECT,

        (
            window_size,
            window_size
        )

    )

    dark_channel = cv2.erode(

        minimum_channel,

        kernel

    )

    return dark_channel


# ============================================================
# ATMOSPHERIC LIGHT
# ============================================================

def estimate_atmospheric_light(
    image,
    dark_channel,
    top_percent=0.001
):

    """
    Estimate atmospheric light from brightest dark-channel pixels.
    """

    height, width = dark_channel.shape

    total_pixels = (

        height
        *
        width

    )

    number_of_pixels = max(

        int(
            total_pixels
            *
            top_percent
        ),

        1

    )


    dark_vector = dark_channel.reshape(

        -1

    )


    image_vector = image.reshape(

        -1,
        3

    )


    indices = np.argsort(

        dark_vector

    )[-number_of_pixels:]


    atmospheric_light = np.mean(

        image_vector[indices],

        axis=0

    )


    atmospheric_light = np.maximum(

        atmospheric_light,

        1e-6

    )


    return atmospheric_light


# ============================================================
# TRANSMISSION ESTIMATION
# ============================================================

def estimate_transmission(
    image,
    atmospheric_light,
    omega=0.95,
    window_size=15
):

    """
    Estimate transmission map using Dark Channel Prior.
    """

    normalized_image = (

        image

        /

        atmospheric_light

    )


    dark_channel = get_dark_channel(

        normalized_image,

        window_size

    )


    transmission = (

        1.0

        -

        omega

        *

        dark_channel

    )


    transmission = np.clip(

        transmission,

        0.05,

        1.0

    )


    return transmission


# ============================================================
# TRANSMISSION REFINEMENT
# ============================================================

def refine_transmission(
    image_bgr,
    transmission,
    radius=40,
    eps=1e-3
):

    """
    Refine transmission using edge-preserving filtering.
    """

    transmission_uint8 = (

        transmission
        *
        255.0

    ).astype(

        np.uint8

    )


    refined = cv2.bilateralFilter(

        transmission_uint8,

        d=9,

        sigmaColor=75,

        sigmaSpace=75

    )


    refined = (

        refined.astype(np.float32)

        /

        255.0

    )


    refined = np.clip(

        refined,

        0.05,

        1.0

    )


    return refined


# ============================================================
# RECOVER DEHAZED IMAGE
# ============================================================

def recover_scene(
    image,
    atmospheric_light,
    transmission,
    transmission_min=0.1
):

    """
    Recover the dehazed image.
    """

    transmission = np.maximum(

        transmission,

        transmission_min

    )


    transmission = transmission[

        :,
        :,
        np.newaxis

    ]


    recovered = (

        image

        -

        atmospheric_light

    )


    recovered = (

        recovered

        /

        transmission

    )


    recovered = (

        recovered

        +

        atmospheric_light

    )


    recovered = np.clip(

        recovered,

        0.0,

        1.0

    )


    return recovered


# ============================================================
# ADAPTIVE TRANSMISSION ADJUSTMENT
# ============================================================

def adaptive_transmission_adjustment(
    transmission,
    haze_strength
):

    """
    Adapt transmission according to estimated haze strength.
    """

    # --------------------------------------------------------
    # Strong haze -> stronger dehazing
    # --------------------------------------------------------

    if haze_strength > 0.65:

        factor = 0.85


    # --------------------------------------------------------
    # Medium haze
    # --------------------------------------------------------

    elif haze_strength > 0.35:

        factor = 0.92


    # --------------------------------------------------------
    # Low haze
    # --------------------------------------------------------

    else:

        factor = 0.97


    adjusted_transmission = (

        transmission

        *

        factor

    )


    adjusted_transmission = np.clip(

        adjusted_transmission,

        0.05,

        1.0

    )


    return adjusted_transmission


# ============================================================
# ESTIMATE HAZE STRENGTH
# ============================================================

def estimate_haze_strength(
    dark_channel
):

    """
    Estimate haze strength from the mean dark channel value.
    """

    haze_strength = float(

        np.mean(

            dark_channel

        )

    )


    haze_strength = np.clip(

        haze_strength,

        0.0,

        1.0

    )


    return haze_strength


# ============================================================
# MAIN DEHAZING FUNCTION
# ============================================================

def adaptive_dehaze(
    image_bgr
):

    """
    Main adaptive image dehazing function.

    INPUT:
        OpenCV BGR uint8 image

    OUTPUT:
        Dehazed BGR uint8 image
    """


    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    if image_bgr is None:

        raise ValueError(

            "Input image is None."

        )


    if not isinstance(

        image_bgr,

        np.ndarray

    ):

        raise TypeError(

            "Input image must be a NumPy array."

        )


    if len(

        image_bgr.shape

    ) != 3:

        raise ValueError(

            "Input image must have 3 color channels."

        )


    # ========================================================
    # CONVERT BGR -> RGB
    # ========================================================

    image_rgb = cv2.cvtColor(

        image_bgr,

        cv2.COLOR_BGR2RGB

    )


    # ========================================================
    # NORMALIZE IMAGE
    # ========================================================

    image = (

        image_rgb.astype(

            np.float32

        )

        /

        255.0

    )


    # ========================================================
    # DARK CHANNEL
    # ========================================================

    dark_channel = get_dark_channel(

        image,

        window_size=15

    )


    # ========================================================
    # HAZE STRENGTH
    # ========================================================

    haze_strength = estimate_haze_strength(

        dark_channel

    )


    # ========================================================
    # ATMOSPHERIC LIGHT
    # ========================================================

    atmospheric_light = (

        estimate_atmospheric_light(

            image,

            dark_channel

        )

    )


    # ========================================================
    # TRANSMISSION MAP
    # ========================================================

    transmission = (

        estimate_transmission(

            image,

            atmospheric_light,

            omega=0.95,

            window_size=15

        )

    )


    # ========================================================
    # ADAPTIVE TRANSMISSION
    # ========================================================

    transmission = (

        adaptive_transmission_adjustment(

            transmission,

            haze_strength

        )

    )


    # ========================================================
    # REFINE TRANSMISSION
    # ========================================================

    transmission = (

        refine_transmission(

            image_bgr,

            transmission

        )

    )


    # ========================================================
    # RECOVER SCENE
    # ========================================================

    recovered = (

        recover_scene(

            image,

            atmospheric_light,

            transmission,

            transmission_min=0.1

        )

    )


    # ========================================================
    # CONVERT RGB -> BGR
    # ========================================================

    recovered_rgb = (

        recovered

        *

        255.0

    ).astype(

        np.uint8

    )


    recovered_bgr = cv2.cvtColor(

        recovered_rgb,

        cv2.COLOR_RGB2BGR

    )


    return recovered_bgr