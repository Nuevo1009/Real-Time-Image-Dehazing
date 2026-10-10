import cv2
import numpy as np


# ============================================================
# DEFAULT PARAMETERS
# ============================================================

from .default_dehaze_params import (
    default_dehaze_params
)


# ============================================================
# DARK CHANNEL
# ============================================================

from .dark_channel import (
    compute_dark_channel
)


# ============================================================
# ATMOSPHERIC LIGHT
# ============================================================

from .atmospheric_light import (
    estimate_atmospheric_light,
    pixelwise_atmospheric_light
)


# ============================================================
# TRANSMISSION
# ============================================================

from .transmission import (
    pixelwise_transmission,
    refine_transmission
)


# ============================================================
# SCENE DEHAZING
# ============================================================

from .scene_dehazing import (
    scene_dehazing
)


# ============================================================
# CLARITY ENHANCEMENT
# ============================================================

from .clarity_enhancement import (
    enhance_clarity
)


# ============================================================
# MAIN ADAPTIVE DEHAZING FUNCTION
# ============================================================

def adaptive_dehaze(
    image_rgb,
    params=None
):
    """
    RGB-ONLY Adaptive Image Dehazing.

    ========================================================
    INPUT
    ========================================================

    image_rgb:

        RGB image.

        Supported formats:

            uint8  -> [0, 255]

            float -> [0, 1]

            float -> [0, 255]

    ========================================================
    OUTPUT
    ========================================================

    Returns:

        dehazed_rgb
            RGB uint8 image [0,255]

        state
            Reserved for future use.

        debug
            Dictionary containing intermediate results.

    ========================================================
    IMPORTANT
    ========================================================

    This function DOES NOT perform:

        BGR -> RGB

    or:

        RGB -> BGR

    The image remains RGB throughout the complete
    dehazing algorithm.

    OpenCV is used only for numerical image processing
    operations such as resizing. No color conversion is used.
    """


    # ========================================================
    # STEP 0
    # PARAMETERS
    # ========================================================

    if params is None:

        params = default_dehaze_params()


    # ========================================================
    # STEP 1
    # VALIDATE INPUT
    # ========================================================

    if image_rgb is None:

        raise ValueError(
            "Input RGB image is None."
        )


    if not isinstance(
        image_rgb,
        np.ndarray
    ):

        raise TypeError(
            "Input image must be a NumPy array."
        )


    if image_rgb.ndim != 3:

        raise ValueError(
            "Input image must have shape H x W x 3."
        )


    if image_rgb.shape[2] != 3:

        raise ValueError(
            "Input image must have exactly "
            "3 RGB channels."
        )


    # ========================================================
    # STEP 2
    # NORMALIZE RGB IMAGE
    # ========================================================

    if np.issubdtype(
        image_rgb.dtype,
        np.integer
    ):

        # ----------------------------------------------------
        # uint8 RGB [0,255]
        # ----------------------------------------------------

        image = (

            image_rgb.astype(
                np.float32
            )

            /

            255.0

        )

    else:

        # ----------------------------------------------------
        # Floating-point RGB image
        # ----------------------------------------------------

        image = np.asarray(
            image_rgb,
            dtype=np.float32
        )


        # ----------------------------------------------------
        # If input is [0,255], normalize it
        # ----------------------------------------------------

        if image.size > 0:

            max_value = float(
                np.max(image)
            )

            if max_value > 1.0:

                image = (

                    image

                    /

                    255.0

                )


    # --------------------------------------------------------
    # Final safety clipping
    # --------------------------------------------------------

    image = np.clip(

        image,

        0.0,

        1.0

    ).astype(

        np.float32

    )


    # ========================================================
    # STEP 3
    # DARK CHANNEL
    # ========================================================

    dark_channel = compute_dark_channel(

        image,

        params.patch_size

    )


    # ========================================================
    # STEP 4
    # HAZE STRENGTH
    # ========================================================

    haze_strength = float(

        np.mean(

            dark_channel

        )

    )


    haze_strength = float(

        np.clip(

            haze_strength,

            0.0,

            1.0

        )

    )


    # ========================================================
    # STEP 5
    # DOWNSAMPLING
    # ========================================================

    down_factor = max(

        1,

        int(

            params.down_factor

        )

    )


    if down_factor > 1:

        height, width = (

            image.shape[:2]

        )


        down_width = max(

            1,

            width // down_factor

        )


        down_height = max(

            1,

            height // down_factor

        )


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # cv2.resize() does NOT change RGB to BGR.
        #
        # It only resizes the three channels.
        # ----------------------------------------------------

        image_down = cv2.resize(

            image,

            (
                down_width,
                down_height
            ),

            interpolation=cv2.INTER_AREA

        )

    else:

        image_down = image.copy()


    # ========================================================
    # STEP 6
    # ATMOSPHERIC LIGHT ESTIMATION
    # ========================================================

    (
        AF,
        AN,
        Th,
        dark_down

    ) = estimate_atmospheric_light(

        image_down,

        params.patch_size,

        th_ref=None

    )


    # ========================================================
    # STEP 7
    # PIXEL-WISE ATMOSPHERIC LIGHT
    # ========================================================

    atmospheric_light = (

        pixelwise_atmospheric_light(

            image,

            AF,

            AN

        )

    )


    # ========================================================
    # STEP 8
    # GAMMA-BASED PIXEL-WISE TRANSMISSION
    # ========================================================

    transmission = (

        pixelwise_transmission(

            image,

            atmospheric_light,

            omega=params.omega,

            gamma=params.gamma,

            t_min=params.t_min

        )

    )


    # ========================================================
    # STEP 9
    # TRANSMISSION REFINEMENT
    # ========================================================

    transmission = (

        refine_transmission(

            transmission,

            radius=params.bilateral_d,

            sigma_color=(
                params.bilateral_sigma_color
            ),

            sigma_space=(
                params.bilateral_sigma_space
            )

        )

    )


    # ========================================================
    # STEP 10
    # SCENE RECOVERY
    # ========================================================

    dehazed = scene_dehazing(

        image,

        atmospheric_light,

        transmission

    )


    # ========================================================
    # STEP 11
    # OPTIONAL CLARITY ENHANCEMENT
    # ========================================================

    if getattr(

        params,

        "enable_clarity",

        False

    ):

        # ----------------------------------------------------
        # clarity_strength is used as the sharpening amount
        # ----------------------------------------------------

        clarity_strength = float(

            getattr(

                params,

                "clarity_strength",

                0.20

            )

        )


        dehazed = enhance_clarity(

            dehazed,

            sharpen_amount=clarity_strength

        )


    # ========================================================
    # STEP 12
    # FINAL RGB UINT8 IMAGE
    # ========================================================

    dehazed = np.clip(

        dehazed,

        0.0,

        1.0

    )


    dehazed_rgb = (

        dehazed

        *

        255.0

    ).round().astype(

        np.uint8

    )


    # ========================================================
    # STEP 13
    # FINAL OUTPUT VALIDATION
    # ========================================================

    if dehazed_rgb.ndim != 3:

        raise RuntimeError(

            "Dehazing output does not have "
            "3 dimensions."

        )


    if dehazed_rgb.shape[2] != 3:

        raise RuntimeError(

            "Dehazing output does not have "
            "3 RGB channels."

        )


    # ========================================================
    # STEP 14
    # DEBUG INFORMATION
    # ========================================================

    debug = {

        # ----------------------------------------------------
        # Atmospheric light
        # ----------------------------------------------------

        "AF":
            AF,

        "AN":
            AN,


        # ----------------------------------------------------
        # Threshold
        # ----------------------------------------------------

        "Th":
            Th,


        # ----------------------------------------------------
        # Haze strength
        # ----------------------------------------------------

        "haze_strength":
            haze_strength,


        # ----------------------------------------------------
        # Dark channel
        # ----------------------------------------------------

        "dark_channel":
            dark_channel,

        "dark_down":
            dark_down,


        # ----------------------------------------------------
        # Atmospheric light image
        # ----------------------------------------------------

        "atmospheric_light":
            atmospheric_light,


        # ----------------------------------------------------
        # Transmission map
        # ----------------------------------------------------

        "tm":
            transmission,


        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        "gamma":
            params.gamma,

        "omega":
            params.omega,

        "t_min":
            params.t_min,

        "patch_size":
            params.patch_size,

        "down_factor":
            params.down_factor,


        # ----------------------------------------------------
        # Clarity
        # ----------------------------------------------------

        "post_clarity_enabled":
            getattr(
                params,
                "enable_clarity",
                False
            ),

        "clarity_strength":
            getattr(
                params,
                "clarity_strength",
                0.20
            ),

        # ----------------------------------------------------
        # Format information
        # ----------------------------------------------------

        "input_format":
            "RGB",

        "output_format":
            "RGB"

    }


    # ========================================================
    # RETURN
    # ========================================================

    return (

        dehazed_rgb,

        None,

        debug

    )