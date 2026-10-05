"""
Run the complete dehazing algorithm on one image
and display the result.
"""

from pathlib import Path
import time
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PROJECT DIRECTORY
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# DEHAZING DIRECTORY
# ============================================================

DEHAZING_DIR = PROJECT_DIR / "dehazing"

if str(DEHAZING_DIR) not in sys.path:
    sys.path.insert(0, str(DEHAZING_DIR))


# ============================================================
# IMPORT DEHAZING MODULE
# ============================================================

import adaptive_transmission_dehaze as dehaze_module


# ============================================================
# INPUT IMAGE
# ============================================================

INPUT_IMAGE = (
    PROJECT_DIR
    / "dataset"
    / "CNN_Dataset"
    / "test"
    / "hazy"
    / "0058_0.8_0.2.jpg"
)


# ============================================================
# OUTPUT IMAGE
# ============================================================

OUTPUT_IMAGE = (
    PROJECT_DIR
    / "output"
    / "demo_dehazed.png"
)


# ============================================================
# READ IMAGE
# ============================================================

def read_image(path):
    """
    Read image in the format expected by adaptive_dehaze().

    OpenCV reads images as:
        BGR
        uint8
        range [0, 255]
    """

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise RuntimeError(
            f"Could not read image:\n{path}"
        )

    return image


# ============================================================
# CONVERT IMAGE FOR DISPLAY
# ============================================================

def prepare_for_display(image):
    """
    Convert an image to RGB format in [0, 1]
    for matplotlib.
    """

    image = np.asarray(image)

    # --------------------------------------------------------
    # FLOAT IMAGE
    # --------------------------------------------------------

    if np.issubdtype(
        image.dtype,
        np.floating
    ):

        image = np.nan_to_num(
            image
        )

        # If image is in [0, 1]
        if image.max() <= 1.0:

            image = (
                np.clip(
                    image,
                    0.0,
                    1.0
                )
                * 255.0
            )

        else:

            image = np.clip(
                image,
                0.0,
                255.0
            )

        image = image.astype(
            np.uint8
        )

    # --------------------------------------------------------
    # INTEGER IMAGE
    # --------------------------------------------------------

    else:

        image = np.clip(
            image,
            0,
            255
        ).astype(
            np.uint8
        )


    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    if image.ndim == 3:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

    return (
        image.astype(np.float32)
        / 255.0
    )


# ============================================================
# SAVE IMAGE
# ============================================================

def save_output(path, image):
    """
    Save dehazed output regardless of whether the
    dehazing function returns uint8 or floating-point data.
    """

    image = np.asarray(
        image
    )

    # --------------------------------------------------------
    # FLOAT OUTPUT
    # --------------------------------------------------------

    if np.issubdtype(
        image.dtype,
        np.floating
    ):

        image = np.nan_to_num(
            image
        )

        if image.max() <= 1.0:

            image = (
                np.clip(
                    image,
                    0.0,
                    1.0
                )
                * 255.0
            )

        else:

            image = np.clip(
                image,
                0.0,
                255.0
            )

    # --------------------------------------------------------
    # INTEGER OUTPUT
    # --------------------------------------------------------

    else:

        image = np.clip(
            image,
            0,
            255
        )


    image = image.astype(
        np.uint8
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(path),
        image
    )


# ============================================================
# GET DEHAZING FUNCTION
# ============================================================

def get_dehazing_function():

    function = getattr(
        dehaze_module,
        "adaptive_dehaze",
        None
    )

    if not callable(function):

        print(
            "\n============================================"
        )

        print(
            " ERROR: adaptive_dehaze() NOT FOUND"
        )

        print(
            "============================================"
        )

        print(
            "\nAvailable functions:"
        )

        for name in dir(
            dehaze_module
        ):

            if name.startswith("_"):
                continue

            obj = getattr(
                dehaze_module,
                name
            )

            if callable(obj):

                print(
                    f"  - {name}"
                )

        raise ImportError(
            "adaptive_dehaze() was not found."
        )

    return function


# ============================================================
# EXTRACT RESULT
# ============================================================

def extract_result(result):

    dehazed_image = None
    state = None
    debug = {}

    # --------------------------------------------------------
    # TUPLE RESULT
    # --------------------------------------------------------

    if isinstance(
        result,
        tuple
    ):

        if len(result) >= 1:

            dehazed_image = result[0]

        if len(result) >= 2:

            state = result[1]

        if len(result) >= 3:

            if isinstance(
                result[2],
                dict
            ):

                debug = result[2]

    # --------------------------------------------------------
    # IMAGE ONLY
    # --------------------------------------------------------

    else:

        dehazed_image = result

    return (
        dehazed_image,
        state,
        debug
    )


# ============================================================
# PRINT DEBUG INFORMATION
# ============================================================

def print_debug_information(debug):

    if not isinstance(
        debug,
        dict
    ):

        return


    print(
        "\n--------------------------------------------"
    )

    print(
        " DEBUG INFORMATION"
    )

    print(
        "--------------------------------------------"
    )


    # --------------------------------------------------------
    # AF
    # --------------------------------------------------------

    if "AF" in debug:

        print(
            f"Far AF           : "
            f"{np.round(debug['AF'], 4)}"
        )


    # --------------------------------------------------------
    # AN
    # --------------------------------------------------------

    if "AN" in debug:

        print(
            f"Near AN          : "
            f"{np.round(debug['AN'], 4)}"
        )


    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    if "Th" in debug:

        print(
            f"Threshold Th     : "
            f"{debug['Th']:.4f}"
        )


    # --------------------------------------------------------
    # TRANSMISSION
    # --------------------------------------------------------

    if "tm" in debug:

        tm = np.asarray(
            debug["tm"]
        )

        print(
            f"Transmission     : "
            f"["
            f"{tm.min():.4f}, "
            f"{tm.max():.4f}"
            f"]"
        )


    # --------------------------------------------------------
    # S-FCAE
    # --------------------------------------------------------

    if "sfcae_used_fallback" in debug:

        if debug[
            "sfcae_used_fallback"
        ]:

            print(
                "S-FCAE           : "
                "fallback to brightest-pixel rule"
            )

        else:

            if (
                "sfcae_sky_candidate_mask_far"
                in debug
            ):

                count = np.count_nonzero(
                    debug[
                        "sfcae_sky_candidate_mask_far"
                    ]
                )

                print(
                    f"S-FCAE           : "
                    f"{count} candidate pixels"
                )


    # --------------------------------------------------------
    # POST GAMMA
    # --------------------------------------------------------

    if "post_gamma_value" in debug:

        print(
            f"Post gamma       : "
            f"{debug['post_gamma_value']:.6f}"
        )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    input_image,
    dehazed_image,
    debug
):

    # --------------------------------------------------------
    # INPUT FOR DISPLAY
    # --------------------------------------------------------

    input_display = prepare_for_display(
        input_image
    )


    # --------------------------------------------------------
    # OUTPUT FOR DISPLAY
    # --------------------------------------------------------

    output_display = prepare_for_display(
        dehazed_image
    )


    # --------------------------------------------------------
    # CHECK DEBUG MAPS
    # --------------------------------------------------------

    has_transmission = (
        isinstance(debug, dict)
        and "tm" in debug
    )

    has_sfcae = (
        isinstance(debug, dict)
        and "sfcae_S_sky" in debug
    )


    # --------------------------------------------------------
    # FIGURE
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8)
    )


    # ========================================================
    # INPUT
    # ========================================================

    axes[0, 0].imshow(
        input_display
    )

    axes[0, 0].set_title(
        "Hazy input"
    )

    axes[0, 0].axis(
        "off"
    )


    # ========================================================
    # OUTPUT
    # ========================================================

    axes[0, 1].imshow(
        output_display
    )

    axes[0, 1].set_title(
        "Dehazed output"
    )

    axes[0, 1].axis(
        "off"
    )


    # ========================================================
    # TRANSMISSION MAP
    # ========================================================

    if has_transmission:

        axes[1, 0].imshow(
            debug["tm"],
            cmap="gray"
        )

        axes[1, 0].set_title(
            "Transmission map"
        )

    else:

        axes[1, 0].text(
            0.5,
            0.5,
            "Transmission map\n"
            "not returned",
            ha="center",
            va="center"
        )

        axes[1, 0].set_title(
            "Transmission map"
        )

    axes[1, 0].axis(
        "off"
    )


    # ========================================================
    # S-FCAE
    # ========================================================

    if has_sfcae:

        axes[1, 1].imshow(
            debug["sfcae_S_sky"],
            cmap="gray"
        )

        axes[1, 1].set_title(
            "S-FCAE sky confidence"
        )

    else:

        axes[1, 1].text(
            0.5,
            0.5,
            "S-FCAE sky confidence\n"
            "not returned",
            ha="center",
            va="center"
        )

        axes[1, 1].set_title(
            "S-FCAE sky confidence"
        )

    axes[1, 1].axis(
        "off"
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    fig.tight_layout()

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # HEADER
    # ========================================================

    print(
        "\n============================================"
    )

    print(
        " SINGLE IMAGE DEHAZING"
    )

    print(
        "============================================"
    )


    # ========================================================
    # INPUT CHECK
    # ========================================================

    print(
        f"\nInput image:\n"
        f"{INPUT_IMAGE}"
    )

    if not INPUT_IMAGE.exists():

        raise FileNotFoundError(
            "\nInput image does not exist:\n"
            f"{INPUT_IMAGE}"
        )


    # ========================================================
    # READ IMAGE
    # ========================================================

    image = read_image(
        INPUT_IMAGE
    )


    print(
        f"\nImage size       : "
        f"{image.shape[1]} x "
        f"{image.shape[0]}"
    )

    print(
        f"Input dtype      : "
        f"{image.dtype}"
    )

    print(
        f"Input channels   : "
        f"{image.shape[2]}"
    )


    # ========================================================
    # GET FUNCTION
    # ========================================================

    dehaze_function = (
        get_dehazing_function()
    )


    print(
        "\nDehazing function: "
        "adaptive_dehaze(image)"
    )


    # ========================================================
    # RUN DEHAZING
    # ========================================================

    print(
        "\nProcessing image..."
    )

    start = time.perf_counter()


    # IMPORTANT:
    #
    # Your adaptive_dehaze() function accepts
    # ONLY ONE ARGUMENT.
    #
    # The input is uint8 BGR because the function
    # internally uses cv2.cvtColor().

    result = dehaze_function(
        image
    )


    elapsed = (
        time.perf_counter()
        - start
    )


    # ========================================================
    # EXTRACT RESULT
    # ========================================================

    (
        dehazed_image,
        state,
        debug
    ) = extract_result(
        result
    )


    # ========================================================
    # CHECK OUTPUT
    # ========================================================

    if dehazed_image is None:

        raise RuntimeError(
            "adaptive_dehaze() did not return "
            "a valid dehazed image."
        )


    dehazed_image = np.asarray(
        dehazed_image
    )


    # ========================================================
    # RESULTS
    # ========================================================

    print(
        "\n============================================"
    )

    print(
        " DEHAZING RESULTS"
    )

    print(
        "============================================"
    )

    print(
        f"Processing time  : "
        f"{elapsed:.3f} s"
    )

    print(
        f"Output shape     : "
        f"{dehazed_image.shape}"
    )

    print(
        f"Output dtype     : "
        f"{dehazed_image.dtype}"
    )


    # ========================================================
    # DEBUG
    # ========================================================

    print_debug_information(
        debug
    )


    # ========================================================
    # SAVE OUTPUT
    # ========================================================

    save_output(
        OUTPUT_IMAGE,
        dehazed_image
    )


    print(
        f"\nSaved            : "
        f"{OUTPUT_IMAGE}"
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    display_results(
        image,
        dehazed_image,
        debug
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()