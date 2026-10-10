"""
Run the complete RGB-only dehazing algorithm on one image,
calculate PSNR, SSIM, FSIM, EPI and residual, and display
the results.
"""

from pathlib import Path
import time
import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# PROJECT DIRECTORY
# ============================================================

PROJECT_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


# ============================================================
# IMPORT PROJECT
# ============================================================

if str(PROJECT_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_DIR)
    )


# ============================================================
# IMPORT DEHAZING
# ============================================================

from dehazing.adaptive_transmission_dehaze import (
    adaptive_dehaze
)

from dehazing.default_dehaze_params import (
    DehazeParams
)

from dehazing.evaluation_metrics import (
    calculate_all_metrics
)


# ============================================================
# INPUT IMAGE
# ============================================================

INPUT_IMAGE = (
    PROJECT_DIR
    / "dataset"
    / "CNN_Dataset"
    / "test"
    / "hazy"
    / "0024_0.85_0.08.jpg"
)


# ============================================================
# GROUND TRUTH / CLEAR IMAGE
# ============================================================

GROUND_TRUTH_IMAGE = (
    PROJECT_DIR
    / "dataset"
    / "CNN_Dataset"
    / "test"
    / "clear"
    / "0024_0.85_0.08.jpg"
)


# ============================================================
# OUTPUT IMAGE
# ============================================================

OUTPUT_IMAGE = (
    PROJECT_DIR
    / "output"
    / "demo_dehazed_rgb_T.png"
)


# ============================================================
# RESIDUAL OUTPUT
# ============================================================

RESIDUAL_IMAGE = (
    PROJECT_DIR
    / "output"
    / "residual_rgb_T.png"
)


# ============================================================
# READ RGB IMAGE
# ============================================================

def read_image(
    path
):
    """
    Read image directly as RGB.

    PIL guarantees:

        channel 0 = R
        channel 1 = G
        channel 2 = B

    No BGR/RGB conversion.
    """

    image = Image.open(
        path
    ).convert(
        "RGB"
    )

    image = np.array(
        image,
        dtype=np.uint8
    )

    return image


# ============================================================
# PREPARE FOR DISPLAY
# ============================================================

def prepare_for_display(
    image
):
    """
    Prepare an already-RGB image for matplotlib.

    No BGR/RGB conversion.
    """

    image = np.asarray(
        image
    )

    if np.issubdtype(
        image.dtype,
        np.floating
    ):

        image = np.nan_to_num(
            image
        )

        if np.max(image) <= 1.0:

            image = np.clip(
                image,
                0.0,
                1.0
            )

        else:

            image = (
                np.clip(
                    image,
                    0.0,
                    255.0
                )
                /
                255.0
            )

    else:

        image = (
            np.clip(
                image,
                0,
                255
            )
            /
            255.0
        )

    return image.astype(
        np.float32
    )


# ============================================================
# SAVE RGB IMAGE
# ============================================================

def save_output(
    path,
    image
):
    """
    Save RGB image using PIL.

    No BGR conversion.
    """

    image = np.asarray(
        image
    )

    if np.issubdtype(
        image.dtype,
        np.floating
    ):

        image = np.nan_to_num(
            image
        )

        if np.max(image) <= 1.0:

            image = (
                np.clip(
                    image,
                    0.0,
                    1.0
                )
                *
                255.0
            )

        else:

            image = np.clip(
                image,
                0.0,
                255.0
            )

    else:

        image = np.clip(
            image,
            0,
            255
        )

    image = image.astype(
        np.uint8
    )

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    Image.fromarray(
        image,
        mode="RGB"
    ).save(
        path
    )


# ============================================================
# SAVE RESIDUAL
# ============================================================

def save_residual_image(
    path,
    residual
):
    """
    Save residual image.

    Residual is in [0,1].
    """

    residual = np.asarray(
        residual,
        dtype=np.float32
    )

    residual = np.clip(
        residual,
        0.0,
        1.0
    )

    residual_uint8 = (
        residual
        *
        255.0
    ).round().astype(
        np.uint8
    )

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    Image.fromarray(
        residual_uint8,
        mode="RGB"
    ).save(
        path
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    input_image,
    ground_truth,
    dehazed_image,
    residual,
    debug,
    metrics
):
    """
    Display:

        Hazy input
        Clear reference
        Dehazed output
        Residual
    """

    input_display = (
        prepare_for_display(
            input_image
        )
    )

    ground_truth_display = (
        prepare_for_display(
            ground_truth
        )
    )

    output_display = (
        prepare_for_display(
            dehazed_image
        )
    )

    residual_display = np.clip(
        residual,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8)
    )

    # ========================================================
    # HAZY INPUT
    # ========================================================

    axes[0, 0].imshow(
        input_display
    )

    axes[0, 0].set_title(
        "Hazy RGB Input"
    )

    axes[0, 0].axis(
        "off"
    )

    # ========================================================
    # CLEAR REFERENCE
    # ========================================================

    axes[0, 1].imshow(
        ground_truth_display
    )

    axes[0, 1].set_title(
        "Clear / Ground Truth"
    )

    axes[0, 1].axis(
        "off"
    )

    # ========================================================
    # DEHAZED
    # ========================================================

    axes[1, 0].imshow(
        output_display
    )

    axes[1, 0].set_title(
        "RGB Dehazed Output"
    )

    axes[1, 0].axis(
        "off"
    )

    # ========================================================
    # RESIDUAL
    # ========================================================

    axes[1, 1].imshow(
        residual_display
    )

    axes[1, 1].set_title(
        "Absolute Residual"
    )

    axes[1, 1].axis(
        "off"
    )

    # ========================================================
    # METRIC TEXT
    # ========================================================

    fig.suptitle(
        (
            f"PSNR: "
            f"{metrics['PSNR (dB)']:.2f} dB   |   "
            f"SSIM: "
            f"{metrics['SSIM']:.4f}   |   "
            f"FSIM: "
            f"{metrics['FSIM']:.4f}   |   "
            f"EPI: "
            f"{metrics['EPI']:.4f}"
        ),
        fontsize=12
    )

    fig.tight_layout()

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "============================================"
    )

    print(
        " RGB-ONLY SINGLE IMAGE DEHAZING"
    )

    print(
        "============================================"
    )

    # ========================================================
    # INPUT CHECK
    # ========================================================

    print()
    print(
        f"Input image:\n"
        f"{INPUT_IMAGE}"
    )

    if not INPUT_IMAGE.exists():

        raise FileNotFoundError(
            f"\nInput image does not exist:\n"
            f"{INPUT_IMAGE}"
        )

    if not GROUND_TRUTH_IMAGE.exists():

        raise FileNotFoundError(
            f"\nGround-truth image does not exist:\n"
            f"{GROUND_TRUTH_IMAGE}"
        )

    # ========================================================
    # READ HAZY RGB
    # ========================================================

    image = read_image(
        INPUT_IMAGE
    )

    # ========================================================
    # READ CLEAR RGB
    # ========================================================

    ground_truth = read_image(
        GROUND_TRUTH_IMAGE
    )

    print()
    print(
        f"Image size       : "
        f"{image.shape[1]} x "
        f"{image.shape[0]}"
    )

    print(
        f"Input dtype      : "
        f"{image.dtype}"
    )

    print(
        f"Input shape      : "
        f"{image.shape}"
    )

    print(
        "Input format     : RGB"
    )

    print(
        f"Ground truth     : "
        f"{ground_truth.shape}"
    )

    # ========================================================
    # CHECK SIZE
    # ========================================================

    if image.shape != ground_truth.shape:

        raise ValueError(
            "\nHazy and ground-truth images "
            "must have the same dimensions.\n"
            f"Hazy : {image.shape}\n"
            f"Clear: {ground_truth.shape}"
        )

    # ========================================================
    # PARAMETERS
    # ========================================================

    params = DehazeParams(

        down_factor=3,

        patch_size=5,

        gamma=2.0,

        omega=0.9375,

        t_min=0.10,

        bilateral_d=9,

        bilateral_sigma_color=0.05,

        bilateral_sigma_space=15,

        atmospheric_top_fraction=0.001,

        # IMPORTANT:
        # Keep false initially.
        enable_clarity=True,

        clarity_strength=0.20
    )

    # ========================================================
    # PRINT PARAMETERS
    # ========================================================

    print()
    print(
        "--------------------------------------------"
    )

    print(
        " DEHAZING PARAMETERS"
    )

    print(
        "--------------------------------------------"
    )

    print(
        f"Gamma            : "
        f"{params.gamma}"
    )

    print(
        f"Omega            : "
        f"{params.omega}"
    )

    print(
        f"Patch size       : "
        f"{params.patch_size}"
    )

    print(
        f"Minimum t        : "
        f"{params.t_min}"
    )

    print(
        f"Down factor      : "
        f"{params.down_factor}"
    )

    print(
        f"Clarity          : "
        f"{params.enable_clarity}"
    )

    # ========================================================
    # RUN DEHAZING
    # ========================================================

    print()
    print(
        "Processing image..."
    )

    start = time.perf_counter()

    result = adaptive_dehaze(
        image,
        params
    )

    elapsed = (
        time.perf_counter()
        -
        start
    )

    # ========================================================
    # EXTRACT RESULT
    # ========================================================

    if isinstance(
        result,
        tuple
    ):

        dehazed_image = result[0]

        if len(result) >= 3:

            debug = result[2]

        else:

            debug = {}

    else:

        dehazed_image = result

        debug = {}

    if dehazed_image is None:

        raise RuntimeError(
            "adaptive_dehaze() did not return "
            "a valid image."
        )

    dehazed_image = np.asarray(
        dehazed_image
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print(
        "============================================"
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

    print(
        "Output format    : RGB"
    )

    # ========================================================
    # DEBUG
    # ========================================================

    print()
    print(
        "--------------------------------------------"
    )

    print(
        " DEBUG INFORMATION"
    )

    print(
        "--------------------------------------------"
    )

    if "AF" in debug:

        print(
            f"Far AF           : "
            f"{np.round(debug['AF'], 4)}"
        )

    if "AN" in debug:

        print(
            f"Near AN          : "
            f"{np.round(debug['AN'], 4)}"
        )

    if "Th" in debug:

        print(
            f"Threshold Th     : "
            f"{debug['Th']:.4f}"
        )

    if "haze_strength" in debug:

        print(
            f"Haze strength    : "
            f"{debug['haze_strength']:.4f}"
        )

    if "tm" in debug:

        tm = np.asarray(
            debug["tm"]
        )

        print(
            f"Transmission     : "
            f"[{tm.min():.4f}, "
            f"{tm.max():.4f}]"
        )

    print(
        f"Gamma used       : "
        f"{params.gamma}"
    )

    print(
        f"Omega used       : "
        f"{params.omega}"
    )

    # ========================================================
    # SAVE DEHAZED OUTPUT
    # ========================================================

    save_output(
        OUTPUT_IMAGE,
        dehazed_image
    )

    print()
    print(
        f"Saved RGB output:\n"
        f"{OUTPUT_IMAGE}"
    )

    # ========================================================
    # QUALITY EVALUATION
    # ========================================================

    print()
    print(
        "============================================"
    )

    print(
        " IMAGE QUALITY EVALUATION"
    )

    print(
        "============================================"
    )

    (
        metrics,
        residual
    ) = calculate_all_metrics(
        ground_truth,
        dehazed_image
    )

    # ========================================================
    # PRINT METRICS
    # ========================================================

    print()
    print(
        "--------------------------------------------"
    )

    print(
        f"PSNR (dB)          : "
        f"{metrics['PSNR (dB)']:.4f}"
    )

    print(
        f"SSIM               : "
        f"{metrics['SSIM']:.4f}"
    )

    print(
        f"FSIM               : "
        f"{metrics['FSIM']:.4f}"
    )

    print(
        f"EPI                : "
        f"{metrics['EPI']:.4f}"
    )

    print(
        f"Residual MAE       : "
        f"{metrics['Residual MAE']:.6f}"
    )

    print(
        f"Residual MSE       : "
        f"{metrics['Residual MSE']:.6f}"
    )

    print(
        f"Maximum Residual   : "
        f"{metrics['Maximum Residual']:.6f}"
    )

    print(
        "--------------------------------------------"
    )

    # ========================================================
    # SAVE RESIDUAL
    # ========================================================

    save_residual_image(
        RESIDUAL_IMAGE,
        residual
    )

    print()
    print(
        f"Residual saved:\n"
        f"{RESIDUAL_IMAGE}"
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    display_results(
        image,
        ground_truth,
        dehazed_image,
        residual,
        debug,
        metrics
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()