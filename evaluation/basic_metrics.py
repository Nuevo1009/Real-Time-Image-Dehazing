import numpy as np

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)


# ============================================================
# RGB -> GRAYSCALE
# ============================================================

def rgb_to_gray(image):
    """
    Convert RGB image to grayscale using RGB luminance weights.

    No BGR conversion is performed.

    Input:
        RGB uint8 or float image

    Output:
        grayscale float32
    """

    image = np.asarray(
        image,
        dtype=np.float32
    )

    if image.max() > 1.0:
        image = image / 255.0

    gray = (
        0.299 * image[:, :, 0]
        +
        0.587 * image[:, :, 1]
        +
        0.114 * image[:, :, 2]
    )

    return gray.astype(
        np.float32
    )


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(
    output_rgb,
    ground_truth_rgb
):
    """
    Calculate PSNR between RGB images.

    Higher is better.
    """

    output = np.asarray(
        output_rgb,
        dtype=np.float32
    )

    ground_truth = np.asarray(
        ground_truth_rgb,
        dtype=np.float32
    )

    score = peak_signal_noise_ratio(
        ground_truth,
        output,
        data_range=255
    )

    return float(score)


# ============================================================
# SSIM
# ============================================================

def calculate_ssim(
    output_rgb,
    ground_truth_rgb
):
    """
    Calculate SSIM using RGB-derived luminance.

    Higher is better.
    """

    output_gray = rgb_to_gray(
        output_rgb
    )

    ground_truth_gray = rgb_to_gray(
        ground_truth_rgb
    )

    min_dimension = min(
        output_gray.shape[0],
        output_gray.shape[1]
    )

    if min_dimension >= 7:

        win_size = 7

    else:

        win_size = min_dimension

        if win_size % 2 == 0:
            win_size -= 1

    if win_size < 3:

        raise ValueError(
            "Image is too small for SSIM."
        )

    score = structural_similarity(
        ground_truth_gray,
        output_gray,
        data_range=1.0,
        win_size=win_size
    )

    return float(score)


# ============================================================
# BASIC METRICS
# ============================================================

def calculate_basic_metrics(
    output_rgb,
    ground_truth_rgb
):
    """
    Calculate PSNR and SSIM.
    """

    return {

        "psnr":
            calculate_psnr(
                output_rgb,
                ground_truth_rgb
            ),

        "ssim":
            calculate_ssim(
                output_rgb,
                ground_truth_rgb
            )
    }