import cv2
import numpy as np


# ============================================================
# RGB -> GRAY
# ============================================================

def rgb_to_gray(
    image
):

    image = image.astype(
        np.float32
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

    return gray


# ============================================================
# EPI
# ============================================================

def calculate_epi(
    output_rgb,
    ground_truth_rgb
):
    """
    Edge Preservation Index.

    Compares edge structures between:
        dehazed output
        ground truth

    Higher is better.
    """

    output_gray = rgb_to_gray(
        output_rgb
    )

    gt_gray = rgb_to_gray(
        ground_truth_rgb
    )

    output_x = cv2.Sobel(
        output_gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    output_y = cv2.Sobel(
        output_gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    gt_x = cv2.Sobel(
        gt_gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gt_y = cv2.Sobel(
        gt_gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    output_gradient = np.sqrt(
        output_x ** 2
        +
        output_y ** 2
    )

    gt_gradient = np.sqrt(
        gt_x ** 2
        +
        gt_y ** 2
    )

    numerator = np.sum(
        output_gradient
        *
        gt_gradient
    )

    denominator = np.sqrt(
        np.sum(
            output_gradient ** 2
        )
        *
        np.sum(
            gt_gradient ** 2
        )
    )

    if denominator <= 1e-12:

        return 0.0

    epi = numerator / denominator

    return float(
        np.clip(
            epi,
            0.0,
            1.0
        )
    )