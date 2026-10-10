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
# FSIM
# ============================================================

def calculate_fsim(
    output_rgb,
    ground_truth_rgb
):
    """
    Feature Similarity approximation.

    Higher is better.

    RGB-only input.
    """

    output_gray = rgb_to_gray(
        output_rgb
    )

    gt_gray = rgb_to_gray(
        ground_truth_rgb
    )

    # --------------------------------------------------------
    # Sobel gradient
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Gradient similarity
    # --------------------------------------------------------

    T = 0.01

    gradient_similarity = (

        2.0
        *
        output_gradient
        *
        gt_gradient
        +
        T

    ) / (

        output_gradient ** 2
        +
        gt_gradient ** 2
        +
        T
    )

    # --------------------------------------------------------
    # Laplacian feature response
    # --------------------------------------------------------

    output_feature = np.abs(
        cv2.Laplacian(
            output_gray,
            cv2.CV_32F
        )
    )

    gt_feature = np.abs(
        cv2.Laplacian(
            gt_gray,
            cv2.CV_32F
        )
    )

    output_feature /= (
        np.max(output_feature)
        +
        1e-8
    )

    gt_feature /= (
        np.max(gt_feature)
        +
        1e-8
    )

    feature_similarity = (

        2.0
        *
        output_feature
        *
        gt_feature
        +
        T

    ) / (

        output_feature ** 2
        +
        gt_feature ** 2
        +
        T
    )

    # --------------------------------------------------------
    # FSIM map
    # --------------------------------------------------------

    similarity_map = (
        gradient_similarity
        *
        feature_similarity
    )

    weight = np.maximum(
        output_feature,
        gt_feature
    )

    denominator = (
        np.sum(weight)
        +
        1e-8
    )

    score = (
        np.sum(
            similarity_map
            *
            weight
        )
        /
        denominator
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0
        )
    )