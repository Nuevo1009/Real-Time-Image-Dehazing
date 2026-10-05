import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def rgb_to_lab(
    image
):
    """
    Converts RGB image to CIELAB.
    """

    image_float = image.astype(
        np.float32
    ) / 255.0

    lab = cv2.cvtColor(
        image_float,
        cv2.COLOR_RGB2LAB
    )

    return lab


def spatial_filter_lab(
    lab_image,
    sigma=1.5
):
    """
    Spatial filtering of CIELAB channels.

    Practical spatial perceptual approximation.
    """

    filtered = np.zeros_like(
        lab_image
    )

    for channel in range(3):

        filtered[:, :, channel] = (
            gaussian_filter(
                lab_image[:, :, channel],
                sigma=sigma
            )
        )

    return filtered


def calculate_scielab(
    output_image,
    ground_truth,
    sigma=1.5
):
    """
    Practical spatial CIELAB error.

    Lower value = closer perceptual match.
    """

    output_lab = rgb_to_lab(
        output_image
    )

    ground_truth_lab = rgb_to_lab(
        ground_truth
    )

    output_filtered = spatial_filter_lab(
        output_lab,
        sigma
    )

    gt_filtered = spatial_filter_lab(
        ground_truth_lab,
        sigma
    )

    delta_l = (
        output_filtered[:, :, 0]
        -
        gt_filtered[:, :, 0]
    )

    delta_a = (
        output_filtered[:, :, 1]
        -
        gt_filtered[:, :, 1]
    )

    delta_b = (
        output_filtered[:, :, 2]
        -
        gt_filtered[:, :, 2]
    )

    delta_e = np.sqrt(
        delta_l ** 2
        +
        delta_a ** 2
        +
        delta_b ** 2
    )

    mean_delta_e = np.mean(
        delta_e
    )

    return float(
        mean_delta_e
    )