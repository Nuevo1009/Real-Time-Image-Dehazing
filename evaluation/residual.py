import numpy as np
import cv2


# ============================================================
# PROCESSING CHANGE / NO-REFERENCE RESIDUAL
# ============================================================

def calculate_processing_change(input_image, output_image):
    """
    Calculate the absolute change introduced by image processing.

    This does NOT require ground truth.

    Parameters
    ----------
    input_image : numpy.ndarray
        Original input image.

    output_image : numpy.ndarray
        Processed/dehazed image.

    Returns
    -------
    change_map : numpy.ndarray
        Absolute difference between input and output.

    mean_change : float
        Average processing change.
    """

    # Ensure both images have the same dimensions
    if input_image.shape != output_image.shape:

        output_image = cv2.resize(
            output_image,
            (
                input_image.shape[1],
                input_image.shape[0]
            )
        )

    # Convert to float
    input_float = input_image.astype(np.float32)
    output_float = output_image.astype(np.float32)

    # Absolute pixel-wise difference
    change_map = np.abs(
        output_float - input_float
    )

    # Mean processing change
    mean_change = np.mean(
        change_map
    )

    return change_map, mean_change


# ============================================================
# GROUND TRUTH RESIDUAL
# ============================================================

def calculate_residual(output_image, ground_truth_image):
    """
    Calculate residual between dehazed output
    and ground truth image.

    Requires ground truth.
    """

    if output_image.shape != ground_truth_image.shape:

        ground_truth_image = cv2.resize(
            ground_truth_image,
            (
                output_image.shape[1],
                output_image.shape[0]
            )
        )

    output_float = output_image.astype(np.float32)
    ground_truth_float = ground_truth_image.astype(
        np.float32
    )

    residual = np.abs(
        output_float - ground_truth_float
    )

    return residual


# ============================================================
# NORMALIZE RESIDUAL FOR SAVING
# ============================================================

def normalize_residual(residual):
    """
    Convert residual image into uint8 format
    for visualization and saving.
    """

    residual = np.asarray(
        residual,
        dtype=np.float32
    )

    max_value = residual.max()

    if max_value > 0:

        residual = (
            residual / max_value
        ) * 255.0

    return residual.astype(np.uint8)