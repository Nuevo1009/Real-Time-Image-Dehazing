import numpy as np

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)


# ==================================================
# CALCULATE BASIC IMAGE QUALITY METRICS
# ==================================================

def calculate_basic_metrics(output_image, ground_truth_image):

    """
    Calculates PSNR and SSIM between the output
    dehazed image and the ground truth clear image.

    Parameters
    ----------
    output_image : numpy.ndarray
        Dehazed/final output image in RGB format.

    ground_truth_image : numpy.ndarray
        Ground truth clear image in RGB format.

    Returns
    -------
    dict
        Dictionary containing PSNR and SSIM.
    """


    # ----------------------------------------------
    # CHECK IMAGE SIZE
    # ----------------------------------------------

    if output_image.shape != ground_truth_image.shape:

        raise ValueError(

            "Output image and ground truth image "
            "must have the same dimensions."
        )


    # ----------------------------------------------
    # CONVERT TO FLOAT
    # ----------------------------------------------

    output_image = output_image.astype(

        np.float32
    )


    ground_truth_image = ground_truth_image.astype(

        np.float32
    )


    # ----------------------------------------------
    # PSNR
    # ----------------------------------------------

    psnr_score = peak_signal_noise_ratio(

        ground_truth_image,

        output_image,

        data_range=255
    )


    # ----------------------------------------------
    # SSIM
    # ----------------------------------------------

    ssim_score = structural_similarity(

        ground_truth_image,

        output_image,

        channel_axis=-1,

        data_range=255
    )


    # ----------------------------------------------
    # RETURN RESULTS
    # ----------------------------------------------

    return {

        "psnr": float(psnr_score),

        "ssim": float(ssim_score)
    }