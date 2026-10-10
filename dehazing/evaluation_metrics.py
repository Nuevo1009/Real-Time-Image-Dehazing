import numpy as np
import cv2

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)



# ============================================================
# PREPARE IMAGE
# ============================================================

def prepare_image(
    image
):
    """
    Convert an already-RGB image to float32 [0,1].

    No BGR/RGB conversion.
    """

    image = np.asarray(
        image
    )

    if image.ndim != 3:
        raise ValueError(
            "Image must have shape H x W x 3."
        )

    if image.shape[2] != 3:
        raise ValueError(
            "Image must have 3 RGB channels."
        )

    if np.issubdtype(
        image.dtype,
        np.floating
    ):

        image = np.nan_to_num(
            image
        )

        if np.max(image) > 1.0:

            image = (
                image / 255.0
            )

    else:

        image = (
            image.astype(
                np.float32
            )
            /
            255.0
        )

    return np.clip(
        image,
        0.0,
        1.0
    ).astype(
        np.float32
    )


# ============================================================
# CHECK IMAGE COMPATIBILITY
# ============================================================

def check_images(
    reference,
    test
):
    """
    Check dimensions of reference and test images.
    """

    if reference.shape != test.shape:

        raise ValueError(
            "\nReference and test images "
            "must have exactly the same dimensions.\n"
            f"Reference: {reference.shape}\n"
            f"Test     : {test.shape}"
        )


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(
    reference,
    test
):
    """
    Peak Signal-to-Noise Ratio.

    Higher is better.
    """

    reference = prepare_image(
        reference
    )

    test = prepare_image(
        test
    )

    check_images(
        reference,
        test
    )

    value = peak_signal_noise_ratio(
        reference,
        test,
        data_range=1.0
    )

    return float(
        value
    )


# ============================================================
# SSIM
# ============================================================

def calculate_ssim(
    reference,
    test
):
    """
    Structural Similarity Index.

    Higher is better.
    """

    reference = prepare_image(
        reference
    )

    test = prepare_image(
        test
    )

    check_images(
        reference,
        test
    )

    value = structural_similarity(
        reference,
        test,
        data_range=1.0,
        channel_axis=2
    )

    return float(
        value
    )


# ============================================================
# FSIM
# ============================================================

# ============================================================
# FSIM
# ============================================================

def calculate_fsim(
    reference,
    test
):
    """
    Feature Similarity Index Measure (FSIM).

    This implementation uses:
        1. Phase congruency approximation
        2. Gradient magnitude similarity

    Higher FSIM = better similarity.

    Input images are RGB.
    No BGR/RGB conversion is performed.
    """

    reference = prepare_image(
        reference
    )

    test = prepare_image(
        test
    )

    check_images(
        reference,
        test
    )

    # --------------------------------------------------------
    # Convert RGB to luminance
    # --------------------------------------------------------

    ref_gray = rgb_luminance(
        reference
    )

    test_gray = rgb_luminance(
        test
    )

    # --------------------------------------------------------
    # Sobel gradients
    # --------------------------------------------------------

    ref_x = cv2.Sobel(
        ref_gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    ref_y = cv2.Sobel(
        ref_gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    test_x = cv2.Sobel(
        test_gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    test_y = cv2.Sobel(
        test_gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    # --------------------------------------------------------
    # Gradient magnitude
    # --------------------------------------------------------

    ref_gradient = np.sqrt(
        ref_x ** 2
        +
        ref_y ** 2
    )

    test_gradient = np.sqrt(
        test_x ** 2
        +
        test_y ** 2
    )

    # --------------------------------------------------------
    # Gradient similarity
    #
    # GSim =
    # (2*G1*G2 + T)
    # /
    # (G1^2 + G2^2 + T)
    # --------------------------------------------------------

    T = 0.01

    gradient_similarity = (
        2.0
        *
        ref_gradient
        *
        test_gradient
        +
        T
    ) / (
        ref_gradient ** 2
        +
        test_gradient ** 2
        +
        T
    )

    # --------------------------------------------------------
    # Approximate phase congruency
    #
    # A simple edge-strength normalization is used
    # as a practical approximation.
    # --------------------------------------------------------

    ref_edge = np.abs(
        cv2.Laplacian(
            ref_gray,
            cv2.CV_32F
        )
    )

    test_edge = np.abs(
        cv2.Laplacian(
            test_gray,
            cv2.CV_32F
        )
    )

    ref_edge = (
        ref_edge
        /
        (
            np.max(
                ref_edge
            )
            +
            1e-8
        )
    )

    test_edge = (
        test_edge
        /
        (
            np.max(
                test_edge
            )
            +
            1e-8
        )
    )

    # --------------------------------------------------------
    # Edge / feature similarity
    # --------------------------------------------------------

    phase_similarity = (
        2.0
        *
        ref_edge
        *
        test_edge
        +
        T
    ) / (
        ref_edge ** 2
        +
        test_edge ** 2
        +
        T
    )

    # --------------------------------------------------------
    # FSIM map
    #
    # FSIM = PCm * Sg
    # --------------------------------------------------------

    fsim_map = (
        phase_similarity
        *
        gradient_similarity
    )

    # --------------------------------------------------------
    # Phase-congruency weighting
    # --------------------------------------------------------

    weight = np.maximum(
        ref_edge,
        test_edge
    )

    denominator = (
        np.sum(weight)
        +
        1e-8
    )

    value = (
        np.sum(
            fsim_map
            *
            weight
        )
        /
        denominator
    )

    return float(
        np.clip(
            value,
            0.0,
            1.0
        )
    )

# ============================================================
# RGB LUMINANCE
# ============================================================

def rgb_luminance(
    image
):
    """
    Calculate luminance directly from RGB.

        Y =
        0.299R +
        0.587G +
        0.114B

    This is not a BGR conversion.
    """

    image = prepare_image(
        image
    )

    luminance = (
        0.299 * image[:, :, 0]
        +
        0.587 * image[:, :, 1]
        +
        0.114 * image[:, :, 2]
    )

    return np.clip(
        luminance,
        0.0,
        1.0
    ).astype(
        np.float32
    )


# ============================================================
# EPI
# ============================================================

def calculate_epi(
    reference,
    test
):
    """
    Edge Preservation Index.

    This implementation measures the correlation
    between gradient-magnitude maps.

    Higher is better.
    """

    reference_gray = rgb_luminance(
        reference
    )

    test_gray = rgb_luminance(
        test
    )

    # --------------------------------------------------------
    # Reference gradients
    # --------------------------------------------------------

    ref_x = cv2.Sobel(
        reference_gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    ref_y = cv2.Sobel(
        reference_gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    # --------------------------------------------------------
    # Test gradients
    # --------------------------------------------------------

    test_x = cv2.Sobel(
        test_gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    test_y = cv2.Sobel(
        test_gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    # --------------------------------------------------------
    # Gradient magnitude
    # --------------------------------------------------------

    ref_gradient = np.sqrt(
        ref_x ** 2
        +
        ref_y ** 2
    )

    test_gradient = np.sqrt(
        test_x ** 2
        +
        test_y ** 2
    )

    ref_gradient = (
        ref_gradient.ravel()
    )

    test_gradient = (
        test_gradient.ravel()
    )

    # --------------------------------------------------------
    # Constant-image protection
    # --------------------------------------------------------

    if (
        np.std(ref_gradient) < 1e-12
        or
        np.std(test_gradient) < 1e-12
    ):

        return 0.0

    epi = np.corrcoef(
        ref_gradient,
        test_gradient
    )[0, 1]

    if not np.isfinite(
        epi
    ):

        return 0.0

    return float(
        np.clip(
            epi,
            -1.0,
            1.0
        )
    )


# ============================================================
# RESIDUAL
# ============================================================

def calculate_residual(
    reference,
    test
):
    """
    Absolute residual:

        R = |Reference - Test|
    """

    reference = prepare_image(
        reference
    )

    test = prepare_image(
        test
    )

    check_images(
        reference,
        test
    )

    residual = np.abs(
        reference
        -
        test
    )

    return residual.astype(
        np.float32
    )


# ============================================================
# RESIDUAL STATISTICS
# ============================================================

def calculate_residual_statistics(
    residual
):
    """
    Calculate residual MAE, MSE and maximum error.
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

    mae = float(
        np.mean(
            residual
        )
    )

    mse = float(
        np.mean(
            residual ** 2
        )
    )

    maximum_error = float(
        np.max(
            residual
        )
    )

    return {
        "MAE": mae,
        "MSE": mse,
        "Maximum Error":
            maximum_error
    }


# ============================================================
# ALL METRICS
# ============================================================

def calculate_all_metrics(
    reference,
    test
):
    """
    Calculate:

        PSNR
        SSIM
        FSIM
        EPI
        Residual

    Returns:

        metrics
        residual
    """

    reference = prepare_image(
        reference
    )

    test = prepare_image(
        test
    )

    check_images(
        reference,
        test
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    psnr_value = calculate_psnr(
        reference,
        test
    )

    ssim_value = calculate_ssim(
        reference,
        test
    )

    fsim_value = calculate_fsim(
        reference,
        test
    )

    epi_value = calculate_epi(
        reference,
        test
    )

    # --------------------------------------------------------
    # Residual
    # --------------------------------------------------------

    residual = calculate_residual(
        reference,
        test
    )

    residual_stats = (
        calculate_residual_statistics(
            residual
        )
    )

    metrics = {

        "PSNR (dB)":
            psnr_value,

        "SSIM":
            ssim_value,

        "FSIM":
            fsim_value,

        "EPI":
            epi_value,

        "Residual MAE":
            residual_stats["MAE"],

        "Residual MSE":
            residual_stats["MSE"],

        "Maximum Residual":
            residual_stats[
                "Maximum Error"
            ]
    }

    return (
        metrics,
        residual
    )