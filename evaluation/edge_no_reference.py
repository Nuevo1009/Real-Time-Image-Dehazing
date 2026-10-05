import cv2
import numpy as np


def edge_strength(image):
    """
    Mean Sobel gradient magnitude.
    Higher value generally means stronger visible edges.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    gray = gray.astype(np.float32) / 255.0

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    return float(np.mean(magnitude))


def sharpness_score(image):
    """
    Variance of Laplacian.
    Higher value generally indicates sharper edges.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    return float(laplacian.var())


def edge_density(image):
    """
    Percentage of pixels detected as edges.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    density = np.count_nonzero(edges) / edges.size

    return float(density)


def saturation_score(image):
    """
    Average saturation in HSV space.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV
    )

    saturation = hsv[:, :, 1].astype(np.float32) / 255.0

    return float(np.mean(saturation))


def gradient_map(image):
    """
    Creates normalized gradient magnitude map.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    gray = gray.astype(np.float32) / 255.0

    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = np.sqrt(
        gx ** 2 + gy ** 2
    )

    return magnitude


def edge_preservation_proxy(
    initial_image,
    final_image
):
    """
    No-reference edge preservation proxy.

    Compares edge maps of initial dehazed image
    and final enhanced image.

    Value near 1 indicates strong edge structure
    has been retained.
    """

    g1 = gradient_map(initial_image)
    g2 = gradient_map(final_image)

    g1 = g1.flatten()
    g2 = g2.flatten()

    numerator = np.sum(g1 * g2)

    denominator = (
        np.sqrt(np.sum(g1 ** 2))
        *
        np.sqrt(np.sum(g2 ** 2))
        +
        1e-8
    )

    score = numerator / denominator

    return float(score)


def analyze_no_reference(
    initial_image,
    final_image
):
    """
    Complete no-reference quality analysis.
    """

    results = {}

    results["initial_edge_strength"] = edge_strength(
        initial_image
    )

    results["final_edge_strength"] = edge_strength(
        final_image
    )

    results["initial_sharpness"] = sharpness_score(
        initial_image
    )

    results["final_sharpness"] = sharpness_score(
        final_image
    )

    results["initial_edge_density"] = edge_density(
        initial_image
    )

    results["final_edge_density"] = edge_density(
        final_image
    )

    results["initial_saturation"] = saturation_score(
        initial_image
    )

    results["final_saturation"] = saturation_score(
        final_image
    )

    results["edge_preservation_proxy"] = (
        edge_preservation_proxy(
            initial_image,
            final_image
        )
    )

    return results