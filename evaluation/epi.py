import cv2
import numpy as np


def calculate_epi(dehazed, ground_truth):
    """
    Calculate Edge Preservation Index (EPI).

    Higher value means better preservation of
    ground-truth edge structure.
    """

    # Match image size
    if dehazed.shape[:2] != ground_truth.shape[:2]:

        dehazed = cv2.resize(
            dehazed,
            (ground_truth.shape[1], ground_truth.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

    # Convert RGB images to grayscale
    dehazed_gray = cv2.cvtColor(
        dehazed,
        cv2.COLOR_RGB2GRAY
    ).astype(np.float32)

    gt_gray = cv2.cvtColor(
        ground_truth,
        cv2.COLOR_RGB2GRAY
    ).astype(np.float32)

    # Sobel gradient for dehazed image
    dx_d = cv2.Sobel(
        dehazed_gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    dy_d = cv2.Sobel(
        dehazed_gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    edge_dehazed = np.sqrt(
        dx_d ** 2 +
        dy_d ** 2
    )

    # Sobel gradient for ground truth
    dx_gt = cv2.Sobel(
        gt_gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    dy_gt = cv2.Sobel(
        gt_gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    edge_gt = np.sqrt(
        dx_gt ** 2 +
        dy_gt ** 2
    )

    # Flatten arrays
    edge_dehazed = edge_dehazed.flatten()
    edge_gt = edge_gt.flatten()

    # Prevent division problems if image has no edges
    if np.std(edge_dehazed) == 0 or np.std(edge_gt) == 0:
        return 0.0

    # Correlation between edge maps
    epi = np.corrcoef(
        edge_dehazed,
        edge_gt
    )[0, 1]

    return float(epi)