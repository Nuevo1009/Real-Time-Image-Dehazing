import torch
import piq
import numpy as np
import cv2


def calculate_fsim(dehazed, ground_truth, device=None):
    """
    Calculate Feature Similarity Index (FSIM).

    Input images should be RGB uint8 arrays.

    Returns FSIM value.
    """

    # Match size
    if dehazed.shape[:2] != ground_truth.shape[:2]:

        dehazed = cv2.resize(
            dehazed,
            (ground_truth.shape[1], ground_truth.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

    # Default device
    if device is None:

        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

    # Convert HWC -> CHW
    dehazed_tensor = torch.from_numpy(
        dehazed.transpose(2, 0, 1)
    ).unsqueeze(0).float()

    gt_tensor = torch.from_numpy(
        ground_truth.transpose(2, 0, 1)
    ).unsqueeze(0).float()

    # Normalize
    dehazed_tensor = dehazed_tensor / 255.0
    gt_tensor = gt_tensor / 255.0

    # Move to device
    dehazed_tensor = dehazed_tensor.to(device)
    gt_tensor = gt_tensor.to(device)

    with torch.no_grad():

        fsim_value = piq.fsim(
            dehazed_tensor,
            gt_tensor,
            data_range=1.0
        )

    return float(fsim_value.cpu().item())