"""
============================================================
GROUND TRUTH IMAGE EVALUATION
============================================================

Evaluates dehazed output images against corresponding
ground-truth clear images.

Metrics:

    PSNR
    SSIM
    FSIM
    EPI
    S-CIELAB
    Residual Image

Important:

    Images are handled as RGB throughout this file.

    Pillow is used for image loading and saving.

    No BGR -> RGB conversion is performed.

Project structure:

    HAZE_CNN/
        evaluation.py

        evaluation/
            basic_metrics.py
            fsim.py
            epi.py
            residual.py
            scielab.py
            edge_no_reference.py

        results/
            final_output/
            residuals/
            metrics/

        ground_truth/
============================================================
"""

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import csv
import re

import numpy as np

from PIL import Image

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)


# ============================================================
# CUSTOM EVALUATION MODULES
# ============================================================

from evaluation.fsim import (
    calculate_fsim
)

from evaluation.epi import (
    calculate_epi
)

from evaluation.scielab import (
    calculate_scielab
)

from evaluation.residual import (
    calculate_residual
)


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent


# ------------------------------------------------------------
# FINAL DEHAZED OUTPUT
# ------------------------------------------------------------

OUTPUT_FOLDER = (
    BASE_DIR
    / "results"
    / "final_output"
)


# ------------------------------------------------------------
# GROUND TRUTH
# ------------------------------------------------------------

GROUND_TRUTH_FOLDER = (
    BASE_DIR
    / "ground_truth"
)


# ------------------------------------------------------------
# RESIDUAL OUTPUT
# ------------------------------------------------------------

RESIDUAL_FOLDER = (
    BASE_DIR
    / "results"
    / "residuals"
)


# ------------------------------------------------------------
# METRICS OUTPUT
# ------------------------------------------------------------

METRICS_FOLDER = (
    BASE_DIR
    / "results"
    / "metrics"
)


# ============================================================
# VALID IMAGE EXTENSIONS
# ============================================================

VALID_EXTENSIONS = {

    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"

}


# ============================================================
# READ RGB IMAGE
# ============================================================

def read_rgb(
    image_path
):
    """
    Read an image directly as RGB.

    Pillow returns the image in RGB order.

    No OpenCV BGR conversion is used.

    Returns:
        RGB uint8 NumPy array
    """

    try:

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

    except Exception as exc:

        raise RuntimeError(
            f"Could not read image:\n"
            f"{image_path}\n"
            f"Reason: {exc}"
        )

    return np.asarray(
        image,
        dtype=np.uint8
    )


# ============================================================
# SAVE RGB IMAGE
# ============================================================

def save_rgb(
    image,
    output_path
):
    """
    Save an RGB image directly.

    No RGB -> BGR conversion.
    """

    image = np.asarray(
        image
    )

    if image.dtype != np.uint8:

        if image.max() <= 1.0:

            image = (
                image
                *
                255.0
            )

        image = np.clip(
            image,
            0,
            255
        ).astype(
            np.uint8
        )

    Image.fromarray(
        image,
        mode="RGB"
    ).save(
        output_path
    )


# ============================================================
# FIND GROUND TRUTH
# ============================================================

def find_ground_truth(
    output_path,
    gt_files
):
    """
    Find the corresponding ground-truth image.

    Matching order:

        1. Exact filename
        2. Exact filename stem
        3. Leading numeric scene ID

    Example:

        Output:
            40_outdoor_dehazed.png

        Ground truth:
            40.jpg

    The numeric scene ID 40 will be used as a fallback.
    """

    output_name = (
        output_path.name.lower()
    )

    output_stem = (
        output_path.stem.lower()
    )

    # ========================================================
    # 1. EXACT FILENAME
    # ========================================================

    for gt_path in gt_files:

        if (
            gt_path.name.lower()
            ==
            output_name
        ):

            return gt_path


    # ========================================================
    # 2. EXACT STEM
    # ========================================================

    for gt_path in gt_files:

        if (
            gt_path.stem.lower()
            ==
            output_stem
        ):

            return gt_path


    # ========================================================
    # 3. REMOVE COMMON OUTPUT SUFFIXES
    # ========================================================

    clean_stem = re.sub(
        r"(_dehazed|_output|_result)$",
        "",
        output_stem
    )


    for gt_path in gt_files:

        if (
            gt_path.stem.lower()
            ==
            clean_stem
        ):

            return gt_path


    # ========================================================
    # 4. LEADING NUMERIC SCENE ID
    # ========================================================

    match = re.match(
        r"^(\d+)",
        output_path.name
    )

    if match:

        scene_id = (
            match.group(1)
        )

        candidates = []

        for gt_path in gt_files:

            gt_match = re.match(
                r"^"
                +
                re.escape(scene_id)
                +
                r"(\D|$)",
                gt_path.name
            )

            if gt_match:

                candidates.append(
                    gt_path
                )

        if candidates:

            return sorted(
                candidates
            )[0]


    # ========================================================
    # NO MATCH
    # ========================================================

    return None


# ============================================================
# RGB IMAGE RESIZE
# ============================================================

def resize_ground_truth(
    ground_truth,
    target_height,
    target_width
):
    """
    Resize RGB ground-truth image if its size does not
    match the dehazed output.

    Pillow is used for RGB resizing.
    """

    height, width = (
        ground_truth.shape[:2]
    )

    if (
        height == target_height
        and
        width == target_width
    ):

        return ground_truth


    resized = Image.fromarray(
        ground_truth,
        mode="RGB"
    ).resize(
        (
            target_width,
            target_height
        ),
        Image.Resampling.LANCZOS
    )

    return np.asarray(
        resized,
        dtype=np.uint8
    )


# ============================================================
# CALCULATE PSNR
# ============================================================

def calculate_psnr(
    output_rgb,
    ground_truth_rgb
):
    """
    Calculate PSNR.

    Higher is better.
    """

    output_float = (
        output_rgb.astype(
            np.float32
        )
    )

    gt_float = (
        ground_truth_rgb.astype(
            np.float32
        )
    )

    score = (
        peak_signal_noise_ratio(
            gt_float,
            output_float,
            data_range=255
        )
    )

    return float(
        score
    )


# ============================================================
# RGB -> GRAYSCALE
# ============================================================

def rgb_to_gray(
    image_rgb
):
    """
    Convert RGB image to luminance.

    This is an RGB luminance calculation.

    No BGR conversion is performed.
    """

    image = (
        image_rgb.astype(
            np.float32
        )
    )

    if image.max() > 1.0:

        image = (
            image
            /
            255.0
        )

    gray = (

        0.299
        *
        image[:, :, 0]

        +

        0.587
        *
        image[:, :, 1]

        +

        0.114
        *
        image[:, :, 2]

    )

    return gray.astype(
        np.float32
    )


# ============================================================
# CALCULATE SSIM
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

    gt_gray = rgb_to_gray(
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
            "Image is too small for SSIM calculation."
        )

    score = structural_similarity(
        gt_gray,
        output_gray,
        data_range=1.0,
        win_size=win_size
    )

    return float(
        score
    )


# ============================================================
# CONVERT METRIC RESULT TO FLOAT
# ============================================================

def metric_to_float(
    value
):
    """
    Convert metric output to a Python float.

    Handles:

        float
        int
        dict
        tuple
        list
        NumPy scalar
        NumPy array
    """

    if isinstance(
        value,
        dict
    ):

        for item in value.values():

            try:

                return metric_to_float(
                    item
                )

            except Exception:

                pass


    if isinstance(
        value,
        tuple
    ):

        if len(value) > 0:

            return metric_to_float(
                value[0]
            )


    if isinstance(
        value,
        list
    ):

        if len(value) > 0:

            return metric_to_float(
                value[0]
            )


    if isinstance(
        value,
        np.ndarray
    ):

        if value.size == 1:

            return float(
                value.flatten()[0]
            )


    return float(
        value
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 70
    )
    print(
        "              GROUND TRUTH IMAGE EVALUATION"
    )
    print(
        "=" * 70
    )

    print()
    print(
        "Project directory:"
    )
    print(
        BASE_DIR
    )

    print()
    print(
        "Final output folder:"
    )
    print(
        OUTPUT_FOLDER
    )

    print()
    print(
        "Ground truth folder:"
    )
    print(
        GROUND_TRUTH_FOLDER
    )

    print()
    print(
        "Residual folder:"
    )
    print(
        RESIDUAL_FOLDER
    )

    print()
    print(
        "Metrics folder:"
    )
    print(
        METRICS_FOLDER
    )


    # ========================================================
    # CHECK DIRECTORIES
    # ========================================================

    if not OUTPUT_FOLDER.exists():

        raise FileNotFoundError(
            "\nFinal output folder not found:\n"
            f"{OUTPUT_FOLDER}"
        )


    if not GROUND_TRUTH_FOLDER.exists():

        raise FileNotFoundError(
            "\nGround truth folder not found:\n"
            f"{GROUND_TRUTH_FOLDER}\n\n"
            "Place the clear ground-truth images inside it."
        )


    # ========================================================
    # CREATE RESULT DIRECTORIES
    # ========================================================

    RESIDUAL_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    METRICS_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # FIND OUTPUT IMAGES
    # ========================================================

    image_files = sorted(

        [

            file

            for file in OUTPUT_FOLDER.iterdir()

            if (

                file.is_file()

                and

                file.suffix.lower()
                in
                VALID_EXTENSIONS

            )

        ]

    )


    # ========================================================
    # FIND GROUND-TRUTH IMAGES
    # ========================================================

    gt_files = sorted(

        [

            file

            for file in GROUND_TRUTH_FOLDER.iterdir()

            if (

                file.is_file()

                and

                file.suffix.lower()
                in
                VALID_EXTENSIONS

            )

        ]

    )


    print()
    print(
        f"Dehazed images found : {len(image_files)}"
    )

    print(
        f"Ground truth found   : {len(gt_files)}"
    )


    if not image_files:

        raise FileNotFoundError(
            "\nNo images found in final output folder."
        )


    if not gt_files:

        raise FileNotFoundError(
            "\nNo ground-truth images found."
        )


    # ========================================================
    # CSV
    # ========================================================

    csv_path = (
        METRICS_FOLDER
        /
        "ground_truth_metrics.csv"
    )


    # ========================================================
    # METRIC STORAGE
    # ========================================================

    psnr_scores = []

    ssim_scores = []

    fsim_scores = []

    epi_scores = []

    scielab_scores = []


    evaluated_count = 0

    missing_ground_truth = 0

    failed_count = 0


    # ========================================================
    # OPEN CSV
    # ========================================================

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )


        # ----------------------------------------------------
        # CSV HEADER
        # ----------------------------------------------------

        writer.writerow(
            [
                "Image",
                "Ground_Truth",
                "PSNR",
                "SSIM",
                "FSIM",
                "EPI",
                "S_CIELAB",
                "Residual"
            ]
        )


        # ====================================================
        # PROCESS EACH OUTPUT
        # ====================================================

        for index, output_path in enumerate(
            image_files,
            start=1
        ):

            print()
            print(
                "-" * 70
            )

            print(
                f"[{index}/{len(image_files)}] "
                f"{output_path.name}"
            )


            # =================================================
            # FIND GROUND TRUTH
            # =================================================

            gt_path = find_ground_truth(
                output_path,
                gt_files
            )


            if gt_path is None:

                print(
                    "    [SKIP] Ground truth missing"
                )

                print(
                    "    Output:"
                )

                print(
                    f"    {output_path.name}"
                )

                missing_ground_truth += 1

                continue


            print(
                f"    GT: {gt_path.name}"
            )


            # =================================================
            # READ RGB IMAGES
            # =================================================

            try:

                output_rgb = read_rgb(
                    output_path
                )

                gt_rgb = read_rgb(
                    gt_path
                )

            except Exception as exc:

                print(
                    f"    [ERROR] Image loading failed: {exc}"
                )

                failed_count += 1

                continue


            # =================================================
            # MATCH IMAGE SIZE
            # =================================================

            if (
                output_rgb.shape[:2]
                !=
                gt_rgb.shape[:2]
            ):

                print(
                    "    Ground truth size differs."
                )

                print(
                    f"    Output size: "
                    f"{output_rgb.shape[1]} x "
                    f"{output_rgb.shape[0]}"
                )

                print(
                    f"    GT size: "
                    f"{gt_rgb.shape[1]} x "
                    f"{gt_rgb.shape[0]}"
                )

                print(
                    "    Resizing ground truth..."
                )

                gt_rgb = resize_ground_truth(

                    gt_rgb,

                    output_rgb.shape[0],

                    output_rgb.shape[1]

                )


            # =================================================
            # CALCULATE METRICS
            # =================================================

            try:

                # ---------------------------------------------
                # PSNR
                # ---------------------------------------------

                psnr_score = calculate_psnr(
                    output_rgb,
                    gt_rgb
                )


                # ---------------------------------------------
                # SSIM
                # ---------------------------------------------

                ssim_score = calculate_ssim(
                    output_rgb,
                    gt_rgb
                )


                # ---------------------------------------------
                # FSIM
                # ---------------------------------------------

                fsim_score = metric_to_float(
                    calculate_fsim(
                        output_rgb,
                        gt_rgb
                    )
                )


                # ---------------------------------------------
                # EPI
                # ---------------------------------------------

                epi_score = metric_to_float(
                    calculate_epi(
                        output_rgb,
                        gt_rgb
                    )
                )


                # ---------------------------------------------
                # S-CIELAB
                # ---------------------------------------------

                scielab_score = metric_to_float(
                    calculate_scielab(
                        output_rgb,
                        gt_rgb
                    )
                )


            except Exception as exc:

                print(
                    f"    [ERROR] Metric calculation failed:"
                    f" {exc}"
                )

                failed_count += 1

                continue


            # =================================================
            # RESIDUAL
            # =================================================

            try:

                residual_rgb = calculate_residual(
                    output_rgb,
                    gt_rgb
                )

                residual_path = (
                    RESIDUAL_FOLDER
                    /
                    f"{output_path.stem}_residual.png"
                )

                save_rgb(
                    residual_rgb,
                    residual_path
                )

            except Exception as exc:

                print(
                    f"    [WARNING] Residual saving failed:"
                    f" {exc}"
                )

                residual_path = None


            # =================================================
            # STORE SCORES
            # =================================================

            psnr_scores.append(
                psnr_score
            )

            ssim_scores.append(
                ssim_score
            )

            fsim_scores.append(
                fsim_score
            )

            epi_scores.append(
                epi_score
            )

            scielab_scores.append(
                scielab_score
            )

            evaluated_count += 1


            # =================================================
            # WRITE CSV
            # =================================================

            writer.writerow(

                [

                    output_path.name,

                    gt_path.name,

                    f"{psnr_score:.6f}",

                    f"{ssim_score:.6f}",

                    f"{fsim_score:.6f}",

                    f"{epi_score:.6f}",

                    f"{scielab_score:.6f}",

                    (
                        residual_path.name
                        if residual_path
                        else
                        "Not saved"
                    )

                ]

            )


            # =================================================
            # PRINT RESULTS
            # =================================================

            print()
            print(
                f"    PSNR       : "
                f"{psnr_score:.6f} dB"
            )

            print(
                f"    SSIM       : "
                f"{ssim_score:.6f}"
            )

            print(
                f"    FSIM       : "
                f"{fsim_score:.6f}"
            )

            print(
                f"    EPI        : "
                f"{epi_score:.6f}"
            )

            print(
                f"    S-CIELAB   : "
                f"{scielab_score:.6f}"
            )

            if residual_path:

                print(
                    f"    Residual   : "
                    f"{residual_path.name}"
                )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "                    FINAL RESULTS"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Total output images       : "
        f"{len(image_files)}"
    )

    print(
        f"Total ground-truth images: "
        f"{len(gt_files)}"
    )

    print(
        f"Successfully evaluated   : "
        f"{evaluated_count}"
    )

    print(
        f"Ground truth missing     : "
        f"{missing_ground_truth}"
    )

    print(
        f"Failed evaluations       : "
        f"{failed_count}"
    )


    # ========================================================
    # AVERAGES
    # ========================================================

    if evaluated_count > 0:

        average_psnr = np.mean(
            psnr_scores
        )

        average_ssim = np.mean(
            ssim_scores
        )

        average_fsim = np.mean(
            fsim_scores
        )

        average_epi = np.mean(
            epi_scores
        )

        average_scielab = np.mean(
            scielab_scores
        )


        print()
        print(
            "-" * 70
        )

        print(
            "AVERAGE METRICS"
        )

        print(
            "-" * 70
        )

        print()

        print(
            f"Average PSNR       : "
            f"{average_psnr:.6f} dB"
        )

        print(
            f"Average SSIM       : "
            f"{average_ssim:.6f}"
        )

        print(
            f"Average FSIM       : "
            f"{average_fsim:.6f}"
        )

        print(
            f"Average EPI        : "
            f"{average_epi:.6f}"
        )

        print(
            f"Average S-CIELAB  : "
            f"{average_scielab:.6f}"
        )


        # ====================================================
        # SUMMARY CSV
        # ====================================================

        summary_path = (
            METRICS_FOLDER
            /
            "average_metrics.csv"
        )

        with open(
            summary_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as summary_file:

            writer = csv.writer(
                summary_file
            )

            writer.writerow(
                [
                    "Metric",
                    "Average"
                ]
            )

            writer.writerow(
                [
                    "PSNR",
                    f"{average_psnr:.6f}"
                ]
            )

            writer.writerow(
                [
                    "SSIM",
                    f"{average_ssim:.6f}"
                ]
            )

            writer.writerow(
                [
                    "FSIM",
                    f"{average_fsim:.6f}"
                ]
            )

            writer.writerow(
                [
                    "EPI",
                    f"{average_epi:.6f}"
                ]
            )

            writer.writerow(
                [
                    "S-CIELAB",
                    f"{average_scielab:.6f}"
                ]
            )


        print()
        print(
            f"CSV saved:"
        )

        print(
            csv_path
        )

        print(
            f"Average metrics saved:"
        )

        print(
            summary_path
        )

    else:

        print()
        print(
            "[ERROR] No images were successfully evaluated."
        )


    print()
    print(
        f"Residual images:"
    )

    print(
        RESIDUAL_FOLDER
    )

    print()
    print(
        "=" * 70
    )

    print(
        "Evaluation completed."
    )

    print(
        "=" * 70
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()