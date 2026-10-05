import cv2
import csv
import numpy as np

from pathlib import Path

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)


# ==========================================================
# IMPORT CUSTOM EVALUATION MODULES
# ==========================================================

import evaluation.fsim as fsim_module
import evaluation.epi as epi_module
import evaluation.scielab as scielab_module


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent


OUTPUT_FOLDER = (
    BASE_DIR
    / "results"
    / "final_output"
)


GROUND_TRUTH_FOLDER = (
    BASE_DIR
    / "ground_truth"
)


RESIDUAL_FOLDER = (
    BASE_DIR
    / "results"
    / "residuals"
)


METRICS_FOLDER = (
    BASE_DIR
    / "results"
    / "metrics"
)


# ==========================================================
# VALID IMAGE EXTENSIONS
# ==========================================================

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)


# ==========================================================
# FIND FUNCTION INSIDE MODULE
#
# This makes evaluate.py more tolerant if your metric files
# use slightly different function names.
# ==========================================================

def get_function(module, possible_names):

    for name in possible_names:

        function = getattr(
            module,
            name,
            None
        )

        if callable(function):

            return function

    return None


# ==========================================================
# CONVERT METRIC OUTPUT TO FLOAT
# ==========================================================

def metric_to_float(value):

    if isinstance(value, dict):

        for key in value:

            try:

                return float(
                    value[key]
                )

            except Exception:

                pass

    if isinstance(value, tuple):

        if len(value) > 0:

            return metric_to_float(
                value[0]
            )

    if isinstance(value, list):

        if len(value) > 0:

            return metric_to_float(
                value[0]
            )

    if isinstance(value, np.ndarray):

        if value.size == 1:

            return float(
                value.flatten()[0]
            )

    return float(value)


# ==========================================================
# CALCULATE PSNR
# ==========================================================

def calculate_psnr(
    output_rgb,
    ground_truth_rgb
):

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

    return float(score)


# ==========================================================
# CALCULATE SSIM
# ==========================================================

def calculate_ssim(
    output_rgb,
    ground_truth_rgb
):

    output_gray = cv2.cvtColor(

        output_rgb,

        cv2.COLOR_RGB2GRAY
    )


    gt_gray = cv2.cvtColor(

        ground_truth_rgb,

        cv2.COLOR_RGB2GRAY
    )


    min_dimension = min(

        output_gray.shape[0],

        output_gray.shape[1]
    )


    # SSIM requires a valid odd window size
    if min_dimension < 7:

        win_size = min_dimension

        if win_size % 2 == 0:

            win_size -= 1

    else:

        win_size = 7


    if win_size < 3:

        raise ValueError(

            "Image is too small for SSIM calculation."
        )


    score = structural_similarity(

        gt_gray,

        output_gray,

        data_range=255,

        win_size=win_size
    )


    return float(score)


# ==========================================================
# CALCULATE RESIDUAL
#
# Residual = absolute difference between
# dehazed image and ground truth.
# ==========================================================

def calculate_residual_image(
    output_rgb,
    ground_truth_rgb
):

    residual = cv2.absdiff(

        ground_truth_rgb,

        output_rgb
    )


    return residual


# ==========================================================
# FIND FSIM FUNCTION
# ==========================================================

FSIM_FUNCTION = get_function(

    fsim_module,

    [

        "calculate_fsim",

        "fsim",

        "compute_fsim",

        "get_fsim"
    ]
)


# ==========================================================
# FIND EPI FUNCTION
# ==========================================================

EPI_FUNCTION = get_function(

    epi_module,

    [

        "calculate_epi",

        "epi",

        "compute_epi",

        "calculate_edge_preservation_index"
    ]
)


# ==========================================================
# FIND SCIELAB FUNCTION
# ==========================================================

SCIELAB_FUNCTION = get_function(

    scielab_module,

    [

        "calculate_scielab",

        "scielab",

        "compute_scielab",

        "calculate_spatial_cielab"
    ]
)


# ==========================================================
# MAIN
# ==========================================================

def main():


    print("\n")

    print("=" * 60)

    print(
        "GROUND TRUTH IMAGE EVALUATION"
    )

    print("=" * 60)


    # ------------------------------------------------------
    # CHECK FINAL OUTPUT FOLDER
    # ------------------------------------------------------

    if not OUTPUT_FOLDER.exists():

        raise FileNotFoundError(

            "\nFinal output folder not found:\n"

            f"{OUTPUT_FOLDER}"
        )


    # ------------------------------------------------------
    # CHECK GROUND TRUTH FOLDER
    # ------------------------------------------------------

    if not GROUND_TRUTH_FOLDER.exists():

        raise FileNotFoundError(

            "\nGround truth folder not found:\n"

            f"{GROUND_TRUTH_FOLDER}\n\n"

            "Create the folder and place the corresponding "
            "clear images inside it."
        )


    # ------------------------------------------------------
    # CHECK METRIC FUNCTIONS
    # ------------------------------------------------------

    if FSIM_FUNCTION is None:

        raise ImportError(

            "\nCould not find an FSIM calculation function.\n"

            "Your evaluation/fsim.py should contain a function "
            "such as:\n\n"

            "calculate_fsim(image1, image2)"
        )


    if EPI_FUNCTION is None:

        raise ImportError(

            "\nCould not find an EPI calculation function.\n"

            "Your evaluation/epi.py should contain a function "
            "such as:\n\n"

            "calculate_epi(image1, image2)"
        )


    if SCIELAB_FUNCTION is None:

        raise ImportError(

            "\nCould not find a SCIELAB calculation function.\n"

            "Your evaluation/scielab.py should contain a function "
            "such as:\n\n"

            "calculate_scielab(image1, image2)"
        )


    # ------------------------------------------------------
    # CREATE OUTPUT FOLDERS
    # ------------------------------------------------------

    RESIDUAL_FOLDER.mkdir(

        parents=True,

        exist_ok=True
    )


    METRICS_FOLDER.mkdir(

        parents=True,

        exist_ok=True
    )


    # ------------------------------------------------------
    # FIND OUTPUT IMAGES
    # ------------------------------------------------------

    image_files = sorted(

        [

            file

            for file in OUTPUT_FOLDER.iterdir()

            if (
                file.is_file()

                and

                file.suffix.lower()

                in VALID_EXTENSIONS
            )

        ]
    )


    if len(image_files) == 0:

        raise FileNotFoundError(

            "\nNo images found in:\n"

            f"{OUTPUT_FOLDER}"
        )


    print(

        f"\nDehazed images found: "

        f"{len(image_files)}"
    )


    # ------------------------------------------------------
    # CSV FILE
    # ------------------------------------------------------

    csv_path = (

        METRICS_FOLDER

        / "ground_truth_metrics.csv"
    )


    # ------------------------------------------------------
    # STORE SCORES FOR AVERAGES
    # ------------------------------------------------------

    psnr_scores = []

    ssim_scores = []

    fsim_scores = []

    epi_scores = []

    scielab_scores = []


    evaluated_count = 0

    missing_ground_truth = 0


    # ======================================================
    # OPEN CSV
    # ======================================================

    with open(

        csv_path,

        "w",

        newline="",

        encoding="utf-8"

    ) as csv_file:


        writer = csv.writer(

            csv_file
        )


        # --------------------------------------------------
        # CSV HEADER
        # --------------------------------------------------

        writer.writerow([

            "Image",

            "PSNR",

            "SSIM",

            "FSIM",

            "EPI",

            "Spatial_CIELAB_Error"
        ])


        # ==================================================
        # PROCESS EACH IMAGE
        # ==================================================

        for index, output_path in enumerate(

            image_files,

            start=1

        ):


            print("\n")

            print("-" * 60)


            print(

                f"[{index}/{len(image_files)}] "

                f"Evaluating: "

                f"{output_path.name}"
            )


            # --------------------------------------------------
            # GROUND TRUTH PATH
            # --------------------------------------------------

            gt_path = (

                GROUND_TRUTH_FOLDER

                / output_path.name
            )


            # --------------------------------------------------
            # CHECK GROUND TRUTH
            # --------------------------------------------------

            if not gt_path.exists():


                print(

                    "GROUND TRUTH MISSING"
                )


                print(

                    f"Expected:\n"

                    f"{gt_path}"
                )


                missing_ground_truth += 1


                continue


            # --------------------------------------------------
            # LOAD DEHAZED IMAGE
            # --------------------------------------------------

            output_bgr = cv2.imread(

                str(output_path),

                cv2.IMREAD_COLOR
            )


            # --------------------------------------------------
            # LOAD GROUND TRUTH
            # --------------------------------------------------

            gt_bgr = cv2.imread(

                str(gt_path),

                cv2.IMREAD_COLOR
            )


            # --------------------------------------------------
            # CHECK IMAGE LOADING
            # --------------------------------------------------

            if output_bgr is None:


                print(

                    "Could not load dehazed image."
                )


                continue


            if gt_bgr is None:


                print(

                    "Could not load ground truth image."
                )


                continue


            # --------------------------------------------------
            # RESIZE GROUND TRUTH IF REQUIRED
            # --------------------------------------------------

            if (

                output_bgr.shape

                !=

                gt_bgr.shape

            ):


                print(

                    "Image sizes differ."
                )


                print(

                    f"Output shape: "

                    f"{output_bgr.shape}"
                )


                print(

                    f"GT shape: "

                    f"{gt_bgr.shape}"
                )


                print(

                    "Resizing ground truth to output size..."
                )


                gt_bgr = cv2.resize(

                    gt_bgr,

                    (

                        output_bgr.shape[1],

                        output_bgr.shape[0]
                    ),

                    interpolation=cv2.INTER_AREA
                )


            # --------------------------------------------------
            # CONVERT BGR → RGB
            # --------------------------------------------------

            output_rgb = cv2.cvtColor(

                output_bgr,

                cv2.COLOR_BGR2RGB
            )


            gt_rgb = cv2.cvtColor(

                gt_bgr,

                cv2.COLOR_BGR2RGB
            )


            # ==================================================
            # PSNR
            # ==================================================

            psnr_score = (

                calculate_psnr(

                    output_rgb,

                    gt_rgb
                )
            )


            # ==================================================
            # SSIM
            # ==================================================

            ssim_score = (

                calculate_ssim(

                    output_rgb,

                    gt_rgb
                )
            )


            # ==================================================
            # FSIM
            # ==================================================

            fsim_result = (

                FSIM_FUNCTION(

                    output_rgb,

                    gt_rgb
                )
            )


            fsim_score = (

                metric_to_float(

                    fsim_result
                )
            )


            # ==================================================
            # EPI
            # ==================================================

            epi_result = (

                EPI_FUNCTION(

                    output_rgb,

                    gt_rgb
                )
            )


            epi_score = (

                metric_to_float(

                    epi_result
                )
            )


            # ==================================================
            # SCIELAB
            # ==================================================

            scielab_result = (

                SCIELAB_FUNCTION(

                    output_rgb,

                    gt_rgb
                )
            )


            scielab_score = (

                metric_to_float(

                    scielab_result
                )
            )


            # ==================================================
            # RESIDUAL
            # ==================================================

            residual_rgb = (

                calculate_residual_image(

                    output_rgb,

                    gt_rgb
                )
            )


            residual_bgr = cv2.cvtColor(

                residual_rgb,

                cv2.COLOR_RGB2BGR
            )


            residual_path = (

                RESIDUAL_FOLDER

                / output_path.name
            )


            cv2.imwrite(

                str(residual_path),

                residual_bgr
            )


            # ==================================================
            # STORE RESULTS
            # ==================================================

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


            # ==================================================
            # WRITE IMAGE RESULTS TO CSV
            # ==================================================

            writer.writerow([

                output_path.name,

                f"{psnr_score:.6f}",

                f"{ssim_score:.6f}",

                f"{fsim_score:.6f}",

                f"{epi_score:.6f}",

                f"{scielab_score:.6f}"
            ])


            # ==================================================
            # PRINT RESULTS
            # ==================================================

            print(

                f"\nPSNR: "

                f"{psnr_score:.6f}"
            )


            print(

                f"SSIM: "

                f"{ssim_score:.6f}"
            )


            print(

                f"FSIM: "

                f"{fsim_score:.6f}"
            )


            print(

                f"EPI: "

                f"{epi_score:.6f}"
            )


            print(

                f"Spatial CIELAB Error: "

                f"{scielab_score:.6f}"
            )


            print(

                f"Residual saved: "

                f"{residual_path.name}"
            )


        # ======================================================
        # CALCULATE AVERAGES
        # ======================================================

        if evaluated_count > 0:


            average_psnr = float(

                np.mean(psnr_scores)
            )


            average_ssim = float(

                np.mean(ssim_scores)
            )


            average_fsim = float(

                np.mean(fsim_scores)
            )


            average_epi = float(

                np.mean(epi_scores)
            )


            average_scielab = float(

                np.mean(scielab_scores)
            )


            # --------------------------------------------------
            # WRITE AVERAGES
            # --------------------------------------------------

            writer.writerow([])


            writer.writerow([

                "AVERAGE",

                f"{average_psnr:.6f}",

                f"{average_ssim:.6f}",

                f"{average_fsim:.6f}",

                f"{average_epi:.6f}",

                f"{average_scielab:.6f}"
            ])


        else:


            average_psnr = None

            average_ssim = None

            average_fsim = None

            average_epi = None

            average_scielab = None


    # ==========================================================
    # FINAL SUMMARY
    # ==========================================================

    print("\n")

    print("=" * 60)

    print(
        "EVALUATION COMPLETED"
    )

    print("=" * 60)


    print(

        f"\nTotal dehazed images: "

        f"{len(image_files)}"
    )


    print(

        f"Successfully evaluated: "

        f"{evaluated_count}"
    )


    print(

        f"Missing ground truth: "

        f"{missing_ground_truth}"
    )


    if evaluated_count > 0:


        print("\n")

        print(
            "AVERAGE METRICS"
        )


        print(

            f"Average PSNR: "

            f"{average_psnr:.6f}"
        )


        print(

            f"Average SSIM: "

            f"{average_ssim:.6f}"
        )


        print(

            f"Average FSIM: "

            f"{average_fsim:.6f}"
        )


        print(

            f"Average EPI: "

            f"{average_epi:.6f}"
        )


        print(

            f"Average Spatial CIELAB Error: "

            f"{average_scielab:.6f}"
        )


    print("\n")

    print(

        f"CSV saved to:\n"

        f"{csv_path}"
    )


    print("\n")

    print(

        f"Residual images saved to:\n"

        f"{RESIDUAL_FOLDER}"
    )


    print("\n")


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":

    main()