"""
============================================================
GAMMA ABLATION STUDY
============================================================

Tests different transmission gamma values and evaluates
the resulting dehazed images.

Metrics:

    PSNR
    SSIM
    FSIM
    S-CIELAB
    EPI
    Edge Retention

Dataset:

    dataset/
        CNN_Dataset/
            test/
                hazy/
                clear/

Dehazing modules are taken from:

    dehazing/

Evaluation modules are taken from:

    evaluation/

Important:

    Images are loaded and saved as RGB.

    Pillow is used for image I/O.

    No BGR -> RGB conversion is performed.

Results are saved to:

    output/
        ablation/
            gamma_1.5/
            gamma_2.0/

            gamma_ablation_results.csv
            gamma_ablation_results.xlsx

            gamma_vs_psnr.png
            gamma_vs_ssim.png
            gamma_vs_fsim.png
            gamma_vs_epi.png
            gamma_vs_scielab.png
            gamma_vs_edge_retention.png
============================================================
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import re

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image


# ============================================================
# DEHAZING MODULES
# ============================================================

from dehazing.default_dehaze_params import (
    default_dehaze_params
)

from dehazing.atmospheric_light import (
    estimate_atmospheric_light,
    pixelwise_atmospheric_light
)

from dehazing.transmission import (
    pixelwise_transmission
)

from dehazing.scene_dehazing import (
    scene_dehazing
)


# ============================================================
# EVALUATION MODULES
# ============================================================

from evaluation.basic_metrics import (
    calculate_basic_metrics
)

from evaluation.fsim import (
    calculate_fsim
)

from evaluation.epi import (
    calculate_epi
)

from evaluation.scielab import (
    calculate_scielab
)

from evaluation.edge_no_reference import (
    edge_preservation_proxy
)


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
)


# ============================================================
# HAZY TEST IMAGES
# ============================================================

HAZY_DIR = (

    BASE_DIR

    /
    "dataset"

    /
    "CNN_Dataset"

    /
    "test"

    /
    "hazy"

)


# ============================================================
# CLEAR / GROUND-TRUTH TEST IMAGES
# ============================================================

GT_DIR = (

    BASE_DIR

    /
    "dataset"

    /
    "CNN_Dataset"

    /
    "test"

    /
    "clear"

)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

ABLATION_DIR = (

    BASE_DIR

    /
    "output"

    /
    "ablation"

)


# ============================================================
# GAMMA VALUES
# ============================================================

GAMMAS = [

    1.5,

    2.0

]


# ============================================================
# SUPPORTED IMAGE FORMATS
# ============================================================

IMAGE_EXTENSIONS = {

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
    path
):
    """
    Read image directly as RGB.

    Returns:

        RGB float32 image
        range [0,1]

    No BGR conversion.
    """

    try:

        image = Image.open(
            path
        ).convert(
            "RGB"
        )

    except Exception as exc:

        raise RuntimeError(
            f"Could not read image:\n"
            f"{path}\n"
            f"Reason: {exc}"
        )

    image = np.asarray(
        image,
        dtype=np.uint8
    )

    return (

        image.astype(
            np.float32
        )

        /

        255.0

    )


# ============================================================
# SAVE RGB IMAGE
# ============================================================

def save_rgb(
    path,
    image
):
    """
    Save RGB float image [0,1] directly.

    No RGB -> BGR conversion.
    """

    image_uint8 = (

        np.clip(

            np.round(

                image
                *
                255.0

            ),

            0,

            255

        )

        .astype(
            np.uint8
        )

    )

    Image.fromarray(
        image_uint8,
        mode="RGB"
    ).save(
        path
    )


# ============================================================
# CONVERT RGB FLOAT TO RGB UINT8
# ============================================================

def rgb_to_uint8(
    image
):
    """
    RGB float [0,1] -> RGB uint8.
    """

    return (

        np.clip(

            image
            *
            255.0,

            0,

            255

        )

        .astype(
            np.uint8
        )

    )


# ============================================================
# FIND GROUND TRUTH
# ============================================================

def find_gt(
    hazy_path,
    gt_files
):
    """
    Find corresponding clear ground truth.

    Matching order:

        1. Exact filename stem
        2. Leading numeric scene ID

    Example:

        Hazy:
            40_outdoor.jpg

        GT:
            40.jpg

    will match using scene ID 40.
    """

    # ========================================================
    # EXACT STEM
    # ========================================================

    hazy_stem = (
        hazy_path.stem.lower()
    )

    for gt_path in gt_files:

        if (
            gt_path.stem.lower()
            ==
            hazy_stem
        ):

            return gt_path


    # ========================================================
    # REMOVE COMMON SUFFIXES
    # ========================================================

    clean_stem = re.sub(

        r"(_hazy|_outdoor|_input)$",

        "",

        hazy_stem

    )


    for gt_path in gt_files:

        if (
            gt_path.stem.lower()
            ==
            clean_stem
        ):

            return gt_path


    # ========================================================
    # LEADING NUMERIC SCENE ID
    # ========================================================

    match = re.match(

        r"^(\d+)",

        hazy_path.name

    )


    if match:

        scene_id = (
            match.group(1)
        )

        candidates = []


        for gt_path in gt_files:

            if re.match(

                r"^"
                +
                re.escape(scene_id)
                +
                r"(\D|$)",

                gt_path.name

            ):

                candidates.append(
                    gt_path
                )


        if candidates:

            return sorted(
                candidates
            )[0]


    return None


# ============================================================
# DEHAZE USING SPECIFIC GAMMA
# ============================================================

def dehaze_with_gamma(
    hazy,
    gamma,
    params
):
    """
    Run the current dehazing pipeline using a specified
    transmission gamma.

    Input:

        hazy:
            RGB float32 [0,1]

    Output:

        dehazed:
            RGB float32 [0,1]

        atmospheric_light

        transmission
    """

    # ========================================================
    # DOWNSAMPLE
    # ========================================================

    down_factor = (
        params.down_factor
    )

    height, width = (
        hazy.shape[:2]
    )

    down_width = max(

        1,

        width
        //
        down_factor

    )

    down_height = max(

        1,

        height
        //
        down_factor

    )


    image_down = cv2.resize(

        hazy,

        (
            down_width,
            down_height
        ),

        interpolation=cv2.INTER_AREA

    )


    # ========================================================
    # ATMOSPHERIC LIGHT
    # ========================================================

    (
        AF,
        AN,
        _,
        _
    ) = estimate_atmospheric_light(

        image_down,

        params.patch_size,

        th_ref=None

    )


    # ========================================================
    # PIXEL-WISE ATMOSPHERIC LIGHT
    # ========================================================

    atmospheric_light = (

        pixelwise_atmospheric_light(

            hazy,

            AF,

            AN

        )

    )


    # ========================================================
    # TRANSMISSION
    # ========================================================

    transmission = (

        pixelwise_transmission(

            hazy,

            atmospheric_light,

            omega=params.omega,

            gamma=gamma,

            t_min=params.t_min

        )

    )


    # ========================================================
    # SCENE RECOVERY
    # ========================================================

    dehazed = (

        scene_dehazing(

            hazy,

            atmospheric_light,

            transmission

        )

    )


    # ========================================================
    # RANGE PROTECTION
    # ========================================================

    dehazed = np.clip(

        dehazed,

        0.0,

        1.0

    ).astype(

        np.float32

    )


    return (

        dehazed,

        atmospheric_light,

        transmission

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
        "                  GAMMA ABLATION STUDY"
    )

    print(
        "=" * 70
    )


    print()
    print(
        "Hazy directory:"
    )

    print(
        HAZY_DIR
    )


    print()
    print(
        "Ground-truth directory:"
    )

    print(
        GT_DIR
    )


    print()
    print(
        "Output directory:"
    )

    print(
        ABLATION_DIR
    )


    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    ABLATION_DIR.mkdir(

        parents=True,

        exist_ok=True

    )


    # ========================================================
    # CHECK DIRECTORIES
    # ========================================================

    if not HAZY_DIR.exists():

        print()
        print(
            "[ERROR] Hazy directory does not exist:"
        )

        print(
            HAZY_DIR
        )

        return


    if not GT_DIR.exists():

        print()
        print(
            "[ERROR] Ground-truth directory does not exist:"
        )

        print(
            GT_DIR
        )

        return


    # ========================================================
    # FIND HAZY FILES
    # ========================================================

    hazy_files = sorted(

        [

            path

            for path
            in
            HAZY_DIR.iterdir()

            if (

                path.is_file()

                and

                path.suffix.lower()
                in
                IMAGE_EXTENSIONS

            )

        ]

    )


    # ========================================================
    # FIND GT FILES
    # ========================================================

    gt_files = sorted(

        [

            path

            for path
            in
            GT_DIR.iterdir()

            if (

                path.is_file()

                and

                path.suffix.lower()
                in
                IMAGE_EXTENSIONS

            )

        ]

    )


    print()
    print(
        f"Found {len(hazy_files)} hazy images."
    )

    print(
        f"Found {len(gt_files)} ground-truth images."
    )


    if not hazy_files:

        print(
            "[ERROR] No hazy images found."
        )

        return


    if not gt_files:

        print(
            "[ERROR] No ground-truth images found."
        )

        return


    # ========================================================
    # RESULT STORAGE
    # ========================================================

    rows = []


    # ========================================================
    # GAMMA LOOP
    # ========================================================

    for gamma in GAMMAS:

        print()
        print(
            "-" * 70
        )

        print(
            f"Testing gamma = {gamma}"
        )

        print(
            "-" * 70
        )


        # ====================================================
        # LOAD PARAMETERS
        # ====================================================

        params = (
            default_dehaze_params()
        )


        # ====================================================
        # SET GAMMA
        # ====================================================

        params.gamma = (
            gamma
        )


        # ====================================================
        # METRIC STORAGE
        # ====================================================

        psnr_values = []

        ssim_values = []

        fsim_values = []

        scielab_values = []

        epi_values = []

        edge_ret_values = []


        # ====================================================
        # OUTPUT DIRECTORY
        # ====================================================

        out_dir = (

            ABLATION_DIR

            /

            f"gamma_{gamma:.1f}"

        )


        out_dir.mkdir(

            parents=True,

            exist_ok=True

        )


        # ====================================================
        # PROCESS EACH HAZY IMAGE
        # ====================================================

        for index, hazy_path in enumerate(

            hazy_files,

            start=1

        ):

            print()

            print(

                f"[{index}/{len(hazy_files)}] "
                f"{hazy_path.name}"

            )


            # =================================================
            # FIND GT
            # =================================================

            gt_path = find_gt(

                hazy_path,

                gt_files

            )


            if gt_path is None:

                print(

                    "    [SKIP] "
                    "No ground truth found."

                )

                continue


            print(

                f"    GT: "
                f"{gt_path.name}"

            )


            # =================================================
            # READ IMAGES
            # =================================================

            try:

                hazy = read_rgb(
                    hazy_path
                )

                gt = read_rgb(
                    gt_path
                )

            except Exception as exc:

                print(

                    f"    [ERROR] "
                    f"Image loading failed: "
                    f"{exc}"

                )

                continue


            # =================================================
            # DEHAZE
            # =================================================

            try:

                (

                    pred,

                    atmospheric_light,

                    transmission

                ) = dehaze_with_gamma(

                    hazy,

                    gamma,

                    params

                )

            except Exception as exc:

                print(

                    f"    [ERROR] "
                    f"Dehazing failed: "
                    f"{exc}"

                )

                continue


            # =================================================
            # MATCH GT SIZE
            # =================================================

            if (

                gt.shape[:2]
                !=
                pred.shape[:2]

            ):

                gt_pil = Image.fromarray(

                    rgb_to_uint8(
                        gt
                    ),

                    mode="RGB"

                )

                gt_pil = gt_pil.resize(

                    (
                        pred.shape[1],
                        pred.shape[0]
                    ),

                    Image.Resampling.LANCZOS

                )

                gt = (

                    np.asarray(
                        gt_pil,
                        dtype=np.uint8
                    )
                    .astype(
                        np.float32
                    )
                    /
                    255.0

                )


            # =================================================
            # RGB UINT8
            # =================================================

            pred_uint8 = (
                rgb_to_uint8(
                    pred
                )
            )

            gt_uint8 = (
                rgb_to_uint8(
                    gt
                )
            )

            hazy_uint8 = (
                rgb_to_uint8(
                    hazy
                )
            )


            # =================================================
            # BASIC METRICS
            # =================================================

            try:

                basic = (

                    calculate_basic_metrics(

                        pred_uint8,

                        gt_uint8

                    )

                )

                psnr = (
                    basic["psnr"]
                )

                ssim = (
                    basic["ssim"]
                )

            except Exception as exc:

                print(

                    f"    [ERROR] "
                    f"PSNR/SSIM failed: "
                    f"{exc}"

                )

                continue


            # =================================================
            # FSIM
            # =================================================

            try:

                fsim = float(

                    calculate_fsim(

                        pred_uint8,

                        gt_uint8

                    )

                )

            except Exception as exc:

                print(

                    f"    [ERROR] "
                    f"FSIM failed: "
                    f"{exc}"

                )

                continue


            # =================================================
            # S-CIELAB
            # =================================================

            try:

                scielab = float(

                    calculate_scielab(

                        pred_uint8,

                        gt_uint8

                    )

                )

            except Exception as exc:

                print(

                    f"    [ERROR] "
                    f"S-CIELAB failed: "
                    f"{exc}"

                )

                continue


            # =================================================
            # EPI
            # =================================================

            try:

                epi = float(

                    calculate_epi(

                        pred_uint8,

                        gt_uint8

                    )

                )

            except Exception as exc:

                print(

                    f"    [ERROR] "
                    f"EPI failed: "
                    f"{exc}"

                )

                continue


            # =================================================
            # EDGE RETENTION
            # =================================================

            try:

                edge_retention = float(

                    edge_preservation_proxy(

                        hazy_uint8,

                        pred_uint8

                    )

                )

            except Exception as exc:

                print(

                    f"    [WARNING] "
                    f"Edge retention failed: "
                    f"{exc}"

                )

                edge_retention = np.nan


            # =================================================
            # STORE METRICS
            # =================================================

            psnr_values.append(
                psnr
            )

            ssim_values.append(
                ssim
            )

            fsim_values.append(
                fsim
            )

            scielab_values.append(
                scielab
            )

            epi_values.append(
                epi
            )

            edge_ret_values.append(
                edge_retention
            )


            # =================================================
            # SAVE DEHAZED IMAGE
            # =================================================

            output_path = (

                out_dir

                /

                f"{hazy_path.stem}_dehazed.png"

            )


            save_rgb(

                output_path,

                pred

            )


            # =================================================
            # PRINT IMAGE RESULTS
            # =================================================

            print()

            print(

                f"    PSNR       : "
                f"{psnr:.4f} dB"

            )

            print(

                f"    SSIM       : "
                f"{ssim:.4f}"

            )

            print(

                f"    FSIM       : "
                f"{fsim:.4f}"

            )

            print(

                f"    S-CIELAB   : "
                f"{scielab:.4f}"

            )

            print(

                f"    EPI        : "
                f"{epi:.4f}"

            )

            print(

                f"    Edge Ret.  : "
                f"{edge_retention:.4f}"

            )


        # ====================================================
        # GAMMA AVERAGE
        # ====================================================

        if not psnr_values:

            print()

            print(

                f"[WARNING] "
                f"No images evaluated for gamma "
                f"{gamma}."

            )

            continue


        # ====================================================
        # SAFE EDGE AVERAGE
        # ====================================================

        valid_edge_values = [

            value

            for value
            in
            edge_ret_values

            if np.isfinite(
                value
            )

        ]


        if valid_edge_values:

            average_edge_retention = float(

                np.mean(
                    valid_edge_values
                )

            )

        else:

            average_edge_retention = np.nan


        # ====================================================
        # RESULT ROW
        # ====================================================

        row = {

            "gamma":
                gamma,

            "n_images":
                len(psnr_values),

            "Average_PSNR_dB":
                float(
                    np.mean(
                        psnr_values
                    )
                ),

            "Average_SSIM":
                float(
                    np.mean(
                        ssim_values
                    )
                ),

            "Average_FSIM":
                float(
                    np.mean(
                        fsim_values
                    )
                ),

            "Average_S_CIELAB_DeltaE":
                float(
                    np.mean(
                        scielab_values
                    )
                ),

            "Average_EPI":
                float(
                    np.mean(
                        epi_values
                    )
                ),

            "Average_Edge_Retention":
                average_edge_retention

        }


        rows.append(
            row
        )


        # ====================================================
        # PRINT GAMMA SUMMARY
        # ====================================================

        print()

        print(
            "=" * 70
        )

        print(
            f"GAMMA {gamma:.1f} SUMMARY"
        )

        print(
            "=" * 70
        )

        print()

        print(

            f"Average PSNR          : "
            f"{row['Average_PSNR_dB']:.6f} dB"

        )

        print(

            f"Average SSIM          : "
            f"{row['Average_SSIM']:.6f}"

        )

        print(

            f"Average FSIM          : "
            f"{row['Average_FSIM']:.6f}"

        )

        print(

            f"Average S-CIELAB     : "
            f"{row['Average_S_CIELAB_DeltaE']:.6f}"

        )

        print(

            f"Average EPI           : "
            f"{row['Average_EPI']:.6f}"

        )

        print(

            f"Average Edge Retention: "
            f"{row['Average_Edge_Retention']:.6f}"

        )


    # ========================================================
    # CHECK RESULTS
    # ========================================================

    if not rows:

        print()

        print(

            "[ERROR] "
            "No gamma values could be evaluated."

        )

        return


    # ========================================================
    # DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        rows
    )


    # ========================================================
    # CSV
    # ========================================================

    csv_path = (

        ABLATION_DIR

        /
        "gamma_ablation_results.csv"

    )


    df.to_csv(

        csv_path,

        index=False

    )


    # ========================================================
    # EXCEL
    # ========================================================

    xlsx_path = (

        ABLATION_DIR

        /
        "gamma_ablation_results.xlsx"

    )


    try:

        df.to_excel(

            xlsx_path,

            index=False

        )

        print()

        print(

            f"Excel saved:"
        )

        print(
            xlsx_path
        )

    except Exception as exc:

        print()

        print(

            f"Excel export skipped: "
            f"{exc}"

        )


    # ========================================================
    # FINAL TABLE
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "              FINAL GAMMA ABLATION RESULTS"
    )

    print(
        "=" * 70
    )

    print()

    print(
        df.to_string(
            index=False
        )
    )


    # ========================================================
    # GRAPH HELPER
    # ========================================================

    def save_bar_graph(
        column,
        ylabel,
        title,
        filename
    ):

        fig, ax = plt.subplots(

            figsize=(
                7,
                5
            )

        )

        ax.bar(

            df["gamma"].astype(
                str
            ),

            df[column]

        )

        ax.set_xlabel(
            "Transmission Gamma"
        )

        ax.set_ylabel(
            ylabel
        )

        ax.set_title(
            title
        )

        fig.tight_layout()

        fig.savefig(

            ABLATION_DIR
            /
            filename,

            dpi=200

        )

        plt.close(
            fig
        )


    # ========================================================
    # PSNR
    # ========================================================

    save_bar_graph(

        "Average_PSNR_dB",

        "Average PSNR (dB)",

        "Gamma Ablation: PSNR",

        "gamma_vs_psnr.png"

    )


    # ========================================================
    # SSIM
    # ========================================================

    save_bar_graph(

        "Average_SSIM",

        "Average SSIM",

        "Gamma Ablation: SSIM",

        "gamma_vs_ssim.png"

    )


    # ========================================================
    # FSIM
    # ========================================================

    save_bar_graph(

        "Average_FSIM",

        "Average FSIM",

        "Gamma Ablation: FSIM",

        "gamma_vs_fsim.png"

    )


    # ========================================================
    # EPI
    # ========================================================

    save_bar_graph(

        "Average_EPI",

        "Average EPI",

        "Gamma Ablation: Edge Preservation Index",

        "gamma_vs_epi.png"

    )


    # ========================================================
    # S-CIELAB
    # ========================================================

    save_bar_graph(

        "Average_S_CIELAB_DeltaE",

        "Average S-CIELAB ΔE",

        "Gamma Ablation: S-CIELAB",

        "gamma_vs_scielab.png"

    )


    # ========================================================
    # EDGE RETENTION
    # ========================================================

    save_bar_graph(

        "Average_Edge_Retention",

        "Average Edge Retention",

        "Gamma Ablation: Edge Retention",

        "gamma_vs_edge_retention.png"

    )


    # ========================================================
    # FINISHED
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "Gamma ablation completed successfully."
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"CSV:"
    )

    print(
        csv_path
    )

    print()

    print(
        f"Excel:"
    )

    print(
        xlsx_path
    )

    print()

    print(
        f"Graphs:"
    )

    print(
        ABLATION_DIR
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()