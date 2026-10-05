"""
Gamma Ablation Study

Compares different transmission gamma values using:

    PSNR
    SSIM
    FSIM
    S-CIELAB DeltaE
    EPI
    Edge Retention

The study uses the current dehazing modules and
CNN_Dataset/test/hazy + clear folders.
"""

from pathlib import Path
import re

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

BASE_DIR = Path(__file__).resolve().parent


# ------------------------------------------------------------
# HAZY TEST IMAGES
# ------------------------------------------------------------

HAZY_DIR = (
    BASE_DIR
    / "dataset"
    / "CNN_Dataset"
    / "test"
    / "hazy"
)


# ------------------------------------------------------------
# CLEAR / GROUND-TRUTH TEST IMAGES
# ------------------------------------------------------------

GT_DIR = (
    BASE_DIR
    / "dataset"
    / "CNN_Dataset"
    / "test"
    / "clear"
)


# ------------------------------------------------------------
# OUTPUT DIRECTORY
# ------------------------------------------------------------

ABLATION_DIR = (
    BASE_DIR
    / "output"
    / "ablation"
)


# ============================================================
# GAMMA VALUES
# ============================================================

# Same values used in your MATLAB ablation study.

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
# IMAGE READING
# ============================================================

def read_rgb(path):
    """
    Read an image using OpenCV and convert BGR -> RGB.

    Returns:
        RGB float32 image in range [0, 1].
    """

    bgr = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR
    )

    if bgr is None:

        raise RuntimeError(
            f"Could not read image: {path}"
        )

    rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB
    )

    rgb = (
        rgb.astype(np.float32)
        / 255.0
    )

    return rgb


# ============================================================
# IMAGE SAVING
# ============================================================

def save_rgb(path, image):
    """
    Save RGB float image [0,1] as PNG.
    """

    image_uint8 = (
        np.clip(
            np.round(
                image * 255.0
            ),
            0,
            255
        )
        .astype(np.uint8)
    )

    bgr = cv2.cvtColor(
        image_uint8,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        str(path),
        bgr
    )


# ============================================================
# FIND GROUND TRUTH
# ============================================================

def find_gt(
    hazy_path,
    gt_files
):
    """
    Find the corresponding clear / GT image.

    Matching order:

    1. Exact filename stem
    2. Leading numeric scene ID

    Example:

        40_outdoor.jpg
        40.jpg

    can still be matched.
    """

    # --------------------------------------------------------
    # Exact filename stem
    # --------------------------------------------------------

    hazy_stem = (
        hazy_path.stem.lower()
    )

    for gt_path in gt_files:

        if (
            gt_path.stem.lower()
            == hazy_stem
        ):

            return gt_path


    # --------------------------------------------------------
    # Leading numeric scene ID
    # --------------------------------------------------------

    match = re.match(
        r"^(\d+)",
        hazy_path.name
    )

    if match:

        scene_id = match.group(1)

        candidates = []

        for gt_path in gt_files:

            if re.match(
                r"^"
                + re.escape(scene_id)
                + r"(\D|$)",
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
# TRANSMISSION DEHAZING FOR A SPECIFIC GAMMA
# ============================================================

def dehaze_with_gamma(
    hazy,
    gamma,
    params
):
    """
    Perform the dehazing pipeline using a specific
    transmission gamma.

    Gamma is passed directly to:

        pixelwise_transmission()

    which implements:

        k = 1 - (minMH / (minMH + maxMH))^gamma

        tm = 1 - omega * ratio * k
    """

    # --------------------------------------------------------
    # Downsample
    # --------------------------------------------------------

    down_factor = params.down_factor

    height, width = hazy.shape[:2]

    down_width = max(
        1,
        width // down_factor
    )

    down_height = max(
        1,
        height // down_factor
    )

    image_down = cv2.resize(
        hazy,
        (
            down_width,
            down_height
        ),
        interpolation=cv2.INTER_AREA
    )


    # --------------------------------------------------------
    # Atmospheric light
    # --------------------------------------------------------

    AF, AN, _, _ = (
        estimate_atmospheric_light(
            image_down,
            params.patch_size,
            th_ref=None
        )
    )


    # --------------------------------------------------------
    # Pixel-wise atmospheric light
    # --------------------------------------------------------

    atmospheric_light = (
        pixelwise_atmospheric_light(
            hazy,
            AF,
            AN
        )
    )


    # --------------------------------------------------------
    # Pixel-wise transmission
    # --------------------------------------------------------

    transmission = (
        pixelwise_transmission(
            hazy,
            atmospheric_light,
            omega=params.omega,
            gamma=gamma,
            t_min=params.t_min
        )
    )


    # --------------------------------------------------------
    # Scene recovery
    # --------------------------------------------------------

    dehazed = (
        scene_dehazing(
            hazy,
            atmospheric_light,
            transmission
        )
    )


    # --------------------------------------------------------
    # Final range protection
    # --------------------------------------------------------

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

    print("=" * 70)
    print("              GAMMA ABLATION STUDY")
    print("=" * 70)

    print()
    print("Hazy directory:")
    print(HAZY_DIR)

    print()
    print("Ground-truth directory:")
    print(GT_DIR)

    print()
    print("Output directory:")
    print(ABLATION_DIR)

    print()


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

        print(
            f"[ERROR] Hazy directory does not exist:\n"
            f"{HAZY_DIR}"
        )

        return


    if not GT_DIR.exists():

        print(
            f"[ERROR] Ground-truth directory does not exist:\n"
            f"{GT_DIR}"
        )

        return


    # ========================================================
    # GET IMAGE FILES
    # ========================================================

    hazy_files = sorted(

        p
        for p in HAZY_DIR.iterdir()

        if (
            p.is_file()
            and
            p.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )


    gt_files = sorted(

        p
        for p in GT_DIR.iterdir()

        if (
            p.is_file()
            and
            p.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )


    # ========================================================
    # CHECK IMAGE COUNTS
    # ========================================================

    print(
        f"Found {len(hazy_files)} hazy images."
    )

    print(
        f"Found {len(gt_files)} ground-truth images."
    )

    print()


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
    # STORE RESULTS
    # ========================================================

    rows = []


    # ========================================================
    # LOOP THROUGH GAMMA VALUES
    # ========================================================

    for gamma in GAMMAS:

        print()
        print("-" * 70)
        print(
            f"Testing gamma = {gamma}"
        )
        print("-" * 70)


        # ----------------------------------------------------
        # Load default parameters
        # ----------------------------------------------------

        params = (
            default_dehaze_params()
        )


        # ----------------------------------------------------
        # Set gamma
        # ----------------------------------------------------

        params.gamma = gamma


        # ----------------------------------------------------
        # Metric storage
        # ----------------------------------------------------

        psnr_values = []
        ssim_values = []
        fsim_values = []
        scielab_values = []
        epi_values = []
        edge_ret_values = []


        # ----------------------------------------------------
        # Output directory for this gamma
        # ----------------------------------------------------

        out_dir = (
            ABLATION_DIR
            / f"gamma_{gamma:.1f}"
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

            print(
                f"[{index}/{len(hazy_files)}] "
                f"{hazy_path.name}"
            )


            # ------------------------------------------------
            # Find GT
            # ------------------------------------------------

            gt_path = find_gt(
                hazy_path,
                gt_files
            )


            if gt_path is None:

                print(
                    f"    [SKIP] "
                    f"No GT found for "
                    f"{hazy_path.name}"
                )

                continue


            print(
                f"    GT: {gt_path.name}"
            )


            # ------------------------------------------------
            # Read images
            # ------------------------------------------------

            hazy = read_rgb(
                hazy_path
            )

            gt = read_rgb(
                gt_path
            )


            # ------------------------------------------------
            # Dehaze using current gamma
            # ------------------------------------------------

            try:

                pred, A, transmission = (
                    dehaze_with_gamma(
                        hazy,
                        gamma,
                        params
                    )
                )

            except Exception as exc:

                print(
                    f"    [ERROR] "
                    f"Dehazing failed: {exc}"
                )

                continue


            # ------------------------------------------------
            # Match GT size
            # ------------------------------------------------

            if (
                gt.shape[:2]
                !=
                pred.shape[:2]
            ):

                gt = cv2.resize(

                    gt,

                    (
                        pred.shape[1],
                        pred.shape[0]
                    ),

                    interpolation=cv2.INTER_AREA
                )


            # ------------------------------------------------
            # Convert to uint8 for evaluation
            # ------------------------------------------------

            pred_uint8 = (
                np.clip(
                    pred * 255.0,
                    0,
                    255
                )
                .astype(np.uint8)
            )


            gt_uint8 = (
                np.clip(
                    gt * 255.0,
                    0,
                    255
                )
                .astype(np.uint8)
            )


            # =================================================
            # PSNR + SSIM
            # =================================================

            basic = (
                calculate_basic_metrics(
                    pred_uint8,
                    gt_uint8
                )
            )


            psnr = basic["psnr"]

            ssim = basic["ssim"]


            # =================================================
            # FSIM
            # =================================================

            fsim = calculate_fsim(
                pred_uint8,
                gt_uint8
            )


            # =================================================
            # S-CIELAB
            # =================================================

            scielab = calculate_scielab(
                pred_uint8,
                gt_uint8
            )


            # =================================================
            # EPI
            # =================================================

            epi = calculate_epi(
                pred_uint8,
                gt_uint8
            )


            # =================================================
            # EDGE RETENTION
            # =================================================

            edge_retention = (
                edge_preservation_proxy(
                    hazy_uint8_from_rgb(hazy),
                    pred_uint8
                )
            )


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
                / f"{hazy_path.stem}_dehazed.png"
            )

            save_rgb(
                output_path,
                pred
            )


            print(
                f"    PSNR={psnr:.2f} dB | "
                f"SSIM={ssim:.4f} | "
                f"FSIM={fsim:.4f} | "
                f"S-CIELAB={scielab:.4f} | "
                f"EPI={epi:.4f}"
            )


        # ====================================================
        # AVERAGE RESULTS
        # ====================================================

        if psnr_values:

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
                    float(
                        np.mean(
                            edge_ret_values
                        )
                    )
            }


            rows.append(
                row
            )


            # ------------------------------------------------
            # Print gamma summary
            # ------------------------------------------------

            print()
            print(
                f"Gamma {gamma:.1f} SUMMARY"
            )

            print(
                f"Average PSNR : "
                f"{row['Average_PSNR_dB']:.4f} dB"
            )

            print(
                f"Average SSIM : "
                f"{row['Average_SSIM']:.4f}"
            )

            print(
                f"Average FSIM : "
                f"{row['Average_FSIM']:.4f}"
            )

            print(
                f"Average S-CIELAB ΔE : "
                f"{row['Average_S_CIELAB_DeltaE']:.4f}"
            )

            print(
                f"Average EPI : "
                f"{row['Average_EPI']:.4f}"
            )

            print(
                f"Average Edge Retention : "
                f"{row['Average_Edge_Retention']:.4f}"
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
    # CREATE DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        rows
    )


    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_path = (
        ABLATION_DIR
        / "gamma_ablation_results.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )


    # ========================================================
    # SAVE EXCEL
    # ========================================================

    xlsx_path = (
        ABLATION_DIR
        / "gamma_ablation_results.xlsx"
    )

    try:

        df.to_excel(
            xlsx_path,
            index=False
        )

        print()
        print(
            f"Excel saved: {xlsx_path}"
        )

    except Exception as exc:

        print(
            f"Excel export skipped: {exc}"
        )


    # ========================================================
    # PRINT FINAL TABLE
    # ========================================================

    print()
    print("=" * 70)
    print(
        "FINAL GAMMA ABLATION RESULTS"
    )
    print("=" * 70)

    print(
        df.to_string(
            index=False
        )
    )


    # ========================================================
    # PSNR GRAPH
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.bar(
        df["gamma"].astype(str),
        df["Average_PSNR_dB"]
    )

    ax.set_xlabel(
        "Transmission Gamma"
    )

    ax.set_ylabel(
        "Average PSNR (dB)"
    )

    ax.set_title(
        "Gamma Ablation: PSNR"
    )

    fig.tight_layout()

    fig.savefig(
        ABLATION_DIR
        / "gamma_vs_psnr.png",
        dpi=200
    )

    plt.close(fig)


    # ========================================================
    # SSIM GRAPH
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.bar(
        df["gamma"].astype(str),
        df["Average_SSIM"]
    )

    ax.set_xlabel(
        "Transmission Gamma"
    )

    ax.set_ylabel(
        "Average SSIM"
    )

    ax.set_title(
        "Gamma Ablation: SSIM"
    )

    fig.tight_layout()

    fig.savefig(
        ABLATION_DIR
        / "gamma_vs_ssim.png",
        dpi=200
    )

    plt.close(fig)


    # ========================================================
    # FSIM GRAPH
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.bar(
        df["gamma"].astype(str),
        df["Average_FSIM"]
    )

    ax.set_xlabel(
        "Transmission Gamma"
    )

    ax.set_ylabel(
        "Average FSIM"
    )

    ax.set_title(
        "Gamma Ablation: FSIM"
    )

    fig.tight_layout()

    fig.savefig(
        ABLATION_DIR
        / "gamma_vs_fsim.png",
        dpi=200
    )

    plt.close(fig)


    # ========================================================
    # EPI GRAPH
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.bar(
        df["gamma"].astype(str),
        df["Average_EPI"]
    )

    ax.set_xlabel(
        "Transmission Gamma"
    )

    ax.set_ylabel(
        "Average EPI"
    )

    ax.set_title(
        "Gamma Ablation: Edge Preservation Index"
    )

    fig.tight_layout()

    fig.savefig(
        ABLATION_DIR
        / "gamma_vs_epi.png",
        dpi=200
    )

    plt.close(fig)


    # ========================================================
    # S-CIELAB GRAPH
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.bar(
        df["gamma"].astype(str),
        df["Average_S_CIELAB_DeltaE"]
    )

    ax.set_xlabel(
        "Transmission Gamma"
    )

    ax.set_ylabel(
        "Average S-CIELAB ΔE"
    )

    ax.set_title(
        "Gamma Ablation: S-CIELAB"
    )

    fig.tight_layout()

    fig.savefig(
        ABLATION_DIR
        / "gamma_vs_scielab.png",
        dpi=200
    )

    plt.close(fig)


    # ========================================================
    # EDGE RETENTION GRAPH
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.bar(
        df["gamma"].astype(str),
        df["Average_Edge_Retention"]
    )

    ax.set_xlabel(
        "Transmission Gamma"
    )

    ax.set_ylabel(
        "Average Edge Retention"
    )

    ax.set_title(
        "Gamma Ablation: Edge Retention"
    )

    fig.tight_layout()

    fig.savefig(
        ABLATION_DIR
        / "gamma_vs_edge_retention.png",
        dpi=200
    )

    plt.close(fig)


    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print("=" * 70)
    print(
        "Gamma ablation completed successfully."
    )
    print("=" * 70)

    print()
    print(
        f"CSV   : {csv_path}"
    )

    print(
        f"Excel : {xlsx_path}"
    )

    print(
        f"Graphs: {ABLATION_DIR}"
    )


# ============================================================
# HELPER
# ============================================================

def hazy_uint8_from_rgb(image):
    """
    Convert RGB float [0,1] -> RGB uint8.
    """

    return (
        np.clip(
            image * 255.0,
            0,
            255
        )
        .astype(np.uint8)
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()