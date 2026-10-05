import cv2
import numpy as np

from pathlib import Path
from tqdm import tqdm

from classifier.predict import (
    load_model,
    classify_image
)

from dehazing.adaptive_transmission_dehaze import (
    adaptive_dehaze
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent


# ============================================================
# INPUT FOLDER
# ============================================================

INPUT_FOLDER = (

    BASE_DIR
    /
    "mixed_images"

)


# ============================================================
# TRAINED CNN MODEL
# ============================================================

MODEL_PATH = (

    BASE_DIR
    /
    "models"
    /
    "best_haze_CNN.pth"

)


# ============================================================
# RESULTS FOLDERS
# ============================================================

RESULTS_FOLDER = (

    BASE_DIR
    /
    "results"

)


CLEAR_FOLDER = (

    RESULTS_FOLDER
    /
    "clear_bypassed"

)


INITIAL_DEHAZED_FOLDER = (

    RESULTS_FOLDER
    /
    "initial_dehazed"

)


FINAL_OUTPUT_FOLDER = (

    RESULTS_FOLDER
    /
    "final_output"

)


# ============================================================
# VALID IMAGE EXTENSIONS
# ============================================================

VALID_EXTENSIONS = (

    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"

)


# ============================================================
# CREATE REQUIRED OUTPUT FOLDERS
# ============================================================

def create_folders():

    CLEAR_FOLDER.mkdir(

        parents=True,

        exist_ok=True

    )


    INITIAL_DEHAZED_FOLDER.mkdir(

        parents=True,

        exist_ok=True

    )


    FINAL_OUTPUT_FOLDER.mkdir(

        parents=True,

        exist_ok=True

    )


# ============================================================
# GET INPUT IMAGES
# ============================================================

def get_image_files():

    image_files = []


    for image_path in INPUT_FOLDER.iterdir():

        if (

            image_path.is_file()

            and

            image_path.suffix.lower()

            in VALID_EXTENSIONS

        ):

            image_files.append(

                image_path

            )


    return sorted(

        image_files

    )


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(

    image_path

):

    image = cv2.imread(

        str(image_path)

    )


    if image is None:

        raise ValueError(

            f"Unable to read image:\n"

            f"{image_path}"

        )


    return image


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(

    image,

    output_path

):

    success = cv2.imwrite(

        str(output_path),

        image

    )


    if not success:

        raise IOError(

            f"Unable to save image:\n"

            f"{output_path}"

        )


# ============================================================
# DEHAZING WRAPPER
# ============================================================

def perform_dehazing(

    image_bgr

):


    # --------------------------------------------------------
    # The existing adaptive_dehaze function may return:
    #
    # output
    #
    # OR
    #
    # output, transmission
    #
    # This handles both cases.
    # --------------------------------------------------------

    result = adaptive_dehaze(

        image_bgr

    )


    # --------------------------------------------------------
    # IF FUNCTION RETURNS TUPLE
    # --------------------------------------------------------

    if isinstance(

        result,

        tuple

    ):


        dehazed_image = result[0]


    # --------------------------------------------------------
    # IF FUNCTION RETURNS ONLY IMAGE
    # --------------------------------------------------------

    else:


        dehazed_image = result


    # --------------------------------------------------------
    # CONVERT TO NUMPY ARRAY
    # --------------------------------------------------------

    dehazed_image = np.asarray(

        dehazed_image

    )


    # --------------------------------------------------------
    # HANDLE FLOAT OUTPUT
    # --------------------------------------------------------

    if dehazed_image.dtype != np.uint8:


        # If values are in 0–1 range

        if dehazed_image.max() <= 1.0:


            dehazed_image = (

                dehazed_image

                *

                255.0

            )


        dehazed_image = np.clip(

            dehazed_image,

            0,

            255

        ).astype(

            np.uint8

        )


    return dehazed_image


# ============================================================
# PROCESS SINGLE IMAGE
# ============================================================

def process_image(

    image_path,

    model,

    device

):


    # --------------------------------------------------------
    # STEP 1
    # CNN + FUZZY CLASSIFICATION
    # --------------------------------------------------------

    classification, haze_score, memberships = (

        classify_image(

            image_path,

            model,

            device

        )

    )


    # --------------------------------------------------------
    # STEP 2
    # LOAD ORIGINAL IMAGE
    # --------------------------------------------------------

    input_image = load_image(

        image_path

    )


    # --------------------------------------------------------
    # STEP 3
    # CREATE OUTPUT PATHS
    # --------------------------------------------------------

    filename = (

        image_path.name

    )


    clear_output_path = (

        CLEAR_FOLDER
        /
        filename

    )


    initial_dehazed_path = (

        INITIAL_DEHAZED_FOLDER
        /
        filename

    )


    final_output_path = (

        FINAL_OUTPUT_FOLDER
        /
        filename

    )


    # ========================================================
    # CLEAR IMAGE
    # ========================================================

    if classification == "CLEAR":


        # ----------------------------------------------------
        # Bypass dehazing
        # ----------------------------------------------------

        output_image = input_image.copy()


        # ----------------------------------------------------
        # Save bypassed clear image
        # ----------------------------------------------------

        save_image(

            output_image,

            clear_output_path

        )


        # ----------------------------------------------------
        # Save final output
        # ----------------------------------------------------

        save_image(

            output_image,

            final_output_path

        )


        processing_action = (

            "BYPASSED"

        )


    # ========================================================
    # UNCERTAIN OR HAZY IMAGE
    # ========================================================

    else:


        # ----------------------------------------------------
        # PERFORM DEHAZING
        # ----------------------------------------------------

        dehazed_image = (

            perform_dehazing(

                input_image

            )

        )


        # ----------------------------------------------------
        # Save initial dehazed result
        # ----------------------------------------------------

        save_image(

            dehazed_image,

            initial_dehazed_path

        )


        # ----------------------------------------------------
        # Currently final output is same as initial dehazed
        #
        # Later:
        #
        # Edge Analysis
        # ↓
        # Adaptive Edge Control
        # ↓
        # Saturation Correction
        # ----------------------------------------------------

        output_image = (

            dehazed_image

        )


        save_image(

            output_image,

            final_output_path

        )


        processing_action = (

            "DEHAZED"

        )


    # ========================================================
    # RETURN INFORMATION
    # ========================================================

    return {

        "filename":

            filename,


        "classification":

            classification,


        "haze_score":

            haze_score,


        "clear_membership":

            memberships["clear"],


        "uncertain_membership":

            memberships["uncertain"],


        "hazy_membership":

            memberships["hazy"],


        "action":

            processing_action

    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():


    print(

        "\n"

        +

        "=" * 60

    )


    print(

        "REAL-WORLD IMAGE DEHAZING PIPELINE"

    )


    print(

        "=" * 60

    )


    # --------------------------------------------------------
    # CHECK INPUT FOLDER
    # --------------------------------------------------------

    if not INPUT_FOLDER.exists():

        raise FileNotFoundError(

            f"\nInput folder not found:\n"

            f"{INPUT_FOLDER}"

        )


    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(

            f"\nTrained model not found:\n"

            f"{MODEL_PATH}"

        )


    # --------------------------------------------------------
    # CREATE OUTPUT FOLDERS
    # --------------------------------------------------------

    create_folders()


    # --------------------------------------------------------
    # GET IMAGES
    # --------------------------------------------------------

    image_files = (

        get_image_files()

    )


    if len(image_files) == 0:

        raise ValueError(

            "\nNo images found in:\n"

            f"{INPUT_FOLDER}"

        )


    print(

        f"\nImages found: "

        f"{len(image_files)}"

    )


    # --------------------------------------------------------
    # LOAD CNN
    # --------------------------------------------------------

    print(

        "\nLoading CNN model..."

    )


    print(

        f"Model path: "

        f"{MODEL_PATH}"

    )


    model, device = (

        load_model(

            MODEL_PATH

        )

    )


    print(

        f"\nUsing device: "

        f"{device}"

    )


    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    total_clear = 0

    total_uncertain = 0

    total_hazy = 0

    total_dehazed = 0

    total_bypassed = 0


    # ========================================================
    # PROCESS IMAGES
    # ========================================================

    print(

        "\nStarting image processing...\n"

    )


    for image_path in tqdm(

        image_files,

        desc="Processing Images"

    ):


        try:


            result = (

                process_image(

                    image_path,

                    model,

                    device

                )

            )


            # ------------------------------------------------
            # UPDATE CLASSIFICATION COUNTERS
            # ------------------------------------------------

            if (

                result["classification"]

                ==

                "CLEAR"

            ):


                total_clear += 1


            elif (

                result["classification"]

                ==

                "UNCERTAIN"

            ):


                total_uncertain += 1


            elif (

                result["classification"]

                ==

                "HAZY"

            ):


                total_hazy += 1


            # ------------------------------------------------
            # UPDATE ACTION COUNTERS
            # ------------------------------------------------

            if (

                result["action"]

                ==

                "DEHAZED"

            ):


                total_dehazed += 1


            else:


                total_bypassed += 1


            # ------------------------------------------------
            # DISPLAY RESULT
            # ------------------------------------------------

            tqdm.write(

                f"\nImage: "

                f"{result['filename']}"

            )


            tqdm.write(

                f"CNN Score: "

                f"{result['haze_score']:.4f}"

            )


            tqdm.write(

                f"Fuzzy Memberships → "

                f"Clear: "

                f"{result['clear_membership']:.3f}, "

                f"Uncertain: "

                f"{result['uncertain_membership']:.3f}, "

                f"Hazy: "

                f"{result['hazy_membership']:.3f}"

            )


            tqdm.write(

                f"Classification: "

                f"{result['classification']}"

            )


            tqdm.write(

                f"Action: "

                f"{result['action']}"

            )


        except Exception as error:


            tqdm.write(

                f"\nERROR processing "

                f"{image_path.name}:"

            )


            tqdm.write(

                str(error)

            )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(

        "\n"

        +

        "=" * 60

    )


    print(

        "PIPELINE COMPLETED"

    )


    print(

        "=" * 60

    )


    print(

        f"\nTotal Images: "

        f"{len(image_files)}"

    )


    print(

        f"\nCLEAR: "

        f"{total_clear}"

    )


    print(

        f"UNCERTAIN: "

        f"{total_uncertain}"

    )


    print(

        f"HAZY: "

        f"{total_hazy}"

    )


    print(

        f"\nBypassed: "

        f"{total_bypassed}"

    )


    print(

        f"Dehazed: "

        f"{total_dehazed}"

    )


    print(

        "\nResults saved in:"

    )


    print(

        f"\nClear bypassed:"

    )


    print(

        CLEAR_FOLDER

    )


    print(

        f"\nInitial dehazed:"

    )


    print(

        INITIAL_DEHAZED_FOLDER

    )


    print(

        f"\nFinal output:"

    )


    print(

        FINAL_OUTPUT_FOLDER

    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()