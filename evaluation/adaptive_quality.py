from evaluation.edge_no_reference import (
    edge_strength,
    sharpness_score,
    saturation_score
)


def calculate_adaptive_parameters(image):
    """
    Automatically determines enhancement parameters
    based on image quality.
    """

    edge = edge_strength(image)

    sharpness = sharpness_score(image)

    saturation = saturation_score(image)

    parameters = {}

    # ------------------------------------------
    # EDGE / SHARPNESS CONTROL
    # ------------------------------------------

    if sharpness < 50:

        parameters["sharpen_amount"] = 1.4
        parameters["sharpen_radius"] = 1.5

    elif sharpness < 150:

        parameters["sharpen_amount"] = 1.0
        parameters["sharpen_radius"] = 1.0

    else:

        parameters["sharpen_amount"] = 0.5
        parameters["sharpen_radius"] = 0.8

    # ------------------------------------------
    # SATURATION CONTROL
    # ------------------------------------------

    if saturation > 0.65:

        parameters["saturation_scale"] = 0.85

    elif saturation < 0.25:

        parameters["saturation_scale"] = 1.05

    else:

        parameters["saturation_scale"] = 1.0

    # ------------------------------------------
    # CLAHE CONTRAST CONTROL
    # ------------------------------------------

    if edge < 0.05:

        parameters["clahe_clip"] = 2.5

    else:

        parameters["clahe_clip"] = 1.5

    return parameters