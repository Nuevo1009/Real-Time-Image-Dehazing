class DehazeParams:
    """
    Parameters for the RGB-only adaptive
    image dehazing algorithm.
    """

    def __init__(
        self,

        # Downsampling factor
        down_factor=3,

        # Dark-channel patch size
        patch_size=5,

        # Gamma parameter
        gamma=2.0,

        # Haze-retention factor
        omega=0.9375,

        # Minimum transmission
        t_min=0.10,

        # Transmission refinement
        bilateral_d=9,
        bilateral_sigma_color=0.05,
        bilateral_sigma_space=15,

        # Atmospheric-light candidate fraction
        atmospheric_top_fraction=0.001,

        # Optional clarity enhancement
        enable_clarity=False,
        clarity_strength=0.20
    ):

        self.down_factor = down_factor

        self.patch_size = patch_size

        self.gamma = gamma

        self.omega = omega

        self.t_min = t_min

        self.bilateral_d = bilateral_d

        self.bilateral_sigma_color = (
            bilateral_sigma_color
        )

        self.bilateral_sigma_space = (
            bilateral_sigma_space
        )

        self.atmospheric_top_fraction = (
            atmospheric_top_fraction
        )

        self.enable_clarity = (
            enable_clarity
        )

        self.clarity_strength = (
            clarity_strength
        )


def default_dehaze_params():
    """
    Return default dehazing parameters.
    """

    return DehazeParams()