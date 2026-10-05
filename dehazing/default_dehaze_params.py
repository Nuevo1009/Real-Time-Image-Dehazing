class DehazeParams:
    """
    Default parameters for the adaptive transmission
    estimation-based dehazing method.
    """

    def __init__(
        self,
        down_factor=3,
        patch_size=5,
        gamma=2.0,
        omega=0.9375,
        t_min=0.1,
        flicker_thresh=3
    ):

        # Down-sampling factor
        self.down_factor = down_factor

        # Dark-channel patch size
        self.patch_size = patch_size

        # Correction-factor exponent
        self.gamma = gamma

        # Haze-retention factor
        self.omega = omega

        # Minimum transmission
        self.t_min = t_min

        # Flicker threshold on 0-255 scale
        self.flicker_thresh = flicker_thresh


def default_dehaze_params():
    """
    Equivalent to MATLAB:
        params = defaultDehazeParams();
    """

    return DehazeParams()