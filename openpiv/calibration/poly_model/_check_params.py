import numpy as np


def _check_parameters(self):
    """Check camera parameters"""

    if not isinstance(self.name, str):
        raise ValueError(
            "Camera name must be a string"
        )

    if len(self.resolution) != 2:
        raise ValueError(
            "Resolution must be a two element tuple"
        )

    if len(self.poly_wi.shape) != 2:
        raise ValueError(
            "World to image polynomial coefficients must be 2 dimensional."
        )

    if self.poly_wi.shape[0] != 19:
        raise ValueError(
            "World to image polynomial coefficients must be ordered in [x, y]'"
        )

    if self.poly_wi.shape[1] != 2:
        raise ValueError(
            "There must be 19 coefficients in the world to image polynomial"
        )

    if len(self.poly_iw_o.shape) != 2:
        raise ValueError(
            "Ray origin polynomial coefficients must be 2 dimensional."
        )

    if self.poly_iw_o.shape[0] != 10:
        raise ValueError(
            "Ray origin polynomial coefficients must be ordered in [x, y]'"
        )

    if self.poly_iw_o.shape[1] != 2:
        raise ValueError(
            "There must be 10 coefficients in the ray origin polynomial"
        )

    if len(self.poly_iw_v.shape) != 2:
        raise ValueError(
            "Ray direction polynomial coefficients must be 2 dimensional."
        )

    if self.poly_iw_v.shape[0] != 10:
        raise ValueError(
            "Ray direction polynomial coefficients must be ordered in [x, y]'"
        )

    if self.poly_iw_v.shape[1] != 2:
        raise ValueError(
            "There must be 10 coefficients in the ray direction polynomial"
        )

    if self.dtype not in ["float32", "float64"]:
        raise ValueError(
            "Dtype is not supported for camera calibration"
        )
