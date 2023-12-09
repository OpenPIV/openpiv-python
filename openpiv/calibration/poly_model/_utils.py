import numpy as np
from typing import Tuple

from ._check_params import _check_parameters


__all__ = [
    "generate_camera_params"
]


def generate_camera_params(
    cam_name: str,
    resolution: Tuple[int, int],
    poly_wi: np.ndarray=np.ones((2,19), dtype="float64").T,
    poly_iw: np.ndarray=np.ones((3,19), dtype="float64").T,
    dtype: str="float64"
    
):
    """Create a camera parameter structure.
    
    Create a camera parameter structure for polynomial calibration.
    
    Parameters
    ----------
    cam_name : str
        Name of camera.
    resolution : tuple[int, int]
        Resolution of camera in x and y axes respectively.
    poly_wi : np.ndarray
        19 coefficients for world to image polynomial calibration in [x, y]'.
    poly_iw : np.ndarray
        19 coefficients for image to world polynomial calibration in [X, Y, Z]'.
    dtype : str
        The dtype used in the projections.
    
    Returns
    -------
    cam_struct : dict
        A dictionary structure of camera parameters.
        
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv import calib_utils, calib_polynomial
    >>> from openpiv.data.test5 import cal_points
    
    >>> obj_x, obj_y, obj_z, img_x, img_y, img_size_x, img_size_y = cal_points()
    
    >>> camera_parameters = calib_polynomial.generate_camera_params(
            name="cam1", 
            [img_size_x, img_size_y]
        )
    
    """    
    cam_struct = {}
    cam_struct["name"] = cam_name
    cam_struct["resolution"] = resolution
    cam_struct["poly_wi"] = poly_wi
    cam_struct["poly_iw"] = poly_iw
    cam_struct["dtype"] = dtype
    
    _check_parameters(cam_struct)
    
    return cam_struct