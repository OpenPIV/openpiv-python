import numpy as np
from typing import Tuple

from ._check_params import _check_parameters
from .._doc_utils import (docstring_decorator,
                          doc_cam_struct)


__all__ = [
    "generate_camera_params"
]


@docstring_decorator(doc_cam_struct)
def generate_camera_params(
    cam_name: str,
    resolution: Tuple[int, int],
    coefficients: np.ndarray=np.identity(3),
    dtype: str="float64"
    
):
    """Create a camera parameter structure.
    
    Create a camera parameter structure for DLT calibration.
    
    Parameters
    ----------
    cam_name : str
        Name of camera.
    resolution : tuple[int, int]
        Resolution of camera in x and y axes respectively.
    coefficients : np.ndarray
        The coefficients for a DLT matrix.
    dtype : str
        The dtype used in the projections.
    
    Returns
    -------
    cam_struct : dict
        {0}
        
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv import calib_utils, dlt_model
    >>> from openpiv.data.test5 import cal_points
    
    >>> obj_x, obj_y, obj_z, img_x, img_y, img_size_x, img_size_y = cal_points()
    
    >>> camera_parameters = dlt_model.generate_camera_params(
            name="cam1", 
            [img_size_x, img_size_y]
        )
    
    """    
    cam_struct = {}
    cam_struct["name"] = cam_name
    cam_struct["resolution"] = resolution
    cam_struct["coefficients"] = coefficients
    cam_struct["dtype"] = dtype
    
    _check_parameters(cam_struct)
    
    return cam_struct