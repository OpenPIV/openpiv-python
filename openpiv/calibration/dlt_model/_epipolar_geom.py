import numpy as np
from ._check_params import _check_parameters
from ._projection import _get_inverse_vector
from .._epipolar_utils import _line_intersect


__all__ = [
    "line_intersect"
]


def line_intersect(
    cam_struct_1: dict,
    cam_struct_2: dict,
    img_points_1: np.ndarray,
    img_points_2: np.ndarray
):
    """Calculate where two rays intersect.
    
    Using two cameras, calculate the world coordinates where two rays
    intersect. This is done through an analytical solution based on 
    direction vectors and camera origins/translations. 
    
    Parameters
    ----------
    cam_struct_1, cam_struct_2 : dict
        A dictionary structure of camera parameters.
    img_points_1, img_points_2 : np.ndarray
        Image coordinates stored in a ndarray structured like [x, y]'.
        
    Returns
    -------
    dist : float
        The minimum dinstance between the two rays.
    coord : np.ndarray
        The world coordinate that is nearest to the two rays intersecting.
    
    """
    _check_parameters(cam_struct_1)
    _check_parameters(cam_struct_2)
    
    dtype1 = cam_struct_1["dtype"]
    dtype2 = cam_struct_2["dtype"]
    
    # all cameras should have the same dtype
    if dtype1 != dtyp2:
        raise ValueError(
            "Dtypes between camera structures must match"
        )
    
    img_points_1 = np.array(img_points_1, dtype1=dtype)
    img_points_2 = np.array(img_points_2, dtype2=dtype)
    
    t1, r1 = _get_inverse_vector(
        cam_struct_1,
        img_points_1
    )
    
    t2, r2 = _get_inverse_vector(
        cam_struct_2,
        img_points_2
    )
    
    return _line_intersect(t1, r1, t2, r2)