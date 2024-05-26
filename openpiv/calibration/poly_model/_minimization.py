import numpy as np

from ._check_params import _check_parameters
from .._doc_utils import (docstring_decorator,
                          doc_obj_coords, doc_img_coords, doc_cam_struct)


__all__ = [
    "minimize_polynomial"
]


@docstring_decorator(doc_cam_struct, doc_obj_coords, doc_img_coords)
def minimize_polynomial(
    cam_struct: dict,
    object_points: list,
    image_points: list,
):
    """Minimize polynomials.
    
    Minimize polynomials using Least Squares minimization.
    
    Parameters
    ----------
    cam_struct : dict
        {0}
    object_points : np.ndarray
        {1}
    image_points : np.ndarray
        {2}
        
    Returns
    -------
    cam_struct : dict
        {0}
        
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv import calib_utils, poly_model
    >>> from openpiv.data.test5 import cal_points
    
    >>> obj_x, obj_y, obj_z, img_x, img_y, img_size_x, img_size_y = cal_points()
    
    >>> camera_parameters = poly_model.generate_camera_params(
            name="cam1", 
            [img_size_x, img_size_y]
        )
    
    >>> camera_parameters = poly_model.minimize_polynomial(
            camera_parameters,
            [obj_x, obj_y, obj_z],
            [img_x, img_y]
        )
        
    >>> calib_utils.get_reprojection_error(
            camera_parameters, 
            poly_model.project_points,
            [obj_x, obj_y, obj_z],
            [img_x, img_y]
        )
        
    """
    _check_parameters(cam_struct)
    
    dtype = cam_struct["dtype"]
    
    object_points = np.array(object_points, dtype=dtype)
    image_points = np.array(image_points, dtype=dtype)
    
    if object_points.shape[1] < 19:
        raise ValueError(
            "Too little points to calibrate"
        )
    
    if object_points.shape[1] != image_points.shape[1]:
        raise ValueError(
            "Object point image point size mismatch"
        )
        
    x = image_points[0]
    y = image_points[1]
    
    X = object_points[0]
    Y = object_points[1]
    Z = object_points[2]

    polynomial_wi = np.array(
        [
            np.ones_like(X),
            X,     Y,     Z, 
            X*Y,   X*Z,   Y*Z,
            X**2,  Y**2,  Z**2,
            X**3,  X*X*Y, X*X*Z,
            Y**3,  X*Y*Y, Y*Y*Z,
            X*Z*Z, Y*Z*Z, X*Y*Z
        ],
        dtype=dtype
    ).T
    
    # in the future, break this into three Z subvolumes to further reduce errors.
    polynomial_iw = np.array(
        [
            np.ones_like(x),
            x,     y,     Z, 
            x*y,   x*Z,   y*Z,
            x**2,  y**2,  Z**2,
            x**3,  x*x*y, x*x*Z,
            y**3,  x*y*y, y*y*Z,
            x*Z*Z, y*Z*Z, x*y*Z
        ],
        dtype=dtype
    ).T
    

    # world to image (forward projection)
    coeff_wi = np.linalg.lstsq(
        polynomial_wi,
        image_points.T, 
        rcond=None
    )[0].astype(dtype, copy=False)
    
    # image to world (back projection)
    coeff_iw = np.linalg.lstsq(
        polynomial_iw,
        object_points.T, 
        rcond=None
    )[0].astype(dtype, copy=False)
    
    # psuedo-inverse based solution to system of equations
#    coeff_wi = np.array(image_points, dtype="float64") @ np.linalg.pinv(polynomial_wi.T)
#    coeff_wi = coeff_wi.T
    
#    coeff_iw = np.array(object_points, dtype="float64") @ np.linalg.pinv(polynomial_iw.T)
#    coeff_iw = coeff_iw.T

    cam_struct["poly_wi"] = coeff_wi
    cam_struct["poly_iw"] = coeff_iw
    
    return cam_struct