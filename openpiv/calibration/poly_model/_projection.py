import numpy as np

from ._check_params import _check_parameters
from .._doc_utils import (docstring_decorator,
                          doc_obj_coords, doc_img_coords, doc_cam_struct)


__all__ = [
    "project_points",
    "project_to_z"
]


@docstring_decorator(doc_cam_struct, doc_obj_coords)
def project_points(
    cam_struct: dict,
    object_points: np.ndarray
):
    """Project object points to image points.
    
    Project object, or real world points, to image points.
    
    Parameters
    ----------
    cam_struct : dict
        {0}
    object_points : 2D np.ndarray
        {1}
        
    Returns
    -------
    x : 1D np.ndarray
        Projected image x-coordinates.
    y : 1D np.ndarray
        Projected image y-coordinates.
        
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv.calibration import calib_utils, poly_model
    >>> from openpiv.data.test5 import cal_points
    
    >>> obj_x, obj_y, obj_z, img_x, img_y, img_size_x, img_size_y = cal_points()
    
    >>> obj_points = np.array([obj_x[0:2], obj_y[0:2], obj_z[0:2]], dtype="float64")
    >>> img_points = np.array([img_x[0:2], img_y[0:2]], dtype="float64")
    
    >>> camera_parameters = poly_model.get_cam_params(
            name="cam1", 
            [img_size_x, img_size_y]
        )
    
    >>> camera_parameters = poly_model.minimize_params(
            camera_parameters,
            [obj_x, obj_y, obj_z],
            [img_x, img_y]
        )
        
    >>> poly_model.project_points(
            camera_parameters,
            obj_points
        )
        
    >>> img_points
    
    """ 
    _check_parameters(cam_struct)
    
    dtype = cam_struct["dtype"]
    
    object_points = np.array(object_points, dtype=dtype)
    
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
    
    img_points = np.dot(
        polynomial_wi,
        cam_struct["poly_wi"]
    ).T
    
    return img_points.astype(dtype, copy=False)


@docstring_decorator(doc_cam_struct, doc_img_coords)
def project_to_z(
    cam_struct: dict,
    image_points: np.ndarray,
    z: float
):
    """Project image points to world points.
    
    Project image points to world points at specified z-plane.
    
    Parameters
    ----------
    cam_struct : dict
        {0}
    image_points : 2D np.ndarray
        {1}
    z : float
        A float specifying the Z (depth) plane to project to.
        
    Returns
    -------
    X : 1D np.ndarray
        Projected world x-coordinates.
    Y : 1D np.ndarray
        Projected world y-coordinates.
    Z : 1D np.ndarray
        Projected world z-coordinates.
    
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv.calibration import calib_utils, poly_model
    >>> from openpiv.data.test5 import cal_points
    
    >>> obj_x, obj_y, obj_z, img_x, img_y, img_size_x, img_size_y = cal_points()
    
    >>> obj_points = np.array([obj_x[0:2], obj_y[0:2], obj_z[0:2]], dtype="float64")
    >>> img_points = np.array([img_x[0:2], img_y[0:2]], dtype="float64")
    
    >>> camera_parameters = poly_model.get_cam_params(
            name="cam1", 
            [img_size_x, img_size_y]
        )
    
    >>> camera_parameters = poly_model.minimize_params(
            camera_parameters,
            [obj_x, obj_y, obj_z],
            [img_x, img_y]
        )
        
    >>> ij = poly_model.project_points(
            camera_parameters,
            obj_points
        )
    >>> ij
    
    >>> poly_model.project_to_z(
            camera_parameters,
            ij,
            z=obj_points[2]
        )
    
    >>> obj_points  
        
    """ 
    _check_parameters(cam_struct)
    
    dtype = cam_struct["dtype"]
    
    image_points = np.array(image_points, dtype=dtype)
    
    x = image_points[0]
    y = image_points[1]
    Z = np.array(z, dtype=dtype)
    
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
    
    obj_points = np.dot(
        polynomial_iw,
        cam_struct["poly_iw"]
    ).T
    
    return obj_points.astype(dtype, copy=False)