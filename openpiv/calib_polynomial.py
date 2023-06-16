import numpy as np
from typing import Tuple


def _check_parameters(
    cam_struct: dict
):
    """Check camera parameters"""
    if type(cam_struct["name"]) != str:
        raise ValueError(
            "Camera name must be a string"
        )
    
    if len(cam_struct["resolution"]) != 2:
        raise ValueError(
            "Resolution must be a two element tuple"
        )
    
    if len(cam_struct["poly_wi"].shape) != 2:
        raise ValueError(
            "World to image polynomial coefficients must be 2 dimensional."
        )
        
    if cam_struct["poly_wi"].shape[0] != 19:
        raise ValueError(
            "World to image polynomial coefficients must be ordered in [x, y]'"
        )
    
    if cam_struct["poly_wi"].shape[1] != 2:
        raise ValueError(
            "There must be 19 coefficients in the world to image polynomial"
        )
    
    if len(cam_struct["poly_iw"].shape) != 2:
        raise ValueError(
            "Image to world polynomial coefficients must be 2 dimensional."
        )
        
    if cam_struct["poly_iw"].shape[0] != 19:
        raise ValueError(
            "Image to world polynomial coefficients must be ordered in [x, y]'"
        )
    
    if cam_struct["poly_iw"].shape[1] != 3:
        raise ValueError(
            "There must be 19 coefficients in the image to world polynomial"
        )


def generate_camera_params(
    cam_name: str,
    resolution: Tuple[int, int],
    poly_wi: np.ndarray=np.ones((2,19), dtype="float64").T,
    poly_iw: np.ndarray=np.ones((3,19), dtype="float64").T
    
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
    
    _check_parameters(cam_struct)
    
    return cam_struct


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
        A dictionary structure of camera parameters.
    object_points : np.ndarray
        A 2D np.ndarray containing [x, y, z] object points.
    image_points : np.ndarray
        A 2D np.ndarray containing [x, y] image points.
        
    Returns
    -------
    cam_struct : dict
        A dictionary structure of optimized camera parameters.
        
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
    
    >>> camera_parameters = calib_polynomial.minimize_polynomial(
            camera_parameters,
            [obj_x, obj_y, obj_z],
            [img_x, img_y]
        )
        
    >>> calib_utils.get_reprojection_error(
            camera_parameters, 
            calib_polynomial.project_points,
            [obj_x, obj_y, obj_z],
            [img_x, img_y]
        )
        
    """
    _check_parameters(cam_struct)
    
    object_points = np.array(object_points, dtype="float64")
    image_points = np.array(image_points, dtype="float64")
    
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

    polynomial_wi = np.array([X*0+1,
                              X,     Y,     Z, 
                              X*Y,   X*Z,   Y*Z,
                              X**2,  Y**2,  Z**2,
                              X**3,  X*X*Y, X*X*Z,
                              Y**3,  X*Y*Y, Y*Y*Z,
                              X*Z*Z, Y*Z*Z, X*Y*Z]).T
    
    # in the future, break this into three Z subvolumes to further reduce errors.
    polynomial_iw = np.array([x*0+1,
                              x,     y,     Z, 
                              x*y,   x*Z,   y*Z,
                              x**2,  y**2,  Z**2,
                              x**3,  x*x*y, x*x*Z,
                              y**3,  x*y*y, y*y*Z,
                              x*Z*Z, y*Z*Z, x*y*Z]).T
    

    # world to image (forward projection)
    coeff_wi, _, _, _ = np.linalg.lstsq(
        polynomial_wi,
        np.array(image_points, dtype="float64").T, 
        rcond=None
    )
    
    # image to world (back projection)
    coeff_iw, _, _, _ = np.linalg.lstsq(
        polynomial_iw,
        np.array(object_points, dtype="float64").T, 
        rcond=None
    )
    
    # SVD based solution to system of equations
#    coeff_wi = np.array(image_points, dtype="float64") @ np.linalg.pinv(polynomial_wi.T)
#    coeff_wi = coeff_wi.T
    
#    coeff_iw = np.array(object_points, dtype="float64") @ np.linalg.pinv(polynomial_iw.T)
#    coeff_iw = coeff_iw.T

    cam_struct["poly_wi"] = coeff_wi
    cam_struct["poly_iw"] = coeff_iw
    
    return cam_struct


def project_points(
    cam_struct: dict,
    object_points: np.ndarray
):
    """Project object points to image points.
    
    Project object, or real world points, to image points.
    
    Parameters
    ----------
    cam_struct : dict
        A dictionary structure of camera parameters.
    object_points : 2D np.ndarray
        Real world coordinates. The ndarray is structured like [X, Y, Z].
        
    Returns
    -------
    x : 1D np.ndarray
        Projected image x-coordinates.
    y : 1D np.ndarray
        Projected image y-coordinates.
        
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv import calib_utils, calib_polynomial
    >>> from openpiv.data.test5 import cal_points
    
    >>> obj_x, obj_y, obj_z, img_x, img_y, img_size_x, img_size_y = cal_points()
    
    >>> obj_points = np.array([obj_x[0:2], obj_y[0:2], obj_z[0:2]], dtype="float64")
    >>> img_points = np.array([img_x[0:2], img_y[0:2]], dtype="float64")
    
    >>> camera_parameters = calib_polynomial.generate_camera_params(
            name="cam1", 
            [img_size_x, img_size_y]
        )
    
    >>> camera_parameters = calib_polynomial.minimize_polynomial(
            camera_parameters,
            [obj_x, obj_y, obj_z],
            [img_x, img_y]
        )
        
    >>> calib_polynomial.project_points(
            camera_parameters,
            obj_points
        )
        
    >>> img_points
    
    """ 
    _check_parameters(cam_struct)
    
    object_points = np.array(object_points, dtype="float64")
    
    X = object_points[0]
    Y = object_points[1]
    Z = object_points[2]
    
    polynomial_wi = np.array([X*0+1,
                              X,     Y,     Z, 
                              X*Y,   X*Z,   Y*Z,
                              X**2,  Y**2,  Z**2,
                              X**3,  X*X*Y, X*X*Z,
                              Y**3,  X*Y*Y, Y*Y*Z,
                              X*Z*Z, Y*Z*Z, X*Y*Z]).T
    
    img_points = np.dot(
        polynomial_wi,
        cam_struct["poly_wi"]
    ).T
    
    return img_points.astype("float64", copy=False)


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
        A dictionary structure of camera parameters.
    image_points : 2D np.ndarray
        Image coordinates. The ndarray is structured like [x, y].
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
    >>> from openpiv import calib_utils, calib_polynomial
    >>> from openpiv.data.test5 import cal_points
    
    >>> obj_x, obj_y, obj_z, img_x, img_y, img_size_x, img_size_y = cal_points()
    
    >>> obj_points = np.array([obj_x[0:2], obj_y[0:2], obj_z[0:2]], dtype="float64")
    >>> img_points = np.array([img_x[0:2], img_y[0:2]], dtype="float64")
    
    >>> camera_parameters = calib_polynomial.generate_camera_params(
            name="cam1", 
            [img_size_x, img_size_y]
        )
    
    >>> camera_parameters = calib_polynomial.minimize_polynomial(
            camera_parameters,
            [obj_x, obj_y, obj_z],
            [img_x, img_y]
        )
        
    >>> ij = calib_polynomial.project_points(
            camera_parameters,
            obj_points
        )
    >>> ij
    
    >>> calib_polynomial.project_to_z(
            camera_parameters,
            ij,
            z=obj_points[2]
        )
    
    >>> obj_points  
        
    """ 
    _check_parameters(cam_struct)
    
    image_points = np.array(image_points, dtype="float64")
    
    x = image_points[0]
    y = image_points[1]
    Z = np.array(z, dtype="float64")
    
    polynomial_iw = np.array([x*0+1,
                              x,     y,     Z, 
                              x*y,   x*Z,   y*Z,
                              x**2,  y**2,  Z**2,
                              x**3,  x*x*y, x*x*Z,
                              y**3,  x*y*y, y*y*Z,
                              x*Z*Z, y*Z*Z, x*y*Z]).T
    
    obj_points = np.dot(
        polynomial_iw,
        cam_struct["poly_iw"]
    ).T
    
    return obj_points.astype("float64", copy=False)