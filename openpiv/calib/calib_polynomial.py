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
    poly_wi: np.ndarray=np.ones((2,19), dtype=float).T,
    poly_iw: np.ndarray=np.ones((3,19), dtype=float).T
    
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
    camera_struct : dict
        A dictionary structure of camera parameters.
    
    """    
    camera_struct = {}
    camera_struct["name"] = cam_name
    camera_struct["resolution"] = resolution
    camera_struct["poly_wi"] = poly_wi
    camera_struct["poly_iw"] = poly_iw
    
    _check_parameters(camera_struct)
    
    return camera_struct


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
    camera_struct : dict
        A dictionary structure of optimized camera parameters.
        
    """
    x = image_points[:, 0]
    y = image_points[:, 1]
    
    X = object_points[:, 0]
    Y = object_points[:, 1]
    Z = object_points[:, 2]

    polynomial_wi=np.array([X*0+1,
                            X,     Y,     Z, 
                            X*Y,   X*Z,   Y*Z,
                            X**2,  Y**2,  Z**2,
                            X**3,  X*X*Y, X*X*Z,
                            Y**3,  X*Y*Y, Y*Y*Z,
                            X*Z*Z, Y*Z*Z, X*Y*Z]).T

    coeff_wi, r_wi, rank_wi, s_wi=np.linalg.lstsq(
        polynomial_wi,
        image_points, 
        rcond=None
    )

    # ignore coefficients for image to world calibration for now
    cam_struct["poly_wi"] = coeff_wi
    
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
        Real world coordinates. The ndarray is structured like [X, Y, Z]`.
        
    Returns
    -------
    x : 1D np.ndarray
        Projected image x-coordinates.
    y : 1D np.ndarray
        Projected image y-coordinates.
    
    """ 
    _check_parameters(cam_struct)
    
    if not isinstance(object_points, np.ndarray):
        object_points = np.array(object_points).T
    
    X = object_points[:, 0]
    Y = object_points[:, 1]
    Z = object_points[:, 2]
    
    polynomial_wi=np.array([X*0+1,
                            X,     Y,     Z, 
                            X*Y,   X*Z,   Y*Z,
                            X**2,  Y**2,  Z**2,
                            X**3,  X*X*Y, X*X*Z,
                            Y**3,  X*Y*Y, Y*Y*Z,
                            X*Z*Z, Y*Z*Z, X*Y*Z]).T
    
    ij = np.dot(
        polynomial_wi,
        cam_struct["poly_wi"]
    )
    
    Xp = ij[:, 0]
    Yp = ij[:, 1]
    
    return np.array([Xp, Yp,]).T