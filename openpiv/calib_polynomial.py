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
                              x*Y,   x*Z,   y*Z,
                              x**2,  y**2,  Z**2,
                              x**3,  x*x*y, x*x*Z,
                              y**3,  x*y*y, y*y*Z,
                              x*Z*Z, y*Z*Z, x*y*Z]).T
    

    # world to image (forward projection)
    coeff_wi, _, _, _ = np.linalg.lstsq(
        polynomial_wi,
        np.array(image_points, dtype=float).T, 
        rcond=None
    )
    
    # image to world (back projection)
    coeff_iw, _, _, _ = np.linalg.lstsq(
        polynomial_iw,
        np.array(object_points, dtype=float).T, 
        rcond=None
    )

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
    
    """ 
    _check_parameters(cam_struct)
    
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
    
    ij = np.dot(
        polynomial_wi,
        cam_struct["poly_wi"]
    )
    
    Xp = ij[:, 0]
    Yp = ij[:, 1]
    
    return np.array([Xp, Yp])


def project_to_z(
    cam_struct: dict,
    image_points: np.ndarray,
    z: np.ndarray
):
    """Project object points to image points.
    
    Project object, or real world points, to image points.
    
    Parameters
    ----------
    cam_struct : dict
        A dictionary structure of camera parameters.
    image_points : 2D np.ndarray
        Image coordinates. The ndarray is structured like [x, y].
    z : int, np.ndarray
        An int or array specifying Z (depth) values to project to.
        
    Returns
    -------
    X : 1D np.ndarray
        Projected world x-coordinates.
    Y : 1D np.ndarray
        Projected world y-coordinates.
    Z : 1D np.ndarray
        Projected world z-coordinates.
        
    
    """ 
    _check_parameters(cam_struct)
    
    x = image_points[0]
    y = image_points[1]
    Z = z
    
    polynomial_iw = np.array([x*0+1,
                              x,     y,     Z, 
                              x*y,   x*Z,   y*Z,
                              x**2,  y**2,  Z**2,
                              x**3,  x*x*y, x*x*Z,
                              y**3,  x*y*y, y*y*Z,
                              x*Z*Z, y*Z*Z, x*y*Z]).T
    
    ijk = np.dot(
        polynomial_iw,
        cam_struct["poly_iw"]
    )
    
    Xp = ijk[:, 0]
    Yp = ijk[:, 1]
    Zp = ijk[:, 2]
    
    return np.array([Xp, Yp, Zp])


# This script was originally from Theo's polynomial calibration repository.
def get_image_mapping(
    cam_struct: dict,
):
    """Get image Mapping.
    
    Get image mapping for rectifying 2D images.
    
    Parameters
    ----------
    cam_struct : dict
        A dictionary structure of camera parameters.
        
    Returns
    -------
    X : 1D np.ndarray
        Projected world x-coordinates.
    Y : 1D np.ndarray
        Projected world y-coordinates.
    scale : float
        Image to world scale factor.
        
    Notes
    -----
    The Scale value is only applicable if the image and object points are 
    not normalized.
    
    """
    
    # create a meshgrid for every x and y pixel for back projection.
    py, px = np.meshgrid(
        np.arange(0, cam_struct["resolution"][1]),
        np.arange(0, cam_struct["resolution"][0]),
        indexing="ij"
    )
    
    image_grid = np.concatenate(
        [py.reshape(-1, 1), px.reshape(-1, 1)],
        axis=1
    )
    
    x = image_grid[:, 1]
    y = image_grid[:, 0]
    
    # We set Z to zero since there is no depth
    Z = x*0.
    
    # project image coordinates to world points
    world_x, world_y, _ = calib_polynomial.project_to_z(
        cam_struct,
        [x, y],
        Z
    )
    
    world_x = world_x.reshape(cam_struct["resolution"], order='C')
    world_y = world_y.reshape(cam_struct["resolution"], order='C')
    
    # get scale
    lower_bound_X = np.min(np.absolute(world_x[:, 0]))
    upper_bound_X = np.min(np.absolute(world_x[:, -1]))
    lower_bound_Y = np.min(np.absolute(world_x[0, :]))
    upper_bound_Y = np.min(np.absolute(world_x[-1, :]))
    
    scale_X = (lower_bound_X + upper_bound_X) / np.size(world_x, 1)
    scale_Y = (lower_bound_Y + upper_bound_Y) / np.size(world_x, 0)
    
    Scale = min(scale_X, scale_Y)
    
    # get border limits
    min_X = np.min(world_x)
    max_X = np.max(world_x)
    min_Y = np.min(world_y)
    max_Y = np.max(world_y)
    
    # create a meshgrid for every x and y point for forward projection.
    X, Y = np.meshgrid(
        np.linspace(
            min_X + scale_X,
            max_X, 
            num=cam_struct["resolution"][0], 
            endpoint=True
        ),
        np.linspace(
            min_Y + scale_Y,
            max_Y, 
            num=cam_struct["resolution"][1], 
            endpoint=True
        )
    )
    
    X = np.squeeze(X.reshape(-1, 1))
    Y = np.squeeze(Y.reshape(-1, 1))
    
    # project world points to image coordinates
    mapped_grid = calib_polynomial.project_points(
        cam_struct,
        [X, Y, Z]
    )
    
    mapped_grid_x = mapped_grid[0].reshape(cam_struct["resolution"])
    mapped_grid_y = mapped_grid[1].reshape(cam_struct["resolution"])
    
    return np.array([mapped_grid_x, mapped_grid_y]), Scale