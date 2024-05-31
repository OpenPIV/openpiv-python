import numpy as np
from typing import Tuple


__all__ = [
    "homogenize",
    "get_reprojection_error",
    "get_los_error",
    "get_image_mapping"
]
    

def homogenize(
    points: np.ndarray
):
    """Homogenize points.
    
    Homogenize points for further processing and correspondence matching.
    Points are homogenized as such:
    [0, 1, 2, 3, ...]
    [0, 1, 2, 3, ...]
    [1, 1, 1, 1, ...] <-- Appended ones
    
    Parameters
    ----------
    points : np.ndarray
        Points to which ones will be appended to the end of. The array
        shape should be [M, N] where M in the number of dimensions and N is
        the number of points.
        
    Returns
    -------
    points : np.ndarray
        Homogenized points of shape (M+1, N].
    
    """
    a1 = np.ones((1, points.shape[1]), dtype = points.dtype)
    
    return np.concatenate([
        points, 
        a1
    ])


def get_reprojection_error(
    cam_struct: dict,
    proj_func: "function",
    object_points: np.ndarray,
    image_points: np.ndarray
):
    """Calculate camera calibration error.
    
    Calculate the camera calibration error by projecting object points into image
    points and calculating the root mean square (RMS) error.
    
    Parameters
    ----------
    cam_struct : dict
        A dictionary structure of camera parameters.
    proj_func : function
        Projection function with the following signature:
        res = func(cam_struct, object_points).
    object_points: 2D np.ndarray
        A numpy array containing [X, Y, Z] object points.
    image_points: 2D np.ndarray
        A numpy array containing [x, y] image points.
        
    Returns
    -------
    RMSE : float
        Root mean square (RMS) error of camera parameters.
        
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv.calibration import calib_utils, poly_model
    >>> from openpiv.data.test5 import cal_points
    
    >>> obj_x, obj_y, obj_z, img_x, img_y, img_size_x, img_size_y = cal_points()
    
    >>> camera_parameters = poly_model.get_cam_params(
            name="cam1", 
            [img_size_x, img_size_y]
        )
        
    >>> camera_parameters = poly_model.minimize_params(
            camera_parameters,
            np.array([obj_x, obj_y, obj_z]),
            np.array([img_x, img_y])
        )
    
    >>> calib_utils.get_reprojection_error(
            camera_parameters2, 
            poly_model.project_points,
            [obj_x, obj_y, obj_z],
            [img_x, img_y]
        )
    
    """ 
    res = proj_func(
        cam_struct,
        object_points
    )
        
    error = res - image_points
    
    RMSE = np.sqrt(
        np.mean(
            np.sum(
                np.square(error),
                axis=0
            )
        )
    )
    
    return RMSE


def get_los_error(
    cam_struct,
    project_to_z_func: "function",
    project_points_func: "function",
    z
):
    """Calculate camera LOS error.
    
    Calculate camera line of sight error at the selected volume depth.
    
    Parameters
    ----------
    cam_struct : dict
        A dictionary structure of camera parameters.
    project_to_z_func : function
        Projection function with the following signiture:
        res = func(cam_struct, image_points, Z).
    project_points_func : function
        Projection function with the following signiture:
        res = func(cam_struct, object_points).
    z : float
        A float specifying the Z (depth) value to project to.
    
    Returns
    -------
    RMSE : float
        Root mean square (RMS) error of camera parameters.
    
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv.calibration import calib_utils, poly_model
    >>> from openpiv.data.test5 import cal_points
    
    >>> obj_x, obj_y, obj_z, img_x, img_y, img_size_x, img_size_y = cal_points()
    
    >>> camera_parameters = poly_model.get_cam_params(
            name="cam1", 
            [img_size_x, img_size_y]
        )
        
    >>> camera_parameters = poly_model.minimize_params(
            camera_parameters,
            np.array([obj_x, obj_y, obj_z]),
            np.array([img_x, img_y])
        )
    
    >>> calib_utils.get_los_error(
            camera_parameters2,
            poly_model.project_to_z,
            poly_model.project_points,
            z = 0
        )
        
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
    
    # get depth
    Z = np.zeros_like(x) + z
    
    # project image coordinates to world points
    X, Y, Z = project_to_z_func(
        cam_struct,
        [x, y],
        Z
    )
    
    # project world points back to image coordinates
    res = project_points_func(
        cam_struct,
        [X, Y, Z]
    )
    
    error = res - np.array([x, y])
    
    RMSE = np.mean(
        np.sqrt(
            np.sum(
                np.square(error),
                axis=0
            )
        )
    )
    
    return RMSE


# This script was originally from Theo's polynomial calibration repository.
def get_image_mapping(
    cam_struct: dict,
    project_to_z_func: "function",
    project_points_func: "function"
):
    """Get image Mapping.
    
    Get image mapping for rectifying 2D images.
    
    Parameters
    ----------
    cam_struct : dict
        A dictionary structure of camera parameters.
    project_to_z_func : function
        Projection function with the following signature:
        res = func(cam_struct, image_points, Z).
    project_points_func : function
        Projection function with the following signature:
        res = func(cam_struct, object_points).
    
    Returns
    -------
    x : 2D np.ndarray
        Mappings for x-coordinates.
    y : 2D np.ndarray
        Mappings for y-coordinates.
    scale : float
        Image to world scale factor.
        
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv import calib_utils, calib_polynomial
    >>> from openpiv.data.test5 import cal_points
    
    >>> obj_x, obj_y, obj_z, img_x, img_y, img_size_x, img_size_y = cal_points()
    
    >>> camera_parameters = poly_model.get_cam_params(
            name="cam1", 
            [img_size_x, img_size_y]
        )
        
    >>> camera_parameters = poly_model.minimize_params(
            camera_parameters,
            np.array([obj_x, obj_y, obj_z]),
            np.array([img_x, img_y])
        )
    
    >>> mappings, scale = calib_utils.get_image_mapping(
            camera_parameters,
            poly_model.project_to_z,
            poly_model.project_points
        )
        
    >>> mappings
    
    >>> scale
    
    """
    field_shape = (
        cam_struct["resolution"][1],
        cam_struct["resolution"][0]
    )
    
    # create a meshgrid for every x and y pixel for back projection.
    py, px = np.meshgrid(
        np.arange(0, cam_struct["resolution"][1]),
        np.arange(0, cam_struct["resolution"][0]),
        indexing="ij"
    )
    
    image_grid = np.concatenate(
        [py.reshape(-1, 1), px.reshape(-1, 1)],
        axis=-1
    ).astype("float64")
    
    x = image_grid[:, 1]
    y = image_grid[:, 0]
    
    # We set Z to zero since there is no depth
    Z = np.zeros_like(x)
    
    # project image coordinates to world points
    world_x, world_y, _ = project_to_z_func(
        cam_struct,
        [x, y],
        Z
    )
    
    world_x = world_x.reshape(field_shape, order='C')
    world_y = world_y.reshape(field_shape, order='C')
    
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
            min_X + Scale,
            max_X, 
            num=cam_struct["resolution"][0], 
            endpoint=True
        ),
        np.linspace(
            min_Y + Scale,
            max_Y, 
            num=cam_struct["resolution"][1], 
            endpoint=True
        )
    )
    
    X = np.squeeze(X.reshape(-1, 1))
    Y = np.squeeze(Y.reshape(-1, 1))
    
    # project world points to image coordinates
    mapped_grid = project_points_func(
        cam_struct,
        [X, Y, Z]
    )
    
    mapped_grid_x = mapped_grid[0].reshape(field_shape)
    mapped_grid_y = mapped_grid[1].reshape(field_shape)
    
    return np.array([mapped_grid_x, mapped_grid_y]), Scale