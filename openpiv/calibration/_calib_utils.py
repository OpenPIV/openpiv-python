import numpy as np
from typing import Tuple
from . import _cal_doc_utils


__all__ = [
    "homogenize",
    "get_rmse",
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
    a1 = np.ones(
        (1, points.shape[1]),
        dtype=points.dtype
    )
    
    return np.concatenate([
        points, 
        a1
    ])


def get_rmse(
    error: np.ndarray
):
    """Get root mean square error (RMSE).
    
    Calculate the root mean square error for statistical purposes.
    
    Parameters
    ----------
    error : np.ndarray
        The residuals between the predicted value and the actual value.
    
    Returns
    -------
    RMSE : float
        The RMSE of the error
    
    """
    
    if len(error.shape) == 2:
        square_error = np.sum(
            np.square(error),
            axis=0
        )

    elif len(error.shape) == 1:
        square_error = np.square(error)
        
    else:
        raise ValueError(
            "Residuals (error) array must be of shape (n) or (2, n), " + 
            f"recieved shape {error.shape}"
        )
        
    rmse = np.sqrt(np.mean(square_error))
    
    return rmse


@_cal_doc_utils.docfiller
def get_reprojection_error(
    cam: "camera",
    object_points: np.ndarray,
    image_points: np.ndarray
):
    """Calculate camera calibration error.
    
    Calculate the camera calibration error by projecting object points into image
    points and calculating the root mean square (RMS) error.
    
    Parameters
    ----------
    cam : camera
        An instance of a camera object.
    %(project_points_func)s
    %(object_points)s
    %(image_points)s
        
    Returns
    -------
    RMSE : float
        Root mean square (RMS) error of camera parameters.
        
    Examples
    --------
    >>> import numpy as np
    >>> from importlib_resources import files
    >>> from openpiv.calibration import dlt_model, calib_utils

    >>> path_to_calib = files('openpiv.data').joinpath('test7/D_Cal.csv')

    >>> obj_x, obj_y, obj_z, img_x, img_y = np.loadtxt(
        path_to_calib,
        unpack=True,
        skiprows=1,
        usecols=range(5),
        delimiter=','
    )

    >>> obj_points = np.array([obj_x[0:3], obj_y[0:3], obj_z[0:3]], dtype="float64")
    >>> img_points = np.array([img_x[0:3], img_y[0:3]], dtype="float64")

    >>> cam = dlt_model.camera(
        'cam1', 
        [4512, 800]
    )

    >>> cam.minimize_params(
        [obj_x, obj_y, obj_z],
        [img_x, img_y]
    )

    >>> calib_utils.get_reprojection_error(
        cam, 
        [obj_x, obj_y, obj_z],
        [img_x, img_y]
    )
    2.6181007833551034e-07
    
    """ 
    res = cam.project_points(
        object_points
    )
        
    error = res - image_points
    
    RMSE = get_rmse(error)
    
    return RMSE


@_cal_doc_utils.docfiller
def get_los_error(
    cam: "camera",
    z: float
):
    """Calculate camera LOS error.
    
    Calculate camera line of sight error at the selected volume depth.
    
    Parameters
    ----------
    %(cam)s
    %(project_to_z_func)s
    %(project_points_func)s
    %(project_z)s
    
    Returns
    -------
    RMSE : float
        Root mean square (RMS) error of camera parameters.
    
    Examples
    --------
    >>> import numpy as np
    >>> from importlib_resources import files
    >>> from openpiv.calibration import dlt_model, calib_utils

    >>> path_to_calib = files('openpiv.data').joinpath('test7/D_Cal.csv')

    >>> obj_x, obj_y, obj_z, img_x, img_y = np.loadtxt(
        path_to_calib,
        unpack=True,
        skiprows=1,
        usecols=range(5),
        delimiter=','
    )

    >>> obj_points = np.array([obj_x[0:3], obj_y[0:3], obj_z[0:3]], dtype="float64")
    >>> img_points = np.array([img_x[0:3], img_y[0:3]], dtype="float64")

    >>> cam = dlt_model.camera(
        'cam1', 
        [4512, 800]
    )

    >>> cam.minimize_params(
        [obj_x, obj_y, obj_z],
        [img_x, img_y]
    )
    
    >>> calib_utils.get_los_error(
            cam,
            z = 0
        )
    1.0097171287719555e-12
    
    """
    # create a meshgrid for every x and y pixel for back projection.
    py, px = np.meshgrid(
        np.arange(0, cam.resolution[1]),
        np.arange(0, cam.resolution[0]),
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
        cam,
        [x, y],
        Z
    )
    
    # project world points back to image coordinates
    res = project_points_func(
        cam,
        [X, Y, Z]
    )
    
    error = res - np.array([x, y])
    
    RMSE = get_rmse(error)
    
    return RMSE


# This script was originally from Theo's polynomial calibration repository.
@_cal_doc_utils.docfiller
def get_image_mapping(
    cam: dict,
    project_to_z_func: "function",
    project_points_func: "function"
):
    """Get image Mapping.
    
    Get image mapping for rectifying 2D images.
    
    Parameters
    ----------
    %(cam)s
    %(project_to_z_func)s
    %(project_points_func)s
    
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
    >>> from importlib_resources import files
    >>> from openpiv.calibration import dlt_model, calib_utils

    >>> path_to_calib = files('openpiv.data').joinpath('test7/D_Cal.csv')

    >>> obj_x, obj_y, obj_z, img_x, img_y = np.loadtxt(
        path_to_calib,
        unpack=True,
        skiprows=1,
        usecols=range(5),
        delimiter=','
    )

    >>> obj_points = np.array([obj_x[0:3], obj_y[0:3], obj_z[0:3]], dtype="float64")
    >>> img_points = np.array([img_x[0:3], img_y[0:3]], dtype="float64")

    >>> cam = dlt_model.camera(
        'cam1', 
        [4512, 800]
    )

    >>> cam.minimize_params(
        [obj_x, obj_y, obj_z],
        [img_x, img_y]
    )
    
    >>> mappings, scale = calib_utils.get_image_mapping(
            cam,
        )
        
    >>> mappings
    
    >>> scale
    
    """
    field_shape = (
        cam.resolution[1],
        cam.resolution[0]
    )
    
    # create a meshgrid for every x and y pixel for back projection.
    py, px = np.meshgrid(
        np.arange(0, cam.resolution[1]),
        np.arange(0, cam.resolution[0]),
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
    world_x, world_y, _ = cam.project_to_z(
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
            num=cam.resolution[0], 
            endpoint=True
        ),
        np.linspace(
            min_Y + Scale,
            max_Y, 
            num=cam.resolution[1], 
            endpoint=True
        )
    )
    
    X = np.squeeze(X.reshape(-1, 1))
    Y = np.squeeze(Y.reshape(-1, 1))
    
    # project world points to image coordinates
    mapped_grid = cam.project_points(
        [X, Y, Z]
    )
    
    mapped_grid_x = mapped_grid[0].reshape(field_shape)
    mapped_grid_y = mapped_grid[1].reshape(field_shape)
    
    return np.array([mapped_grid_x, mapped_grid_y]), Scale