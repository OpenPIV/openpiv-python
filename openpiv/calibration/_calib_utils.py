
from typing import Any

import numpy as np

from . import _cal_doc_utils

__all__ = [
    'get_image_mapping',
    'get_los_error',
    'get_reprojection_error',
    'get_rmse',
    'homogenize'
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
        msg = 'Residuals (error) array must be of shape (n) or (2, n), ' + \
             f'recieved shape {error.shape}'
        raise ValueError(msg)

    return np.sqrt(np.mean(square_error))



@_cal_doc_utils.docfiller
def get_reprojection_error(
    cam: Any,
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

    >>> obj_points = np.array([obj_x[0:3], obj_y[0:3], obj_z[0:3]], dtype=np.float64)
    >>> img_points = np.array([img_x[0:3], img_y[0:3]], dtype=np.float64)

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

    return get_rmse(error)



@_cal_doc_utils.docfiller
def get_los_error(
    cam: Any,
    z: float
):
    """Calculate camera LOS error.

    Calculate camera line of sight error at the selected volume depth.

    Parameters
    ----------
    cam : camera
        An instance of a camera object.
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

    >>> obj_points = np.array([obj_x[0:3], obj_y[0:3], obj_z[0:3]], dtype=np.float64)
    >>> img_points = np.array([img_x[0:3], img_y[0:3]], dtype=np.float64)

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
    X, Y, Z = cam.project_to_z(
        [x, y],
        Z
    )

    # project world points back to image coordinates
    res = cam.project_points(
        [X, Y, Z]
    )

    error = res - np.array([x, y])

    return get_rmse(error)



# This script was originally from Theo's polynomial calibration repository.
@_cal_doc_utils.docfiller
def get_image_mapping(
    cam,
    z: float = 0.0,
):
    """Get image Mapping.

    Get image mapping for rectifying 2D images. Fixed version:
    correct world-Y bounds (previously used world_x), true
    min/max extent scale (previously centered-abs heuristic) and no
    off-by-``Scale`` linspace offset. Extra legacy func args are
    accepted for backward compatibility and ignored (methods on
    ``cam`` are used).

    Parameters
    ----------
    cam : camera
        An instance of a camera object.

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

    >>> obj_points = np.array([obj_x[0:3], obj_y[0:3], obj_z[0:3]], dtype=np.float64)
    >>> img_points = np.array([img_x[0:3], img_y[0:3]], dtype=np.float64)

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
    ).astype(np.float64)

    x = image_grid[:, 1]
    y = image_grid[:, 0]

    # project to the requested sheet plane
    Z = np.zeros_like(x, dtype=np.float64) + float(z)

    # project image coordinates to world points
    world_x, world_y, _ = cam.project_to_z(
        [x, y],
        Z
    )

    world_x = np.asarray(world_x, dtype=np.float64).reshape(field_shape, order='C')
    world_y = np.asarray(world_y, dtype=np.float64).reshape(field_shape, order='C')

    # get border limits (fixed: Y bounds from world_y, not world_x)
    finite = np.isfinite(world_x) & np.isfinite(world_y)
    if not np.any(finite):
        raise ValueError('Back-projection produced no finite world points')

    min_X = float(np.min(world_x[finite]))
    max_X = float(np.max(world_x[finite]))
    min_Y = float(np.min(world_y[finite]))
    max_Y = float(np.max(world_y[finite]))

    # true extent-based scale (world units per rectified pixel)
    scale_X = (max_X - min_X) / cam.resolution[0]
    scale_Y = (max_Y - min_Y) / cam.resolution[1]
    Scale = float(min(scale_X, scale_Y))

    # rectified world grid spanning the full extent (fixed: no +Scale offset)
    X, Y = np.meshgrid(
        np.linspace(
            min_X,
            max_X,
            num=cam.resolution[0],
            endpoint=True
        ),
        np.linspace(
            min_Y,
            max_Y,
            num=cam.resolution[1],
            endpoint=True
        )
    )

    X = np.squeeze(X.reshape(-1, 1))
    Y = np.squeeze(Y.reshape(-1, 1))
    Z = np.zeros_like(X, dtype=np.float64) + float(z)

    # project world points to image coordinates
    mapped_grid = cam.project_points(
        [X, Y, Z]
    )

    mapped_grid_x = np.asarray(mapped_grid[0], dtype=np.float64).reshape(field_shape)
    mapped_grid_y = np.asarray(mapped_grid[1], dtype=np.float64).reshape(field_shape)

    return np.array([mapped_grid_x, mapped_grid_y]), Scale
