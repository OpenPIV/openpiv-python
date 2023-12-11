import numpy as np

from ._check_params import _check_parameters
from ._distortion import *


_all__ = [
    "project_points",
    "project_to_z",
    "_normalize_world_points",
    "_normalize_image_points",
    "_get_inverse_vector"
]


def _normalize_world_points(
    cam_struct: dict,
    object_points: np.ndarray
):
    R = cam_struct["rotation"]
    T = cam_struct["translation"]
    dtype = cam_struct["dtype"]

    # transformation to camera coordinates
    Wc = np.dot(
        R.T,
        object_points
    ) - np.dot(R.T, T[:, np.newaxis])
        
    # the camera coordinates
    Wc_x = Wc[0, :]
    Wc_y = Wc[1, :]
    Wc_h = Wc[2, :]
        
    # normalize coordinates
    Wn_x = Wc_x / Wc_h
    Wn_y = Wc_y / Wc_h 
    
    return np.array([Wn_x, Wn_y], dtype=dtype)


def _normalize_image_points(
    cam_struct: dict,
    image_points: np.ndarray
):
    fx, fy = cam_struct["focal"]
    cx, cy = cam_struct["principal"]
    dtype = cam_struct["dtype"]
    
    x, y = image_points
    
    # normalize image coordinates
    Wn_x = (x - cx) / fx
    Wn_y = (y - cy) / fy
    
    return np.array([Wn_x, Wn_y], dtype=dtype)


def project_points(
    cam_struct: dict,
    object_points: np.ndarray,
    correct_distortion: bool = True
):
    """Project object points to image points.
    
    Project object, or real world points, to image points.
    
    Parameters
    ----------
    cam_struct : dict
        A dictionary structure of camera parameters.
    object_points : 2D np.ndarray
        Real world coordinates. The ndarray is structured like [X, Y, Z].
    correct_distortion : bool
        If true, perform distortion correction.
        
    Returns
    -------
    x : 1D np.ndarray
        Projected image x-coordinates.
    y : 1D np.ndarray
        Projected image y-coordinates.
    
    Notes
    -----
    World coordinates are projected to pixel coordinates using the following:
    
    |x_c|   |r11 r21 r31|^-1  |X|   |r11 r21 r31|^-1  |tx|
    |y_c| = |r12 r22 r32|  .  |Y| - |r12 r22 r32|  .  |ty|
    | h |   |r13 r23 r33|     |Z|   |r13 r23 r33|     |tz|
    
    x_nd = x_c / h
    y_nd = y_c / h
    
    |x_n|                        |x_nd|
    |y_n| = undistort_points --> |y_nd|
    | 1 |                        | 1  |
    
    |x|   |fx 0  cx|   |x_n|
    |y| = |0  fy cy| . |y_n|
    |1|   |0  0  1 |   | 1 |
    
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv import calib_utils, calib_pinhole
    >>> from openpiv.data.test5 import cal_points
    
    >>> obj_x, obj_y, obj_z, img_x, img_y, img_size_x, img_size_y = cal_points()
    
    >>> obj_points = np.array([obj_x[0:2], obj_y[0:2], obj_z[0:2]], dtype="float64")
    >>> img_points = np.array([img_x[0:2], img_y[0:2]], dtype="float64")
    
    >>> camera_parameters = calib_pinhole.generate_camera_params(
            name="cam1", 
            [img_size_x, img_size_y]
        )
    
    >>> camera_parameters = calib_pinhole.minimize_camera_params(
            camera_parameters, 
            [obj_x, obj_y, obj_z],
            [img_x, img_y],
            correct_focal = True,
            correct_distortion = False,
            iterations=5
        )
    
    >>> camera_parameters = calib_pinhole.minimize_camera_params(
            camera_parameters, 
            [obj_x, obj_y, obj_z],
            [img_x, img_y],
            correct_focal = True,
            correct_distortion = True,
            iterations=5
        )
        
    >>> ij = calib_pinhole.project_points(
            camera_parameters,
            obj_points
        )
    >>> ij
    
    >>> img_points
    
    """ 
    _check_parameters(cam_struct)    
    
    fx, fy = cam_struct["focal"]
    cx, cy = cam_struct["principal"]
    dtype = cam_struct["dtype"]
    
    object_points = np.array(object_points, dtype=dtype)
    
    Wn_x, Wn_y = _normalize_world_points(
        cam_struct,
        object_points
    )
    
    if correct_distortion == True:
        if cam_struct["distortion_model"].lower() == "brown":
            Wn_x, Wn_y = _undistort_points_brown(
                cam_struct,
                Wn_x,
                Wn_y
            )
        else:
            Wn_x, Wn_y = _undistort_points_poly(
                cam_struct,
                Wn_x,
                Wn_y
            )
    
    # rescale coordinates
    x = Wn_x * fx + cx
    y = Wn_y * fy + cy
    
    return np.array([x, y], dtype=dtype)
    

def _get_inverse_vector(
    cam_struct: dict,
    image_points: np.ndarray
):
    """Get pixel to direction vector.
    
    Calculate a direction vector from a pixel. This vector can be used for
    forward projection along a ray using y + ar where y is the origin of
    the camera, a is the z plane along the ray, and r is the direction vector.
    
    Parameters
    ----------
    cam_struct : dict
        A dictionary structure of camera parameters.
    image_points : 2D np.ndarray
        Image coordinates. The ndarray is structured like [x, y].
        
    Returns
    -------
    dx : 1D np.ndarray
        Direction vector for x-axis.
    dy : 1D np.ndarray
        Direction vector for y-axis.
    dz : 1D np.ndarray
        Direction vector for z-axis.
    
    Notes
    -----
    The direction vector is not normalized.
    
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv import calib_utils, calib_pinhole
    >>> from openpiv.data.test5 import cal_points
    
    >>> obj_x, obj_y, obj_z, img_x, img_y, img_size_x, img_size_y = cal_points()
    
    >>> obj_points = np.array([obj_x[0:2], obj_y[0:2], obj_z[0:2]], dtype="float64")
    >>> img_points = np.array([img_x[0:2], img_y[0:2]], dtype="float64")
    
    >>> camera_parameters = calib_pinhole.generate_camera_params(
            name="cam1", 
            [img_size_x, img_size_y]
        )
    
    >>> camera_parameters = calib_pinhole.minimize_camera_params(
            camera_parameters, 
            [obj_x, obj_y, obj_z],
            [img_x, img_y],
            correct_focal = True,
            correct_distortion = False,
            iterations=5
        )
    
    >>> camera_parameters = calib_pinhole.minimize_camera_params(
            camera_parameters, 
            [obj_x, obj_y, obj_z],
            [img_x, img_y],
            correct_focal = True,
            correct_distortion = True,
            iterations=5
        )
        
    >>> ij = calib_pinhole._get_direction_vector(
            camera_parameters,
            img_points
        )
    >>> ij
    
    """    
    R = cam_struct["rotation"]
    dtype = cam_struct["dtype"]
    
    image_points = np.array(image_points, dtype=dtype)
    
    Wn_x, Wn_y = _normalize_image_points(
        cam_struct,
        image_points
    )
    
    if cam_struct["distortion_model"].lower() == "brown":
        Wn_x, Wn_y = _distort_points_brown(
            cam_struct,
            Wn_x,
            Wn_y
        )
    else:
        Wn_x, Wn_y = _distort_points_poly(
            cam_struct,
            Wn_x,
            Wn_y
        )
        
    # inverse rotation
    dx, dy, dz = np.dot(
        R,
        [Wn_x, Wn_y, np.ones_like(Wn_x)]
    )
    
    return np.array([dx, dy, dz], dtype=dtype)

    
def project_to_z(
    cam_struct: dict,
    image_points: np.ndarray,
    z
):
    """Project image points to world points.
    
    Project image points to world points at specified z-plane using a closed
    form solution (when omiting distortion correction). This means under ideal
    circumstances with no distortion, the forward project coordinates would be
    accurate down to machine precision or numerical round off errors.
    
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
    
    Notes
    -----
    Pixel coordinates are projected to world coordinates using the following:
    
    |x_n|   |fx 0  cx|^-1  |x|
    |y_n| = |0  fy cy|   . |y|
    | 1 |   |0  0  1 |     |1|
    
    |x_nd|                      |x_n|
    |y_nd| = distort_points --> |y_n|
    | 1  |                      | 1 |
    
    |dx|   |r11 r21 r31|   |x_nd|
    |dy| = |r12 r22 r32| . |y_nd|
    |dz|   |r13 r23 r33|   | 1  |
    
    a = ((z - tz) / dz
    
    X = a*dx + tx
    Y = a*dy + ty
    Z = np.zeros_like(X) + z
    
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv import calib_utils, calib_pinhole
    >>> from openpiv.data.test5 import cal_points
    
    >>> obj_x, obj_y, obj_z, img_x, img_y, img_size_x, img_size_y = cal_points()
    
    >>> obj_points = np.array([obj_x[0:2], obj_y[0:2], obj_z[0:2]], dtype="float64")
    >>> img_points = np.array([img_x[0:2], img_y[0:2]], dtype="float64")
    
    >>> camera_parameters = calib_pinhole.generate_camera_params(
            name="cam1", 
            [img_size_x, img_size_y]
        )
    
    >>> camera_parameters = calib_pinhole.minimize_camera_params(
            camera_parameters, 
            [obj_x, obj_y, obj_z],
            [img_x, img_y],
            correct_focal = True,
            correct_distortion = False,
            iterations=5
        )
    
    >>> camera_parameters = calib_pinhole.minimize_camera_params(
            camera_parameters, 
            [obj_x, obj_y, obj_z],
            [img_x, img_y],
            correct_focal = True,
            correct_distortion = True,
            iterations=5
        )
        
    >>> ij = calib_pinhole.project_points(
            camera_parameters,
            obj_points
        )
    >>> ij
    
    >>> calib_pinhole.project_to_z(
            camera_parameters,
            ij,
            z=obj_points[2]
        )
    
    >>> obj_points
        
    """
    _check_parameters(cam_struct) 
    
    dtype = cam_struct["dtype"]
    
    dx, dy, dz = _get_inverse_vector(
        cam_struct,
        image_points
    )
    
    tx, ty, tz = cam_struct["translation"]
    
    a = (z - tz) / dz
    
    X = a*dx + tx
    Y = a*dy + ty

    return np.array([X, Y, np.zeros_like(X) + z], dtype=dtype)