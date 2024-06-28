import numpy as np

from ._check_params import _check_parameters
from ._distortion import (_undistort_points_brown, _distort_points_brown,
                          _undistort_points_poly,  _distort_points_poly)
from .. import _cal_doc_utils


_all__ = [
    "project_points",
    "project_to_z"
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


@_cal_doc_utils.docfiller
def project_points(
    cam_struct: dict,
    object_points: np.ndarray,
    correct_distortion: bool = True
):
    """Project object points to image points.
    
    Project object, or real world points, to image points.
    
    Parameters
    ----------
    %(cam_struct)s
    %(object_points)s
    correct_distortion : bool
        If true, perform distortion correction.
        
    Returns
    -------
    %(x_img_coord)s
    %(y_img_coord)s
    
    Examples
    --------
    >>> import numpy as np
    >>> from importlib_resources import files
    >>> from openpiv.calibration import pinhole_model
    
    >>> path_to_calib = files('openpiv.data').joinpath('test7/D_Cal.csv')

    >>> obj_x, obj_y, obj_z, img_x, img_y = np.loadtxt(
        path_to_calib,
        unpack=True,
        skiprows=1,
        usecols=range(5), # get first 5 columns of data
        delimiter=','
    )

    >>> obj_points = np.array([obj_x[0:3], obj_y[0:3], obj_z[0:3]], dtype="float64")
    >>> img_points = np.array([img_x[0:3], img_y[0:3]], dtype="float64")

    >>> cam_params = pinhole_model.get_cam_params(
        'cam1', 
        [4512, 800]
    )

    >>> cam_params = pinhole_model.minimize_params(
            cam_params, 
            [obj_x, obj_y, obj_z],
            [img_x, img_y],
            correct_focal = True,
            correct_distortion = False,
            iterations=5
        )
    
    >>> cam_params = pinhole_model.minimize_params(
            cam_params, 
            [obj_x, obj_y, obj_z],
            [img_x, img_y],
            correct_focal = True,
            correct_distortion = True,
            iterations=5
        )

    >>> pinhole_model.project_points(
        cam_params,
        obj_points
    )
    array([[-44.33764399, -33.67518588, -22.97467733],
           [ 89.61102874, 211.88636408, 334.59805555]])

    >>> img_points
    array([[-44.33764398, -33.67518587, -22.97467733],
           [ 89.61102873, 211.8863641 , 334.5980555 ]])
    
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


@_cal_doc_utils.docfiller
def _get_inverse_vector(
    cam_struct: dict,
    image_points: np.ndarray
):
    """Get pixel to direction vector.
    
    Calculate a direction vector from a pixel. This vector can be used for
    forward projection along a ray using y + ar where y is the origin of
    the camera, a is the z plane along the ray, and r is the direction
    vector.
    
    Parameters
    ----------
    %(cam_struct)s
    %(image_points)s
        
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


@_cal_doc_utils.docfiller
def project_to_z(
    cam_struct: dict,
    image_points: np.ndarray,
    z: np.ndarray
):
    """Project image points to world points.
    
    Project image points to world points at specified z-plane using a
    closed form solution (when omiting distortion correction). This means
    under ideal circumstances with no distortion, the forward project
    coordinates would be accurate down to machine precision or numerical
    round off errors.
    
    Parameters
    ----------
    %(cam_struct)s
    %(image_points)s
    %(project_z)s
        
    Returns
    -------
    %(x_lab_coord)s
    %(y_lab_coord)s
    %(z_lab_coord)s
    
    Examples
    --------
    >>> import numpy as np
    >>> from importlib_resources import files
    >>> from openpiv.calibration import pinhole_model
    
    >>> path_to_calib = files('openpiv.data').joinpath('test7/D_Cal.csv')

    >>> obj_x, obj_y, obj_z, img_x, img_y = np.loadtxt(
        path_to_calib,
        unpack=True,
        skiprows=1,
        usecols=range(5), # get first 5 columns of data
        delimiter=','
    )

    >>> obj_points = np.array([obj_x[0:3], obj_y[0:3], obj_z[0:3]], dtype="float64")
    >>> img_points = np.array([img_x[0:3], img_y[0:3]], dtype="float64")

    >>> cam_params = pinhole_model.get_cam_params(
        'cam1', 
        [4512, 800]
    )

    >>> cam_params = pinhole_model.minimize_params(
            cam_params, 
            [obj_x, obj_y, obj_z],
            [img_x, img_y],
            correct_focal = True,
            correct_distortion = False,
            iterations=5
        )
    
    >>> cam_params = pinhole_model.minimize_params(
            cam_params, 
            [obj_x, obj_y, obj_z],
            [img_x, img_y],
            correct_focal = True,
            correct_distortion = True,
            iterations=5
        )
        
    >>> ij = pinhole_model.project_points(
            cam_params,
            obj_points
        )
    
    >>> ij
    array([[-44.33764399, -33.67518588, -22.97467733],
           [ 89.61102874, 211.88636408, 334.59805555]])
    
    >>> pinhole_model.project_to_z(
            cam_params,
            ij,
            z=obj_points[2]
        )
    array([[-105., -105., -105.],
           [ -15.,  -10.,   -5.],
           [ -10.,  -10.,  -10.]])
    
    >>> obj_points
    array([[-105., -105., -105.],
           [ -15.,  -10.,   -5.],
           [ -10.,  -10.,  -10.]])
        
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