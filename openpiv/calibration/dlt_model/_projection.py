import numpy as np

from ._check_params import _check_parameters
from .._calib_utils import homogenize
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
    """Project lab coordinates to image points.
        
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
    >>> from openpiv import calib_utils, dlt_model
    >>> from openpiv.data.test5 import cal_points
    
    >>> obj_x, obj_y, obj_z, img_x, img_y, img_size_x, img_size_y = cal_points()
    
    >>> obj_points = np.array([obj_x[0:2], obj_y[0:2], obj_z[0:2]], dtype="float64")
    >>> img_points = np.array([img_x[0:2], img_y[0:2]], dtype="float64")
    
    >>> camera_parameters = dlt_model.get_cam_params(
            name="cam1", 
            [img_size_x, img_size_y]
        )
    
    >>> camera_parameters = dlt_model.minimize_params(
            camera_parameters,
            [obj_x, obj_y, obj_z],
            [img_x, img_y]
        )
        
    >>> dlt_model.project_points(
            camera_parameters,
            obj_points
        )
        
    >>> img_points
    
    """ 
    _check_parameters(cam_struct)
    
    dtype = cam_struct["dtype"]
    
    object_points = np.array(object_points, dtype=dtype)
        
    H = cam_struct["coefficients"]
    
    ndim = H.shape[1] - 1
    
    if ndim == 2:        
        object_points = object_points[:2, :]
    else:
        object_points = object_points[:3, :]
        
    # compute RMSE error
    xy = np.dot(
        H, 
        homogenize(object_points)
    )
    
    xy /= xy[2, :]
    img_points = xy[:2, :]
    
    return img_points.astype(dtype, copy=False)


@docstring_decorator(doc_cam_struct, doc_img_coords)
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
    cam_struct : dict
        {0}
    image_points : 2D np.ndarray
        {1}
        
    Returns
    -------
    tx, ty, tz : 1D np.ndarray
        Direction vector for each respective axis.
    dx, dy, dz : 1D np.ndarray
        The camera center for each respective axis.
    
    Notes
    -----
    The direction vector is not normalized.
    
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv.calibration import calib_utils, dlt_model
    >>> from openpiv.data.test5 import cal_points
    
    >>> obj_x, obj_y, obj_z, img_x, img_y, img_size_x, img_size_y = cal_points()
    
    >>> obj_points = np.array([obj_x[0:2], obj_y[0:2], obj_z[0:2]], dtype="float64")
    >>> img_points = np.array([img_x[0:2], img_y[0:2]], dtype="float64")
    
    >>> camera_parameters = dlt_model.get_cam_params(
            name="cam1", 
            [img_size_x, img_size_y]
        )
    
    >>> camera_parameters = dlt_model.minimize_params(
            camera_parameters, 
            [obj_x, obj_y, obj_z],
            [img_x, img_y]
        )
        
    >>> t, r = dlt_model._get_direction_vector(
            camera_parameters,
            img_points
        )
    >>> t
    >>> r
    
    """    
    dtype = cam_struct["dtype"]
    p = cam_struct["coefficients"]
    
    assert(p.shape == (3, 4))
        
    m = p[:, :3]
    t_un = p[:, 3]
    m_inv = np.linalg.inv(m)

    # direction vector
    r = np.dot(m_inv, homogenize(image_points)).astype(dtype, copy=False)
    
    # camera center/translation
    t = np.dot(-m_inv, t_un).astype(dtype, copy=False)
    
    return t, r


@docstring_decorator(doc_cam_struct, doc_img_coords)
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
    >>> from openpiv.calibration import calib_utils, dlt_model
    >>> from openpiv.data.test5 import cal_points
    
    >>> obj_x, obj_y, obj_z, img_x, img_y, img_size_x, img_size_y = cal_points()
    
    >>> obj_points = np.array([obj_x[0:2], obj_y[0:2], obj_z[0:2]], dtype="float64")
    >>> img_points = np.array([img_x[0:2], img_y[0:2]], dtype="float64")
    
    >>> camera_parameters = dlt_model.get_cam_params(
            name="cam1", 
            [img_size_x, img_size_y]
        )
    
    >>> camera_parameters = dlt_model.minimize_params(
            camera_parameters, 
            [obj_x, obj_y, obj_z],
            [img_x, img_y]
        )
        
    >>> ij = dlt_model.project_points(
            camera_parameters,
            obj_points
        )
    >>> ij
    
    >>> dlt_model.project_to_z(
            camera_parameters,
            ij,
            z=obj_points[2]
        )
    
    >>> obj_points
        
    """
    dtype = cam_struct["dtype"]
        
    image_points = np.array(image_points, dtype=dtype)
    
    t, r = _get_inverse_vector(
        cam_struct,
        image_points
    )
    
    tx, ty, tz = t
    dx, dy, dz = r
        
    a = (z - tz) / dz
    
    X = a*dx + tx
    Y = a*dy + ty

    return np.array([X, Y, np.zeros_like(X) + z], dtype=dtype)