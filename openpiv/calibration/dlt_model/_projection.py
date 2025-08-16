import numpy as np

from .._calib_utils import homogenize
from .. import _cal_doc_utils


@_cal_doc_utils.docfiller
def _project_points(
    self,
    object_points: np.ndarray
):
    """Project lab coordinates to image points.
        
    Parameters
    ----------
    camera : class
        An instance of a DLT camera.
    %(object_points)s
        
    Returns
    -------
    %(x_img_coord)s
    %(y_img_coord)s
        
    Examples
    --------
    >>> import numpy as np
    >>> from importlib_resources import files
    >>> from openpiv.calibration import dlt_model
    
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

    >>> cam = dlt_model.camera(
        'cam1', 
        [4512, 800]
    )

    >>> cam.minimize_params(
        [obj_x, obj_y, obj_z],
        [img_x, img_y]
    )

    >>> cam.project_points(
        obj_points
    )
    array([[-44.33764399, -33.67518588, -22.97467733],
           [ 89.61102874, 211.88636408, 334.59805555]])

    >>> img_points
    array([[-44.33764398, -33.67518587, -22.97467733],
           [ 89.61102873, 211.8863641 , 334.5980555 ]])
    
    """ 
    self._check_parameters()
    
    dtype = self.dtype
    H = self.coeffs
    
    object_points = np.array(object_points, dtype=dtype)
    
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


@_cal_doc_utils.docfiller
def _get_inverse_vector(
    self,
    image_points: np.ndarray
):
    """Get pixel to direction vector.
    
    Calculate a direction vector from a pixel. This vector can be used for
    forward projection along a ray using y + ar where y is the origin of
    the camera, a is the z plane along the ray, and r is the direction
    vector.
    
    Parameters
    ----------
    %(image_points)s
        
    Returns
    -------
    tx, ty, tz : 1D np.ndarray
        Direction vector for each respective axis.
    dx, dy, dz : 1D np.ndarray
        The camera center for each respective axis.
    
    Notes
    -----
    The direction vector is not normalized.
    
    """    
    dtype = self.dtype
    p = self.coeffs
    
    if p.shape != (3, 4):
        raise ValueError(
            "DLT coefficients must be of shape (3, 4); recieved shape " +
            f"{p.shape}"
        )
        
    m = p[:, :3]
    t_un = p[:, 3]
    m_inv = np.linalg.inv(m)

    # direction vector
    r = np.dot(m_inv, homogenize(image_points)).astype(dtype, copy=False)
    
    # camera center/translation
    t = np.dot(-m_inv, t_un).astype(dtype, copy=False)
    
    return t, r


@_cal_doc_utils.docfiller
def _project_to_z(
    self,
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
    >>> from openpiv.calibration import dlt_model
    
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

    >>> cam = dlt_model.camera(
        'cam1', 
        [4512, 800]
    )

    >>> cam.minimize_params(
        [obj_x, obj_y, obj_z],
        [img_x, img_y]
    )
        
    >>> ij = cam.project_points(
            obj_points
        )
    
    >>> ij
    array([[-44.33764399, -33.67518588, -22.97467733],
           [ 89.61102874, 211.88636408, 334.59805555]])
    
    >>> cam.project_to_z(
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
    self._check_parameters()
    
    dtype = self.dtype
        
    image_points = np.array(image_points, dtype=dtype)
    
    t, r = self._get_inverse_vector(
        image_points
    )
    
    tx, ty, tz = t
    dx, dy, dz = r
        
    a = (z - tz) / dz
    
    X = a*dx + tx
    Y = a*dy + ty

    return np.array([X, Y, np.zeros_like(X) + z], dtype=dtype)