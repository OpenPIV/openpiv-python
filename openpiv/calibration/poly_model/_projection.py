import numpy as np

from .. import _cal_doc_utils


@_cal_doc_utils.docfiller
def _project_points(
    self,
    object_points: np.ndarray
):
    """Project object points to image points.
    
    Project object, or real world points, to image points.
    
    Parameters
    ----------
    %(object_points)s
        
    Returns
    -------
    %(x_img_coord)s
    %(y_img_coord)s
        
    Examples
    --------
    >>> import numpy as np
    >>> from importlib_resources import files
    >>> from openpiv.calibration import poly_model
    
    >>> path_to_calib = files('openpiv.data').joinpath('test7/D_Cal.csv')

    >>> obj_x, obj_y, obj_z, img_x, img_y = np.loadtxt(
        path_to_calib,
        unpack=True,
        skiprows=1,
        usecols=range(5), # get first 5 columns of data
        delimiter=','
    )

    >>> obj_points = np.array([obj_x[0:3], obj_y[0:3], obj_z[0:3]], dtype=np.float64)
    >>> img_points = np.array([img_x[0:3], img_y[0:3]], dtype=np.float64)

    >>> cam = poly_model.camera(
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
    array([[-44.24281474, -33.56231972, -22.84229244],
           [ 89.63444964, 211.90372246, 334.60601499]])
        
    >>> img_points
    array([[-44.33764398, -33.67518587, -22.97467733],
           [ 89.61102873, 211.8863641 , 334.5980555 ]])
    
    """ 
    self._check_parameters()
    
    dtype = self.dtype
    
    object_points = np.array(object_points, dtype=dtype)
    
    X = object_points[0]
    Y = object_points[1]
    Z = object_points[2]
    
    polynomial_wi = np.array(
        [
            np.ones_like(X),
            X,     Y,     Z, 
            X*Y,   X*Z,   Y*Z,
            X**2,  Y**2,  Z**2,
            X**3,  X*X*Y, X*X*Z,
            Y**3,  X*Y*Y, Y*Y*Z,
            X*Z*Z, Y*Z*Z, X*Y*Z
        ],
        dtype=dtype
    ).T
    
    return np.dot(
        polynomial_wi,
        self.poly_wi
    ).T


@_cal_doc_utils.docfiller
def _project_to_z(
    self,
    image_points: np.ndarray,
    z: float
):
    """Project image points to world points.
    
    Project image points to world points at specified z-plane.
    
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
    >>> from openpiv.calibration import poly_model
    
    >>> path_to_calib = files('openpiv.data').joinpath('test7/D_Cal.csv')

    >>> obj_x, obj_y, obj_z, img_x, img_y = np.loadtxt(
        path_to_calib,
        unpack=True,
        skiprows=1,
        usecols=range(5), # get first 5 columns of data
        delimiter=','
    )

    >>> obj_points = np.array([obj_x[0:3], obj_y[0:3], obj_z[0:3]], dtype=np.float64)
    >>> img_points = np.array([img_x[0:3], img_y[0:3]], dtype=np.float64)

    >>> cam = poly_model.camera(
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
    array([[-44.24281474, -33.56231972, -22.84229244],
           [ 89.63444964, 211.90372246, 334.60601499]])
    
    >>> cam.project_to_z(
            ij,
            z=obj_points[2]
        )
    array([[-105.0088358 , -105.00967895, -105.01057378],
           [ -15.0022918 ,  -10.00166811,   -5.00092353],
           [ -10.00000004,  -10.00000003,  -10.00000002]])
    
    >>> obj_points
    array([[-105., -105., -105.],
           [ -15.,  -10.,   -5.],
           [ -10.,  -10.,  -10.]]) 
        
    """ 
    self._check_parameters()
    
    dtype = self.dtype
    
    image_points = np.array(image_points, dtype=dtype)
    
    x = image_points[0]
    y = image_points[1]
    
    if isinstance(z, np.ndarray):
        Z = np.array(z, dtype=dtype)
    elif isinstance(z, (float, int)):
        Z = np.zeros_like(x) + z
    else:
        msg = 'Unsupported value entered for `z`. `z` must be either a scalar ' +\
              'or numpy ndarray'
        raise ValueError(msg)
    
    t, r = self._get_inverse_vector(image_points)
    a = (Z - t[2]) / r[2]

    X = a*r[0] + t[0]
    Y = a*r[1] + t[1]

    return np.array([X, Y, Z], dtype=dtype)


def _get_inverse_vector(
    self,
    image_points: np.ndarray
):
    """Return per-pixel origins and direction vectors, both shaped [3, N].

    Origins lie on Z=0. Directions are not normalized; their Z component
    is one. Coefficients use the raw image coordinates, as in project_points.
    """
    image_points = np.array(image_points, dtype=self.dtype)
    x = image_points[0]
    y = image_points[1]

    polynomial_iw = np.array(
        [
            np.ones_like(x),
            x,     y,
            x*x,   x*y,   y*y,
            x*x*x, x*x*y, x*y*y, y*y*y
        ],
        dtype=self.dtype
    ).T

    X, Y = np.dot(polynomial_iw, self.poly_iw_o).T
    dx, dy = np.dot(polynomial_iw, self.poly_iw_v).T

    t = np.array([X, Y, np.zeros_like(X)], dtype=self.dtype)
    r = np.array([dx, dy, np.ones_like(dx)], dtype=self.dtype)

    return t, r
