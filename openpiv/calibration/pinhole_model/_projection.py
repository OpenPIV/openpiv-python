import numpy as np

from .. import _cal_doc_utils


_all__ = [
    "project_points",
    "project_to_z"
]


def _normalize_world_points(
    self,
    object_points: np.ndarray
):
    R = self.rotation
    T = self.translation
    dtype = self.dtype

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
    self,
    image_points: np.ndarray
):
    fx, fy = self.focal
    cx, cy = self.principal
    dtype = self.dtype
    
    x, y = image_points
    
    # normalize image coordinates
    Wn_x = (x - cx) / fx
    Wn_y = (y - cy) / fy
    
    return np.array([Wn_x, Wn_y], dtype=dtype)


@_cal_doc_utils.docfiller
def _project_points(
    self,
    object_points: np.ndarray,
    correct_distortion: bool = True
):
    """Project object points to image points.
    
    Project object, or real world points, to image points.
    
    Parameters
    ----------
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

    >>> cam = pinhole_model.camera(
        'cam1', 
        [4512, 800],
        translation = [-340, 125, 554],
        orientation = [0., 0, np.pi],
        focal = [15310, 15310],
    )
    
     >>> cam.minimize_params(
            [obj_x, obj_y, obj_z],
            [img_x, img_y],
            correct_focal = True,
            correct_distortion = False,
            iterations=5
        )
    
    >>> cam.minimize_params(
            [obj_x, obj_y, obj_z],
            [img_x, img_y],
            correct_focal = True,
            correct_distortion = True,
            iterations=5
        )

    >>> cam.project_points(
        obj_points
    )
    array([[-44.18942498, -33.54150164, -22.85390086],
           [ 89.60689939, 211.87910151, 334.5884999 ]])

    >>> img_points
    array([[-44.33764398, -33.67518587, -22.97467733],
           [ 89.61102873, 211.8863641 , 334.5980555 ]])
    
    """ 
    self._check_parameters()    
    
    fx, fy = self.focal
    cx, cy = self.principal
    dtype = self.dtype
    
    object_points = np.array(object_points, dtype=dtype)
    
    Wn_x, Wn_y = self._normalize_world_points(
        object_points
    )
    
    if correct_distortion == True:
        if self.distortion_model.lower() == "brown":
            Wn_x, Wn_y = self._undistort_points_brown(
                Wn_x,
                Wn_y
            )
        else:
            Wn_x, Wn_y = self._undistort_points_poly(
                Wn_x,
                Wn_y
            )
    
    # rescale coordinates
    x = Wn_x * fx + cx
    y = Wn_y * fy + cy
    
    return np.array([x, y], dtype=dtype)


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
    R = self.rotation
    dtype = self.dtype
    
    image_points = np.array(image_points, dtype=dtype)
    
    Wn_x, Wn_y = self._normalize_image_points(
        image_points
    )
    
    if self.distortion_model.lower() == "brown":
        Wn_x, Wn_y = self._distort_points_brown(
            Wn_x,
            Wn_y
        )
    else:
        Wn_x, Wn_y = self._distort_points_poly(
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

    >>> cam = pinhole_model.camera(
        'cam1', 
        [4512, 800],
        translation = [-340, 125, 554],
        orientation = [0., 0, np.pi],
        focal = [15310, 15310],
    )
    
     >>> cam.minimize_params(
            [obj_x, obj_y, obj_z],
            [img_x, img_y],
            correct_focal = True,
            correct_distortion = False,
            iterations=5
        )
    
    >>> cam.minimize_params(
            [obj_x, obj_y, obj_z],
            [img_x, img_y],
            correct_focal = True,
            correct_distortion = True,
            iterations=5
        )
        
    >>> ij = cam.project_points(
            obj_points
        )
    
    >>> ij
    array([[-44.18942498, -33.54150164, -22.85390086],
           [ 89.60689939, 211.87910151, 334.5884999 ]])
    
    >>> cam.project_to_z(
            ij,
            z=obj_points[2]
        )
    array([[-104.99762952, -104.99828339, -104.99883109],
           [ -14.99969016,   -9.99966432,   -4.99960968],
           [ -10.        ,  -10.        ,  -10.        ]])
    
    >>> obj_points
    array([[-105., -105., -105.],
           [ -15.,  -10.,   -5.],
           [ -10.,  -10.,  -10.]])
        
    """
    self._check_parameters() 
    
    dtype = self.dtype
    
    dx, dy, dz = self._get_inverse_vector(
        image_points
    )
    
    tx, ty, tz = self.translation
    
    a = (z - tz) / dz
    
    X = a*dx + tx
    Y = a*dy + ty

    return np.array([X, Y, np.zeros_like(X) + z], dtype=dtype)