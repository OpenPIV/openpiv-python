import numpy as np
from typing import Tuple

from calib_utils import get_reprojection_error


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
    
    if cam_struct["translation"].shape != (3,):
        raise ValueError(
            "Translation must be a three element 1D numpy ndarray"
        )
    
    if cam_struct["orientation"].shape != (3,):
        raise ValueError(
            "Orientation must be a three element 1D numpy ndarray"
        )
        
    if cam_struct["rotation"].shape != (3,3):
        raise ValueError(
            "Rotation must be a 3x3 numpy array"
        )
    
    if cam_struct["distortion"].shape != (8,):
        raise ValueError(
            "Distortion coefficients must be an 8 element 1D numpy ndarray"
        )
    
    if not isinstance(cam_struct["focal"], (tuple, list, np.ndarray)):
        raise ValueError(
            "Focal point must be a tuple or list"
        )
            
    if len(cam_struct["focal"]) != 2:
        raise ValueError(
            "Focal point must be a two element tuple or list"
        )
            
    if not isinstance(cam_struct["principal"], (tuple, list, np.ndarray)):
        raise ValueError(
            "Principal point must be a tuple or list"
        )
    
    if len(cam_struct["principal"]) != 2:
        raise ValueError(
            "Principal point must be a two element tuple or list"
        )
            

def generate_camera_params(
    cam_name: str,
    resolution: Tuple[int, int],
    translation: np.ndarray=[0, 0, 1],
    orientation: np.ndarray=np.zeros(3, dtype="float64"),
    rotation: np.ndarray=np.eye(3, 3, dtype="float64"),
    distortion: np.ndarray=np.zeros(8, dtype="float64"),
    focal: Tuple[float, float]=[1.0, 1.0],
    principal: Tuple[float, float]=None
    
):
    """Create a camera parameter structure.
    
    Create a camera parameter structure for calibration.
    
    Parameters
    ----------
    cam_name : str
        Name of camera.
    resolution : tuple[int, int]
        Resolution of camera in x and y axes respectively.
    translation : 1D np.ndarray-like
        Location of camera origin/center in x, y, and z axes respectively.
    orientation : 1D np.ndarray-like
        Orientation of camera in x, y, z axes respectively.
    rotation : 2D np.ndarray
        Rotational camera parameter for camera system.
    distortion : 2D np.ndarray
        Distortion compensation matrix for a camera.
    focal : tuple[float, float]
        Focal distance/magnification of camera-lense system for x any y axis
        espectively.
    principal : tuple[float, float]
        Principal point offset for x any y axis respectively.
    
    Returns
    -------
    cam_struct : dict
        A dictionary structure of camera parameters.
    
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv import calib_utils, calib_pinhole
    >>> from openpiv.data.test5 import cal_points
    
    >>> obj_x, obj_y, obj_z, img_x, img_y, img_size_x, img_size_y = cal_points()
    
    >>> camera_parameters = calib_pinhole.generate_camera_params(
            name="cam1", 
            [img_size_x, img_size_y]
        )
    
    """    
    # default principal point is half of image resolution
    if principal is None:
        principal = [resolution[0] / 2, resolution[1] / 2]
    
    # cast to arrays
    translation = np.array(translation, dtype="float64")
    orientation = np.array(orientation, dtype="float64")
        
    # create the dictionary structure
    cam_struct = {}
    cam_struct["name"] = cam_name
    cam_struct["resolution"] = resolution
    cam_struct["translation"] = translation
    cam_struct["orientation"] = orientation
    cam_struct["rotation"] = rotation
    cam_struct["distortion"] = distortion
    cam_struct["focal"] = focal
    cam_struct["principal"] = principal
    
    _check_parameters(cam_struct)
            
    return cam_struct


def get_rotation_matrix(
    cam_struct: dict
):
    """Calculate a rotation matrix for a camera.
    
    Calculate a rotation matrix for a camera. The matrix is a 3x3 numpy ndarray
    like such:
    
    [ r1 r2 r3 ]
    [ r4 r5 r6 ]
    [ r7 r8 r9 ]
    
    where
    
    r1 = cos(tz) * cos(ty)
    r2 = -sin(tz) * cos(ty)
    r3 = sin(ty)
    r4 = cos(tz) * sin(tx) * sin(ty) + sin(tz) * cos(tx)
    r5 = cos(tz) * cos(tx) - sin(tz) * sin(tx) * sin(ty)
    r6 = -sin(tx) * cos(ty)
    r7 = sin(tz) * sin(tx) - cos(tz) * cos(tx) * sin(ty)
    r8 = sin(tz) * cos(tx) * sin(ty) + cos(tz) * sin(tx)
    r9 = cos(tx) * cos(ty)
    
    Parameters
    ----------
    cam_struct : dict
        A dictionary structure of camera parameters.
    
    Rewturns
    --------
    rotation_matrix : 2D np.ndarray
        A 3x3 rotation matrix.
    
    """
    _check_parameters(cam_struct)
    
    # Orientation is composed of angles, or theta, for each axes.
    # Theta for each dimensions is abbreviated as t<axis>.
    tx, ty, tz = cam_struct["orientation"]
    
    # We compute the camera patrix based off of this website.
    # https://support.pix4d.com/hc/en-us/articles/202559089-How-are-the-Internal-and-External-Camera-Parameters-defined
    
    rot_x = np.array(
        [
            [1,        0,          0],
            [0, np.cos(tx),-np.sin(tx)],
            [0, np.sin(tx), np.cos(tx)]
        ],
        dtype="float64"
    )
    
    rot_y = np.array(
        [
            [ np.cos(ty), 0, np.sin(ty)],
            [        0,   1,        0],
            [-np.sin(ty), 0, np.cos(ty)]
        ], 
        dtype="float64"
    )
    
    rot_z = np.array(
        [
            [np.cos(tz),-np.sin(tz), 0],
            [np.sin(tz), np.cos(tz), 0],
            [       0,          0,   1]
        ], 
        dtype="float64"
    )
    
    rotation_matrix = np.dot(
        np.dot(rot_x, rot_y), 
        rot_z
    )
    
    return rotation_matrix


def undistort_points(
    cam_struct: dict,
    x: np.ndarray, 
    y: np.ndarray
):
    """Undistort normalized points.
    
    Undistort normalized camera points using a radial and tangential distortion
    model. 
    
    Parameters
    ----------
    cam_struct : dict
        A dictionary structure of camera parameters.
    x : 1D np.ndarray
        Distorted x-coordinates.
    y : 1D np.ndarray
        Distorted y-coordinates.
        
    Returns
    -------
    xd : 1D np.ndarray
        Undistorted x-coordinates.
    yd : 1D np.ndarray
        Undistorted y-coordinates.
    
    Notes
    -----
    Distortion model is based off of OpenCV. The direct link where the distortion 
    model was accessed is provided below.
    https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html
    
    """
    _check_parameters(cam_struct)  
    
    k = cam_struct["distortion"]
    
    r2 = x*x + y*y
    r4 = r2 * r2
    r6 = r4 * r2
    
    num = 1 + k[0]*r2 + k[1]*r4 + k[4]*r6
    den = 1 + k[5]*r2 + k[6]*r4 + k[7]*r6
    
    delta_x = k[2]*2*x*y + k[3]*(r2 + 2*x*x)
    delta_y = k[3]*2*x*y + k[2]*(r2 + 2*y*y)
    
    xd = x*(num/den) + delta_x
    yd = y*(num/den) + delta_y
    
    return xd, yd


def distort_points(
    cam_struct: dict,
    xd: np.ndarray, 
    yd: np.ndarray
):
    """Distort normalized points.
    
    Distort normalized camera points using a radial and tangential distortion
    model. 
    
    Parameters
    ----------
    cam_struct : dict
        A dictionary structure of camera parameters.
    xd : 1D np.ndarray
        Undistorted x-coordinates.
    yd : 1D np.ndarray
        Undistorted y-coordinates.
        
    Returns
    -------
    x : 1D np.ndarray
        Distorted x-coordinates.
    y : 1D np.ndarray
        Distorted y-coordinates.
    
    Notes
    -----
    Distortion model is based off of OpenCV. The direct link where the distortion 
    model was accessed is provided below.
    https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html
    
    """
    _check_parameters(cam_struct)  
    
    k = cam_struct["distortion"]
    
    r2 = xd*xd + yd*yd
    r4 = r2 * r2
    r6 = r4 * r2
    
    den = 1 + k[0]*r2 + k[1]*r4 + k[4]*r6
    num = 1 + k[5]*r2 + k[6]*r4 + k[7]*r6
    
    delta_x = k[2]*2*xd*yd + k[3]*(r2 + 2*xd*xd)
    delta_y = k[3]*2*xd*yd + k[2]*(r2 + 2*yd*yd)

    x = (xd - delta_x) * (num/den)
    y = (yd - delta_y) * (num/den)
    
    return x, y


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
            correct_instrinsic = True,
            correct_distortion = False,
            iterations=5
        )
    
    >>> camera_parameters = calib_pinhole.minimize_camera_params(
            camera_parameters, 
            [obj_x, obj_y, obj_z],
            [img_x, img_y],
            correct_instrinsic = True,
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
    
    object_points = np.array(object_points, dtype="float64")
    
    R = cam_struct["rotation"]
    T = cam_struct["translation"]
    fx, fy = cam_struct["focal"]
    cx, cy = cam_struct["principal"]

    # transformation to camera coordinates
    Wc = np.dot(
        R,
        object_points
    ) + T[:, np.newaxis]
        
    # the camera coordinates
    Wc_x = Wc[0, :]
    Wc_y = Wc[1, :]
    Wc_h = Wc[2, :]
        
    # normalize coordinates
    Wn_x = Wc_x / Wc_h
    Wn_y = Wc_y / Wc_h 
    
    # distortion correction
    Wd_x, Wd_y = undistort_points(
        cam_struct,
        Wn_x,
        Wn_y
    )
    
    # rescale coordinates
    x = Wd_x * fx + cx
    y = Wd_y * fy + cy
    
    return np.array([x, y])
    
    
def project_to_z(
    cam_struct: dict,
    image_points: np.ndarray,
    z
):
    """Project image points to world points.
    
    Project image points to world points at specified z-plane.
    
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
            correct_instrinsic = True,
            correct_distortion = False,
            iterations=5
        )
    
    >>> camera_parameters = calib_pinhole.minimize_camera_params(
            camera_parameters, 
            [obj_x, obj_y, obj_z],
            [img_x, img_y],
            correct_instrinsic = True,
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
    
    x, y = np.array(image_points, dtype="float64")
    
    R = cam_struct["rotation"]
    T = cam_struct["translation"]
    fx, fy = cam_struct["focal"]
    cx, cy = cam_struct["principal"]
    
    # normalize image coordinates
    Wd_x = (x - cx) / fx
    Wd_y = (y - cy) / fy
    
    # distort points
    Wn_x, Wn_y = distort_points(
        cam_struct,
        Wd_x,
        Wd_y
    )
    
    # inverse rotation
    dx, dy, dz = np.dot(
        R.T,
        [Wn_x, Wn_y, np.ones_like(Wn_x)]
    )
    
    # inverse translation
    Tx, Ty, Tz = np.dot(
        R.T,
        -T[:, np.newaxis]
    )
    
    # camera coordinates to world coordinates
    X = ((-Tz + z) / dz)*dx + Tx
    Y = ((-Tz + z) / dz)*dy + Ty

    return np.array([X, Y, np.ones_like(X) * z], dtype="float64")


def minimize_camera_params(
    cam_struct: dict,
    object_points: list,
    image_points: list,
    correct_instrinsic: bool = False,
    correct_distortion: bool = False,
    max_iter: int = 1000,
    iterations: int = 3
):
    """Minimize camera parameters.
    
    Minimize camera parameters using Nelder-Mead optimization. To do this,
    the root mean square error (RMS error) is calculated for each iteration. The
    set of parameters with the lowest RMS error is returned (which is hopefully correct
    the minimum).
    
    Parameters
    ----------
    cam_struct : dict
        A dictionary structure of camera parameters.
    object_points : np.ndarray
        A 2D np.ndarray containing [x, y, z] object points.
    image_points : np.ndarray
        A 2D np.ndarray containing [x, y] image points.
    correct_instrinsic : bool
        If true, minimize the instrinsic matrix.
    correct_distortion : bool
        If true, mininmize the distortion model.
    max_iter : int
        Maximum amount of iterations in Nelder-Mead minimization.
    iterations : int
        Number of iterations iterations to perform.
        
    Returns
    -------
    cam_struct : dict
        A dictionary structure of optimized camera parameters.
    
    Notes
    -----
    When minimizing the camera parameters, it is important that the parameters are
    estimated first before distortion correction. This allows a better estimation
    of the camera parameters and distortion coefficients. For instance, if one were
    to calibrate the camera intrinsic and distortion coefficients before moving the
    camera to the lab apparatus, it would be important to calibrate the camera
    parameters before the distortion model to ensure a better convergence and thus,
    lower root mean sqaure (RMS) errors. This can be done in the following procedure:
    
    1. Place the camera directly in front of a planar calibration plate.
    
    2. Estimate camera parameters with out distortion correction.
    
    3. Estimate distortion model coefficients and refine camera parameters.
    
    4. Place camera in lab apparatus.
    
    5. Calibrate camera again without distortion or intrinsic correction.
    
    A brief example is shown below. More can be found in the example PIV lab
    experiments.
    
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv import calib_utils, calib_pinhole
    >>> from openpiv.data.test5 import cal_points
    
    >>> obj_x, obj_y, obj_z, img_x, img_y, img_size_x, img_size_y = cal_points()
    
    >>> camera_parameters = calib_pinhole.generate_camera_params(
            name="cam1", 
            [img_size_x, img_size_y]
        )
    
     >>> camera_parameters = calib_pinhole.minimize_camera_params(
            camera_parameters, 
            [obj_x, obj_y, obj_z],
            [img_x, img_y],
            correct_instrinsic = True,
            correct_distortion = False,
            iterations=5
        )
    
    >>> camera_parameters = calib_pinhole.minimize_camera_params(
            camera_parameters, 
            [obj_x, obj_y, obj_z],
            [img_x, img_y],
            correct_instrinsic = True,
            correct_distortion = True,
            iterations=5
        )
    
    >>> calib_utils.get_reprojection_error(
            camera_parameters, 
            calib_pinhole.project_points,
            [obj_x, obj_y, obj_z],
            [img_x, img_y]
        )
    
    """
    _check_parameters(cam_struct)
    
    from scipy.optimize import minimize
    
    object_points = np.array(object_points, dtype="float64")
    image_points = np.array(image_points, dtype="float64")
    
    # For each iteration, calculate the RMS error of this function. The input is a numpy
    # array to meet the requirements of scipy's minimization functions.
    def func_to_minimize(x):
        cam_struct["translation"] = x[0:3]
        cam_struct["orientation"] = x[3:6]
        
        if correct_instrinsic == True:
            cam_struct["principal"] = x[6:8]
            cam_struct["focal"] = x[8:10]
            
        if correct_distortion == True:
            cam_struct["distortion"]= x[10:19]
        
        cam_struct["rotation"] = get_rotation_matrix(
            cam_struct
        )
        
        RMS_error = get_reprojection_error(
            cam_struct,
            project_points,
            object_points,
            image_points
        )
        
        return RMS_error
    
    # Create a numpy array since we cannot pass a dictionary to scipy's minimize function.
    params_to_minimize = [
        cam_struct["translation"],
        cam_struct["orientation"],
        cam_struct["principal"],
        cam_struct["focal"],
        cam_struct["distortion"]
    ]
    
    params_to_minimize = np.hstack(
        params_to_minimize
    )
    
    # Peform multiple iterations to hopefully attain a better calibration.
    for _ in range(iterations):
        # Discard output of minimization as we are interested in the camera params dict.
        res = minimize(
            func_to_minimize,
            params_to_minimize,
            method="BFGS",
            options={"maxiter": max_iter},
            jac = "2-point"
        )
        
    return cam_struct


