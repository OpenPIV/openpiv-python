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
    
    if cam_struct["distortion"].shape != (2,5):
        raise ValueError(
            "Distortion correction matrix must be a 3x5 numpy array"
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
    translation: np.ndarray=np.ones(3, dtype=float),
    orientation: np.ndarray=np.ones(3, dtype=float),
    rotation: np.ndarray=np.zeros((3,3), dtype=float),
    distortion: np.ndarray=np.zeros((2,5), dtype=float),
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
    camera_struct : dict
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
    
    translation = np.array(translation, dtype=float)
    orientation = np.array(orientation, dtype=float)
    
    camera_struct = {}
    camera_struct["name"] = cam_name
    camera_struct["resolution"] = resolution
    camera_struct["translation"] = translation
    camera_struct["orientation"] = orientation
    camera_struct["rotation"] = rotation
    camera_struct["distortion"] = distortion
    camera_struct["focal"] = focal
    camera_struct["principal"] = principal
    
    _check_parameters(camera_struct)
            
    return camera_struct


def calculate_rotation_matrix(
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
        dtype=float
    )
    
    rot_y = np.array(
        [
            [ np.cos(ty), 0, np.sin(ty)],
            [        0,   1,        0],
            [-np.sin(ty), 0, np.cos(ty)]
        ], 
        dtype=float
    )
    
    rot_z = np.array(
        [
            [np.cos(tz),-np.sin(tz), 0],
            [np.sin(tz), np.cos(tz), 0],
            [       0,          0,   1]
        ], 
        dtype=float
    )
    
    rotation_matrix = np.dot(
        np.dot(rot_x, rot_y), 
        rot_z
    )
    
    return rotation_matrix


# Copyright (c) 2022 Ron Shnapp
# Originally incorporated from myPTV as the previous non-linear
# distortion model had undefined behavior. In the furure, we could incorporate
# the distortion model used by OpenCV.

# @ErichZimemr - Changes (May 30, 2023):
# Added parameters checks, modified function for use with this module.
def eta_zeta_from_bRinv(
    cam_struct: dict,
    eta_: np.ndarray, 
    zeta_: np.ndarray
):
    """Non-linear distortion correction
    
    Non-linear distortion correction solved by linearizing the error
    term with the Taylor series expantion. 
    
    Parameters
    ----------
    cam_struct : dict
        A dictionary structure of camera parameters.
    x : 1D np.ndarray
        Projected image x-coordinates.
    y : 1D np.ndarray
        Projected image y-coordinates.
        
    Returns
    -------
    x : 1D np.ndarray
        Projected image x-coordinates.
    y : 1D np.ndarray
        Projected image y-coordinates.
    
    Notes
    -----
    Copyright (c) 2022 Ron Shnapp
    This function was taken from myPTV and is copyrighted under MIT licenses.
    The link to the repository is provided below.
    https://github.com/ronshnapp/MyPTV
    
    """
    _check_parameters(camera_struct)
    
    Z3 = np.array([eta_, zeta_, eta_**2, zeta_**2, eta_ * zeta_])

    e_ = np.dot(cam_struct["distortion"], Z3)

    # calculating the derivatives of the error term:
    e_0 = e_[0]
    a, b, c, d, ee = cam_struct["distortion"][0,:]
    e_eta_0 = a + 2*c*eta_ + ee*zeta_
    e_zeta_0 = b + 2*d*zeta_ + ee*eta_

    e_1 = e_[1]
    a, b, c, d, ee = cam_struct["distortion"][1,:]
    e_eta_1 = a + 2*c*eta_ + ee*zeta_
    e_zeta_1 = b + 2*d*zeta_ + ee*eta_

    A11 = 1.0 + e_eta_0
    A12 = e_zeta_0
    A21 = e_eta_1
    A22 = 1.0 + e_zeta_1

    rhs1 = eta_*(1.0 + e_eta_0) + zeta_*e_zeta_0 - e_0
    rhs2 = zeta_*(1.0 + e_zeta_1) + eta_*e_eta_1 - e_1

    Ainv = np.array([[A22, -A12],[-A21, A11]]) / (A11*A22 - A12*A21)
    
    eta = Ainv[0, 0] * rhs1 + Ainv[0, 1] * rhs1
    zeta = Ainv[1, 0] * rhs2 + Ainv[1, 1] * rhs2

    return eta, zeta


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
    
    >>> obj_points = np.array([obj_x[0:2], obj_y[0:2], obj_z[0:2]], dtype=float)
    >>> img_points = np.array([img_x[0:2], img_y[0:2]], dtype=float)
    
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
    
    object_points = np.array(object_points, dtype=float)
    
    R = cam_struct["rotation"]
    T = cam_struct["translation"]
    fx, fy = cam_struct["focal"]
    cx, cy = cam_struct["principal"]

    # camera transformation to camera coordinates
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
    Wn_y = Wc_x / Wc_h 
    
    # distortion correction
    Wd_x, Wd_y = calib_pinhole.eta_zeta_from_bRinv(
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
    
    >>> obj_points = np.array([obj_x[0:2], obj_y[0:2], obj_z[0:2]], dtype=float)
    >>> img_points = np.array([img_x[0:2], img_y[0:2]], dtype=float)
    
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
    
    x, y = np.array(image_points, dtype=float)
    
    R = cam_struct["rotation"]
    T = cam_struct["translation"]
    fx, fy = cam_struct["focal"]
    cx, cy = cam_struct["principal"]
    
    # normalize image coordinates
    Wn_x = (x - cx) / fx
    Wn_y = (y - cy) / fy
    Wn_z = np.ones_like(Wn_x)
    
    # inverse rotation
    dx, dy, dz = np.dot(
        R.T,
        [Wn_x, Wn_y, Wn_z]
    )
    
    # inverse translation
    Tx, Ty, Tz = np.dot(
        R.T,
        -T[:, np.newaxis]
    )
    
    # camera coordinates to world coordinates
    X = ((-Tz + z) / dz)*dx + Tx
    Y = ((-Tz + z) / dz)*dy + Ty

    return np.array([X, Y, np.ones_like(X) * z], dtype=float)


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
    camera_struct : dict
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
    
    object_points = np.array(object_points, dtype=float)
    image_points = np.array(image_points, dtype=float)
    
    # For each iteration, calculate the RMS error of this function. The input is a numpy
    # array to meet the requirements of scipy's minimization functions.
    def func_to_minimize(x):
        cam_struct["translation"] = x[0:3]
        cam_struct["orientation"] = x[3:6]
        
        if correct_instrinsic == True:
            cam_struct["principal"] = x[6:8]
            cam_struct["focal"] = x[8:10]
            
        if correct_distortion == True:
            cam_struct["distortion"][0, :] = x[10:15]
            cam_struct["distortion"][1, :] = x[15:20]
        
        cam_struct["rotation"] = calculate_rotation_matrix(
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
        cam_struct["distortion"][0, :],
        cam_struct["distortion"][1, :]
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


