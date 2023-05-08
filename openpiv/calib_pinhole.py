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
    
    if cam_struct["distortion"].shape != (3,5):
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
    orientation: np.ndarray= np.ones(3, dtype=float),
    rotation: np.ndarray=np.zeros((3,3), dtype=float),
    distortion: np.ndarray=np.zeros((3,5), dtype=float),
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
    translation : tuple[int, int, int]
        Location of camera origin/center in x, y, and z axes respectively.
    orientation : np.ndarray
        Orientation of camera in x, y, z axes respectively.
    rotation : np.ndarray
        Rotational camera parameter for camera system.
    distortion : np.ndarray
        Radial distortion compensation matrix for a camera.
    focal : tuple[float, float]
        Focal distance/magnification of camera-lense system for x any y axis
        espectively.
    principal : tuple[float, float]
        Principal point offset for x any y axis respectively.
    
    Returns
    -------
    camera_struct : dict
        A dictionary structure of camera parameters.
    
    """    
    # default principal point is half of image resolution
    if principal is None:
        principal = [resolution[0] / 2, resolution[1] / 2]
    
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
    
    """ 
    _check_parameters(cam_struct)
    
    cx, cy = cam_struct["principal"]
    
    
    R = cam_struct["rotation"]
    T = cam_struct["translation"]
    fx, fy = cam_struct["focal"]
    
    # Here, we create a transformation matrix and homogenize the object points.
    # Then, we transform the object points to camera points via dot product.
    # The transformation matrix is defined by the following:
    # [ R     | T ]
    # [ 0 0 0 | 1 ]
    # where R is the rotation matrix and T is the translation vector.
    # The object points are like the following:
    # [P]
    # [1]
    # where P is the object points.
    
    Wc = np.dot(
        np.concatenate(
            [
                np.hstack((R, T[:, np.newaxis])),
                np.array([[0, 0, 0, 1]])
            ], 
            axis=0
        ),
        np.concatenate(
            (object_points, np.ones((1, object_points.shape[1]))),
            axis=0
        )
    )
    
    # the camera coordinates
    Xc = Wc[0, :]
    Yc = Wc[1, :]
    Zc = Wc[2, :]
    
    # Wc = R'(object_points - T)
#    Xc = R[0, 0] * (object_points[:, 0] - T[0]) +\
#         R[0, 1] * (object_points[:, 1] - T[1]) +\
#         R[0, 2] * (object_points[:, 2] - T[2])
    
#    Yc = R[1, 0] * (object_points[:, 0] - T[0]) +\
#         R[1, 1] * (object_points[:, 1] - T[1]) +\
#         R[1, 2] * (object_points[:, 2] - T[2])
    
#    Zc = R[2, 0] * (object_points[:, 0] - T[0]) +\
#         R[2, 1] * (object_points[:, 1] - T[1]) +\
#         R[2, 2] * (object_points[:, 2] - T[2])
    
    # normalize coordinates
    Xn = Xc / Zc
    Yn = Yc / Zc
    
    # distortion correction
    Xd, Yd = eta_zeta_from_bRinv(
        cam_struct,
        Xn,
        Yn
    )
    
    # intrinsic matrix
    K = np.array(
       [[fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]],
        dtype=float
    )
    
    # projection to image coordinates
    ij = np.dot(
        K, 
        np.array([Xd, Yd, Xd * 0. + 1.])
    )
    
    # image coordinates
    Xp = ij[0, :]
    Yp = ij[1, :]
    
    return np.array([Xp, Yp,])


# Copyright (c) 2022 Ron Shnapp
# Originally incorporated from myPTV as the previous non-linear
# distortion model had undefined behavior. In the furure, we could incorporate
# the distortion model used by OpenCV.
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


def minimize_camera_params(
    cam_struct: dict,
    object_points: list,
    image_points: list,
    correct_focal: bool = False,
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
    correct_focal : bool
        If true, minimize the focal distance.
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
    
    """
    _check_parameters(cam_struct)
    
    from scipy.optimize import minimize
    
    # For each iteration, calculate the RMS error of this function. The input is a numpy
    # array to meet the requirements of scipy's minimization functions.
    def func_to_minimize(x):
        cam_struct["translation"] = x[0:3]
        cam_struct["orientation"] = x[3:6]
        cam_struct["principal"] = x[6:8]
        
        if correct_focal == True:
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
#        cam_struct["distortion"][2, :]
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