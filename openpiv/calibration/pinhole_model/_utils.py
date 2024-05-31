import numpy as np
from os.path import join
from typing import Tuple

from ._check_params import _check_parameters
from .._doc_utils import (docstring_decorator,
                          doc_cam_struct)


__all__ = [
    "get_cam_params",
    "get_rotation_matrix",
    "save_parameters",
    "load_parameters"
]


@docstring_decorator(doc_cam_struct)
def get_cam_params(
    name: str,
    resolution: Tuple[int, int],
    translation: np.ndarray=[0, 0, 1],
    orientation: np.ndarray=np.zeros(3, dtype="float64"),
    rotation: np.ndarray=np.eye(3, 3, dtype="float64"),
    distortion_model: str="polynomial",
    distortion1: np.ndarray=np.zeros(8, dtype="float64"),
    distortion2: np.ndarray=None,
    focal: Tuple[float, float]=[1.0, 1.0],
    principal: Tuple[float, float]=None,
    dtype: str="float64"
    
):
    """Create a camera parameter structure.
    
    Create a camera parameter structure for calibration.
    
    Parameters
    ----------
    name : str
        Name of camera.
    resolution : tuple[int, int]
        Resolution of camera in x and y axes respectively.
    translation : 1D np.ndarray-like
        Location of camera origin/center in x, y, and z axes respectively.
    orientation : 1D np.ndarray-like
        Orientation of camera in x, y, z axes respectively.
    rotation : 2D np.ndarray
        Rotational camera parameter for camera system.
    distortion_model : str
        The type of distortion model to use.
        
        ``brown``
        The Brown model follows the distortion model incorporated by OpenCV.
        It consists of a radial and tangential model to compensate for distortion.
        
        ``polynomial``
        The polynomial model is used for general distortion compensation. It
        consists of a 2nd order polynomial in the x and y axes.
        
        Both models do not attempt to correct distortions along the z-plane.
        
    distortion1 : 2D np.ndarray
       Radial and tangential distortion compensation matrix for a camera.
    distortion2 : 2D np.ndarray
       2nd order polynomial distortion compensation matrix for a camera.
    focal : tuple[float, float]
        Focal distance/magnification of camera-lens system for x any y axis
        respectively.
    principal : tuple[float, float]
        Principal point offset for x any y axis respectively.
    dtype : str
        The dtype used in the projections. All data is copied if the dtype is
        different. It is highly unadvisable to change this parameter.
    
    Returns
    -------
    cam_struct : dict
        {0}
    
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv import calib_utils, calib_pinhole
    >>> from openpiv.data.test5 import cal_points
    
    >>> obj_x, obj_y, obj_z, img_x, img_y, img_size_x, img_size_y = cal_points()
    
    >>> camera_parameters = calib_pinhole.get_cam_params(
            name="cam1", 
            [img_size_x, img_size_y]
        )
    
    """    
    # cast to arrays
    translation = np.array(translation, dtype=dtype)
    orientation = np.array(orientation, dtype=dtype)
    distortion1 = np.array(distortion1, dtype=dtype)
    
    if distortion2 is None:
        distortion2 = np.array(
            [
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0]
            ],
            dtype=dtype
        )
        
    # create the dictionary structure
    cam_struct = {}
    cam_struct["name"] = name
    cam_struct["resolution"] = resolution
    cam_struct["translation"] = translation
    cam_struct["orientation"] = orientation
    cam_struct["rotation"] = rotation
    cam_struct["distortion_model"] = distortion_model
    cam_struct["distortion1"] = distortion1
    cam_struct["distortion2"] = distortion2
    cam_struct["focal"] = focal
    
    if principal is not None:
        cam_struct["principal"] = principal
    else:
        # temporary place holder
        cam_struct["principal"] = [0, 0]
    
    cam_struct["dtype"] = dtype
    
    # check parameters here
    _check_parameters(cam_struct)
    
    # default principal point is half of image resolution
    if principal is None:
        cam_struct["principal"] = [resolution[0] / 2, resolution[1] / 2]
        
    return cam_struct


@docstring_decorator(doc_cam_struct)
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
        {0}
    
    Rewturns
    --------
    rotation_matrix : 2D np.ndarray
        A 3x3 rotation matrix.
    
    """
    _check_parameters(cam_struct)
    
    # Orientation is composed of angles, or theta, for each axes.
    # Theta for each dimensions is abbreviated as t<axis>.
    tx, ty, tz = cam_struct["orientation"]
    dtype = cam_struct["dtype"]
    
    # We compute the camera patrix based off of this website.
    # https://support.pix4d.com/hc/en-us/articles/202559089-How-are-the-Internal-and-External-Camera-Parameters-defined
    
    rot_x = np.array(
        [
            [1,        0,          0],
            [0, np.cos(tx),-np.sin(tx)],
            [0, np.sin(tx), np.cos(tx)]
        ],
        dtype=dtype
    )
    
    rot_y = np.array(
        [
            [ np.cos(ty), 0, np.sin(ty)],
            [        0,   1,        0],
            [-np.sin(ty), 0, np.cos(ty)]
        ], 
        dtype=dtype
    )
    
    rot_z = np.array(
        [
            [np.cos(tz),-np.sin(tz), 0],
            [np.sin(tz), np.cos(tz), 0],
            [       0,          0,   1]
        ], 
        dtype=dtype
    )
    
    rotation_matrix = np.dot(
        np.dot(rot_x, rot_y), 
        rot_z
    )
    
    return rotation_matrix


@docstring_decorator(doc_cam_struct)
def save_parameters(
    cam_struct: dict,
    file_path: str,
    file_name: str=None
):
    """Save pinhole camera parameters.
    
    Save the pinhole camera parameters to a text file.
    
    Parameters
    ----------
    cam_struct : dict
        {0}
    file_path : str
        File path where the camera parameters are saved.
    file_name : str, optional
        If specified, override the default file name.
        
    Returns
    -------
    None
    
    """
    _check_parameters(cam_struct)
    
    if file_name is None:
        file_name = cam_struct["name"]
    
    full_path = join(file_path, file_name)
    
    with open(full_path, 'w') as f:
        f.write(cam_struct["name"] + '\n')
        
        _r = ''
        for i in range(2):
            _r += str(cam_struct["resolution"][i]) + ' '
            
        f.write(_r + '\n')
        
        _t = ''
        for i in range(3):
            _t += str(cam_struct["translation"][i]) + ' '
        
        f.write(_t + '\n')
        
        _o = ''
        for i in range(3):
            _o += str(cam_struct["orientation"][i]) + ' '
        
        f.write(_o + '\n')
        
        f.write(cam_struct["distortion_model"] + '\n')
        
        _d1 = ''
        for i in range(8):
            _d1 += str(cam_struct["distortion1"][i]) + ' '
        
        f.write(_d1 + '\n')
        
        for i in range(4):
            _d2 = ''
            for j in range(6):
                _d2 += str(cam_struct["distortion2"][i, j]) + ' '
                
            f.write(_d2 + '\n')
            
        _f = ''
        for i in range(2):
            _f += str(cam_struct["focal"][i]) + ' '
            
        f.write(_f + '\n')
        
        _p = ''
        for i in range(2):
            _p += str(cam_struct["principal"][i]) + ' '
            
        f.write(_p + '\n')
        
        f.write(cam_struct["dtype"] + '\n')
        
    return None
        

@docstring_decorator(doc_cam_struct)
def load_parameters(
    file_path: str,
    file_name: str
):
    """Load pinhole camera parameters.
    
    Load the pinhole camera parameters from a text file.
    
    Parameters
    ----------
    file_path : str
        File path where the camera parameters are saved.
    file_name : str
        Name of the file that contains the camera parameters.
        
    Returns
    -------
    cam_struct : dict
        {0}
    
    """
    full_path = join(file_path, file_name)
    
    with open(full_path, 'r') as f:
        
        name = f.readline()[:-1]
        
        _r = f.readline()[:-2]
        resolution = np.array([float(s) for s in _r.split()])
        
        _t = f.readline()[:-2]
        translation = np.array([float(s) for s in _t.split()])
        
        _o = f.readline()[:-2]
        orientation = np.array([float(s) for s in _o.split()])
        
        distortion_model = f.readline()[:-1]
        
        _d1 = f.readline()[:-2]
        distortion1 = np.array([float(s) for s in _d1.split()])
        
        distortion2 = []
        for i in range(4):
            _d2 = f.readline()[:-2]
            distortion2.append(np.array([float(s) for s in _d2.split()]))
            
        distortion2 = np.array(distortion2, dtype = "float64")
        
        _f = f.readline()[:-2]
        focal = np.array([float(s) for s in _f.split()])
        
        _p = f.readline()[:-2]
        principal = np.array([float(s) for s in _p.split()])
        
        dtype = f.readline()[:-1]

    cam_struct = get_cam_params(
        name,
        resolution,
        translation=translation,
        orientation=orientation,
        distortion_model=distortion_model,
        distortion1=distortion1,
        distortion2=distortion2,
        focal=focal,
        principal=principal,
        dtype=dtype
    )
    
    cam_struct["rotation"] = get_rotation_matrix(
        cam_struct
    )
    
    return cam_struct