import numpy as np
from os.path import join
from typing import Tuple

from ._check_params import _check_parameters
from .._doc_utils import (docstring_decorator,
                          doc_cam_struct)


__all__ = [
    "get_cam_params",
    "save_parameters",
    "load_parameters"
]


@docstring_decorator(doc_cam_struct)
def get_cam_params(
    cam_name: str,
    resolution: Tuple[int, int],
    poly_wi: np.ndarray=np.ones((2,19), dtype="float64").T,
    poly_iw: np.ndarray=np.ones((3,19), dtype="float64").T,
    dtype: str="float64"
    
):
    """Create a camera parameter structure.
    
    Create a camera parameter structure for polynomial calibration.
    
    Parameters
    ----------
    cam_name : str
        Name of camera.
    resolution : tuple[int, int]
        Resolution of camera in x and y axes respectively.
    poly_wi : np.ndarray
        19 coefficients for world to image polynomial calibration in [x, y]'.
    poly_iw : np.ndarray
        19 coefficients for image to world polynomial calibration in [X, Y, Z]'.
    dtype : str
        The dtype used in the projections.
    
    Returns
    -------
    cam_struct : dict
        {0}
        
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv.calibration import calib_utils, poly_model
    >>> from openpiv.data.test5 import cal_points
    
    >>> obj_x, obj_y, obj_z, img_x, img_y, img_size_x, img_size_y = cal_points()
    
    >>> camera_parameters = poly_model.get_cam_params(
            name="cam1", 
            [img_size_x, img_size_y]
        )
    
    """    
    cam_struct = {}
    cam_struct["name"] = cam_name
    cam_struct["resolution"] = resolution
    cam_struct["poly_wi"] = poly_wi
    cam_struct["poly_iw"] = poly_iw
    cam_struct["dtype"] = dtype
    
    _check_parameters(cam_struct)
    
    return cam_struct


@docstring_decorator(doc_cam_struct)
def save_parameters(
    cam_struct: dict,
    file_path: str,
    file_name: str=None
):
    """Save polynomial camera parameters.
    
    Save the polynomial camera parameters to a text file.
    
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
    if file_name is None:
        file_name = cam_struct["name"]
    
    full_path = join(file_path, file_name)
    
    with open(full_path, 'w') as f:
        f.write(cam_struct["name"] + '\n')
        
        _r = ''
        for i in range(2):
            _r += str(cam_struct["resolution"][i]) + ' '
            
        f.write(_r + '\n')
        
        for i in range(19):
            _d2 = ''
            for j in range(2):
                _d2 += str(cam_struct["poly_wi"][i, j]) + ' '
                
            f.write(_d2 + '\n')
            
        for i in range(19):
            _d2 = ''
            for j in range(3):
                _d2 += str(cam_struct["poly_iw"][i, j]) + ' '
                
            f.write(_d2 + '\n')
        
        f.write(cam_struct["dtype"] + '\n')
        
    return None
        

@docstring_decorator(doc_cam_struct)
def load_parameters(
    file_path: str,
    file_name: str
):
    """Load polynomial camera parameters.
    
    Load the polynomial camera parameters from a text file.
    
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
            
        poly_wi = []
        for i in range(19):
            _d2 = f.readline()[:-2]
            poly_wi.append(np.array([float(s) for s in _d2.split()]))
        
        poly_iw = []
        for i in range(19):
            _d2 = f.readline()[:-2]
            poly_iw.append(np.array([float(s) for s in _d2.split()]))
        
        dtype = f.readline()[:-1]
        
        poly_wi = np.array(poly_wi, dtype=dtype)
        poly_iw = np.array(poly_iw, dtype=dtype)

    cam_struct = get_cam_params(
        name,
        resolution,
        poly_wi=poly_wi,
        poly_iw=poly_iw,
        dtype=dtype
    )

    return cam_struct