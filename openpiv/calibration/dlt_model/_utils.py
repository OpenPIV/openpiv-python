import numpy as np
from os.path import join
from typing import Tuple

from ._check_params import _check_parameters
from .. import _cal_doc_utils


__all__ = [
    "get_cam_params",
    "save_parameters",
    "load_parameters"
]


@_cal_doc_utils.docfiller
def get_cam_params(
    cam_name: str,
    resolution: Tuple[int, int],
    coefficients: np.ndarray=np.identity(3),
    dtype: str="float64"
    
):
    """Create a camera parameter structure.
    
    Create a camera parameter structure for DLT calibration.
    
    Parameters
    ----------
    cam_name : str
        Name of camera.
    resolution : tuple[int, int]
        Resolution of camera in x and y axes respectively.
    coefficients : np.ndarray
        The coefficients for a DLT matrix.
    dtype : str
        The dtype used in the projections.
    
    Returns
    -------
    %(cam_struct)s
        
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv import calib_utils, dlt_model
    >>> from openpiv.data.test5 import cal_points
    
    >>> obj_x, obj_y, obj_z, img_x, img_y, img_size_x, img_size_y = cal_points()
    
    >>> camera_parameters = dlt_model.get_cam_params(
            name="cam1", 
            [img_size_x, img_size_y]
        )
    
    """    
    cam_struct = {}
    cam_struct["name"] = cam_name
    cam_struct["resolution"] = resolution
    cam_struct["coefficients"] = coefficients
    cam_struct["dtype"] = dtype
    
    _check_parameters(cam_struct)
    
    return cam_struct


@_cal_doc_utils.docfiller
def save_parameters(
    cam_struct: dict,
    file_path: str,
    file_name: str=None
):
    """Save DLT camera parameters.
    
    Save the DLT camera parameters to a text file.
    
    Parameters
    ----------
    %(cam_struct)s
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
        
        for i in range(3):
            _c = ''
            for j in range(cam_struct["coefficients"].shape[1]):
                _c += str(cam_struct["coefficients"][i, j]) + ' '
                
            f.write(_c + '\n')
        
        f.write(cam_struct["dtype"] + '\n')
        
    return None
        

@_cal_doc_utils.docfiller
def load_parameters(
    file_path: str,
    file_name: str
):
    """Load DLT camera parameters.
    
    Load the DLT camera parameters from a text file.
    
    Parameters
    ----------
    file_path : str
        File path where the camera parameters are saved.
    file_name : str
        Name of the file that contains the camera parameters.
        
    Returns
    -------
    %(cam_struct)s
    
    """
    full_path = join(file_path, file_name)
    
    with open(full_path, 'r') as f:
        
        name = f.readline()[:-1]
        
        _r = f.readline()[:-2]
        resolution = np.array([float(s) for s in _r.split()])
            
        coefficients = []
        for i in range(3):
            _c = f.readline()[:-2]
            coefficients.append(np.array([float(s) for s in _c.split()]))
                    
        dtype = f.readline()[:-1]
        
        coefficients = np.array(coefficients, dtype = dtype)

    cam_struct = get_cam_params(
        name,
        resolution,
        coefficients=coefficients,
        dtype=dtype
    )

    return cam_struct