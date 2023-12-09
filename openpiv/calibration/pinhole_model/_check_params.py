import numpy as np


__all__ = [
    "_check_parameters"
]


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
    
    if cam_struct["distortion_model"].lower() not in ["brown", "polynomial"]:
        raise ValueError(
            "Distortion model must be either 'brown' or 'polynomial', not '{}'.".format(cam_struct["distortion_model"])
        )
        
    if cam_struct["distortion1"].shape != (8,):
        raise ValueError(
            "Radial and tangential distortion coefficients must be " +\
             "an 8 element 1D numpy ndarray"
        )
    
    if not isinstance(cam_struct["distortion2"], np.ndarray):
        raise ValueError(
            "Polynomial distortion coefficients must be a numpy ndarray"
        )
        
    if cam_struct["distortion2"].shape != (4, 6):
        raise ValueError(
            "Polynomial distortion coefficients must be a 4x6 numpy ndarray"
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
        
    if cam_struct["dtype"] not in ["float32", "float64"]:
        raise ValueError(
            "Dtype is not supported for camera calibration"
        )