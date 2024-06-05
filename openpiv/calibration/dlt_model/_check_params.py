import numpy as np


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
    
    if len(cam_struct["coefficients"].shape) != 2:
        raise ValueError(
            "DLT coefficients must be 2 dimensional"
        )
    
    if cam_struct["coefficients"].shape[0] != 3:
        raise ValueError(
            "DLT coefficients axis 0 must be of size 3"
        )
    
    if cam_struct["coefficients"].shape[1] not in [3, 4]:
        raise ValueError(
            "DLT coefficients axis 1 must be of size 3 for 2D " +
            "and size 4 for 3D transformations"
        )
        
    if cam_struct["dtype"] not in ["float32", "float64"]:
        raise ValueError(
            "Dtype is not supported for camera calibration"
        )