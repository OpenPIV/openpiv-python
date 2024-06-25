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
    
    if len(cam_struct["poly_wi"].shape) != 2:
        raise ValueError(
            "World to image polynomial coefficients must be 2 dimensional."
        )
        
    if cam_struct["poly_wi"].shape[0] != 19:
        raise ValueError(
            "World to image polynomial coefficients must be ordered in [x, y]'"
        )
    
    if cam_struct["poly_wi"].shape[1] != 2:
        raise ValueError(
            "There must be 19 coefficients in the world to image polynomial"
        )
    
    if len(cam_struct["poly_iw"].shape) != 2:
        raise ValueError(
            "Image to world polynomial coefficients must be 2 dimensional."
        )
    
    if cam_struct["poly_iw"].shape[0] != 19:
        raise ValueError(
            "Image to world polynomial coefficients must be ordered in [x, y]'"
        )
    
    if cam_struct["poly_iw"].shape[1] != 3:
        raise ValueError(
            "There must be 19 coefficients in the image to world polynomial"
        )
    
    if len(cam_struct["dlt"].shape) != 2:
        raise ValueError(
            "DLT coefficients must be 2 dimensional."
        )
        
    if cam_struct["dlt"].shape != (3, 4):
        raise ValueError(
            "DLT coefficients must be of shape (3, 4)"
        )
        
    if cam_struct["dtype"] not in ["float32", "float64"]:
        raise ValueError(
            "Dtype is not supported for camera calibration"
        )