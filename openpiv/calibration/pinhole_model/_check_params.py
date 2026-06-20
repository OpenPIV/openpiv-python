import numpy as np


def _check_parameters(self):
    """Check camera parameters"""
    
    if type(self.name) != str:
        raise ValueError(
            "Camera name must be a string"
        )
    
    if len(self.resolution) != 2:
        raise ValueError(
            "Resolution must be a two element tuple"
        )
    
    if self.translation.shape != (3,):
        raise ValueError(
            "Translation must be a three element 1D numpy ndarray"
        )
    
    if self.orientation.shape != (3,):
        raise ValueError(
            "Orientation must be a three element 1D numpy ndarray"
        )
    
    if self.distortion_model.lower() not in ["brown", "polynomial"]:
        raise ValueError(
            "Distortion model must be either 'brown' or 'polynomial', not " +
            f"'{self.distortion_model}'."
        )
        
    if self.distortion1.shape != (8,):
        raise ValueError(
            "Radial and tangential distortion coefficients must be " +\
             "an 8 element 1D numpy ndarray"
        )
    
    if not isinstance(self.distortion2, np.ndarray):
        raise ValueError(
            "Polynomial distortion coefficients must be a numpy ndarray"
        )
        
    if self.distortion2.shape != (2, 5):
        raise ValueError(
            "Polynomial distortion coefficients must be a 2x5 numpy ndarray"
        )
    
    if not isinstance(self.focal, (tuple, list, np.ndarray)):
        raise ValueError(
            "Focal point must be a tuple or list"
        )
            
    if len(self.focal) != 2:
        raise ValueError(
            "Focal point must be a two element tuple or list"
        )
            
    if not isinstance(self.principal, (tuple, list, np.ndarray)):
        raise ValueError(
            "Principal point must be a tuple or list"
        )
    
    if len(self.principal) != 2:
        raise ValueError(
            "Principal point must be a two element tuple or list"
        )
        
    if self.dtype not in ["float32", "float64"]:
        raise ValueError(
            "Dtype is not supported for camera calibration"
        )