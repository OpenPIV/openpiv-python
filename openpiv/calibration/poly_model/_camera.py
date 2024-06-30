import numpy as np
from typing import Tuple

from ._check_params import _check_parameters
from ._minimization import _minimize_params
from ._projection import _project_points, _project_to_z
from ._utils import _save_parameters, _load_parameters


__all__ = ["camera"]


class camera(object):
    """An instance of a DLT camera model.
    
    Create an instance of a DLT camera. The DLT camera model is based on
    the direct linear transformation where image points and object points
    are mapped through a 3x3 or 3x4 matrix.
    
    Attributes
    ----------
    cam_name : str
        Name of camera.
    resolution : tuple[int, int]
        Resolution of camera in x and y axes respectively.
    poly_wi : np.ndarray
        19 coefficients for world to image polynomial calibration in [x, y]'.
    poly_iw : np.ndarray
        19 coefficients for image to world polynomial calibration in [X, Y, Z]'.
    dlt : np.ndarray
        12 coefficients for direct linear transformation.
    dtype : str
        The dtype used in the projections.
        
    Methods
    -------
    minimize_params
    project_points
    project_to_z
    save_parameters
    load_parameters
        
    """
    def __init__(
        self,
        name: str,
        resolution: Tuple[int, int],
        poly_wi: np.ndarray=np.ones((2,19), dtype="float64").T,
        poly_iw: np.ndarray=np.ones((3,19), dtype="float64").T,
        dlt: np.ndarray=np.eye(3, 4),
        dtype: str="float64"
    ):
        self.name = name
        self.resolution = resolution
        self.poly_wi = poly_wi
        self.poly_iw = poly_iw
        self.dlt = dlt
        self.dtype = dtype
        
        # method definitions
        camera._check_parameters = _check_parameters
        camera.minimize_params = _minimize_params
        camera.project_points = _project_points
        camera.project_to_z = _project_to_z
        camera.save_parameters = _save_parameters
        camera.load_parameters = _load_parameters
        
        # check camera params at init
        self._check_parameters()