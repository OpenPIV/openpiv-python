import numpy as np
from typing import Tuple

from ._check_params import _check_parameters
from ._minimization import _minimize_params
from ._projection import _project_points, _project_to_z, _get_inverse_vector
from ._utils import _save_parameters, _load_parameters


__all__ = ["camera"]


class camera(object):
    """An instance of a polynomial camera model.
    
    Create an instance of a polynomial camera. The polynomial camera model
    is based on the Soloff camera model where a polymial mapping function
    is created with third order polynomials in the x- and y-axis and a second
    order polynomial along the z-axis. The polynomial camera models also has
    ray-based estimator for 3D point triangulation.
    
    Attributes
    ----------
    cam_name : str
        Name of camera.
    resolution : tuple[int, int]
        Resolution of camera in x and y axes respectively.
    poly_wi : np.ndarray
        19 coefficients for world to image polynomial calibration in [x, y]'.
    poly_iw_o : np.ndarray
        10 coefficients for ray origins in [X, Y]'.
    poly_iw_v : np.ndarray
        10 coefficients for ray slopes in [dX/dZ, dY/dZ]'.
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
        poly_iw_o: np.ndarray=np.ones((2,10), dtype="float64").T,
        poly_iw_v: np.ndarray=np.ones((2,10), dtype="float64").T,
        dtype: str="float64"
    ):
        self.name = name
        self.resolution = resolution
        self.poly_wi = poly_wi
        self.poly_iw_o = poly_iw_o
        self.poly_iw_v = poly_iw_v
        self.dtype = dtype
        
        # method definitions
        camera._check_parameters   = _check_parameters
        camera.minimize_params     = _minimize_params
        camera.project_points      = _project_points
        camera.project_to_z        = _project_to_z
        camera._get_inverse_vector = _get_inverse_vector
        camera.save_parameters     = _save_parameters
        camera.load_parameters     = _load_parameters
        
        # check camera params at init
        self._check_parameters()