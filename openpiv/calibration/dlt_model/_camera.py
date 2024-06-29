import numpy as np

from ._check_params import _check_parameters
from ._minimization import _minimize_params
from ._projection import _project_points, _project_to_z, _get_inverse_vector
from ._utils import _save_parameters, _load_parameters


__all__ = ["camera"]


class camera(object):
    """An instance of a DLT camera model.
    
    Create an instance of a DLT camera. The DLT camera model is based on
    the direct linear transformation where image points and object points
    are mapped through a 3x3 or 3x4 matrix.
    
    Attributes
    ----------
    name : str
        The name of the camera.
    resolution : tuple[int, int]
        Resolution of camera in x and y axes respectively.
    coeffs : np.ndarray
        The coefficients for a DLT matrix.
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
        coeffs: np.ndarray=np.identity(3),
        dtype: str="float64"
    ):
        self.name = name
        self.resolution = resolution
        self.coeffs = coeffs
        self.dtype = dtype
        
        # method definitions
        camera._check_parameters = _check_parameters
        camera._get_inverse_vector = _get_inverse_vector
        camera.minimize_params = _minimize_params
        camera.project_points = _project_points
        camera.project_to_z = _project_to_z
        camera.save_parameters = _save_parameters
        camera.load_parameters = _load_parameters