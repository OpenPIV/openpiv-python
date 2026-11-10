
import numpy as np
from numpy.typing import DTypeLike

from .._camera import BaseCamera
from ._check_params import _check_parameters
from ._minimization import _minimize_params
from ._projection import _get_inverse_vector, _project_points, _project_to_z
from ._utils import _load_parameters, _save_parameters


__all__ = [
    'PolyCamera'
]


class PolyCamera(BaseCamera):
    """An instance of a polynomial camera model.

    Create an instance of a polynomial camera. The polynomial camera model
    is based on the Soloff camera model where a polynomial mapping function
    is created with third order polynomials in the x- and y-axis and a second
    order polynomial along the z-axis. The polynomial camera models also has
    ray-based estimator for 3D point triangulation.

    .. rubric:: Methods

    * :meth:`minimize_params` - Fit camera parameters to calibration points.
    * :meth:`project_points` - Project world points to image coordinates.
    * :meth:`project_to_z` - Project image points to a world Z plane.
    * :meth:`save_parameters` - Save camera parameters to a file.
    * :meth:`load_parameters` - Load camera parameters from a file.

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
    dtype : np.dtype
        The dtype used in the projections.

    """
    # method definitions
    _check_parameters   = _check_parameters
    minimize_params     = _minimize_params
    project_points      = _project_points
    project_to_z        = _project_to_z
    _get_inverse_vector = _get_inverse_vector
    save_parameters     = _save_parameters
    load_parameters     = _load_parameters

    def __init__(
        self,
        name: str,
        resolution: tuple[int, int],
        poly_wi: np.ndarray=np.ones((2,19), dtype=np.float64).T,
        poly_iw_o: np.ndarray=np.ones((2,10), dtype=np.float64).T,
        poly_iw_v: np.ndarray=np.ones((2,10), dtype=np.float64).T,
        dtype: DTypeLike=np.float64
    ):
        super().__init__(name, resolution, dtype)
        self.poly_wi = np.array(poly_wi, dtype=self.dtype, copy=True)
        self.poly_iw_o = np.array(poly_iw_o, dtype=self.dtype, copy=True)
        self.poly_iw_v = np.array(poly_iw_v, dtype=self.dtype, copy=True)

        # check camera params at init
        self._check_parameters()
