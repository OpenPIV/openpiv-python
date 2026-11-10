from copy import deepcopy

import numpy as np

from .. import _cal_doc_utils
from .._optimizers import optimize_general
from ._optimize import (
    _camera_columns,
    _camera_vector,
    _set_camera_vector
)


@_cal_doc_utils.docfiller
def _minimize_params(
    self,
    object_points: list,
    image_points: list,
    correct_focal: bool=False,
    correct_distortion: bool=False,
    max_iter: int=1000,
    iterations: int=3
):
    """Refine pinhole camera parameters using NumPy QR Levenberg-Marquardt.

    Fit the selected camera parameters to image observations of fixed
    world points. The world points may be distributed throughout a volume.

    Parameters
    ----------
    %(object_points)s
    %(image_points)s
    correct_focal : bool, optional
        Refine both focal lengths. Default is False.
    correct_distortion : bool, optional
        Include distortion in projection and refine the enabled Brown and
        tilt coefficients. When False, fit an ideal pinhole projection,
        ignoring the stored distortion coefficients. Default is False.
    max_iter : int, optional
        Maximum LM steps per refinement pass. Default is 1000.
    iterations : int, optional
        Number of successive refinement passes. Default is 3.

    Returns
    -------
    None
        The camera parameters are updated in place.

    Raises
    ------
    ValueError
        If camera parameters are invalid, point arrays have invalid
        shapes, too few calibration points are supplied, or max_iter
        is not positive when running a refinement pass.
    RuntimeError
        If a Levenberg-Marquardt refinement pass fails.

    """
    self._check_parameters()

    object_points = np.array(object_points, dtype=np.float64)
    image_points = np.array(image_points, dtype=np.float64)

    if object_points.ndim != 2 or object_points.shape[0] != 3:
        raise ValueError('Object points must have shape (3, N)')

    if image_points.shape != (2, object_points.shape[1]):
        raise ValueError('Object point image point size mismatch')

    columns = _camera_columns(
        self,
        correct_focal=correct_focal,
        correct_distortion=correct_distortion
    )
    n_parameters = len(columns)
    min_points = max(9, (n_parameters + 1) // 2)

    if object_points.shape[1] < min_points:
        msg = f'Too few points to calibrate. Need at least {min_points} points'
        raise ValueError(msg)

    camera = deepcopy(self)
    camera.dtype = np.dtype(np.float64)
    values = _camera_vector(camera).astype(np.float64)

    def set_parameters(parameters):
        values[columns] = parameters
        _set_camera_vector(camera, values)

    def residuals(parameters):
        set_parameters(parameters)

        if np.any(camera.focal <= 0):
            return np.full(image_points.size, np.inf)

        projected = camera.project_points(
            object_points,
            correct_distortion=correct_distortion
        )

        return (projected - image_points).T.ravel()

    def jacobian(parameters):
        set_parameters(parameters)

        _, projection_jacobian, _ = camera.project_jacobian(
            object_points,
            correct_distortion=correct_distortion
        )

        # Interleave x and y rows to match the expected format of the residual vector.
        return projection_jacobian[:, :, columns].reshape(
            -1,
            n_parameters
        )

    parameters = values[columns]

    for _ in range(iterations):
        parameters, result = optimize_general(
            residuals,
            parameters,
            jac=jacobian,
            max_iter=max_iter
        )

        if not result['success']:
            msg = f"Pinhole refinement failed: {result['message']}"
            raise RuntimeError(msg)

    values[columns] = parameters
    _set_camera_vector(self, values.astype(self.dtype))