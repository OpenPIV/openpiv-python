import numpy as np
from scipy.optimize import least_squares

from .. import _cal_doc_utils


def _refine_func(
    parameters: np.ndarray,
    camera,
    object_points: np.ndarray,
    image_points: np.ndarray,
    correct_focal: bool,
    correct_distortion: bool
):
    """Apply trial parameters and return image reprojection residuals."""
    camera.translation = parameters[0:3]
    camera.orientation = parameters[3:6]
    camera.principal = parameters[6:8]

    offset = 8

    if correct_focal:
        camera.focal = parameters[offset:offset + 2]
        offset += 2

    if correct_distortion:
        if camera.distortion_model.lower() == "brown":
            camera.distortion1 = parameters[offset:offset + 8]
        else:
            camera.distortion2 = parameters[offset:offset + 10].reshape(
                (2, 5)
            )

    camera._get_rotation_matrix()

    projected_points = camera.project_points(object_points)

    return (projected_points - image_points).T.ravel()


@_cal_doc_utils.docfiller
def _minimize_params(
    self,
    object_points: list,
    image_points: list,
    correct_focal: bool = False,
    correct_distortion: bool = False,
    max_iter: int = 1000,
    iterations: int = 3
):
    """Refine pinhole camera parameters using Levenberg-Marquardt.

    Minimize image reprojection residuals using a two-point numerical
    Jacobian. Translation, orientation, and principal point are always
    refined. Focal lengths and distortion coefficients are optionally
    refined.

    Parameters
    ----------
    %(object_points)s
    %(image_points)s
    correct_focal : bool, optional
        If True, refine both focal lengths. Default is False.
    correct_distortion : bool, optional
        If True, refine coefficients of the selected distortion model.
        Otherwise, keep the existing coefficients fixed. Default is False.
    max_iter : int, optional
        Maximum function evaluations per refinement pass. Default is 1000.
    iterations : int, optional
        Number of successive refinement passes. Default is 3.

    Returns
    -------
    None
        The camera parameters are updated in place.
    """
    self._check_parameters()

    object_points = np.array(object_points, dtype=self.dtype)
    image_points = np.array(image_points, dtype=self.dtype)

    if object_points.shape[1] != image_points.shape[1]:
        raise ValueError(
            "Object point image point size mismatch"
        )

    # Pack only parameters that will be refined.
    parameters = [
        self.translation,
        self.orientation,
        self.principal
    ]

    if correct_focal:
        parameters.append(self.focal)

    if correct_distortion:
        if self.distortion_model.lower() == "brown":
            parameters.append(self.distortion1)
        else:
            parameters.append(self.distortion2.ravel())

    params_to_minimize = np.hstack(parameters)

    # Retain the existing nine-point minimum while ensuring that LM
    # has at least as many scalar residuals as free parameters.
    min_points = max(
        9,
        (params_to_minimize.size + 1) // 2
    )

    if object_points.shape[1] < min_points:
        raise ValueError(
            f"Too few points to calibrate. Need at least {min_points} points"
        )

    args = (
        self,
        object_points,
        image_points,
        correct_focal,
        correct_distortion
    )

    for _ in range(iterations):
        res = least_squares(
            _refine_func,
            x0=params_to_minimize,
            args=args,
            method="lm",
            jac="2-point",
            x_scale="jac",
            max_nfev=max_iter
        )

        # Update the camera with the latest and most valid result
        _refine_func(res.x, *args)

        if not res.success:
            raise RuntimeError(
                f"Pinhole refinement failed: {res.message}"
            )

        # Continue subsequent passes from the returned solution.
        params_to_minimize = res.x
