import numpy as np

from ._distortion import _distort_normalized
from ._utils import _rotation


def _evaluate_projection(
    self,
    object_points: np.ndarray,
    correct_distortion: bool=True,
    jacobian: bool=False
):
    self._check_parameters()

    object_points = np.asarray(
        object_points,
        dtype=np.float64
    )

    if object_points.ndim != 2 or object_points.shape[0] != 3:
        msg = 'Object points must have shape (3, N)'
        raise ValueError(msg)

    if not np.all(np.isfinite(object_points)):
        msg = 'Object points must contain only finite values'
        raise ValueError(msg)

    n_points = object_points.shape[1]

    rotation, rotation_derivatives = _rotation(self.orientation)
    self.rotation = rotation.astype(self.dtype)

    camera_center = np.asarray(self.translation).reshape(3, 1)
    centered_points = object_points - camera_center
    camera_points = rotation.T @ centered_points

    depth = camera_points[2]
    x = camera_points[0] / depth
    y = camera_points[1] / depth

    if correct_distortion:
        if jacobian:
            normalized, distortion_jacobian, coefficient_jacobian = _distort_normalized(
                self,
                x,
                y,
                jacobian=True
            )
        else:
            normalized = _distort_normalized(
                self,
                x,
                y
            )
    else:
        normalized = np.array([x, y])

        if jacobian:
            distortion_jacobian = np.broadcast_to(
                np.eye(2),
                (n_points, 2, 2)
            )
            coefficient_jacobian = np.zeros((n_points, 2, 8))

    fx, fy = self.focal
    cx, cy = self.principal

    pixels = np.array([
        fx * normalized[0] + cx,
        fy * normalized[1] + cy
    ])

    if not jacobian:
        return pixels.astype(self.dtype)

    # Derivatives of normalized x and y with respect to camera X, Y, Z.
    perspective_jacobian = np.zeros((n_points, 2, 3))

    perspective_jacobian[:, 0, 0] = 1 / depth
    perspective_jacobian[:, 1, 1] = 1 / depth
    perspective_jacobian[:, 0, 2] = -x / depth
    perspective_jacobian[:, 1, 2] = -y / depth

    camera_point_jacobian = distortion_jacobian @ perspective_jacobian
    camera_point_jacobian[:, 0, :] *= fx
    camera_point_jacobian[:, 1, :] *= fy

    world_point_jacobian = camera_point_jacobian @ rotation.T
    camera_jacobian = np.zeros((n_points, 2, 18))

    # Moving the camera reverses the world-point change.
    camera_jacobian[:, :, :3] = -world_point_jacobian

    # Get camera-coordinate derivative for each angle.
    for i in range(3):
        point_derivative = rotation_derivatives[i].T @ centered_points
        point_derivative = point_derivative.T
        point_derivative = point_derivative[:, :, np.newaxis]

        orientation_derivative = camera_point_jacobian @ point_derivative
        camera_jacobian[:, :, 3+i] = orientation_derivative[:, :, 0]

    # Focal lengths and principal point.
    camera_jacobian[:, 0, 6] = normalized[0]
    camera_jacobian[:, 1, 7] = normalized[1]
    camera_jacobian[:, 0, 8] = 1
    camera_jacobian[:, 1, 9] = 1

    # Distortion coefficients
    camera_jacobian[:, 0, 10:] = fx * coefficient_jacobian[:, 0, :]
    camera_jacobian[:, 1, 10:] = fy * coefficient_jacobian[:, 1, :]

    return pixels, camera_jacobian, world_point_jacobian


def _project_jacobian(
    self,
    object_points: np.ndarray,
    correct_distortion: bool=True
):
    """Project points and return analytic pixel derivatives.

    Parameters
    ----------
    object_points : np.ndarray
        World coordinates with shape (3, N).
    correct_distortion : bool, optional
        Include Brown and the configured sensor tilt. Default is True.

    Returns
    -------
    image_points : np.ndarray
        Projected coordinates with shape (2, N).
    camera_jacobian : np.ndarray
        Shape (N, 2, 18), with columns ordered as camera center (3), Euler
        orientation (3), focal (2), principal (2), and distortion1 (8).
        Disabled distortion columns are zero.
    point_jacobian : np.ndarray
        Derivatives with respect to world coordinates, shape (N, 2, 3).

    """
    return _evaluate_projection(
        self,
        object_points,
        correct_distortion=correct_distortion,
        jacobian=True
    )