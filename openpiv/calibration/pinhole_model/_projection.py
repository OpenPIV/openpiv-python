import numpy as np

from .. import _cal_doc_utils
from ._jacobian import _evaluate_projection


def _normalize_world_points(
    self,
    object_points: np.ndarray
):
    self._get_rotation_matrix()

    points = np.asarray(object_points)
    camera_center = self.translation.reshape(3, 1)
    centered_points = points - camera_center
    camera_points = self.rotation.T @ centered_points

    depth = camera_points[2]
    normalized = camera_points[:2] / depth

    return normalized.astype(self.dtype)


def _normalize_image_points(
    self,
    image_points: np.ndarray
):
    fx, fy = self.focal
    cx, cy = self.principal
    x, y = image_points

    normalized_x = (x - cx) / fx
    normalized_y = (y - cy) / fy

    return np.array([
        normalized_x,
        normalized_y
    ], dtype=self.dtype)


@_cal_doc_utils.docfiller
def _project_points(
    self,
    object_points: np.ndarray,
    correct_distortion: bool=True
):
    """Project world points to pixels through Brown, tilt, then intrinsics.

    Parameters
    ----------
    %(object_points)s
    correct_distortion : bool, optional
        Apply the configured Brown and sensor-tilt model. Default is True.
        False projects through ideal pinhole intrinsics only.

    Returns
    -------
    image_points : np.ndarray
        Pixel coordinates with shape (2, N).

    Raises
    ------
    ValueError
        If camera parameters are invalid or object coordinates cannot
        be converted to an array with three rows.

    Examples
    --------
    >>> from openpiv.calibration import pinhole_model
    >>> cam = pinhole_model.PinCam(
    ...     'camera0',
    ...     (640, 480),
    ...     translation=[0, 0, 0],
    ...     focal=(800, 800)
    ... )
    >>> cam.project_points([[10], [20], [100]])
    array([[400.],
           [400.]])

    """
    return _evaluate_projection(
        self,
        object_points,
        correct_distortion=correct_distortion
    )


@_cal_doc_utils.docfiller
def _get_inverse_vector(
    self,
    image_points: np.ndarray,
    correct_distortion: bool=True
):
    self._get_rotation_matrix()

    dtype = self.dtype

    image_points = np.array(
        image_points,
        dtype=dtype
    )

    if image_points.ndim != 2 or image_points.shape[0] != 2:
        msg = 'Image points must have shape (2, N)'
        raise ValueError(msg)

    if not np.all(np.isfinite(image_points)):
        msg = 'Image points must contain only finite values'
        raise ValueError(msg)

    x, y = self._normalize_image_points(image_points)

    if correct_distortion:
        x, y = self._undistort_points_brown(
            x,
            y
        )

    camera_directions = np.array([
        x,
        y,
        np.ones_like(x)
    ])
    world_directions = self.rotation @ camera_directions

    return world_directions.astype(self.dtype)


@_cal_doc_utils.docfiller
def _project_to_z(
    self,
    image_points: np.ndarray,
    z: float | np.ndarray,
    correct_distortion: bool=True
):
    """Back-project pixels onto the specified world Z plane or planes.

    Parameters
    ----------
    %(image_points)s
    %(project_z)s
    correct_distortion : bool, optional
        Reverse the configured Brown and tilt model. Default is True.

    Returns
    -------
    object_points : np.ndarray
        World coordinates with shape (3, N).

    Raises
    ------
    ValueError
        If camera parameters or image-coordinate dimensions are invalid,
        or the supplied Z values cannot broadcast over the image points.
    RuntimeError
        If Brown distortion inversion fails to converge.
    np.linalg.LinAlgError
        If reversing tilt or solving a Brown inversion step encounters
        a singular matrix.

    Examples
    --------
    >>> from openpiv.calibration import pinhole_model
    >>> cam = pinhole_model.PinCam(
    ...     'camera0',
    ...     (640, 480),
    ...     translation=[0, 0, 0],
    ...     focal=(800, 800)
    ... )
    >>> cam.project_to_z([[400], [400]], z=100)
    array([[ 10.],
           [ 20.],
           [100.]])

    """
    self._check_parameters()

    dtype = self.dtype

    directions = self._get_inverse_vector(
        image_points,
        correct_distortion=correct_distortion
    )

    z = np.array(
        z,
        dtype=dtype
    )
    n_points = directions.shape[1]

    if z.ndim > 1 or (z.ndim == 1 and z.size not in (1, n_points)):
        msg = 'Z must be a scalar or a 1D array matching the image points'
        raise ValueError(msg)

    if not np.all(np.isfinite(z)):
        msg = 'Z must contain only finite values'
        raise ValueError(msg)

    direction_x, direction_y, direction_z = directions
    center_x, center_y, center_z = self.translation

    distance = (z - center_z) / direction_z

    world_x = distance * direction_x + center_x
    world_y = distance * direction_y + center_y
    world_z = np.zeros_like(world_x) + z

    return np.array([
        world_x,
        world_y,
        world_z
    ], dtype=dtype)