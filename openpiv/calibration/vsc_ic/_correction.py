import numpy as np

from .._camera import BaseCamera


__all__ = [
    'fix_disparity'
]


def _make_calibration_grid(disparity: dict) -> np.ndarray:
    X, Y = disparity['grid']
    z = disparity['z_planes']

    range_x = np.linspace(X.min(), X.max(), 5)
    range_y = np.linspace(Y.min(), Y.max(), 5)
    range_z = np.linspace(z[0], z[-1], 5)

    X, Y, Z = np.meshgrid(
        range_x,
        range_y,
        range_z,
        indexing='ij'
    )

    return np.array(
        [X.ravel(), Y.ravel(), Z.ravel()],
        dtype=np.float64
    )


def fix_disparity(
    cams: list[BaseCamera],
    disparity: dict,
    object_points: np.ndarray | None = None,
) -> dict:
    """Calculate image and object points for camera recalibration.

    Calculate corrected image points by projecting object points into each
    camera and adding the measured image disparities. Return the image and
    object point correspondences for use with the camera calibration methods.

    Parameters
    ----------
    cams : list of openpiv.calibration.BaseCamera
        A list of calibrated camera instances used to calculate the disparity.
    disparity : dict
        A valid triple-image disparity measurement returned by
        get_disparity.
    object_points : 2D np.ndarray, optional
        2D object points of [X, Y, Z] in world coordinates, with shape (3, n).
        If None, a 5 by 5 by 5 grid is generated over the measurement volume.
        Default is None.

    Returns
    -------
    correction : dict
        A dictionary containing the image and object points for recalibration.
        object_points contains a copy of the world coordinates, shape (3, n).
        image_points contains one corrected (2, n) array of [x, y] image
        coordinates for each camera. image_offsets contains the applied
        sensor pixel shifts, shape (n_cameras, 2).

    Examples
    --------
    Given calibrated cameras, a valid disparity measurement and object points:

    >>> from openpiv.calibration import vsc_ic

    >>> correction = vsc_ic.fix_disparity(cams, disparity, object_points)
    >>> for cam, image_points in zip(cams, correction['image_points']):
    ...     cam.minimize_params(correction['object_points'], image_points)

    """
    if disparity.get('method') != 'triple' or not disparity.get('valid'):
        msg = 'Correction requires a valid triple-image disparity measurement'
        raise ValueError(msg)

    if disparity.get('image_offsets') is None:
        raise ValueError('Disparity measurement has no image offsets')

    image_offsets = np.array(disparity['image_offsets'], dtype=np.float64)

    if image_offsets.shape != (len(cams), 2):
        raise ValueError('Image offsets must have shape (n_cameras, 2)')

    if not np.all(np.isfinite(image_offsets)):
        raise ValueError('Image offsets must contain only finite values')

    if object_points is None:
        object_points = _make_calibration_grid(disparity)
    else:
        object_points = np.array(object_points, dtype=np.float64)

    if object_points.ndim != 2 or object_points.shape[0] != 3:
        raise ValueError('Object points must have shape (3, N)')

    if object_points.shape[1] == 0:
        raise ValueError('Object points must contain at least one point')

    if not np.all(np.isfinite(object_points)):
        raise ValueError('Object points must contain only finite values')

    image_points = []

    for cam, offset in zip(cams, image_offsets, strict=True):
        image_points_proj = cam.project_points(object_points)
        image_points_proj = np.array(image_points_proj, dtype=np.float64)

        if image_points_proj.shape != (2, object_points.shape[1]):
            raise ValueError('Camera projections must have shape (2, N)')

        image_points_proj += offset[:, None]

        if not np.all(np.isfinite(image_points_proj)):
            msg = 'Camera projections must contain only finite values'
            raise ValueError(msg)

        image_points.append(image_points_proj)

    return {
        'object_points': object_points,
        'image_points': image_points,
        'image_offsets': image_offsets
    }
