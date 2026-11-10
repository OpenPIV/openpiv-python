import numpy as np
from scipy.optimize import least_squares

from .._camera import BaseCamera


def _get_grid_spacing(
    X: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.ndim != 2 or X.shape != Y.shape or min(X.shape) < 8:
        msg = 'World grid must contain matching 2D arrays of at least 8 by 8'
        raise ValueError(msg)

    spacing = np.array([
        X[0, 1] - X[0, 0],
        Y[1, 0] - Y[0, 0],
    ])

    if not np.all(np.isfinite([X, Y])) or np.any(spacing <= 0):
        raise ValueError('World grid must be finite and increasing in X and Y')

    expected_x = X[0, 0] + np.arange(X.shape[1]) * spacing[0]
    expected_y = Y[0, 0] + np.arange(Y.shape[0]) * spacing[1]

    if not (
        np.allclose(X, expected_x[None, :])
        and np.allclose(Y, expected_y[:, None])
    ):
        raise ValueError('World grid must be a uniform Cartesian meshgrid')

    return spacing


def _get_los_slopes(
    cams: list[BaseCamera],
    reference: np.ndarray,
    step: float,
) -> np.ndarray:
    # Equation 6; differentiate invers LOS at fixed image coords
    point = np.asarray(reference, dtype=float).reshape(3, 1)
    
    slopes = []
    for cam in cams:
        image = cam.project_points(point)

        z_high = np.array([point[2, 0] + step])
        z_low = np.array([point[2, 0] - step])

        high = np.asarray(cam.project_to_z(image, z_high))
        low = np.asarray(cam.project_to_z(image, z_low))

        slope = (high - low)[:2, 0] / (2 * step)
        slopes.append(slope)

    slopes = np.asarray(slopes)

    if not np.all(np.isfinite(slopes)):
        msg = 'Camera back-projection produced non-finite LOS slopes'
        raise ValueError(msg)

    return slopes


def _get_streak_geometry(
    slopes: np.ndarray,
    pair: tuple,
    spacing: np.ndarray,
) -> dict:
    # Equation 7-8, 10; compute the streak vector
    i, j = pair

    direction = (slopes[i] - slopes[j]) / spacing
    length = np.linalg.norm(direction)

    if length <= np.finfo(float).eps:
        msg = f'Camera pair {pair} has no measurable depth parallax'
        raise ValueError(msg)
    
    tangent = direction / length

    return {
        'depth_direction': direction,
        'tangent': tangent,
        'normal': np.array([-tangent[1], tangent[0]]),
    }


def _world_to_image_offsets(
    cams: list[BaseCamera],
    reference: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    # Equation 15; convert plane offset into pixel offsets
    point = np.asarray(reference, dtype=float).reshape(3, 1)
    result = []

    for cam, offset in zip(cams, offsets, strict=True):
        shifted = point.copy()
        shifted[:2, 0] += offset

        shifted_image = cam.project_points(shifted)
        reference_image = cam.project_points(point)
        image_offset = (shifted_image - reference_image)[:, 0]

        result.append(image_offset)

    return np.asarray(result)


def _minimize_image_disparities(
    cams: list[BaseCamera],
    reference: np.ndarray,
    offsets: np.ndarray,
    slopes: np.ndarray,
    exclude_camera: int | None = None,
) -> dict:
    # Equation 20-22; try to minimize disparities between three cameras
    offsets = np.asarray(offsets, dtype=float)
   
    keep = np.ones(len(cams), dtype=bool)

    if exclude_camera is not None:
        if not 0 <= exclude_camera < len(cams):
            raise ValueError('Excluded camera index is out of range')

        keep[exclude_camera] = False

    def shifted(translation):
        return offsets + translation[:2] + slopes * translation[2]

    def residual(translation):
        world_offsets = shifted(translation)
        image_offsets = _world_to_image_offsets(cams, reference, world_offsets)
        
        return image_offsets[keep].ravel()

    fit = least_squares(
        residual,
        np.zeros(3),
        x_scale='jac',
    )

    if not fit.success:
        msg = f'Image disparity minimization failed: {fit.message}'
        raise RuntimeError(msg)
    
    world_offsets = shifted(fit.x)
    image_offsets = _world_to_image_offsets(cams, reference, world_offsets)

    return {
        'world_offsets': world_offsets,
        'image_offsets': image_offsets,
        'coordinate_shift': fit.x,
    }
