import numpy as np

from .._camera import BaseCamera
from ..stereo import (
    dewarp_image,
    get_common_world_grid,
    get_dewarp_mapping,
)
from ._geometry import _get_grid_spacing


def _prepare_image_stacks(
    cams: list[BaseCamera],
    images: list,
) -> list:
    if len(cams) != len(images) or len(cams) < 2:
        msg = 'Need matching cameras and images for at least two cameras'
        raise ValueError(msg)

    stacks = []

    for cam, images_i in zip(cams, images, strict=True):
        stack = np.asarray(images_i)

        if stack.ndim == 2:
            stack = stack[None]

        expected = (int(cam.resolution[1]), int(cam.resolution[0]))

        if stack.ndim != 3 or stack.shape[1:] != expected or stack.shape[0] == 0:
            name = cam.name
            msg = f'Images for camera {name} must have sensor shape {expected}'
            raise ValueError(msg)

        if not np.all(np.isfinite(stack)) or np.any(stack < 0):
            msg = 'Particle intensities must be positive definite'
            raise ValueError(msg)

        stacks.append(stack)

    frame_counts = {stack.shape[0] for stack in stacks}

    if len(frame_counts) != 1:
        msg = 'Every camera must have the same number of synchronized frames'
        raise ValueError(msg)

    return stacks


def _prepare_world_grid(
    cams: list[BaseCamera],
    z: float,
    window_size: int | tuple,
    world_grid: tuple | None,
) -> tuple:
    if world_grid is None:
        X, Y, _ = get_common_world_grid(cams, z=z)

        if np.ndim(window_size) == 0:
            shape = (window_size, window_size)
        else:
            shape = window_size

        if len(shape) != 2 or any(int(n) != n or n < 8 for n in shape):
            msg = 'window_size must be at least 8 pixels'
            raise ValueError(msg)
        
        height, width = map(int, shape)

        if height > X.shape[0] or width > X.shape[1]:
            msg = 'window_size exceeds the common grid; supply a world_grid'
            raise ValueError(msg)
        
        row = (X.shape[0] - height) // 2
        col = (X.shape[1] - width) // 2

        X = X[row:row + height, col:col + width]
        Y = Y[row:row + height, col:col + width]

    else:
        X, Y = world_grid
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

    spacing = _get_grid_spacing(X, Y)

    return X, Y, spacing


def _get_window_mapping(
    cam: BaseCamera,
    X: np.ndarray,
    Y: np.ndarray,
    z: float,
    world_offset: np.ndarray | None=None,
) -> tuple:
    if world_offset is None:
        offset = np.zeros(2)
    else:
        offset = world_offset

    map_x, map_y = get_dewarp_mapping(
        cam,
        X + offset[0],
        Y + offset[1],
        z=z,
    )

    width, height = cam.resolution

    inside = (
        (map_x >= 0)
        & (map_x <= width - 1)
        & (map_y >= 0)
        & (map_y <= height - 1)
    )

    return map_x, map_y, inside


def _dewarp_window(
    image: np.ndarray,
    mapping: tuple,
) -> np.ndarray:
    # Equation 4; Sample sensor intensities onto world planes
    map_x, map_y, inside = mapping

    image = np.asarray(image, dtype=float)
    map_x = np.where(inside, map_x, -1)
    map_y = np.where(inside, map_y, -1)

    result = dewarp_image(image, map_x, map_y)

    return np.where(inside, result, 0.0)
