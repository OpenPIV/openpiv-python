import numpy as np

from ._distortion import _active_distortion


def _camera_columns(
    cam,
    correct_focal: bool=True,
    correct_distortion: bool=True,
    pose: bool=True
):
    if pose:
        columns = list(range(6))
    else:
        columns = []

    if correct_focal:
        columns += [6, 7]

    columns += [8, 9]

    if correct_distortion:
        columns += list(10 + _active_distortion(cam))

    return np.array(columns, dtype=int)


def _camera_vector(
    cam
):
    return np.hstack([
        cam.translation,
        cam.orientation,
        cam.focal,
        cam.principal,
        cam.distortion1
    ])


def _set_camera_vector(
    cam,
    values: np.ndarray
):
    cam.translation = values[:3].copy()
    cam.orientation = values[3:6].copy()

    cam.focal = values[6:8].copy()
    cam.principal = values[8:10].copy()

    cam.distortion1 = values[10:18].copy()

    cam._get_rotation_matrix()