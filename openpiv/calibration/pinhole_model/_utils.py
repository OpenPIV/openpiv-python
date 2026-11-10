from os.path import join

import numpy as np

from .. import _cal_doc_utils


def _rotation(
    angles: np.ndarray
):
    cos_x, cos_y, cos_z = np.cos(angles)
    sin_x, sin_y, sin_z = np.sin(angles)

    rotation_x = np.array([
        [1, 0, 0],
        [0, cos_x, -sin_x],
        [0, sin_x, cos_x]
    ])

    rotation_y = np.array([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y]
    ])

    rotation_z = np.array([
        [cos_z, -sin_z, 0],
        [sin_z, cos_z, 0],
        [0, 0, 1]
    ])

    derivative_x = np.array([
        [0, 0, 0],
        [0, -sin_x, -cos_x],
        [0, cos_x, -sin_x]
    ])

    derivative_y = np.array([
        [-sin_y, 0, cos_y],
        [0, 0, 0],
        [-cos_y, 0, -sin_y]
    ])

    derivative_z = np.array([
        [-sin_z, -cos_z, 0],
        [cos_z, -sin_z, 0],
        [0, 0, 0]
    ])

    rotation = rotation_x @ rotation_y @ rotation_z

    derivatives = np.array([
        derivative_x @ rotation_y @ rotation_z,
        rotation_x @ derivative_y @ rotation_z,
        rotation_x @ rotation_y @ derivative_z
    ])

    return rotation, derivatives


def _rotation_angles(
    rotation: np.ndarray
):
    cosine_y = np.hypot(rotation[0, 0], rotation[0, 1])
    angle_y = np.arctan2(rotation[0, 2], cosine_y)

    if cosine_y > 1e-12:
        angle_x = np.arctan2(-rotation[1, 2], rotation[2, 2])
        angle_z = np.arctan2(-rotation[0, 1], rotation[0, 0])
    else:
        angle_x = np.arctan2(rotation[2, 1], rotation[1, 1])
        angle_z = 0.0

    return np.array([
        angle_x,
        angle_y,
        angle_z
    ])


def _get_rotation_matrix(
    self
):
    self._check_parameters()

    rotation, _ = _rotation(self.orientation)
    self.rotation = rotation.astype(self.dtype)


@_cal_doc_utils.docfiller
def _save_parameters(
    self,
    file_path: str,
    file_name: str | None=None
):
    """Save pinhole parameters to a text file.

    Parameters
    ----------
    file_path : str
        Directory in which to save parameters.
    file_name : str, optional
        File name. Defaults to the camera name.

    Returns
    -------
    None
        The camera parameters are written to the file.

    Raises
    ------
    ValueError
        If camera parameters are invalid.
    OSError
        If the parameter file cannot be opened or written.

    """
    self._check_parameters()

    if file_name is None:
        file_name = self.name

    parameter_path = join(file_path, file_name)

    with open(parameter_path, 'w') as parameter_file:
        parameter_file.write(self.name + '\n')

        for values in (self.resolution, self.translation, self.orientation):
            line = ' '.join(str(value) for value in values)
            parameter_file.write(line + '\n')

        tilt_flag = int(self.tilt)
        configuration = f'{self.distortion_model} {tilt_flag} {self.refraction}'
        parameter_file.write(configuration + '\n')

        for values in (self.distortion1, self.focal, self.principal):
            line = ' '.join(str(value) for value in values)
            parameter_file.write(line + '\n')

        parameter_file.write(self.dtype.name + '\n')


@_cal_doc_utils.docfiller
def _load_parameters(
    self,
    file_path: str,
    file_name: str
):
    """Load pinhole parameters from the Brown/tilt text format.

    Parameters
    ----------
    file_path : str
        Directory containing the parameters.
    file_name : str
        File name of the saved camera.

    Returns
    -------
    None
        The camera parameters and rotation matrix are updated in place.

    Raises
    ------
    OSError
        If the parameter file cannot be opened or read.
    ValueError
        If the file contains invalid parameter values or malformed
        configuration fields.
    TypeError
        If the stored dtype cannot be interpreted by NumPy.

    """
    parameter_path = join(file_path, file_name)

    with open(parameter_path) as parameter_file:
        self.name = parameter_file.readline().strip()
        self.resolution = np.fromstring(parameter_file.readline(), sep=' ').astype(int)
        self.translation = np.fromstring(parameter_file.readline(), sep=' ')
        self.orientation = np.fromstring(parameter_file.readline(), sep=' ')

        model, tilt_flag, refraction = parameter_file.readline().split()
        self.refraction = refraction
        self.distortion1 = np.fromstring(parameter_file.readline(), sep=' ')
        self.distortion_model = model
        self.tilt = bool(int(tilt_flag))

        self.focal = np.fromstring(parameter_file.readline(), sep=' ')
        self.principal = np.fromstring(parameter_file.readline(), sep=' ')

        dtype_name = parameter_file.readline().strip()
        self.dtype = np.dtype(dtype_name)

    array_names = (
        'translation',
        'orientation',
        'distortion1',
        'focal',
        'principal'
    )

    for name in array_names:
        values = getattr(self, name)
        values = values.astype(self.dtype)
        setattr(self, name, values)

    self._get_rotation_matrix()