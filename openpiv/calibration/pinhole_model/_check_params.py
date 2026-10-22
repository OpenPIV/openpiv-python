import numpy as np


def _check_parameters(self):
    if not isinstance(self.name, str):
        raise ValueError('Camera name must be a string')

    if len(self.resolution) != 2:
        raise ValueError('Resolution must be a two element tuple')

    if self.translation.shape != (3,):
        msg = 'Translation must be a three element 1D numpy ndarray'
        raise ValueError(msg)

    if self.orientation.shape != (3,):
        msg = 'Orientation must be a three element 1D numpy ndarray'
        raise ValueError(msg)

    if self.distortion_model.lower() not in ['brown', 'polynomial']:
        msg = 'Distortion model must be either "brown" or "polynomial", not ' +\
             f'{self.distortion_model}'
        raise ValueError(msg)

    if self.distortion1.shape != (8,):
        msg = 'Radial and tangential distortion coefficients must be ' +\
              'an 8 element 1D numpy ndarray'
        raise ValueError(msg)

    if not isinstance(self.distortion2, np.ndarray):
        msg = 'Polynomial distortion coefficients must be a numpy ndarray'
        raise ValueError(msg)

    if self.distortion2.shape != (2, 5):
        msg = 'Polynomial distortion coefficients must be a 2x5 numpy ndarray'
        raise ValueError(msg)

    if not isinstance(self.focal, (tuple, list, np.ndarray)):
        raise ValueError('Focal point must be a tuple or list')

    if len(self.focal) != 2:
        raise ValueError('Focal point must be a two element tuple or list')

    if not isinstance(self.principal, (tuple, list, np.ndarray)):
        raise ValueError('Principal point must be a tuple or list')

    if len(self.principal) != 2:
        msg = 'Principal point must be a two element tuple or list'
        raise ValueError(msg)

    if self.dtype not in [np.float32, np.float64]:
        raise ValueError('Dtype is not supported for camera calibration')
