import numpy as np


def _check_parameters(self):    
    if type(self.name) != str:
        raise ValueError('Camera name must be a string')
    
    if len(self.resolution) != 2:
        raise ValueError('Resolution must be a two element tuple')
    
    if len(self.poly_wi.shape) != 2:
        msg = 'World to image polynomial coefficients must be 2 dimensional.'
        raise ValueError(msg)
        
    if self.poly_wi.shape[0] != 19:
        msg = 'World to image polynomial coefficients must be ordered ' +\
              "in [x, y]'"
        raise ValueError(msg)
    
    if self.poly_wi.shape[1] != 2:
        msg = 'There must be 19 coefficients in the world to image polynomial'
        raise ValueError(msg)
    
    if len(self.poly_iw_o.shape) != 2:
        msg = 'Ray origin polynomial coefficients must be 2 dimensional.'
        raise ValueError(msg)
    
    if self.poly_iw_o.shape[0] != 10:
        msg = "Ray origin polynomial coefficients must be ordered in [x, y]'"
        raise ValueError(msg)
    
    if self.poly_iw_o.shape[1] != 2:
        msg = 'There must be 10 coefficients in the ray origin polynomial'
        raise ValueError(msg)

    if len(self.poly_iw_v.shape) != 2:
        msg = 'Ray direction polynomial coefficients must be 2 dimensional.'
        raise ValueError(msg)

    if self.poly_iw_v.shape[0] != 10:
        msg = "Ray direction polynomial coefficients must be ordered in [x, y]'"
        raise ValueError(msg)

    if self.poly_iw_v.shape[1] != 2:
        msg = 'There must be 10 coefficients in the ray direction polynomial'
        raise ValueError(msg)

    if self.dtype not in [np.float32, np.float64]:
        raise ValueError('Dtype is not supported for camera calibration')