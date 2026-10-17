import numpy as np
from os.path import join
from typing import Tuple


def _save_parameters(
    self,
    file_path: str,
    file_name: str=None
):
    """Save polynomial camera parameters.
    
    Save the polynomial camera parameters to a text file.
    
    Parameters
    ----------
    file_path : str
        File path where the camera parameters are saved.
    file_name : str, optional
        If specified, override the default file name.
        
    Returns
    -------
    None
    
    """
    if file_name is None:
        file_name = self.name
    
    full_path = join(file_path, file_name)
    
    with open(full_path, 'w') as f:
        f.write(self.name + '\n')
        
        _r = ''
        for i in range(2):
            _r += str(self.resolution[i]) + ' '
            
        f.write(_r + '\n')
        
        for i in range(19):
            _d2 = ''
            for j in range(2):
                _d2 += str(self.poly_wi[i, j]) + ' '
                
            f.write(_d2 + '\n')
            
        for i in range(10):
            _d2 = ''
            for j in range(2):
                _d2 += str(self.poly_iw_o[i, j]) + ' '

            f.write(_d2 + '\n')

        for i in range(10):
            _d2 = ''
            for j in range(2):
                _d2 += str(self.poly_iw_v[i, j]) + ' '
                
            f.write(_d2 + '\n')
        
        f.write(self.dtype + '\n')
        

def _load_parameters(
    self,
    file_path: str,
    file_name: str
):
    """Load polynomial camera parameters.
    
    Load the polynomial camera parameters from a text file.
    
    Parameters
    ----------
    file_path : str
        File path where the camera parameters are saved.
    file_name : str
        Name of the file that contains the camera parameters.
    
    """
    full_path = join(file_path, file_name)
    
    with open(full_path, 'r') as f:
        
        name = f.readline()[:-1]
        
        _r = f.readline()[:-2]
        resolution = np.array([float(s) for s in _r.split()])
            
        poly_wi = []
        for i in range(19):
            _d2 = f.readline()[:-2]
            poly_wi.append(np.array([float(s) for s in _d2.split()]))
        
        poly_iw_o = []
        for i in range(10):
            _d2 = f.readline()[:-2]
            poly_iw_o.append(np.array([float(s) for s in _d2.split()]))
        if any(len(row) != 2 for row in poly_iw_o):
            raise ValueError("Old poly_iw file: recalibrate to obtain volume ray coefficients")

        poly_iw_v = []
        for i in range(10):
            _d2 = f.readline()[:-2]
            poly_iw_v.append(np.array([float(s) for s in _d2.split()]))
        
        dtype = f.readline()[:-1]
        
        poly_wi = np.array(poly_wi, dtype=dtype)
        poly_iw_o = np.array(poly_iw_o, dtype=dtype)
        poly_iw_v = np.array(poly_iw_v, dtype=dtype)

    self.name = name
    self.resolution = resolution
    self.poly_wi = poly_wi
    self.poly_iw_o = poly_iw_o
    self.poly_iw_v = poly_iw_v
    self.dtype = dtype