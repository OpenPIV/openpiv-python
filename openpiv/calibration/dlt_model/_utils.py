import numpy as np
from os.path import join
from typing import Tuple

from .. import _cal_doc_utils


def _save_parameters(
    self,
    file_path: str,
    file_name: str=None
):
    """Save DLT camera parameters.
    
    Save the DLT camera parameters to a text file.
    
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
        
        for i in range(3):
            _c = ''
            for j in range(self.coeffs.shape[1]):
                _c += str(self.coeffs[i, j]) + ' '
                
            f.write(_c + '\n')
        
        f.write(self.dtype + '\n')
                

@_cal_doc_utils.docfiller
def _load_parameters(
    self,
    file_path: str,
    file_name: str
):
    """Load DLT camera parameters.
    
    Load the DLT camera parameters from a text file.
    
    Parameters
    ----------
    file_path : str
        File path where the camera parameters are saved.
    file_name : str
        Name of the file that contains the camera parameters.
        
    Returns
    -------
    None
    
    """
    full_path = join(file_path, file_name)
    
    with open(full_path, 'r') as f:
        
        name = f.readline()[:-1]
        
        _r = f.readline()[:-2]
        resolution = np.array([float(s) for s in _r.split()])
            
        coefficients = []
        for i in range(3):
            _c = f.readline()[:-2]
            coefficients.append(np.array([float(s) for s in _c.split()]))
                    
        dtype = f.readline()[:-1]
        
        coefficients = np.array(coefficients, dtype = dtype)
        
    self.name = name
    self.resolution = resolution
    self.coeffs = coefficients
    self.dtype = dtype