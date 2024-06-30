import numpy as np
from os.path import join
from typing import Tuple

from .. import _cal_doc_utils


def _get_rotation_matrix(self):
    """Calculate a rotation matrix for a camera.
    
    Calculate a rotation matrix for a camera. The matrix is a 3x3 numpy ndarray
    like such:
    
    [ r1 r2 r3 ]
    [ r4 r5 r6 ]
    [ r7 r8 r9 ]
    
    where
    
    r1 = cos(tz) * cos(ty)
    r2 = -sin(tz) * cos(ty)
    r3 = sin(ty)
    r4 = cos(tz) * sin(tx) * sin(ty) + sin(tz) * cos(tx)
    r5 = cos(tz) * cos(tx) - sin(tz) * sin(tx) * sin(ty)
    r6 = -sin(tx) * cos(ty)
    r7 = sin(tz) * sin(tx) - cos(tz) * cos(tx) * sin(ty)
    r8 = sin(tz) * cos(tx) * sin(ty) + cos(tz) * sin(tx)
    r9 = cos(tx) * cos(ty)
    
    """
    self._check_parameters()
    
    # Orientation is composed of angles, or theta, for each axes.
    # Theta for each dimensions is abbreviated as t<axis>.
    tx, ty, tz = self.orientation
    dtype = self.dtype
    
    # We compute the camera patrix based off of this website.
    # https://support.pix4d.com/hc/en-us/articles/202559089-How-are-the-Internal-and-External-Camera-Parameters-defined
    
    rot_x = np.array(
        [
            [1,        0,          0],
            [0, np.cos(tx),-np.sin(tx)],
            [0, np.sin(tx), np.cos(tx)]
        ],
        dtype=dtype
    )
    
    rot_y = np.array(
        [
            [ np.cos(ty), 0, np.sin(ty)],
            [        0,   1,        0],
            [-np.sin(ty), 0, np.cos(ty)]
        ], 
        dtype=dtype
    )
    
    rot_z = np.array(
        [
            [np.cos(tz),-np.sin(tz), 0],
            [np.sin(tz), np.cos(tz), 0],
            [       0,          0,   1]
        ], 
        dtype=dtype
    )
    
    rotation_matrix = np.dot(
        np.dot(rot_x, rot_y), 
        rot_z
    )
    
    self.rotation = rotation_matrix


@_cal_doc_utils.docfiller
def _save_parameters(
    self,
    file_path: str,
    file_name: str=None
):
    """Save pinhole camera parameters.
    
    Save the pinhole camera parameters to a text file.
    
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
    self._check_parameters()
    
    if file_name is None:
        file_name = self.name
    
    full_path = join(file_path, file_name)
    
    with open(full_path, 'w') as f:
        f.write(self.name + '\n')
        
        _r = ''
        for i in range(2):
            _r += str(self.resolution[i]) + ' '
            
        f.write(_r + '\n')
        
        _t = ''
        for i in range(3):
            _t += str(self.translation[i]) + ' '
        
        f.write(_t + '\n')
        
        _o = ''
        for i in range(3):
            _o += str(self.orientation[i]) + ' '
        
        f.write(_o + '\n')
        
        f.write(self.distortion_model + '\n')
        
        _d1 = ''
        for i in range(8):
            _d1 += str(self.distortion1[i]) + ' '
        
        f.write(_d1 + '\n')
        
        for i in range(4):
            _d2 = ''
            for j in range(6):
                _d2 += str(self.distortion2[i, j]) + ' '
                
            f.write(_d2 + '\n')
            
        _f = ''
        for i in range(2):
            _f += str(self.focal[i]) + ' '
            
        f.write(_f + '\n')
        
        _p = ''
        for i in range(2):
            _p += str(self.principal[i]) + ' '
            
        f.write(_p + '\n')
        
        f.write(self.dtype + '\n')
        

@_cal_doc_utils.docfiller
def _load_parameters(
    self,
    file_path: str,
    file_name: str
):
    """Load pinhole camera parameters.
    
    Load the pinhole camera parameters from a text file.
    
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
        
        _t = f.readline()[:-2]
        translation = np.array([float(s) for s in _t.split()])
        
        _o = f.readline()[:-2]
        orientation = np.array([float(s) for s in _o.split()])
        
        distortion_model = f.readline()[:-1]
        
        _d1 = f.readline()[:-2]
        distortion1 = np.array([float(s) for s in _d1.split()])
        
        distortion2 = []
        for i in range(4):
            _d2 = f.readline()[:-2]
            distortion2.append(np.array([float(s) for s in _d2.split()]))
            
        distortion2 = np.array(distortion2, dtype = "float64")
        
        _f = f.readline()[:-2]
        focal = np.array([float(s) for s in _f.split()])
        
        _p = f.readline()[:-2]
        principal = np.array([float(s) for s in _p.split()])
        
        dtype = f.readline()[:-1]

    self.name = name
    self.resolution = resolution
    self.translation = translation
    self.orientation = orientation
    self.distortion_model = distortion_model
    self.distortion1 = distortion1
    self.distortion2 = distortion2
    self.focal = focal
    self.principal = principal
    self.dtype = dtype
    
    self._get_rotation_matrix()