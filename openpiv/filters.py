"""The openpiv.filters module contains some filtering/smoothing routines."""

__licence_ = """
Copyright (C) 2011  www.openpiv.net

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from openpiv.lib import replace_nans
import numpy as np
from scipy.signal import convolve
  
    
def _gaussian_kernel( size ):
    """A normalized 2D Gaussian kernel array
    
    Parameters
    ----------
    size : int
        the half width of the kernel. Kernel
        has shape 2*size+1
        
    Examples
    --------
    
    >>> from openpiv.filters import _gaussian_kernel
    >>> _gaussian_kernel(1)
    array([[ 0.04491922,  0.12210311,  0.04491922],
       [ 0.12210311,  0.33191066,  0.12210311],
       [ 0.04491922,  0.12210311,  0.04491922]])
   
    
    """
    size = int(size)
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp(-(x**2/float(size)+y**2/float(size)))
    return g / g.sum()



def gaussian( u, v, size) :
    """Smooths the velocity field with a Gaussian kernel.
    
    Parameters
    ----------
    u : 2d np.ndarray
        the u velocity component field
        
    v : 2d np.ndarray
        the v velocity component field
        
    size : int
        the half width of the kernel. Kernel
        has shape 2*size+1
        
    Returns
    -------
    uf : 2d np.ndarray
        the smoothed u velocity component field
        
    vf : 2d np.ndarray
        the smoothed v velocity component field    
        
    """
    g = _gaussian_kernel( size=size )
    uf = convolve( u, g, mode='same')
    vf = convolve( v, g, mode='same')
    return uf, vf
    
def replace_outliers( u, v, method='localmean', max_iter=5, tol=1e-3, kernel_size=1):
    """Replace invalid vectors in an velocity field using an iterative image inpainting algorithm.
    
    The algorithm is the following:
    
    1) For each element in the arrays of the ``u`` and ``v`` components, replace it by a weighted average
       of the neighbouring elements which are not invalid themselves. The weights depends
       of the method type. If ``method=localmean`` weight are equal to 1/( (2*kernel_size+1)**2 -1 )
       
    2) Several iterations are needed if there are adjacent invalid elements.
       If this is the case, inforation is "spread" from the edges of the missing
       regions iteratively, until the variation is below a certain threshold. 
    
    Parameters
    ----------
    
    u : 2d np.ndarray
        the u velocity component field
        
    v : 2d np.ndarray
        the v velocity component field
        
    max_iter : int
        the number of iterations
    fil
    kernel_size : int
        the size of the kernel, default is 1
        
    method : str
        the type of kernel used for repairing missing vectors
        
    Returns
    -------
    uf : 2d np.ndarray
        the smoothed u velocity component field, where invalid vectors have been replaced
        
    vf : 2d np.ndarray
        the smoothed v velocity component field, where invalid vectors have been replaced    
        
    """
    
    u = replace_nans( u, method=method, max_iter=max_iter, tol=tol, kernel_size=kernel_size )
    v = replace_nans( v, method=method, max_iter=max_iter, tol=tol, kernel_size=kernel_size )
    
    return u, v
