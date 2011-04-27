#!/usr/bin/python

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

import numpy as np
import scipy.signal
from openpiv.lib import replace_nans
  
    
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
    uf = scipy.signal.convolve( u, g, mode='same')
    vf = scipy.signal.convolve( v, g, mode='same')
    return uf, vf
    
def replace_outliers( u, v, method='localmean', n_iter=5, kernel_size=1):
    """Replace nans in an velocity field using an iterative image inpainting algorithm.
    
    The algorithm is the following:
    
    1) For each element in the arrays of the ``u`` and ``v`` components, replace it by a weighted average
       of the neighbouring elements which are not nan. The weights depends
       of the method type. If ``method=localmean`` weight are equal to 1/( (2*kernel_size+1)**2 -1 )
       
    2) Several iterations are needed if there are adjacent nan elements.
       If this is the case, inforation is "spread" from the edges of the missing
       regions iteratively, until the variation is below a certain threshold. 
    
    Parameters
    ----------
    
    u : 2d np.ndarray
        the u velocity component field
        
    v : 2d np.ndarray
        the v velocity component field
        
    n_iter : int
        the number of iterations
    
    kernel_size : int
        the size of the kernel, default is 1
        
    method : str
        the method used to replace nans. Valid options are
        `localmean`.
        
    Returns
    -------
    uf : 2d np.ndarray
        the smoothed u velocity component field, where nans have been replaced
        
    vf : 2d np.ndarray
        the smoothed v velocity component field, where nans have been replaced    
        
    """
    if method == 'localmean':
        u = replace_nans( u, method=method, n_iter=n_iter, kernel_size=kernel_size )
        v = replace_nans( v, method=method, n_iter=n_iter, kernel_size=kernel_size )
    else:
        raise ValueError( 'method not valid. Should be one of `localmean`.')
    
    return u, v
    
    
    
