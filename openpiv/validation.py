#!/usr/bin/python

"""A module for spurious vector detection."""

__licence__ = """
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


def global_val( u, v, u_thresholds, v_thresholds ):
    """Eliminate spurious vectors with a global threshold.
    
    This validation method tests for the spatial consistency of the data
    and outliers vector are replaced with zero if at least one of the 
    two velocity components is out of a specified global range.
    
    Parameters
    ----------
    u : 2d np.ndarray
        a two dimensional array containing the u velocity component.
        
    v : 2d np.ndarray
        a two dimensional array containing the v velocity component.
        
    u_thresholds: two elements tuple
        u_thresholds = (u_min, u_max). If ``u<u_min`` or ``u>u_max``
        the vector is treated as an outlier.
        
    v_thresholds: two elements tuple
        ``v_thresholds = (v_min, v_max)``. If ``v<v_min`` or ``v>v_max``
        the vector is treated as an outlier.
        
    Returns
    -------
    u : 2d np.ndarray
        a two dimensional array containing the u velocity component, 
        where spurious vectors have been replaced by zero
        
    v : 2d np.ndarray
        a two dimensional array containing the v velocity component, 
        where spurious vectors have been replaced by zero
        
    mask : boolean 2d np.ndarray 
        a boolean array. True elements corresponds to outliers.
        
    """
    
    ind = u < u_thresholds[0] or u > u_thresholds[1] or v < v_thresholds[0] or v > v_thresholds[1]
    u[ind] = 0.0
    v[ind] = 0.0
    
    mask = np.zeros(u.shape, dtype=bool)
    mask[ind] = True
    
    return u, v, mask
    
def global_std( u, v, std_threshold=3 ):
    """Eliminate spurious vectors with a global threshold defined by the standard deviation
    
    This validation method tests for the spatial consistency of the data
    and outliers vector are replaced with zero if at least one of the 
    two velocity components is out of a specified global range.
    
    Parameters
    ----------
    u : 2d np.ndarray
        a two dimensional array containing the u velocity component.
        
    v : 2d np.ndarray
        a two dimensional array containing the v velocity component.
        
    std_threshold: float
        If the length of the vector (actually the sum of squared components) is 
        larger than std_threshold times standard deviation of the flow field, 
        then the vector is treated as an outlier. [default = 3]
        
        
    Returns
    -------
    u : 2d np.ndarray
        a two dimensional array containing the u velocity component, 
        where spurious vectors have been replaced by zero
        
    v : 2d np.ndarray
        a two dimensional array containing the v velocity component, 
        where spurious vectors have been replaced by zero
        
    mask : boolean 2d np.ndarray 
        a boolean array. True elements corresponds to outliers.
        
    """
    
    vel_magnitude = u**2 + v**2
    ind = vel_magnitude > std_threshold*np.std(vel_magnitude)
    
    u[ind] = 0.0
    v[ind] = 0.0
    
    mask = np.zeros(u.shape, dtype=bool)
    mask[ind] = True
    
    return u, v, mask

def sig2noise_val( u, v, sig2noise, threshold=1.3):
    """Eliminate spurious vectors from cross-correlation signal to noise ratio.
    
    Replace spurious vectors with zero if signal to noise ratio
    is below a specified threshold.
    
    Parameters
    ----------
    u : 2d np.ndarray
        a two dimensional array containing the u velocity component.
        
    v : 2d np.ndarray
        a two dimensional array containing the v velocity component.
        
    sig2noise : 2d np.ndarray
        a two dimensional array containing the value  of the signal to 
        noise ratio from cross-correlation function.
        
    threshold: float
        the signal to noise ratio threshold value.
        
    Returns
    -------
    u : 2d np.ndarray
        a two dimensional array containing the u velocity component, 
        where spurious vectors have been replaced by zero
        
    v : 2d np.ndarray
        a two dimensional array containing the v velocity component, 
        where spurious vectors have been replaced by zero
        
    mask : boolean 2d np.ndarray 
        a boolean array. True elements corresponds to outliers.
    
    References
    ----------
    R. D. Keane and R. J. Adrian, Measurement Science & Technology,1990, 1, 1202-1215.
    
    """
    
    ind = sig2noise < threshold
    u[ind] = 0.0
    v[ind] = 0.0
    
    mask = np.zeros(u.shape, dtype=bool)
    mask[ind] = True
    
    return u, v, mask
    
