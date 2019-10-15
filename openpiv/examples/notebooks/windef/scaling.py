#!/usr/bin/env python
"""Scaling utilities
"""

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


def uniform( x, y, u, v, scaling_factor ):
    """
    Apply an uniform scaling
    
    Parameters
    ----------
    x : 2d np.ndarray
    
    y : 2d np.ndarray
    
    u : 2d np.ndarray
    
    v : 2d np.ndarray
    
    scaling_factor : float
        the image scaling factor in pixels per meter
    
    Return
    ----------
    x : 2d np.ndarray
    
    y : 2d np.ndarray
    
    u : 2d np.ndarray
    
    v : 2d np.ndarray
        
    """
    return x/scaling_factor, y/scaling_factor, u/scaling_factor, v/scaling_factor
