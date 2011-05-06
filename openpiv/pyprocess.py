#!/usr/bin/env python

"""
This module contains a pure python implementation of the cross-correlation
algorithm for PIV image processing.
"""

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


import numpy.lib.stride_tricks
import numpy as np
import openpiv.process

def moving_window_array( array, window_size, overlap ):
    """
    This is a nice numpy trick. The concept of numpy strides should be 
    clear to understand this code.
    
    Basically, we have a 2d array and we want to perform cross-correlation
    over the interrogation windows. An approach could be to loop over the array
    but loops are expensive in python. So we create from the array a new array
    with three dimension, of size (n_windows, window_size, window_size), in which 
    each slice, (along the first axis) is an interrogation window. 
    
    """
    sz = array.itemsize
    shape = array.shape
    
    strides = (sz*shape[1]*(window_size-overlap), sz*(window_size-overlap), sz*shape[1], sz)
    shape = ( int((shape[0] - window_size)/(window_size-overlap))+1, int((shape[1] - window_size)/(window_size-overlap))+1 , window_size, window_size)
    
    return numpy.lib.stride_tricks.as_strided( array, strides=strides, shape=shape ).reshape(-1, window_size, window_size)

def piv ( frame_a, frame_b, window_size=32, overlap=16, dt=1.0, corr_method = 'fft', subpixel_method='gaussian', sig2noise_method=None, nfftx=None, nffty=None, width=2):
    """Standard PIV cross-correlation algorithm.
    
    This is a pure python implementation of the standard PIV cross-correlation
    algorithm. It is a zero order displacement predictor, and no iterative process
    is performed.
        
    Parameters
    ----------
    frame_a : 2d np.ndarray
        an two dimensions array of integers containing grey levels of 
        the first frame.
        
    frame_b : 2d np.ndarray
        an two dimensions array of integers containing grey levels of 
        the second frame.
        
    window_size : int
        the size of the (square) interrogation window, [default: 32 pix].
        
    overlap : int
        the number of pixels by which two adjacent windows overlap
        [default: 16 pix].
        
    dt : float
        the time delay separating the two frames [default: 1.0].
    
    corr_method : string
        one of the two methods implemented: 'fft' or 'direct',
        [default: 'fft'].
        
    subpixel_method : string
         one of the following methods to estimate subpixel location of the peak: 
         'centroid' [replaces default if correlation map is negative], 
         'gaussian' [default if correlation map is positive], 
         'parabolic'.
    
    sig2noise_method : string 
        defines the method of signal-to-noise-ratio measure,
        ('peak2peak' or 'peak2mean'. If None, no measure is performed.)
        
    nfftx   : int
        the size of the 2D FFT in x-direction, 
        [default: 2 x windows_a.shape[0] is recommended]
        
    nffty   : int
        the size of the 2D FFT in y-direction, 
        [default: 2 x windows_a.shape[1] is recommended]
        
    width : int
        the half size of the region around the first
        correlation peak to ignore for finding the second
        peak. [default: 2]. Only used if ``sig2noise_method==peak2peak``.
    
    
    Returns
    -------
    u : 2d np.ndarray
        a two dimensional array containing the u velocity component,
        in pixels/seconds.
        
    v : 2d np.ndarray
        a two dimensional array containing the v velocity component,
        in pixels/seconds.
    
    sig2noise : 2d np.ndarray, ( optional: only if sig2noise_method is not None )
        a two dimensional array the signal to noise ratio for each 
        window pair.
        
    """
    
    # transform the arrray into a more covenient form for looping
    windows_a = moving_window_array( frame_a, window_size, overlap )
    windows_b = moving_window_array( frame_b, window_size, overlap )
    
    # get shape of the output so that we can preallocate 
    # memory for velocity array
    n_rows, n_cols = openpiv.process.get_field_shape( image_size=frame_a.shape, window_size=window_size, overlap=overlap )
    
    u = np.empty(n_rows*n_cols)
    v = np.empty(n_rows*n_cols)
    
    # if we want sig2noise information, allocate memory
    if sig2noise_method:
        sig2noise = np.empty(n_rows*n_cols)
    
    # for each interrogation window
    for i in range(windows_a.shape[0]):
        # get correlation window
        corr = openpiv.process.correlate_windows( windows_a[i], windows_b[i], corr_method = corr_method, nfftx=nfftx, nffty=nffty )
        
        # get subpixel approximation for peak position row and column index
        row, col = openpiv.process.find_subpixel_peak_position( corr, subpixel_method=subpixel_method)
        
        # get displacements
        u[i], v[i] = -(col - corr.shape[1]/2), (row - corr.shape[0]/2)
        
        # get signal to noise ratio
        if sig2noise_method:
            sig2noise[i] = openpiv.process.sig2noise_ratio( corr, sig2noise_method=sig2noise_method, width=width )
    
    # return output depending if user wanted sig2noise information
    if sig2noise_method:
        return u.reshape(n_rows, n_cols)/dt, v.reshape(n_rows, n_cols)/dt, sig2noise.reshape(n_rows, n_cols)
    else:
        return u.reshape(n_rows, n_cols)/dt, v.reshape(n_rows, n_cols)/dt
