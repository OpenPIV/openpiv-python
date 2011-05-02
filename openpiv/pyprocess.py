#!/usr/bin/env python

"""
This module contains a pure python implementation of the cross-correlation
algorithm for PIV image processing. It also contains some useful helper functions.
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


import numpy as np
import numpy.ma as ma
from numpy.fft import rfft2,irfft2,fftshift
import numpy.lib.stride_tricks
from scipy import signal
from math import log
import pdb




def get_coordinates( image_size, window_size, overlap ):
    """Compute the x, y coordinates of the centers of the interrogation windows.
    
    Parameters
    ----------
    
    image_size: two elements tuple
        a two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is 
        the number of columns
        
    window_size: int
        the final size of the interrogation window
        
    overlap: int
        the number of pixel by which two adjacent interrogation
        window overlap.
        
        
    Returns
    -------
    x : 2d np.ndarray
        a two dimensional array containing the x coordinates of the 
        interrogation window centers, in pixels.
        
    y : 2d np.ndarray
        a two dimensional array containing the y coordinates of the 
        interrogation window centers, in pixels.
        
    """

    # get shape of the resulting flow field
    field_shape = get_field_shape( image_size, window_size, overlap )

    # compute grid coordinates of the interrogation window centers
    x = np.arange( field_shape[1] )*(window_size-overlap) + (window_size-1)/2.0
    y = np.arange( field_shape[0] )*(window_size-overlap) + (window_size-1)/2.0
    
    return np.meshgrid(x,y[::-1])

def get_field_shape ( image_size, window_size, overlap ):
    """Compute the shape of the resulting flow field.
    
    Given the image size, the interrogation window size and
    the overlap size, it is possible to calcualte the number 
    of rows and columns of the resulting flow field.
    
    Parameters
    ----------
    image_size: two elements tuple
        a two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is 
        the number of columns
        
    window_size: int
        the final size of the interrogation window
        
    overlap: int
        the number of pixel by which two adjacent interrogation
        window overlap.
        
        
    Returns
    -------
    field_shape : two elements tuple
        the shape of the resulting flow field
        
    
    """
    
    return ( (image_size[0] - window_size)//(window_size-overlap)+1, 
             (image_size[1] - window_size)//(window_size-overlap)+1 )

def moving_window_array( array, window_size, overlap ):
    """
    This is a nice numpy trick. The concept of numpy strides should be 
    clear to understand this code.
    
    Basically, we have a 2d array and we want to perform cross-correlation
    over the interrogation windows. An approach could be to loop over the array
    but loops are expensive in python. So we create from the array a new array
    with three dimension, of size (n_windows, window_size, window_size), in which 
    each slice, (along the first axis) is an interrogation window. 
    
    To understand what i mean run this code in ipython:
    
    >>> a = np.arange(100, dtype=np.int).reshape(10,10)
        
    >>> a
         array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
                [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
                [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
                [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
                [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
                [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]])
                
                
    >>> b = moving_window_array( a, 5, 1 )
    
    >>>  moving_window_array(a,6,2)

    array([[[ 0,  1,  2,  3,  4,  5],
            [10, 11, 12, 13, 14, 15],
            [20, 21, 22, 23, 24, 25],
            [30, 31, 32, 33, 34, 35],
            [40, 41, 42, 43, 44, 45],
            [50, 51, 52, 53, 54, 55]],

           [[ 4,  5,  6,  7,  8,  9],
            [14, 15, 16, 17, 18, 19],
            [24, 25, 26, 27, 28, 29],
            [34, 35, 36, 37, 38, 39],
            [44, 45, 46, 47, 48, 49],
            [54, 55, 56, 57, 58, 59]],
    
           [[40, 41, 42, 43, 44, 45],
            [50, 51, 52, 53, 54, 55],
            [60, 61, 62, 63, 64, 65],
            [70, 71, 72, 73, 74, 75],
            [80, 81, 82, 83, 84, 85],
            [90, 91, 92, 93, 94, 95]],
    
           [[44, 45, 46, 47, 48, 49],
            [54, 55, 56, 57, 58, 59],
            [64, 65, 66, 67, 68, 69],
            [74, 75, 76, 77, 78, 79],
            [84, 85, 86, 87, 88, 89],
            [94, 95, 96, 97, 98, 99]]])


   
    """
    sz = array.itemsize
    shape = array.shape
    
    strides = (sz*shape[1]*(window_size-overlap), sz*(window_size-overlap), sz*shape[1], sz)
    shape = ( int((shape[0] - window_size)/(window_size-overlap))+1, int((shape[1] - window_size)/(window_size-overlap))+1 , window_size, window_size)
    
    return numpy.lib.stride_tricks.as_strided( array, strides=strides, shape=shape ).reshape(-1, window_size, window_size)

def find_first_peak ( corr ):
    """
    Find row and column indices of the first correlation peak.
    
    Parameters
    ----------
    corr : np.ndarray
        the correlation map
    """    
    i = corr.argmax() // corr.shape[1] 
    j = corr.argmax() %  corr.shape[1]
    
    return i, j, corr.max()
    
def find_second_peak ( corr, i=None, j=None ):
    """
    Find the value of the second largest peak
    
    Parameters
    ----------
    corr: np.ndarray
          the correlation map
    i,j : ints
          row and column location of the first peak [optional]
    """
    
    if i is None or j is None:
        i,j,tmp = find_first_peak( corr )
        
    # create a masked view of the corr
    tmp = corr.view(ma.MaskedArray)
    
    # set 3x3 square submatrix around the first correlation peak as masked 
    tmp[ i-1:i+2,j-1:j+2 ] = ma.masked
    i,j,corr_max2 = find_first_peak( tmp )
    
    return i,j,corr_max2
    
    
def find_pixel_peak_position( corr, sig2noise_method = 'peak2peak', sig2noise_lim = 1.0 ):
    """
    Find pixel approximation of the correlation peak.
    
    This function returns a subpixels approximation of the correlation
    peak by fitting a two dimensional Gaussian curve in a 3x3 square region
    around the correlation peak.
    
    Parameters
    ----------
    corr : np.ndarray
        the correlation map
    
    sig2noise_method : string 
        defines the method of signal-to-noise-ratio measure, ('peak2peak' or 'peak2mean')
        
    sig2noise_lim : float
        the minimum signal/noise ratio that can be accepted
        
    default_peak_position : two elements tuple
        the default peak position, if something
        goes wrong in the peak detection algorithm.
        Default is (corr.shape[0]/2,corr.shape[1]/2)
        
    Returns
    -------
    
    peak_position : two elements tuple
        the sub-pixel approximation of the correlation peak
        
    sig2noise : float 
        the signal to noise ratio

    """

    # initialization
    default_peak_position=(corr.shape[0]/2,corr.shape[1]/2)
        
    # find first peak
    peak1_i, peak1_j, corr_max1 = find_first_peak( corr )
    peak_position = (peak1_i, peak1_j)
    
    if sig2noise_method == 'peak2peak':
        # find second peak height
        # if it's an empty interrogation window 
        # if the image is lacking particles, totally black it will correlate to very low value, but not zero
        # if the first peak is on the borders, the correlation map is also wrong
        peak2_i, peak2_j, corr_max2 = find_second_peak( corr , peak1_i, peak1_j )
        if  corr_max1 <  1e-3 or (peak1_i == 0 or peak1_j == corr.shape[0] or peak1_j == 0 or peak1_j == corr.shape[1] or 
                                  peak2_i == 0 or peak2_j == corr.shape[0] or peak2_j == 0 or peak2_j == corr.shape[1]): 
            peak_position = default_peak_position
            sig2noise = np.inf
            return peak_position[0], peak_position[1], sig2noise
    elif sig2noise_method == 'peak2mean':
            # find mean of the correlation map
        corr_max2 = corr.mean()
    else:
        raise ValueError('wrong sig2noise_method')


    # avoid dividing by zero
    try:
        sig2noise = corr_max1/corr_max2
    except ValueError:
        sig2noise = np.inf
        
    # if signal/noise ratio is higher than a certain limit
    if sig2noise < sig2noise_lim :
        peak_position = default_peak_position


    return peak_position[0], peak_position[1], sig2noise
    

def find_subpixel_peak_position( corr, peak_indices, subpixel_method = 'gaussian' ):
    """
    Find subpixel approximation of the correlation peak.
    
    This function returns a subpixels approximation of the correlation
    peak by fitting a two dimensional Gaussian curve in a 3x3 square region
    around the correlation peak.
    
    Parameters
    ----------
    corr : np.ndarray
        the correlation map
        
    peak_indices: two elements tuple
        the row and column indices of the first correlation peak
        
    subpixel_method : string
         one of the following methods to estimate subpixel location of the peak: 
         'centroid' [replaces default if correlation map is negative], 
         'gaussian' [default if correlation map is positive], 
         'parabolic'
        
    Returns
    -------
    
    subp_peak_position : two elements tuple
        the sub-pixel approximation of the correlation peak
    """
    
    # initialization
    default_peak_position=(corr.shape[0]/2,corr.shape[1]/2)

    # the peak locations
    peak1_i, peak1_j = peak_indices
    
    # the peak and its neighbours: left, right, down, up
    c = corr[peak1_i,   peak1_j]
    cl = corr[peak1_i-1, peak1_j]
    cr = corr[peak1_i+1, peak1_j]
    cd = corr[peak1_i,   peak1_j-1] 
    cu = corr[peak1_i,   peak1_j+1]
    
    
    # gaussian fit
    if np.any ( [c,cl,cr,cd,cu] < 0 ) and subpixel_method == 'gaussian':
        subpixel_method = 'centroid'
    
    
    try: 
         if subpixel_method == 'centroid':
             subp_peak_position = (((peak1_i-1)*cl+peak1_i*c+(peak1_i+1)*cr)/(cl+c+cr),
                                   ((peak1_j-1)*cd+peak1_j*c+(peak1_j+1)*cu)/(cd+c+cu))
    
         elif subpixel_method == 'gaussian':
             subp_peak_position = (peak1_i + ( (log(cl)-log(cr) )/( 2*log(cl) - 4*log(c) + 2*log(cr) )),
                                  peak1_j + ( (log(cd)-log(cu) )/( 2*log(cd) - 4*log(c) + 2*log(cu) ))) 
      
         elif subpixel_method == 'parabolic':
              subp_peak_position = (peak1_i +  (cl-cr)/(2*cl-4*c+2*cr),
                                    peak1_j +  (cd-cu)/(2*cd-4*c+2*cu)) 

    except: 
        subp_peak_position = default_peak_position

    return subp_peak_position[0], subp_peak_position[1]
    
def xcorrf2(a,b,nfftx,nffty):
    """ XCORRF2 computes 2D cross correlation using FFT method 
    
    Parameters
    ----------
    a,b : 2D np.ndarrays of the same size
    
    nfftx, nffty : size of the output correlation map
    """
    # return signal.correlate2d(a,b,'full')
    return 
    

def correlate_windows( window_a, window_b, corr_method = 'fft', nfftx = None, nffty = None ):
    """Compute correlation function between two interrogation windows.
    
    Use the correlation theorem to speed up the computation of the 
    correlation function.
    
    Parameters
    ----------
    window_a : 2d np.ndarray
        a two dimensions array for the first interrogation window
        
    window_b : 2d np.ndarray
        a two dimensions array for the second interrogation window
        
    corr_method   : string
        one of the two methods currently implemented: 'fft' or 'direct'
        
    nfftx   : int
        a size of the 2D FFT in x-direction, 
        [default: 2 x windows_a.shape[0] is recommended]
        
    nffty   : int
        a size of the 2D FFT in y-direction, 
        [default: 2 x windows_a.shape[1] is recommended]
        
        
    Returns
    -------
    corr : 2d np.ndarray
        a two dimensions array with the correlation function
    
    """
    
    if corr_method == 'fft':
        if nfftx is None:
            nfftx = 2*window_a.shape[0]
        if nffty is None:
            nffty = 2*window_a.shape[1]
        
        return fftshift(irfft2(rfft2(normalize_intensity(window_a),shape=(nfftx,nffty))*np.conj(rfft2(normalize_intensity(window_b),shape=(nfftx,nffty)))).real, axes=(0,1)  )
    elif corr_method == 'direct':
        return signal.convolve(normalize_intensity(window_a), normalize_intensity(window_b[::-1,::-1]), 'full')
    else:
        raise ValueError('method is not implemented')
    
    
def normalize_intensity( window ):
    """Remove mean value from window and masks negative, dark pixels """
     
    return window - window.mean()

def piv ( frame_a, frame_b, window_size=32, overlap=16, dt=1.0, corr_method = 'fft', sig2noise_method = 'peak2peak', sig2noise_lim=1.0):
    """Basic python implementation of the PIV cross-correlation
    algorithm.
        
    Parameters
    ----------
    frame_a : 2d np.ndarray
        an two dimensions array of integers containing grey levels of 
        the first frame.
        
    frame_b : 2d np.ndarray
        an two dimensions array of integers containing grey levels of 
        the second frame.
        
    window_size : int
        the size of the (square) interrogation window 
        [default: 32 pix]
        
    overlap : int
        the number of pixels by which two adjacent windows overlap
        [default: 16 pix]
        
    dt : float
        the time delay separating the two frames [default: 1.0]
    
    corr_method : string
        one of the two methods implemented: 'fft' or 'direct'
    
    sig2noise_method : string 
        defines the method of signal-to-noise-ratio measure, 'peak2peak' (default) or 'peak2mean'
    
    sig2noise_lim: float 
        the limit of the signal to noise ratio, 1.0 [default] to 
        ignore this limit
        
    nfftx   : int
        the size of the 2D FFT in x-direction, 
        [default: 2 x windows_a.shape[0] is recommended]
        
    nffty   : int
        the size of the 2D FFT in y-direction, 
        [default: 2 x windows_a.shape[1] is recommended]
    
    
    Returns
    -------
    u : 2d np.ndarray
        a two dimensional array containing the u velocity component,
        in pixels/seconds.
        
    v : 2d np.ndarray
        a two dimensional array containing the v velocity component,
        in pixels/seconds.
    """
    
    # transform the arrray into a more covenient form for looping
    windows_a = moving_window_array( frame_a, window_size, overlap )
    windows_b = moving_window_array( frame_b, window_size, overlap )
    
    # get shape of the output so that we can preallocate memory
    n_rows, n_cols = get_field_shape( image_size=frame_a.shape, window_size=window_size, overlap=overlap )
    
    u = np.empty(n_rows*n_cols)
    v = np.empty(n_rows*n_cols)
    
    
    # for each interrogation window
    for i in range(windows_a.shape[0]):
        print i
        # get correlation window
        corr = correlate_windows( windows_a[i], windows_b[i], corr_method = corr_method, nfftx=window_size*2, nffty=window_size*2 )
        
        # get pixel approximation for peak position row and column index
        row, col, sig2noise = find_pixel_peak_position( corr, sig2noise_method = sig2noise_method, sig2noise_lim = sig2noise_lim)
        
        # get subpixel approximation for peak position row and column index
        if sig2noise < sig2noise_lim:
            u[i], v[i] = 0.0, 0.0
        else:
            row, col = find_subpixel_peak_position( corr, (row, col) )
            u[i], v[i] = -(col - corr.shape[1]/2), (row - corr.shape[0]/2)
    
    return (u.reshape(n_rows, n_cols)/dt, v.reshape(n_rows, n_cols)/dt)
