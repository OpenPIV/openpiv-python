"""A cython module for fast advanced algorithms for PIV image analysis."""

import numpy as np
import openpiv.pyprocess

cimport numpy as np
cimport cython

DTYPEi = np.int64
ctypedef np.int64_t DTYPEi_t

DTYPEf = np.float64
ctypedef np.float64_t DTYPEf_t

def extended_search_area_piv( np.ndarray[DTYPEi_t, ndim=2] frame_a, np.ndarray[DTYPEi_t, ndim=2] frame_b, int window_size, int overlap, float dt, int search_area_size, int nfftx=0, int nffty=0):
    """
    The implementation of the one-step direct correlation with different 
    size of the interrogation window and the search area. The increased
    size of the search areas cope with the problem of loss of pairs due
    to in-plane motion, allowing for a smaller interrogation window size,
    without increasing the number of outlier vectors.
    
    See:
    
    Particle-Imaging Techniques for Experimental Fluid Mechanics

    Annual Review of Fluid Mechanics
    Vol. 23: 261-304 (Volume publication date January 1991)
    DOI: 10.1146/annurev.fl.23.010191.001401    
    
    Parameters
    ----------
    frame_a : 2d np.ndarray
        an two dimensions array of integers containing grey levels of 
        the first frame.
        
    frame_b : 2d np.ndarray
        an two dimensions array of integers containing grey levels of 
        the second frame.
        
    window_size : int
        the size of the (square) interrogation window.
        
    overlap : int
        the number of pixels by which two adjacent windows overlap.
        
    dt : float
        the time delay separating the two frames.
    
    search_area_size : int
        the size of the (square) interrogation window from the second frame
    
    nfftx   : int
        the size of the 2D FFT in x-direction, 
        [default: 2 x search_area_size is recommended].
        
    nffty   : int
        the size of the 2D FFT in y-direction, 
        [default: 2 x search_area_size is recommended].
    
    
    Returns
    -------
    u : 2d np.ndarray
        a two dimensional array containing the u velocity component,
        in pixels/seconds.
        
    v : 2d np.ndarray
        a two dimensional array containing the v velocity component,
        in pixels/seconds.
        
    sig2noise : 2d np.ndarray
        a two dimensional array containing the signal to noise ratio
        from the cross correlation function.
        
    Examples
    --------
    
    >>> u, v, sn = openpiv.lib.extended_search_area_piv( frame_a, frame_b, window_size=16, overlap=8, search_area_size=48, dt=0.1)
        
        
    """
    
    cdef int i, j, k, l, I, J
    cdef float i_peak, j_peak
    cdef float s2n
    
    cdef int n_cols, n_rows
    
    n_rows, n_cols = openpiv.pyprocess.get_field_shape( (frame_a.shape[0], frame_a.shape[1]), window_size, overlap )
    
    cdef np.ndarray[DTYPEi_t, ndim=2] window_a = np.zeros([window_size, window_size], dtype=DTYPEi)
    cdef np.ndarray[DTYPEi_t, ndim=2] search_area = np.zeros([search_area_size, search_area_size], dtype=DTYPEi)
    cdef np.ndarray[DTYPEf_t, ndim=2] corr = np.zeros([search_area_size, search_area_size], dtype=DTYPEf)
        
    cdef np.ndarray[DTYPEf_t, ndim=2] u = np.zeros([n_rows, n_cols], dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] v = np.zeros([n_rows, n_cols], dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] sig2noise = np.zeros([n_rows, n_cols], dtype=DTYPEf)
    
    # loop over the interrogation windows
    # i, j are the row, column indices of the top left corner
    I = 0
    for i in range( 0, frame_a.shape[0]-window_size, window_size-overlap ):
        J = 0
        for j in range( 0, frame_a.shape[1]-window_size, window_size-overlap ):
            
          
            # get interrogation window matrix from frame a
            for k in range( window_size ):
                for l in range( window_size ):
                    window_a[k,l] = frame_a[i+k, j+l]
                    
                    
            # get search area using frame b
            for k in range( search_area_size ):
                for l in range( search_area_size ):
                    
                    # fill with zeros if we are out of the borders
                    if i+window_size/2-search_area_size/2+k < 0 or i+window_size/2-search_area_size/2+k >= frame_b.shape[0]:
                        search_area[k,l] = 0
                    elif j+window_size/2-search_area_size/2+l < 0 or j+window_size/2-search_area_size/2+l >= frame_b.shape[1]:
                        search_area[k,l] = 0
                    else:
                        search_area[k,l] = frame_b[ i+window_size/2-search_area_size/2+k, j+window_size/2-search_area_size/2+l ]
                        
            
            # compute correlation map 
            corr = openpiv.pyprocess.correlate_windows( search_area, window_a )
            
            # find subpixel approximation of the peak center
            i_peak, j_peak, s2n = openpiv.pyprocess.find_pixel_peak_position( corr )
            i_peak, j_peak = openpiv.pyprocess.find_subpixel_peak_position( corr, (i_peak, j_peak) )
            
            # velocities
            v[I,J] = -( (i_peak - corr.shape[0]/2) - (search_area_size-window_size)/2 ) / dt
            u[I,J] =  ( (j_peak - corr.shape[0]/2) - (search_area_size-window_size)/2 ) / dt
            
            # compute signal to noise ratio
            sig2noise[I,J] = s2n
            
            # go to next vector
            J = J + 1
                
        # go to next vector
        I = I + 1
            
    return u, v, sig2noise
    
