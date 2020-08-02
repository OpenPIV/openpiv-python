"""This module is dedicated to advanced algorithms for PIV image analysis."""
from __future__ import division
import numpy as np
import numpy.ma as ma
from numpy.fft import rfft2,irfft2,fftshift
from math import log
from scipy.signal import convolve
import time
import openpiv
import warnings
from progressbar import *

cimport numpy as np
cimport cython

DTYPEi = np.int32
ctypedef np.int32_t DTYPEi_t

DTYPEf = np.float64
ctypedef np.float64_t DTYPEf_t


def extended_search_area_piv( np.ndarray[DTYPEi_t, ndim=2] frame_a, 
                              np.ndarray[DTYPEi_t, ndim=2] frame_b,
                              int window_size,
                              int overlap=0,
                              float dt=1.0,
                              int search_area_size=0,
                              str subpixel_method='gaussian',
                              str sig2noise_method=None,
                              int width=2,
                              int nfftx=0,
                              int nffty=0):
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
    frame_a : 2d np.ndarray, dtype=np.int32
        an two dimensions array of integers containing grey levels of 
        the first frame.
        
    frame_b : 2d np.ndarray, dtype=np.int32
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
    
    subpixel_method : string
         one of the following methods to estimate subpixel location of the peak: 
         'centroid' [replaces default if correlation map is negative], 
         'gaussian' [default if correlation map is positive], 
         'parabolic'.
    
    sig2noise_method : string 
        defines the method of signal-to-noise-ratio measure,
        ('peak2peak' or 'peak2mean'. If None, no measure is performed.)
        
    width : int
        the half size of the region around the first
        correlation peak to ignore for finding the second
        peak. [default: 2]. Only used if ``sig2noise_method==peak2peak``.
        
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
        
    sig2noise : 2d np.ndarray, optional
        a two dimensional array containing the signal to noise ratio
        from the cross correlation function. This array is returned if
        sig2noise_method is not None.
        
    Examples
    --------
    
    >>> u, v = openpiv.process.extended_search_area_piv( frame_a, frame_b, window_size=16, overlap=8, search_area_size=48, dt=0.1)
    """
                      
    # check the inputs for validity
    
    if search_area_size == 0:
        search_area_size = window_size
    
    if overlap >= window_size:
        raise ValueError('Overlap has to be smaller than the window_size')
    
    if search_area_size < window_size:
        raise ValueError('Search size cannot be smaller than the window_size')
    
        
    if (window_size > frame_a.shape[0]) or (window_size > frame_a.shape[1]):
        raise ValueError('window size cannot be larger than the image')
                                

    cdef int i, j, k, l, I, J
    
    # subpixel peak location
    cdef float i_peak, j_peak
    
    # signal to noise ratio
    cdef float s2n
    
    # shape of the resulting flow field
    cdef int n_cols, n_rows
    
    # get field shape
    n_rows, n_cols = get_field_shape( (frame_a.shape[0], frame_a.shape[1]), window_size, overlap )
    
    # define arrays
    cdef np.ndarray[DTYPEi_t, ndim=2] window_a = np.zeros([window_size, window_size], dtype=DTYPEi)
    cdef np.ndarray[DTYPEi_t, ndim=2] search_area = np.zeros([search_area_size, search_area_size], dtype=DTYPEi)
    cdef np.ndarray[DTYPEf_t, ndim=2] corr = np.zeros([search_area_size, search_area_size], dtype=DTYPEf)
        
    cdef np.ndarray[DTYPEf_t, ndim=2] u = np.zeros([n_rows, n_cols], dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] v = np.zeros([n_rows, n_cols], dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] sig2noise = np.zeros([n_rows, n_cols], dtype=DTYPEf)
    
    # loop over the interrogation windows
    # i, j are the row, column indices of the top left corner
    I = 0
    for i in range( 0, frame_a.shape[0]-window_size+1, window_size-overlap ):
        J = 0
        for j in range( 0, frame_a.shape[1]-window_size+1, window_size-overlap ):

            # get interrogation window matrix from frame a
            for k in range( window_size ):
                for l in range( window_size ):
                    window_a[k,l] = frame_a[i+k, j+l]
                    
            # get search area using frame b
            for k in range( search_area_size ):
                for l in range( search_area_size ):
                    
                    # fill with zeros if we are out of the borders
                    if i+window_size/2-search_area_size//2+k < 0 or \
                        i+window_size//2-search_area_size//2+k >= frame_b.shape[0]:
                        search_area[k,l] = 0
                    elif j+window_size//2-search_area_size//2+l < 0 or \
                        j+window_size//2-search_area_size//2+l >= frame_b.shape[1]:
                        search_area[k,l] = 0
                    else:
                        search_area[k,l] = frame_b[ i+window_size//2-search_area_size//2+k,
                            j+window_size//2-search_area_size//2+l ]
                        
            # compute correlation map 
            if any(window_a.flatten()):
                corr = correlate_windows( search_area, window_a, nfftx=nfftx, nffty=nffty )
                c = CorrelationFunction( corr )
            
                # find subpixel approximation of the peak center
                i_peak, j_peak = c.subpixel_peak_position( subpixel_method )
            
                # velocities
                v[I,J] = -( (i_peak - corr.shape[0]/2) - (search_area_size-window_size)/2 ) / dt
                u[I,J] =  ( (j_peak - corr.shape[0]/2) - (search_area_size-window_size)/2 ) / dt
            
                # compute signal to noise ratio
                if sig2noise_method:
                    sig2noise[I,J] = c.sig2noise_ratio( sig2noise_method, width )
            else:
                v[I,J] = 0.0
                u[I,J] = 0.0
                # compute signal to noise ratio
                if sig2noise_method:
                    sig2noise[I,J] = np.inf
                
            # go to next vector
            J = J + 1
                
        # go to next vector
        I = I + 1

    if sig2noise_method:
        return u, v, sig2noise
    else:
        return u, v
    
class CorrelationFunction( ):
    def __init__ ( self, corr ):
        """A class representing a cross correlation function.
        
        Parameters
        ----------
        corr : 2d np.ndarray
            the correlation function array
        
        """
        self.data = corr
        self.shape = self.data.shape
        
        # get first peak
        self.peak1, self.corr_max1 = self._find_peak( self.data )
        
    def _find_peak ( self, array ):
        """Find row and column indices of the highest peak in an array."""    
        ind = array.argmax()
        s = array.shape[1] 
        
        i = ind // s 
        j = ind %  s
        
        return  (i, j),  array.max()
        
    def _find_second_peak ( self, width ):
        """
        Find the value of the second largest peak.
        
        The second largest peak is the height of the peak in 
        the region outside a ``width * width`` submatrix around 
        the first correlation peak.
        
        Parameters
        ----------
        width : int
            the half size of the region around the first correlation 
            peak to ignore for finding the second peak.
              
        Returns
        -------
        i, j : two elements tuple
            the row, column index of the second correlation peak.
            
        corr_max2 : int
            the value of the second correlation peak.
        
        """ 
        # create a masked view of the self.data array
        tmp = self.data.view(ma.MaskedArray)
        
        # set width x width square submatrix around the first correlation peak as masked.
        # Before check if we are not too close to the boundaries, otherwise we have negative indices
        iini = max(0, self.peak1[0]-width)
        ifin = min(self.peak1[0]+width+1, self.data.shape[0])
        jini = max(0, self.peak1[1]-width)
        jfin = min(self.peak1[1]+width+1, self.data.shape[1])
        tmp[ iini:ifin, jini:jfin ] = ma.masked
        peak, corr_max2 = self._find_peak( tmp )
        
        return peak, corr_max2  
            
    def subpixel_peak_position( self, method='gaussian' ):
        """
        Find subpixel approximation of the correlation peak.
        
        This function returns a subpixels approximation of the correlation
        peak by using one of the several methods available. 
        
        Parameters
        ----------            
        method : string
             one of the following methods to estimate subpixel location of the peak: 
             'centroid' [replaces default if correlation map is negative], 
             'gaussian' [default if correlation map is positive], 
             'parabolic'.
             
        Returns
        -------
        subp_peak_position : two elements tuple
            the fractional row and column indices for the sub-pixel
            approximation of the correlation peak.
        """
    
        # the peak and its neighbours: left, right, down, up
        try:
            c  = self.data[self.peak1[0]  , self.peak1[1]  ]
            cl = self.data[self.peak1[0]-1, self.peak1[1]  ]
            cr = self.data[self.peak1[0]+1, self.peak1[1]  ]
            cd = self.data[self.peak1[0]  , self.peak1[1]-1] 
            cu = self.data[self.peak1[0]  , self.peak1[1]+1]
        except IndexError:
            # if the peak is near the border do not 
            # do subpixel approximation
            return self.peak1
            
        # if all zero or some is NaN, don't do sub-pixel search:
        tmp = np.array([c,cl,cr,cd,cu])
        if np.any( np.isnan(tmp) ) or np.all ( tmp == 0 ):
            return self.peak1
            
        # if correlation is negative near the peak, fall back 
        # to a centroid approximation
        if np.any ( tmp  < 0 ) and method == 'gaussian':
            method = 'centroid'
        
        # choose method
        if method == 'centroid':
            subp_peak_position = (((self.peak1[0]-1)*cl+self.peak1[0]*c+(self.peak1[0]+1)*cr)/(cl+c+cr),
                                ((self.peak1[1]-1)*cd+self.peak1[1]*c+(self.peak1[1]+1)*cu)/(cd+c+cu))
    
        elif method == 'gaussian':
            subp_peak_position = (self.peak1[0] + ( (np.log(cl)-np.log(cr) )/( 2*np.log(cl) - 4*np.log(c) + 2*np.log(cr) )),
                                self.peak1[1] + ( (np.log(cd)-np.log(cu) )/( 2*np.log(cd) - 4*np.log(c) + 2*np.log(cu) ))) 
    
        elif method == 'parabolic':
            subp_peak_position = (self.peak1[0] +  (cl-cr)/(2*cl-4*c+2*cr),
                                    self.peak1[1] +  (cd-cu)/(2*cd-4*c+2*cu)) 
        else:
            raise ValueError( "method not understood. Can be 'gaussian', 'centroid', 'parabolic'." )
        
        return subp_peak_position
        
    def sig2noise_ratio( self, method='peak2peak', width=2 ):
        """Computes the signal to noise ratio.
        
        The signal to noise ratio is computed from the correlation map with
        one of two available method. It is a measure of the quality of the 
        matching between two interogation windows.
        
        Parameters
        ----------
        sig2noise_method: string
            the method for evaluating the signal to noise ratio value from 
            the correlation map. Can be `peak2peak`, `peak2mean` or None
            if no evaluation should be made.
            
        width : int, optional
            the half size of the region around the first
            correlation peak to ignore for finding the second
            peak. [default: 2]. Only used if ``sig2noise_method==peak2peak``.
            
        Returns
        -------
        sig2noise : float 
            the signal to noise ratio from the correlation map.
            
        """

        # if the image is lacking particles, totally black it will correlate to very low value, but not zero
        # return zero, since we have no signal.
        if self.corr_max1 <  1e-3:
            return 0.0
            
        # if the first peak is on the borders, the correlation map is wrong
        # return zero, since we have no signal.
        if ( 0 in self.peak1 or self.data.shape[0] in self.peak1 or self.data.shape[1] in self.peak1):
            return 0.0
        
        # now compute signal to noise ratio
        if method == 'peak2peak':
            # find second peak height
            peak2, corr_max2 = self._find_second_peak( width=width )
            
        elif method == 'peak2mean':
            # find mean of the correlation map
            corr_max2 = self.data.mean()
            
        else:
            raise ValueError('wrong sig2noise_method')
    
        # avoid dividing by zero
        try:
            sig2noise = self.corr_max1/corr_max2
        except ValueError:
            sig2noise = np.inf    
            
        return sig2noise

def get_coordinates( image_size, window_size, overlap ):
    """Compute the x, y coordinates of the centers of the interrogation windows.
    
    Parameters
    ----------
    image_size: two elements tuple
        a two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is 
        the number of columns.
        
    window_size: int
        the size of the interrogation windows.
        
    overlap: int
        the number of pixel by which two adjacent interrogation
        windows overlap.
        
        
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
    x = np.arange( field_shape[1] )*(window_size-overlap) + window_size/2.0
    y = np.arange( field_shape[0] )*(window_size-overlap) + window_size/2.0
    
    return np.meshgrid(x,y)

def get_field_shape ( image_size, window_size, overlap ):
    """Compute the shape of the resulting flow field.
    
    Given the image size, the interrogation window size and
    the overlap size, it is possible to calculate the number 
    of rows and columns of the resulting flow field.
    
    Parameters
    ----------
    image_size: two elements tuple
        a two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is 
        the number of columns.
        
    window_size: int
        the size of the interrogation window.
        
    overlap: int
        the number of pixel by which two adjacent interrogation
        windows overlap.
        
        
    Returns
    -------
    field_shape : two elements tuple
        the shape of the resulting flow field
    """
    
    return ( (image_size[0] - window_size)//(window_size-overlap)+1, 
             (image_size[1] - window_size)//(window_size-overlap)+1 )

def correlate_windows( window_a, window_b, corr_method = 'fft', int nfftx = 0, int nffty = 0 ):
    """Compute correlation function between two interrogation windows.
    
    The correlation function can be computed by using the correlation 
    theorem to speed up the computation.
    
    Parameters
    ----------
    window_a : 2d np.ndarray
        a two dimensions array for the first interrogation window.
        
    window_b : 2d np.ndarray
        a two dimensions array for the second interrogation window.
        
    corr_method   : string
        one of the two methods currently implemented: 'fft' or 'direct'.
        Default is 'fft', which is much faster.
        
    nfftx   : int
        the size of the 2D FFT in x-direction, 
        [default: 2 x windows_a.shape[0] is recommended].
        
    nffty   : int
        the size of the 2D FFT in y-direction, 
        [default: 2 x windows_a.shape[1] is recommended].
        
        
    Returns
    -------
    corr : 2d np.ndarray
        a two dimensions array for the correlation function.
    
    """
    
    if corr_method == 'fft':
        if nfftx == 0:
            nfftx = 2*window_a.shape[0]
        if nffty == 0:
            nffty = 2*window_a.shape[1]
        return fftshift(irfft2(rfft2(normalize_intensity(window_a),\
            s=(nfftx,nffty))*np.conj(rfft2(normalize_intensity(window_b),\
            s=(nfftx,nffty)))).real, axes=(0,1)  )
    elif corr_method == 'direct':
        return convolve(normalize_intensity(window_a), normalize_intensity(window_b[::-1,::-1]), 'full')
    else:
        raise ValueError('method is not implemented')

def normalize_intensity( window ):
    """Normalize interrogation window by removing the mean value.
    
    Parameters
    ----------
    window :  2d np.ndarray
        the interrogation window array
        
    Returns
    -------
    window :  2d np.ndarray
        the interrogation window array, with mean value equal to zero.
    
    """
    return window - window.mean()








##################################################################
def WiDIM( np.ndarray[DTYPEi_t, ndim=2] frame_a, 
           np.ndarray[DTYPEi_t, ndim=2] frame_b,
           np.ndarray[DTYPEi_t, ndim=2] mark,
           int min_window_size,
           float overlap_ratio,
           int coarse_factor,
           float dt,
           str validation_method='mean_velocity',
           int trust_1st_iter=1,
           int validation_iter = 1,
           float tolerance = 1.5,
           int nb_iter_max=3,
           str subpixel_method='gaussian',
           str sig2noise_method='peak2peak',
           float sig2noise_threshold = 1.5,
           int width=2,
           int nfftx=0,
           int nffty=0):
    """
    Implementation of the WiDIM algorithm (Window Displacement Iterative Method).
    This is an iterative  method to cope with  the lost of pairs due to particles 
    motion and get rid of the limitation in velocity range due to the window size.
    The possibility of window size coarsening is implemented.
    Example : minimum window size of 16*16 pixels and coarse_level of 2 gives a 1st
    iteration with a window size of 64*64 pixels, then 32*32 then 16*16.
        ----Algorithm : At each step, a predictor of the displacement (dp) is applied based on the results of the previous iteration.
                        Each window is correlated with a shifted window.
                        The displacement obtained from this correlation is the residual displacement (dc)
                        The new displacement (d) is obtained with dx = dpx + dcx and dy = dpy + dcy
                        The velocity field is validated and wrong vectors are replaced by mean value of surrounding vectors from the previous iteration (or by bilinear interpolation if the window size of previous iteration was different)
                        The new predictor is obtained by bilinear interpolation of the displacements of the previous iteration:
                            dpx_k+1 = dx_k

    Reference:
        F. Scarano & M. L. Riethmuller, Iterative multigrid approach in PIV image processing with discrete window offset, Experiments in Fluids 26 (1999) 513-523
    
    Parameters
    ----------
    frame_a : 2d np.ndarray, dtype=np.int32
        an two dimensions array of integers containing grey levels of 
        the first frame.
        
    frame_b : 2d np.ndarray, dtype=np.int32
        an two dimensions array of integers containing grey levels of 
        the second frame.
        
    mark : 2d np.ndarray, dtype=np.int32
        an two dimensions array of integers with values 0 for the background, 1 for the flow-field. If the center of a window is on a 0 value the velocity is set to 0.

    min_window_size : int
        the size of the minimum (final) (square) interrogation window.
        
    overlap_ratio : float
        the ratio of overlap between two windows (between 0 and 1).
        
    dt : float
        the time delay separating the two frames.
    
    validation_method : string
        the method used for validation (in addition to the sig2noise method). Only the mean velocity method is implemented now
    
    trust_1st_iter : int = 0 or 1
        0 if the first iteration need to be validated. With a first window size following the 1/4 rule, the 1st iteration can be trusted and the value should be 1 (Default value)
     
    validation_iter : int
        number of iterations per validation cycle.
       
    tolerance : float
        the threshold for the validation method chosen. This does not concern the sig2noise for which the threshold is 1.5; [nb: this could change in the future]
    
    nb_iter_max : int
        global number of iterations.
       
    subpixel_method : string
         one of the following methods to estimate subpixel location of the peak: 
         'centroid' [replaces default if correlation map is negative], 
         'gaussian' [default if correlation map is positive], 
         'parabolic'.
    
    sig2noise_method : string 
        defines the method of signal-to-noise-ratio measure,
        ('peak2peak' or 'peak2mean'. If None, no measure is performed.)
        
    sig2noise_threshold : float, recommended to use 1.5
        defines a threshold of a local outlier vector that is below the signal-to-noise threshold,
        see the sig2noise_method above for better understanding of this value.
        
    width : int
        the half size of the region around the first
        correlation peak to ignore for finding the second
        peak. [default: 2]. Only used if ``sig2noise_method==peak2peak``.
        
    nfftx   : int
        the size of the 2D FFT in x-direction, 
        [default: 2 x windows_a.shape[0] is recommended]
        
    nffty   : int
        the size of the 2D FFT in y-direction, 
        [default: 2 x windows_a.shape[1] is recommended]

    
    Returns
    -------

    x : 2d np.ndarray
        a two dimensional array containing the x-axis component of the interpolations locations.
        
    y : 2d np.ndarray
        a two dimensional array containing the y-axis component of the interpolations locations.
        
    u : 2d np.ndarray
        a two dimensional array containing the u velocity component,
        in pixels/seconds.
        
    v : 2d np.ndarray
        a two dimensional array containing the v velocity component,
        in pixels/seconds.
        
    mask : 2d np.ndarray
        a two dimensional array containing the boolean values (True for vectors interpolated from previous iteration)
        
    Example
    --------
    
    >>> x,y,u,v, mask = openpiv.process.WiDIM( frame_a, frame_b, mark, min_window_size=16, overlap_ratio=0.25, 
      coarse_factor=2, dt=0.02, validation_method='mean_velocity', trust_1st_iter=1, validation_iter=2, tolerance=0.7, 
      nb_iter_max=4, sig2noise_method='peak2peak', sig2noise_threshold = 1.5)

    --------------------------------------
    Method of implementation : to improve the speed of the programm,
    all data have been placed in the same huge 4-dimensions 'F' array.
    (this prevent the definition of a new array for each iteration)
    However, during the coarsening process a large part of the array is not used.
    Structure of array F:
    --The 1st index is the main iteration (K)   --> length is nb_iter_max
        -- 2nd index (I) is row (of the map of the interpolations locations of iteration K) --> length (effectively used) is Nrow[K]
            --3rd index (J) is column  --> length (effectively used) is Ncol[K]
                --4th index represent the type of data stored at this point:
                            | 0 --> x         |
                            | 1 --> y         | 
                            | 2 --> xb        |
                            | 3 --> yb        | 
                            | 4 --> dx        |
                            | 5 --> dy        | 
                            | 6 --> dpx       |
                            | 7 --> dpy       | 
                            | 8 --> dcx       |
                            | 9 --> dcy       | 
                            | 10 --> u        |
                            | 11 --> v        | 
                            | 12 --> si2noise | 
    Storage of data with indices is not good for comprehension so its very important to comment on each single operation.
    A python dictionary type could have been used (and would be much more intuitive)
    but its equivalent in c language (type map) is very slow compared to a numpy ndarray.
    """

    #initializations
    # warnings.warn("deprecated", RuntimeWarning)
    if nb_iter_max <= coarse_factor:
        raise ValueError( "Please provide a nb_iter_max that is greater than the coarse_level" )
    cdef int K #main iteration index
    cdef int I, J #interrogation locations indices
    cdef int L, M #inside window indices
    cdef int O, P #frame indices corresponding to I and J
    cdef int i, j #dumb indices for various works
    cdef float i_peak, j_peak, mean_u, mean_v, rms_u, rms_v, residual_0
    cdef int residual, nbwind
    cdef np.ndarray[DTYPEi_t, ndim=1] Nrow = np.zeros(nb_iter_max, dtype=DTYPEi)
    cdef np.ndarray[DTYPEi_t, ndim=1] Ncol = np.zeros(nb_iter_max, dtype=DTYPEi)
    cdef np.ndarray[DTYPEi_t, ndim=1] W = np.zeros(nb_iter_max, dtype=DTYPEi)
    cdef np.ndarray[DTYPEi_t, ndim=1] Overlap = np.zeros(nb_iter_max, dtype=DTYPEi)
    pic_size=frame_a.shape
    #window sizes list initialization
    for K in range(coarse_factor+1):
        W[K]=np.power(2,coarse_factor-K)*min_window_size
    for K in range(coarse_factor+1,nb_iter_max):
        W[K]=W[K-1]
    #overlap init
    for K in range(nb_iter_max):
        Overlap[K]=int(np.floor(overlap_ratio*W[K]))
    #Ncol and Nrow init
    for K in range(nb_iter_max):
        Nrow[K]=((pic_size[0]-W[K])//(W[K]-Overlap[K]))+1
        Ncol[K]=((pic_size[1]-W[K])//(W[K]-Overlap[K]))+1
    #writting the parameters to the screen
    if validation_iter==0:
        validation_method='None'
    cdef float startTime = launch(method='WiDIM', names=['Size of image', 'total number of iterations', 'overlap ratio', 'coarse factor', 'time step', 'validation method', 'number of validation iterations', 'subpixel_method','Nrow', 'Ncol', 'Window sizes', 'overlaps'], arg=[[pic_size[0], pic_size[1]], nb_iter_max, overlap_ratio, coarse_factor, dt, validation_method, validation_iter,  subpixel_method, Nrow, Ncol, W, Overlap])
    #define the main array F that contains all the data
    cdef np.ndarray[DTYPEf_t, ndim=4] F = np.zeros([nb_iter_max, Nrow[nb_iter_max-1], Ncol[nb_iter_max-1], 14], dtype=DTYPEf)
    #define mask - bool array don't exist in cython so we go to lower level with cast
    #you can access mask with (<object>mask)[I,J]
    cdef np.ndarray[np.uint8_t, ndim=2, cast=True] mask = np.empty([Nrow[nb_iter_max-1], Ncol[nb_iter_max-1]], dtype=np.bool)
    #define u,v, x,y fields (only used as outputs of this programm)
    cdef np.ndarray[DTYPEf_t, ndim=2] u = np.zeros([Nrow[nb_iter_max-1], Ncol[nb_iter_max-1]], dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] v = np.zeros([Nrow[nb_iter_max-1], Ncol[nb_iter_max-1]], dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] x = np.zeros([Nrow[nb_iter_max-1], Ncol[nb_iter_max-1]], dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] y = np.zeros([Nrow[nb_iter_max-1], Ncol[nb_iter_max-1]], dtype=DTYPEf)
    #define two small arrays used for the validation process
    cdef np.ndarray[DTYPEf_t, ndim=3] neighbours = np.zeros([2,3,3], dtype=DTYPEf)
    cdef np.ndarray[DTYPEi_t, ndim=2] neighbours_present = np.zeros([3,3], dtype=DTYPEi)
    #initialize x and y values
    for K in range(nb_iter_max):
        for I in range(Nrow[K]):
            for J in range(Ncol[K]):
                #x unit vector corresponds to rows
                #y unit vector corresponds to columns
                if I==0:
                    F[K,I,J,0]=W[K]/2 #init x on 1st row
                else:
                    F[K,I,J,0]=F[K,I-1,J,0] + W[K] - Overlap[K] #init x
                if J==0:
                    F[K,I,J,1]=W[K]/2 #init y on first column
                else:
                    F[K,I,J,1]=F[K,I,J-1,1] + W[K] - Overlap[K] #init y
    #end of the initializations
    ####################################################
    #main loop
    for K in range(nb_iter_max):
        print("ITERATION # ", K)
        window_a, window_b = define_windows(W[K])
        #a simple progress bar
        # widgets = ['Computing the displacements : ', Percentage(), ' ', Bar(marker='-',left='[',right=']'),
        #    ' ', ETA(), ' ', FileTransferSpeed()]
        # pbar = ProgressBar(widgets=widgets, maxval=100)
        # pbar.start()
        residual = 0
        for I in range(Nrow[K]):#run through interpolations locations
            # pbar.update(100*I/Nrow[K])#progress update
            for J in range(Ncol[K]):
                
                #compute xb, yb:
                F[K,I,J,2]=F[K,I,J,0]+F[K,I,J,6]#xb=xa+dpx
                F[K,I,J,3]=F[K,I,J,1]+F[K,I,J,7]#yb=yb+dpy
                #look for corrupted window (ie. going outside of the picture) and relocate them with 0 displacement:
                if F[K,I,J,2] + W[K]/2 > pic_size[0]-1 or F[K,I,J,2] - W[K]/2 < 0: #if corrupted on x-axis do:
                    F[K,I,J,2]=F[K,I,J,0]#xb=x
                    F[K,I,J,3]=F[K,I,J,1]#yb=y
                    F[K,I,J,6]=0.0#dpx=0
                    F[K,I,J,7]=0.0#dpy=0
                elif F[K,I,J,3] + W[K]/2 > pic_size[1]-1 or F[K,I,J,3] - W[K]/2 < 0: #if corrupted on y-axis do the same
                    F[K,I,J,2]=F[K,I,J,0]#xb=x
                    F[K,I,J,3]=F[K,I,J,1]#yb=y
                    F[K,I,J,6]=0.0#dpx=0
                    F[K,I,J,7]=0.0#dpy=0
                #fill windows a and b
                for L in range(W[K]):
                    for M in range(W[K]):
                        window_a[L,M]=frame_a[np.int(F[K,I,J,0] - W[K]/2 + L), \
                                            np.int(F[K,I,J,1] - W[K]/2 + M)]
                        window_b[L,M]=frame_b[np.int(F[K,I,J,2] - W[K]/2 + L), \
                                            np.int(F[K,I,J,3] - W[K]/2 + M)]
                #perform correlation of the two windows
                corr = correlate_windows( window_b, window_a, nfftx=nfftx, nffty=nffty )
                c = CorrelationFunction( corr )
                F[K,I,J,12] = c.sig2noise_ratio( sig2noise_method, width )#compute sig2noise
                i_peak, j_peak = c.subpixel_peak_position( subpixel_method )#get peak position
                if np.any(np.isnan((i_peak, j_peak))) or mark[np.int(F[K,I,J,0]), np.int(F[K,I,J,1])] == 0:#prevent 'Not a Number' peak location
                #if np.any(np.isnan((i_peak, j_peak))):
                    F[K,I,J,8]=0.0
                    F[K,I,J,9]=0.0
                else:
                    #find residual displacement dcx and dcy
                    F[K,I,J,8]=i_peak - corr.shape[0]/2#dcx
                    F[K,I,J,9]=j_peak - corr.shape[1]/2#dcy
                residual = residual + np.sqrt(F[K,I,J,8]*F[K,I,J,8]+F[K,I,J,9]*F[K,I,J,9])
                #get new displacement prediction
                F[K,I,J,4]=F[K,I,J,6]+F[K,I,J,8]#dx=dpx+dcx
                F[K,I,J,5]=F[K,I,J,7]+F[K,I,J,9]#dy=dpy+dcy
                #get new velocity vectors
                F[K,I,J,10]=F[K,I,J,5] / dt #u=dy/dt
                F[K,I,J,11]=-F[K,I,J,4] / dt #v=-dx/dt
        # pbar.finish()#close progress bar
        print("..[DONE]")
        if K==0:
            residual_0 = residual/np.float(Nrow[K]*Ncol[K])
        print(" --residual : ", (residual/np.float(Nrow[K]*Ncol[K]))/residual_0)
        #####################################################
        #validation of the velocity vectors with 3*3 filtering
        if K==0 and trust_1st_iter:#1st iteration can generally be trust if it follows the 1/4 rule
            print("no validation : trusting 1st iteration")
        else: 
            print("Starting validation..")
            for I in range(Nrow[nb_iter_max-1]):#init mask to False
                for J in range(Ncol[nb_iter_max-1]):
                    (<object>mask)[I,J]=False
            for i in range(validation_iter):#real validation starts
                print("Validation, iteration number ",i)
                print(" ")
                # widgets = ['Validation : ', Percentage(), ' ', Bar(marker='-',left='[',right=']'),
           # ' ', ETA(), ' ', FileTransferSpeed()]
                # pbar = ProgressBar(widgets=widgets, maxval=100)
                # pbar.start()
                for I in range(Nrow[K]):#run through locations
                    # pbar.update(100*I/Nrow[K])                    
                    for J in range(Ncol[K]):
                        neighbours_present = find_neighbours(I, J, Nrow[K]-1, Ncol[K]-1)#get a map of the neighbouring locations
                        for L in range(3):#get the velocity of the neighbours in a 2*3*3 array
                            for M in range(3):
                                if neighbours_present[L,M]:
                                    neighbours[0,L,M]=F[K,I+L-1,J+M-1,10]#u
                                    neighbours[1,L,M]=F[K,I+L-1,J+M-1,11]#v
                                else:
                                    neighbours[0,L,M]=0
                                    neighbours[1,L,M]=0

                            if np.sum(neighbours_present) !=0 and mark[np.int(F[K,I,J,0]), np.int(F[K,I,J,1])] == 1:
                            #if np.sum(neighbours_present):
                                mean_u = np.sum(neighbours[0])/np.float(np.sum(neighbours_present))#computing the mean velocity
                                mean_v = np.sum(neighbours[1])/np.float(np.sum(neighbours_present))
                                if F[K,I,J,12] < sig2noise_threshold:#validation with the sig2noise ratio, 1.5 is a recommended minimum value
                                    if K==0:#if in 1st iteration, no interpolation is needed so just replace by the mean
                                        F[K,I,J,10] = mean_u
                                        F[K,I,J,11] = mean_v
                                        (<object>mask)[I,J]=True
                                        F[K,I,J,4] = -F[K,I,J,11]*dt#recompute displacement from velocity
                                        F[K,I,J,5] = F[K,I,J,10]*dt
                                    elif K>0 and (Nrow[K] != Nrow[K-1] or Ncol[K] != Ncol[K-1]):#perform interpolation using previous iteration (which is supposed to be already validated -> this prevents error propagation)
                                        F[K,I,J,10] = interpolate_surroundings(F,Nrow,Ncol,K-1,I,J, 10)
                                        F[K,I,J,11] = interpolate_surroundings(F,Nrow,Ncol,K-1,I,J, 11)
                                if validation_method=='mean_velocity':#add a validation with the mean and rms values
                                    rms_u = np.sqrt(sumsquare_array(neighbours[0])/np.float(np.sum(neighbours_present)))#get rms of u
                                    rms_v = np.sqrt(sumsquare_array(neighbours[1])/np.float(np.sum(neighbours_present)))
                                    if rms_u==0 or rms_v==0:
                                            F[K,I,J,10] = mean_u
                                            F[K,I,J,11] = mean_v
                                    elif ((F[K,I,J,10] - mean_u)/rms_u) > tolerance or ((F[K,I,J,11] - mean_v)/rms_v) > tolerance:
                                        if K==0:
                                            F[K,I,J,10] = mean_u
                                            F[K,I,J,11] = mean_v
                                            (<object>mask)[I,J]=True
                                            F[K,I,J,4] = -F[K,I,J,11]*dt
                                            F[K,I,J,5] = F[K,I,J,10]*dt
                                        elif K>0 and (Nrow[K] != Nrow[K-1] or Ncol[K] != Ncol[K-1]):#case if different dimensions : interpolation using previous iteration
                                            F[K,I,J,10] = interpolate_surroundings(F,Nrow,Ncol,K-1,I,J, 10)
                                            F[K,I,J,11] = interpolate_surroundings(F,Nrow,Ncol,K-1,I,J, 11)
                                            (<object>mask)[I,J]=True
                                            F[K,I,J,4] = -F[K,I,J,11]*dt
                                            F[K,I,J,5] = F[K,I,J,10]*dt
                                        elif K>0 and (Nrow[K] == Nrow[K-1] or Ncol[K] == Ncol[K-1]):#case if same dimensions
                                            for L in range(3):
                                                for M in range(3):
                                                    if neighbours_present[L,M]:
                                                        neighbours[0,L,M]=F[K-1,I+L-1,J+M-1,10]#u
                                                        neighbours[1,L,M]=F[K-1,I+L-1,J+M-1,11]#v
                                                    else:
                                                        neighbours[0,L,M]=0
                                                        neighbours[1,L,M]=0
                                            if np.sum(neighbours_present) !=0:
                                                mean_u = np.sum(neighbours[0])/np.float(np.sum(neighbours_present))
                                                mean_v = np.sum(neighbours[1])/np.float(np.sum(neighbours_present))
                                                F[K,I,J,10] = mean_u
                                                F[K,I,J,11] = mean_v
                                                (<object>mask)[I,J]=True
                                                F[K,I,J,4] = -F[K,I,J,11]*dt
                                                F[K,I,J,5] = F[K,I,J,10]*dt
            # pbar.finish()                    
            print("..[DONE]")
            print(" ")
        #end of validation
        ##############################################################################
        #stop process if this is the last iteration
        if K==nb_iter_max-1:
            print("//////////////////////////////////////////////////////////////////")
            print("end of iterative process.. Re-arranging vector fields..")
            for I in range(Nrow[K]):#assembling the u,v and x,y fields for outputs
                for J in range(Ncol[K]):
                    x[I,J]=F[K,I,J,1]
                    y[I,J]=F[K,Nrow[K]-I-1,J,0]
                    u[I,J]=F[K,I,J,10]
                    v[I,J]=F[K,I,J,11]
            print("...[DONE]")
            end(startTime)
            return x, y, u, v, (<object>mask)
        #############################################################################
        #go to next iteration : compute the predictors dpx and dpy from the current displacements
        print("going to next iteration.. ")
        print("performing interpolation of the displacement field")
        print(" ")
        # widgets = ['Performing interpolations : ', Percentage(), ' ', Bar(marker='-',left='[',right=']'),
        #    ' ', ETA(), ' ', FileTransferSpeed()]
        # pbar = ProgressBar(widgets=widgets, maxval=100)
        # pbar.start()
        for I in range(Nrow[K+1]):
            # pbar.update(100*I/Nrow[K+1])
            for J in range(Ncol[K+1]):
                if Nrow[K+1]==Nrow[K] and Ncol[K+1]==Ncol[K]:
                    F[K+1,I,J,6] = np.floor(F[K,I,J,4])#dpx_k+1 = dx_k
                    F[K+1,I,J,7] = np.floor(F[K,I,J,5])#dpy_k+1 = dy_k
                else:#interpolate if dimensions do not agree
                    F[K+1,I,J,6] = np.floor(interpolate_surroundings(F,Nrow,Ncol,K,I,J, 4))
                    F[K+1,I,J,7] = np.floor(interpolate_surroundings(F,Nrow,Ncol,K,I,J, 5))
        # pbar.finish()
        print("..[DONE] -----> going to iteration ",K+1)
        print(" ")


def interpolate_surroundings(np.ndarray[DTYPEf_t, ndim=4] F,
                             np.ndarray[DTYPEi_t, ndim=1] Nrow,
                             np.ndarray[DTYPEi_t, ndim=1] Ncol,
                             int K,
                             int I,
                             int J,
                             int dat):
    """Perform interpolation of between to iterations of the F 4d-array for a specific location I,J and the data type dat.
    
    Parameters
    ----------
    F :  4d np.ndarray
        The main array of the WIDIM algorithm.

    Nrow : 1d np.ndarray
        list of the numbers of row for each iteration K
       
    Ncol : 1d np.ndarray
        list of the numbers of column for each iteration K
    
    K : int
        the iteration that contains the valid data. K+1 will be the iteration at which the interpolation is needed.
    
    I,J : int
        indices of the point that need interpolation (in iteration K+1)
    
    dat : int
        the index of the data to interpolate.
    
    Returns
    -------
    the interpolated data (type float)
    
    """
    #interpolate data dat from previous iteration
    cdef float lower_lim_previous_x = F[K,0,0,0]
    cdef float lower_lim_previous_y = F[K,0,0,1]
    cdef float upper_lim_previous_x = F[K,Nrow[K]-1,Ncol[K]-1,0]
    cdef float upper_lim_previous_y = F[K,Nrow[K]-1,Ncol[K]-1,1]
    cdef float pos_now_x = F[K+1,I,J,0]
    cdef float pos_now_y = F[K+1,I,J,1]
    cdef np.ndarray[DTYPEi_t, ndim=1] Q1 = np.zeros(2, dtype=DTYPEi)
    cdef np.ndarray[DTYPEi_t, ndim=1] Q4 = np.zeros(2, dtype=DTYPEi)
    if pos_now_x < lower_lim_previous_x:#top row
        if pos_now_y < lower_lim_previous_y:#top left corner
            return F[K,0,0,dat]
        elif pos_now_y > upper_lim_previous_y:#top right corner
            return F[K,0,Ncol[K]-1,dat]
        else:#top row no corners
            low_y, high_y = F_dichotomy(F,K,Ncol,'y_axis',pos_now_y)
            if low_y == high_y:
                return F[K,0,low_y,dat]
            else:
                return linear_interpolation(F[K,0,low_y,1], F[K,0,high_y,1], pos_now_y, F[K,0,low_y,dat], F[K,0,high_y,dat])
    elif pos_now_x > upper_lim_previous_x:#bottom row
        if pos_now_y < lower_lim_previous_y:#bottom left corner
            return F[K,Nrow[K]-1,0,dat]
        elif pos_now_y > upper_lim_previous_y:#bottom right corner
            return F[K,Nrow[K]-1,Ncol[K]-1,dat]
        else:#bottom row no corners
            low_y, high_y = F_dichotomy(F,K,Ncol,'y_axis',pos_now_y)
            #print low_y, high_y
            if low_y == high_y:
                return F[K,Nrow[K]-1,low_y,dat]
            else:
                return linear_interpolation(F[K,0,low_y,1], F[K,0,high_y,1], pos_now_y, F[K,Nrow[K]-1,low_y,dat], F[K,Nrow[K]-1,high_y,dat])
    elif pos_now_y < lower_lim_previous_y:#left column no corners
        low_x, high_x = F_dichotomy(F,K,Nrow,'x_axis',pos_now_x)
        if low_x == high_x:
            return F[K,low_x,0,dat]
        else:
            return linear_interpolation(F[K,low_x,0,0], F[K,high_x,0,0], pos_now_x, F[K,low_x,0,dat], F[K,high_x,0,dat])
    elif pos_now_y > upper_lim_previous_y:#right column no corners
        low_x, high_x = F_dichotomy(F,K,Nrow,'x_axis',pos_now_x)
        if low_x == high_x:
            return F[K,low_x,Ncol[K]-1,dat]
        else:
            return linear_interpolation(F[K,low_x,0,0], F[K,high_x,0,0], pos_now_x, F[K,low_x,Ncol[K]-1,dat], F[K,high_x,Ncol[K]-1,dat])
    else:#interior grid
        low_x, high_x = F_dichotomy(F,K,Nrow,'x_axis',pos_now_x)
        low_y, high_y = F_dichotomy(F,K,Ncol,'y_axis',pos_now_y)
        Q1[0] = F[K,low_x,0,0] 
        Q1[1] = F[K,0,low_y,1]
        Q4[0] = F[K,high_x,0,0]
        Q4[1] = F[K,0,high_y,1]
        if pos_now_x >= Q1[0] and pos_now_x <= Q4[0] and pos_now_y >= Q1[1] and pos_now_y <= Q4[1]:
            return bilinear_interpolation(Q1[0],Q4[0],Q1[1],Q4[1],pos_now_x,pos_now_y,F[K,low_x,low_y,dat],F[K,low_x,high_y,dat],F[K,high_x,low_y,dat],F[K,high_x,high_y,dat])
        else:
            raise ValueError( "cannot perform interpolation, a problem occured" )






def bilinear_interpolation(int x1, int x2, int y1, int y2, int x, int y, float f1, float f2, float f3, float f4):
    """Perform a bilinear interpolation between 4 points 
    
    Parameters
    ----------
    x1,x2,y1,y2 :  int
        x-axis and y-axis locations of the 4 points. (ie. location in the frame) 
        (Note that the x axis is vertical and pointing down while the y-axis is horizontal)

    x,y : int
        locations of the target point for the interpolation (in the frame)
       
    f1,f2,f3,f4 : float
        value at each point : f1=f(x1,y1), f2=f(x1,y2), f3=f(x2, y1), f4=f(x2,y2)
       
    Returns
    -------
    the interpolated data (type float)
    
    """
    if x1 == x2:
        if y1 == y2:
            return f1
        else:
            return linear_interpolation(y1,y2,y,f1,f2)
    elif y1 == y2:
        return linear_interpolation(x1,x2,x,f1,f3)
    else:
        return (f1*(x2-x)*(y2-y)+f2*(x2-x)*(y-y1)+f3*(x-x1)*(y2-y)+f4*(x-x1)*(y-y1))/(np.float(x2-x1)*np.float(y2-y1))





def linear_interpolation(int x1, int x2, int x, int f1, int f2):
    """Perform a linear interpolation between 2 points 
    
    Parameters
    ----------
    x1,x2 :  int
        locations of the 2 points. (along any axis) 

    x : int
        locations of the target point for the interpolation (along the same axis as x1 and x2)
       
    f1,f2 : float
        value at each point : f1=f(x1), f2=f(x2)
       
    Returns
    -------
    the interpolated data (type float)
    
    """
    return ((x2-x)/np.float(x2-x1))*f1 + ((x-x1)/np.float(x2-x1))*f2







def F_dichotomy( np.ndarray[DTYPEf_t, ndim=4] F, int K, np.ndarray[DTYPEi_t, ndim=1] N, str side, int pos_now):
    """Look for the position at the iteration K of the points surrounding a given point in the frame.
    
    Parameters
    ----------
    F :  4d np.ndarray
        The main array of the WIDIM algorithm.

    K : int
        the iteration of interest (1st index for F).
    
    N : 1d np.ndarray
        list of the numbers of row or column (depending on the specified value of 'side') for each iteration K

    side : string
        the axis of interest : can be either 'x_axis' or 'y_axis'    

    pos_now : int
        position of the point in the frame (along the axis 'side').
    
    Returns
    -------
    low : int
        largest index at the iteration K along the 'side' axis so that the position of index low in the frame is less than or equal to pos_now.    

    high : int
        smallest index at the iteration K along the 'side' axis so that the position of index low in the frame is greater than or equal to pos_now.                                                        
    
    """
    #print "starting dichotomy"
    cdef int low
    cdef int high
    cdef int minlow = 0
    cdef int maxhigh = N[K]-1
    cdef int searching
    low = np.floor(maxhigh/2)
    high = low + 1
    searching=1
    if side == 'x_axis':
        while searching:#start dichotomy
            if pos_now == F[K,low,0,0] or (low == 0 and pos_now < F[K,low,0,0]):
                searching = 0
                high = low
            elif pos_now == F[K,high,0,0] or (high == N[K]-1 and pos_now > F[K,high,0,0]):
                searching = 0
                low = high
            elif pos_now > F[K,low,0,0] and pos_now < F[K,high,0,0]:
                searching = 0
            elif pos_now < F[K,low,0,0]:
                maxhigh = low
                low = np.floor((low-minlow)/2)
                high = low + 1
            else:
                minlow=high
                low = low + np.floor((maxhigh-low)/2)
                high = low + 1
        return low, high
    elif side == 'y_axis':
        while searching:#start dichotomy
            if pos_now == F[K,0,low,1] or (low == 0 and pos_now < F[K,0,low,1]):
                searching = 0
                high = low
            elif pos_now == F[K,0,high,1] or (high == N[K]-1 and pos_now > F[K,0,high,1]):
                searching = 0
                low = high
            elif pos_now > F[K,0,low,1] and pos_now < F[K,0,high,1]:
                searching = 0
            elif pos_now < F[K,0,low,1]:
                maxhigh = low
                low = np.floor((low-minlow)/2)
                high = low + 1
            else:
                minlow=high
                low = low + np.floor((maxhigh-low)/2)
                high = low + 1
        return low, high
    else:
        raise ValueError( "no valid side for F-dichotomy!" )







def define_windows( int size ):
    """Define two windows of a given size (trick to allow the use of cdef during an iterative process)
    
    Parameters
    ----------
    size : int
        size of the two windows
       
    Returns
    -------
    window_a, window_b : two 2d np.ndarray of zero (integer)
    
    """
    cdef np.ndarray[DTYPEi_t, ndim=2] window_a = np.zeros([size, size], dtype=DTYPEi)
    cdef np.ndarray[DTYPEi_t, ndim=2] window_b = np.zeros([size, size], dtype=DTYPEi)
    return window_a, window_b






def find_neighbours(int I, int J, int Imax, int Jmax):
    """Find the neighbours of a point I,J in an array of size Imax+1, Jmax+1
    
    Parameters
    ----------
    I,J : int
        indices of the point of interest
       
    Imax,Jmax : int
        max indices for the neighbours (ie. (size of the array) - 1)
       
    Returns
    -------
    neighbours : 2d np.ndarray of size 3*3 
        containing value 1 if neighbour is present, 0 if not. Value is 0 at the center (corresponds to point I,J).
    
    """
    cdef np.ndarray[DTYPEi_t, ndim=2] neighbours = np.zeros([3,3], dtype=DTYPEi)
    cdef int k,l
    for k in range(3):
        for l in range(3):
            neighbours[k,l]=1
    neighbours[1,1]=0
    if I == 0:
        neighbours[0,0]=0
        neighbours[0,1]=0
        neighbours[0,2]=0
    if J == 0:
        neighbours[0,0]=0
        neighbours[1,0]=0
        neighbours[2,0]=0
    if I == Imax:
        neighbours[2,0]=0
        neighbours[2,1]=0
        neighbours[2,2]=0
    if J == Jmax:
        neighbours[0,2]=0
        neighbours[1,2]=0
        neighbours[2,2]=0
    return neighbours








def sumsquare_array(arr1):
    """Compute the sum of the square of the elements of a given array or list
    
    Parameters
    ----------
    arr1 : array or list of any size

    Returns
    -------
    result : float
        = sum( arr1_i * arr1_i)
    
    """
    cdef float result
    cdef I, J    
    result = 0
    for I in range(arr1.shape[0]):
        for J in range(arr1.shape[1]):
            result = result + arr1[I,J]*arr1[I,J]
    return result






def launch( str method, names, arg ):
    """A nice launcher for any openpiv function, printing a header in terminal with a list of the parameters used.
    
    Parameters
    ----------
    method : string
        the name of the algorithm used

    names : list of string
        names of the parameters to print

    arg : list of parameters of different types

    Returns
    -------
    StartTime : float 
        the current time --> can be used to print the execution time of the programm at the end.
    
    """
    print( '----------------------------------------------------------')
    print('|----->     ||   The Open Source  P article              |')
    print('| Open      ||                    I mage                 |')              
    print('|     PIV   ||                    V elocimetry  Toolbox  |')                                                  
    print('|     <-----||   www.openpiv.net          version 1.0    |')                
    print('----------------------------------------------------------') 
    print(" ")
    print("Algorithm : ", method)
    print(" ")
    print("Parameters   ")
    print("-----------------------------------")
    for i in range(len(arg)-1):
        print("     ", names[i], " | ", arg[i])
    print("-----------------------------------")
    print("|           STARTING              |")
    print("-----------------------------------")
    cdef float StartTime  = time.time()
    return StartTime





def end( float startTime ):
    """A function that prints the time since startTime. Used to end nicely a programm
    
    Parameters
    ----------
    startTime : float
        a time
    
    """
    print("-------------------------------------------------------------")
    print("[DONE] ..after ", (time.time() - startTime), "seconds ")
    print("-------------------------------------------------------------")
    






