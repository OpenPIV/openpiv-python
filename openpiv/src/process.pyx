"""This module is dedicated to advanced algorithms for PIV image analysis."""

import numpy as np
import numpy.ma as ma
from numpy.fft import rfft2,irfft2,fftshift
from math import log
from scipy.signal import convolve

cimport numpy as np
cimport cython

DTYPEi = np.int32
ctypedef np.int32_t DTYPEi_t

DTYPEf = np.float64
ctypedef np.float64_t DTYPEf_t

def extended_search_area_piv( np.ndarray[DTYPEi_t, ndim=2] frame_a, 
                              np.ndarray[DTYPEi_t, ndim=2] frame_b,
                              int window_size,
                              int overlap,
                              float dt,
                              int search_area_size,
                              str subpixel_method='gaussian',
                              sig2noise_method=None,
                              int width=2,
                              nfftx=None,
                              nffty=None):
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
            
        # if correlation is negative near the peak, fall back 
        # to a centroid approximation
        if np.any ( np.array([c,cl,cr,cd,cu]) < 0 ) and method == 'gaussian':
            method = 'centroid'
        
        # choose method
        if method == 'centroid':
            subp_peak_position = (((self.peak1[0]-1)*cl+self.peak1[0]*c+(self.peak1[0]+1)*cr)/(cl+c+cr),
                                ((self.peak1[1]-1)*cd+self.peak1[1]*c+(self.peak1[1]+1)*cu)/(cd+c+cu))
    
        elif method == 'gaussian':
            subp_peak_position = (self.peak1[0] + ( (log(cl)-log(cr) )/( 2*log(cl) - 4*log(c) + 2*log(cr) )),
                                self.peak1[1] + ( (log(cd)-log(cu) )/( 2*log(cd) - 4*log(c) + 2*log(cu) ))) 
    
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
    x = np.arange( field_shape[1] )*(window_size-overlap) + (window_size-1)/2.0
    y = np.arange( field_shape[0] )*(window_size-overlap) + (window_size-1)/2.0
    
    return np.meshgrid(x,y[::-1])

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

def correlate_windows( window_a, window_b, corr_method = 'fft', nfftx = None, nffty = None ):
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
        if nfftx is None:
            nfftx = 2*window_a.shape[0]
        if nffty is None:
            nffty = 2*window_a.shape[1]
        return fftshift(irfft2(rfft2(normalize_intensity(window_a),s=(nfftx,nffty))*np.conj(rfft2(normalize_intensity(window_b),s=(nfftx,nffty)))).real, axes=(0,1)  )
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
