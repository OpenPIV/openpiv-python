"""This module contains a pure python implementation of the basic
cross-correlation algorithm for PIV image processing."""

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
from numpy.fft import rfft2, irfft2
from numpy import ma
from scipy.signal import convolve2d
from numpy import log


def get_coordinates(image_size, search_area_size, window_size, overlap):
    """Compute the x, y coordinates of the centers of the interrogation windows.

    Parameters
    ----------
    image_size: two elements tuple
        a two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is
        the number of columns.

    search_area_size: int
        the size of the search area windows.

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
    field_shape = get_field_shape(image_size, search_area_size, window_size, overlap)

    # compute grid coordinates of the search area window centers
    # compute grid coordinates of the search area window centers
    x = np.arange(field_shape[1]) * (window_size - overlap) + (search_area_size - 1)/2.0
    y = np.arange(field_shape[0]) * (window_size - overlap) + (search_area_size - 1)/2.0

    # moving coordinates further to the center, so that the points at the extreme left/right or top/bottom
    # have the same distance to the window edges. For simplicity only integer movements are allowed.
    x += (image_size[1] - 1 - ((field_shape[1] - 1) * (window_size - overlap) + (search_area_size - 1))) // 2
    y += (image_size[0] - 1 - ((field_shape[0] - 1) * (window_size - overlap) + (search_area_size - 1))) // 2

    return np.meshgrid(x, y)


def get_field_shape(image_size, search_area_size, window_size, overlap):
    """Compute the shape of the resulting flow field.

    Given the image size, the interrogation window size, the search_area size and
    the overlap size, it is possible to calculate the number
    of rows and columns of the resulting flow field.

    Parameters
    ----------
    image_size: two elements tuple
        a two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is
        the number of columns.

    search_area_size: int
        the size of the search area window.

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

    return ((image_size[0] - search_area_size) // (window_size - overlap) + 1,
            (image_size[1] - search_area_size) // (window_size - overlap) + 1)


def moving_window_array(array, window_size, overlap):
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
    array = np.ascontiguousarray(array)

    strides = (sz * shape[1] * (window_size - overlap),
               sz * (window_size - overlap), sz * shape[1], sz)
    shape = (int((shape[0] - window_size) / (window_size - overlap)) + 1, int(
        (shape[1] - window_size) / (window_size - overlap)) + 1, window_size, window_size)

    return numpy.lib.stride_tricks.as_strided(array, strides=strides, shape=shape).reshape(-1, window_size, window_size)


def find_first_peak(corr):
    """
    Find row and column indices of the first correlation peak.

    Parameters
    ----------
    corr : np.ndarray
        the correlation map

    Returns
    -------
    i : int
        the row index of the correlation peak

    j : int
        the column index of the correlation peak

    corr_max1 : int
        the value of the correlation peak

    """
    ind = corr.argmax()
    s = corr.shape[1]

    i = ind // s
    j = ind % s

    return i, j, corr.max()


def find_second_peak(corr, i=None, j=None, width=2):
    """
    Find the value of the second largest peak.

    The second largest peak is the height of the peak in
    the region outside a 3x3 submatrxi around the first
    correlation peak.

    Parameters
    ----------
    corr: np.ndarray
          the correlation map.

    i,j : ints
          row and column location of the first peak.

    width : int
        the half size of the region around the first correlation
        peak to ignore for finding the second peak.

    Returns
    -------
    i : int
        the row index of the second correlation peak.

    j : int
        the column index of the second correlation peak.

    corr_max2 : int
        the value of the second correlation peak.

    """

    if i is None or j is None:
        i, j, tmp = find_first_peak(corr)

    # create a masked view of the corr
    tmp = corr.view(ma.MaskedArray)

    # set width x width square submatrix around the first correlation peak as masked.
    # Before check if we are not too close to the boundaries, otherwise we
    # have negative indices
    iini = max(0, i - width)
    ifin = min(i + width + 1, corr.shape[0])
    jini = max(0, j - width)
    jfin = min(j + width + 1, corr.shape[1])
    tmp[iini:ifin, jini:jfin] = ma.masked
    i, j, corr_max2 = find_first_peak(tmp)

    return i, j, corr_max2


def find_subpixel_peak_position(corr, subpixel_method='gaussian'):
    """
    Find subpixel approximation of the correlation peak.

    This function returns a subpixels approximation of the correlation
    peak by using one of the several methods available. If requested,
    the function also returns the signal to noise ratio level evaluated
    from the correlation map.

    Parameters
    ----------
    corr : np.ndarray
        the correlation map.

    subpixel_method : string
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

    # initialization
    # default_peak_position = (np.floor(corr.shape[0] / 2.), np.floor(corr.shape[1] / 2.))
    default_peak_position = (0, 0)
    
    # the peak locations
    peak1_i, peak1_j, dummy = find_first_peak(corr)

    try:
        # the peak and its neighbours: left, right, down, up
        c = corr[peak1_i,   peak1_j]
        cl = corr[peak1_i - 1, peak1_j]
        cr = corr[peak1_i + 1, peak1_j]
        cd = corr[peak1_i,   peak1_j - 1]
        cu = corr[peak1_i,   peak1_j + 1]

        # gaussian fit
        if np.any(np.array([c, cl, cr, cd, cu]) < 0) and subpixel_method == 'gaussian':
            subpixel_method = 'centroid'

        try:
            if subpixel_method == 'centroid':
                subp_peak_position = (((peak1_i - 1) * cl + peak1_i * c + (peak1_i + 1) * cr) / (cl + c + cr),
                                      ((peak1_j - 1) * cd + peak1_j * c + (peak1_j + 1) * cu) / (cd + c + cu))

            elif subpixel_method == 'gaussian':
                subp_peak_position = (peak1_i + ((log(cl) - log(cr)) / (2 * log(cl) - 4 * log(c) + 2 * log(cr))),
                                      peak1_j + ((log(cd) - log(cu)) / (2 * log(cd) - 4 * log(c) + 2 * log(cu))))

            elif subpixel_method == 'parabolic':
                subp_peak_position = (peak1_i + (cl - cr) / (2 * cl - 4 * c + 2 * cr),
                                      peak1_j + (cd - cu) / (2 * cd - 4 * c + 2 * cu))

        except:
            subp_peak_position = default_peak_position

    except IndexError:
        subp_peak_position = default_peak_position
        
    return subp_peak_position[0] - default_peak_position[0], subp_peak_position[1] - default_peak_position[1]


def sig2noise_ratio(corr, sig2noise_method='peak2peak', width=2):
    """
    Computes the signal to noise ratio from the correlation map.

    The signal to noise ratio is computed from the correlation map with
    one of two available method. It is a measure of the quality of the
    matching between to interrogation windows.

    Parameters
    ----------
    corr : 2d np.ndarray
        the correlation map.

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

    # compute first peak position
    peak1_i, peak1_j, corr_max1 = find_first_peak(corr)

    # now compute signal to noise ratio
    if sig2noise_method == 'peak2peak':
        # find second peak height
        peak2_i, peak2_j, corr_max2 = find_second_peak(
            corr, peak1_i, peak1_j, width=width)

        # if it's an empty interrogation window
        # if the image is lacking particles, totally black it will correlate to very low value, but not zero
        # if the first peak is on the borders, the correlation map is also
        # wrong
        if corr_max1 < 1e-3 or (peak1_i == 0 or peak1_j == corr.shape[0] or peak1_j == 0 or peak1_j == corr.shape[1] or
                                peak2_i == 0 or peak2_j == corr.shape[0] or peak2_j == 0 or peak2_j == corr.shape[1]):
            # return zero, since we have no signal.
            return 0.0

    elif sig2noise_method == 'peak2mean':
        # find mean of the correlation map
        corr_max2 = corr.mean()

    else:
        raise ValueError('wrong sig2noise_method')

    # avoid dividing by zero
    try:
        sig2noise = corr_max1 / corr_max2
    except ValueError:
        sig2noise = np.inf

    return sig2noise


def correlate_windows(window_a, window_b, corr_method='fft', nfftx=0, nffty=0):
    """Compute correlation function between two interrogation windows.

    The correlation function can be computed by using the correlation
    theorem to speed up the computation.

    Parameters
    ----------
    window_a : 2d np.ndarray
        a two dimensions array for the first interrogation window, 

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
    
    Note that due to the wish to use 2^N windows for faster FFT
    we use a slightly different convention for the size of the 
    correlation map. The theory says it is M+N-1, and the 
    'direct' method gets this size out
    the FFT-based method returns M+N size out, where M is the window_size
    and N is the search_area_size
    It leads to inconsistency of the output 
    """
    
    if corr_method == 'fft':
        window_b = np.conj(window_b[::-1, ::-1])
        if nfftx == 0:
            nfftx = nextpower2(window_b.shape[0] + window_a.shape[0])  
        if nffty == 0:
            nffty = nextpower2(window_b.shape[1] + window_a.shape[1]) 
        
        f2a = rfft2(normalize_intensity(window_a), s=(nfftx, nffty))
        f2b = rfft2(normalize_intensity(window_b), s=(nfftx, nffty))
        corr = irfft2(f2a * f2b).real
        corr = corr[:window_a.shape[0] + window_b.shape[0] - 1,
                    :window_b.shape[1] + window_a.shape[1] - 1]
        return corr
    elif corr_method == 'direct':
        return convolve2d(normalize_intensity(window_a),
                          normalize_intensity(window_b[::-1, ::-1]), 'full')

    else:
        raise ValueError('method is not implemented')


def normalize_intensity(window):
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


def extended_search_area_piv(
        frame_a, frame_b, 
        window_size, 
        overlap=0, 
        dt=1.0,
        search_area_size=0, 
        corr_method='fft',
        subpixel_method='gaussian', 
        sig2noise_method=None,
        width=2, 
        nfftx=0, nffty=0):
    """Standard PIV cross-correlation algorithm, with an option for 
    extended area search that increased dynamic range. The search region
    in the second frame is larger than the interrogation window size in the 
    first frame. For Cython implementation see 
    openpiv.process.extended_search_area_piv

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
    
    search_area_size : int 
       the size of the interrogation window in the second frame, 
       default is the same interrogation window size and it is a 
       fallback to the simplest FFT based PIV


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
    
    # check the inputs for validity
    if search_area_size == 0:
        search_area_size = window_size
    
    if overlap >= window_size:
        raise ValueError('Overlap has to be smaller than the window_size')
    
    if search_area_size < window_size:
        raise ValueError('Search size cannot be smaller than the window_size')
        
    if (window_size > frame_a.shape[0]) or (window_size > frame_a.shape[1]):
        raise ValueError('window size cannot be larger than the image')
        
    # get field shape
    n_rows, n_cols = get_field_shape(frame_a.shape,
                                     search_area_size, window_size, overlap)

    # create empty arrays for deformation in x and y direction
    u, v = np.zeros((n_rows, n_cols)), np.zeros((n_rows, n_cols))
    
    # if we want sig2noise information, allocate memory
    if sig2noise_method is not None:
        sig2noise = np.zeros((n_rows, n_cols))

    # shift for x and y coordinates of the search area windows so that the centers of search area windows have
    # the same distances to the image edge at all sides. For simplicity only shifts by integers are allowed
    x_centering = (frame_a.shape[1] - 1 - ((n_cols - 1) * (window_size - overlap) + (search_area_size - 1))) // 2
    y_centering = (frame_a.shape[0] - 1 - ((n_rows - 1) * (window_size - overlap) + (search_area_size - 1))) // 2

    # iterate through interrogation widows and search areas
    for k in range(n_rows):
        for m in range(n_cols):

            # centers of search area. (window_size - overlap) defines the distance between each center
            # and (search_area_size - 1)/2.0 moves the center points away from the left or top image edge
            y = k * (window_size - overlap) + (search_area_size - 1)/2.0
            x = m * (window_size - overlap) + (search_area_size - 1)/2.0

            # moving the coordinates a bit to the center, to guarantee that the distance
            #  of a extreme point at the image edges is symmetric all all edges
            x += x_centering
            y += y_centering

            # left, right, top, bottom indices of the search area edges
            # note that x - (search_area_size +/- 1)/2  always returns an integer due to the definition of x and y
            # see also "get_coordinates()"
            il = int(y - (search_area_size - 1) / 2.0)
            ir = int(y + (search_area_size + 1) / 2.0)
            it = int(x - (search_area_size - 1) / 2.0)
            ib = int(x + (search_area_size + 1) / 2.0)
            # picking the search area from frame b
            window_b = frame_b[il:ir, it:ib]

            # left, right, top, bottom indices of the interrogation window
            # Sometimes the interrogation window cannot be placed in the middle of the search area, e.g.
            # in the case of window_size=3 search_area_size=4. In this case the interrogation window
            # is shifted 0.5 pixels to the left/top, which is achieved by rounding
            # the indices down during the int() conversion
            il = int(y - (window_size - 1) / 2)
            ir = int(y + (window_size + 1) / 2)
            it = int(x - (window_size - 1) / 2)
            ib = int(x + (window_size + 1) / 2)
            # picking the interrogation window from frame a
            window_a = frame_a[il:ir, it:ib]

            if np.any(window_a):
                corr = correlate_windows(window_a, window_b,
                                         corr_method=corr_method,
                                         nfftx=nfftx, nffty=nffty)
                #                 plt.figure()
                #                 plt.contourf(corr)
                #                 plt.show()
                # get subpixel approximation for peak position row and column index
                row, col = find_subpixel_peak_position(corr,
                                                       subpixel_method=subpixel_method)

                row -= (search_area_size + window_size - 1) // 2
                col -= (search_area_size + window_size - 1) // 2

                # get displacements, apply coordinate system definition
                u[k, m], v[k, m] = -col, row

                # get signal to noise ratio
                if sig2noise_method is not None:
                    sig2noise[k, m] = sig2noise_ratio(
                        corr, sig2noise_method=sig2noise_method, width=width)

    # return output depending if user wanted sig2noise information
    if sig2noise_method is not None:
        return u/dt, v/dt, sig2noise
    else:
        return u/dt, v/dt


def nextpower2(i):
    """ Find 2^n that is equal to or greater than. """
    n = 1
    while n < i:
        n *= 2
    return n
