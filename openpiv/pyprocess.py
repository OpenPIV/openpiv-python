import numpy.lib.stride_tricks
import numpy as np
from scipy.fft import rfft2 as rfft2_, irfft2 as irfft2_, fftshift as fftshift_
from numpy import ma
from scipy.signal import convolve2d as conv_
from numpy import log

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


def get_field_shape(image_size, search_area_size, overlap):
    """Compute the shape of the resulting flow field.

    Given the image size, the interrogation window size and
    the overlap size, it is possible to calculate the number
    of rows and columns of the resulting flow field.

    Parameters
    ----------
    image_size: two elements tuple
        a two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is
        the number of columns, easy to obtain using .shape

    search_area_size: tuple
        the size of the interrogation windows (if equal in frames A,B)
        or the search area (in frame B), the largest  of the two

    overlap: tuple
        the number of pixel by which two adjacent interrogation
        windows overlap.


    Returns
    -------
    field_shape : three elements tuple
        the shape of the resulting flow field
    """
    field_shape = (np.array(image_size) - np.array(search_area_size)) // (
        np.array(search_area_size) - np.array(overlap)
    ) + 1
    
    return field_shape


def get_coordinates(image_size, search_area_size, overlap, center_on_field = True):
    """Compute the x, y coordinates of the centers of the interrogation windows.
    the origin (0,0) is like in the image, top left corner
    positive x is an increasing column index from left to right
    positive y is increasing row index, from top to bottom


    Parameters
    ----------
    image_size: two elements tuple
        a two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is
        the number of columns.

    search_area_size: int
        the size of the search area windows, sometimes it's equal to
        the interrogation window size in both frames A and B

    overlap: int = 0 (default is no overlap)
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

        Coordinate system 0,0 is at the top left corner, positive
        x to the right, positive y from top downwards, i.e.
        image coordinate system

    """

    # get shape of the resulting flow field
    field_shape = get_field_shape(image_size,
                                  search_area_size,
                                  overlap)

    # compute grid coordinates of the search area window centers
    # note the field_shape[1] (columns) for x
    x = (
        np.arange(field_shape[1]) * (search_area_size - overlap)
        + (search_area_size) / 2.0
    )
    # note the rows in field_shape[0]
    y = (
        np.arange(field_shape[0]) * (search_area_size - overlap)
        + (search_area_size) / 2.0
    )

    # moving coordinates further to the center, so that the points at the
    # extreme left/right or top/bottom
    # have the same distance to the window edges. For simplicity only integer
    # movements are allowed.
    if center_on_field == True:
        x += (
            image_size[1]
            - 1
            - ((field_shape[1] - 1) * (search_area_size - overlap) +
                (search_area_size - 1))
        ) // 2
        y += (
            image_size[0] - 1
            - ((field_shape[0] - 1) * (search_area_size - overlap) +
               (search_area_size - 1))
        ) // 2

        # the origin 0,0 is at top left
        # the units are pixels

    return np.meshgrid(x, y)


def get_rect_coordinates(frame_a, window_size, overlap, center_on_field = False):
    '''
    Rectangular grid version of get_coordinates.
    '''
    if isinstance(window_size, tuple) == False and isinstance(window_size, list) == False:
        window_size = [window_size, window_size]
    if isinstance(overlap, tuple) == False and isinstance(overlap, list) == False:
        overlap = [overlap, overlap]
    _, y = get_coordinates(frame_a, window_size[0], overlap[0], center_on_field = False)
    x, _ = get_coordinates(frame_a, window_size[1], overlap[1], center_on_field = False)
    
    return np.meshgrid(x[0,:], y[:,0])


def sliding_window_array(image, window_size = 64, overlap = 32):
    '''
    This version does not use numpy as_strided and is much more memory efficient.
    Basically, we have a 2d array and we want to perform cross-correlation
    over the interrogation windows. An approach could be to loop over the array
    but loops are expensive in python. So we create from the array a new array
    with three dimension, of size (n_windows, window_size, window_size), in
    which each slice, (along the first axis) is an interrogation window. 
    '''
    if isinstance(window_size, tuple) == False and isinstance(window_size, list) == False:
        window_size = [window_size, window_size]
    if isinstance(overlap, tuple) == False and isinstance(overlap, list) == False:
        overlap = [overlap, overlap]

    x, y = get_rect_coordinates(image.shape, window_size, overlap, center_on_field = False)
    x = (x - window_size[1]//2).astype(int); y = (y - window_size[0]//2).astype(int)
    x, y = np.reshape(x, (-1,1,1)), np.reshape(y, (-1,1,1))

    win_x, win_y = np.meshgrid(np.arange(0, window_size[1]), np.arange(0, window_size[0]))
    win_x = win_x[np.newaxis,:,:] + x
    win_y = win_y[np.newaxis,:,:] + y
    windows = image[win_y, win_x]
    
    return windows


def moving_window_array(array, window_size, overlap):
    """
    This is a nice numpy trick. The concept of numpy strides should be
    clear to understand this code.

    Basically, we have a 2d array and we want to perform cross-correlation
    over the interrogation windows. An approach could be to loop over the array
    but loops are expensive in python. So we create from the array a new array
    with three dimension, of size (n_windows, window_size, window_size), in
    which each slice, (along the first axis) is an interrogation window.

    """
    sz = array.itemsize
    shape = array.shape
    array = np.ascontiguousarray(array)

    strides = (
        sz * shape[1] * (window_size - overlap),
        sz * (window_size - overlap),
        sz * shape[1],
        sz,
    )
    shape = (
        int((shape[0] - window_size) / (window_size - overlap)) + 1,
        int((shape[1] - window_size) / (window_size - overlap)) + 1,
        window_size,
        window_size,
    )

    return numpy.lib.stride_tricks.as_strided(
        array, strides=strides, shape=shape
    ).reshape(-1, window_size, window_size)


def find_first_peak(corr):
    """
    Find row and column indices of the first correlation peak.

    Parameters
    ----------
    corr : np.ndarray
        the correlation map fof the strided images (N,K,M) where
        N is the number of windows, KxM is the interrogation window size

    Returns
    -------
        (i,j) : integers, index of the peak position
        peak  : amplitude of the peak
    """

    return np.unravel_index(np.argmax(corr), corr.shape), corr.max()


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
        (i, j), tmp = find_first_peak(corr)

    # create a masked view of the corr
    tmp = corr.view(ma.MaskedArray)

    # set width x width square submatrix around the first correlation peak as
    # masked.
    # Before check if we are not too close to the boundaries, otherwise we
    # have negative indices
    iini = max(0, i - width)
    ifin = min(i + width + 1, corr.shape[0])
    jini = max(0, j - width)
    jfin = min(j + width + 1, corr.shape[1])
    tmp[iini:ifin, jini:jfin] = ma.masked
    (i, j), corr_max2 = find_first_peak(tmp)

    return (i, j), corr_max2


def find_all_first_peaks(corr):
    '''
    Find row and column indices of the first correlation peak.

    Parameters
    ----------
    corr : np.ndarray
        the correlation map fof the strided images (N,K,M) where
        N is the number of windows, KxM is the interrogation window size

    Returns
    -------
        index_list : integers, index of the peak position in (N,i,j)
        peaks_max  : amplitude of the peak
    '''
    ind = corr.reshape(corr.shape[0], -1).argmax(-1)
    peaks = np.array(np.unravel_index(ind, corr.shape[-2:]))
    peaks = np.vstack((peaks[0], peaks[1])).T
    index_list = [(i, v[0], v[1]) for i, v in enumerate(peaks)]
    peaks_max = np.nanmax(corr, axis = (-2, -1))
    return np.array(index_list), np.array(peaks_max)


def find_all_second_peaks(corr, width = 2, return_corr = False):
    '''
    Find row and column indices of the first correlation peak.

    Parameters
    ----------
    corr : np.ndarray
        the correlation map fof the strided images (N,K,M) where
        N is the number of windows, KxM is the interrogation window size
        
    width : int
        the half size of the region around the first correlation
        peak to ignore for finding the second peak
    
    return_corr: bool
        return the correlation matix. Can be used to find second peaks
        for subpixel approximation.
        
    Returns
    -------
        index_list : integers, index of the peak position in (N,i,j)
        peaks_max  : amplitude of the peak
    '''
    indexes = find_all_first_peaks(corr)[0].astype(int)
    ind = indexes[:, 0]
    x = indexes[:, 1]
    y = indexes[:, 2]
    iini = x - width
    ifin = x + width + 1
    jini = y - width
    jfin = y + width + 1
    iini[iini < 0] = 0 # border checking
    ifin[ifin > corr.shape[1]] = corr.shape[1]
    jini[jini < 0] = 0
    jfin[jfin > corr.shape[2]] = corr.shape[2]
    # create a masked view of the corr, is too slow?
    #tmp = corr.view(np.ma.MaskedArray)
    tmp = corr.copy()
    for i in ind:
        tmp[i, iini[i]:ifin[i], jini[i]:jfin[i]] = 0 #ma.masked 
    indexes, peaks = find_all_first_peaks(tmp)
    if return_corr == True:
        return tmp
    else:
        return indexes, peaks

    
def find_subpixel_peak_position(corr, subpixel_method="gaussian"):
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
         one of the following methods to estimate subpixel location of the
         peak:
         'centroid' [replaces default if correlation map is negative],
         'gaussian' [default if correlation map is positive],
         'parabolic'.

    Returns
    -------
    subp_peak_position : two elements tuple
        the fractional row and column indices for the sub-pixel
        approximation of the correlation peak.
        If the first peak is on the border of the correlation map
        or any other problem, the returned result is a tuple of NaNs.
    """

    # initialization
    # default_peak_position = (np.floor(corr.shape[0] / 2.),
    # np.floor(corr.shape[1] / 2.))
    # default_peak_position = np.array([0,0])
    eps = 1e-7
    # subp_peak_position = tuple(np.floor(np.array(corr.shape)/2))
    subp_peak_position = (np.nan, np.nan)  # any wrong position will mark nan

    # check inputs
    if subpixel_method not in ("gaussian", "centroid", "parabolic"):
        raise ValueError(f"Method not implemented {subpixel_method}")

    # the peak locations
    (peak1_i, peak1_j), _ = find_first_peak(corr)

    # import pdb; pdb.set_trace()

    # the peak and its neighbours: left, right, down, up
    # but we have to make sure that peak is not at the border
    # @ErichZimmer noticed this bug for the small windows

    if ((peak1_i == 0) | (peak1_i == corr.shape[0]-1) |
       (peak1_j == 0) | (peak1_j == corr.shape[1]-1)):
        return subp_peak_position
    else:
        corr += eps  # prevents log(0) = nan if "gaussian" is used (notebook)
        c = corr[peak1_i, peak1_j]
        cl = corr[peak1_i - 1, peak1_j]
        cr = corr[peak1_i + 1, peak1_j]
        cd = corr[peak1_i, peak1_j - 1]
        cu = corr[peak1_i, peak1_j + 1]

        # gaussian fit
        if np.logical_and(np.any(np.array([c, cl, cr, cd, cu]) < 0),
                          subpixel_method == "gaussian"):
            subpixel_method = "parabolic"

        # try:
        if subpixel_method == "centroid":
            subp_peak_position = (
                ((peak1_i - 1) * cl + peak1_i * c + (peak1_i + 1) * cr) /
                (cl + c + cr),
                ((peak1_j - 1) * cd + peak1_j * c + (peak1_j + 1) * cu) /
                (cd + c + cu),
            )

        elif subpixel_method == "gaussian":
            nom1 = log(cl) - log(cr)
            den1 = 2 * log(cl) - 4 * log(c) + 2 * log(cr)
            nom2 = log(cd) - log(cu)
            den2 = 2 * log(cd) - 4 * log(c) + 2 * log(cu)

            subp_peak_position = (
                peak1_i + np.divide(nom1, den1, out=np.zeros(1),
                                    where=(den1 != 0.0))[0],
                peak1_j + np.divide(nom2, den2, out=np.zeros(1),
                                    where=(den2 != 0.0))[0],
            )

        elif subpixel_method == "parabolic":
            subp_peak_position = (
                peak1_i + (cl - cr) / (2 * cl - 4 * c + 2 * cr),
                peak1_j + (cd - cu) / (2 * cd - 4 * c + 2 * cu),
            )

        return subp_peak_position


def sig2noise_ratio(correlation, sig2noise_method="peak2peak", width=2):
    """
    Computes the signal to noise ratio from the correlation map.

    The signal to noise ratio is computed from the correlation map with
    one of two available method. It is a measure of the quality of the
    matching between to interrogation windows.

    Parameters
    ----------
    corr : 3d np.ndarray
        the correlation maps of the image pair, concatenated along 0th axis

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
    sig2noise : np.array
        the signal to noise ratios from the correlation maps.

    """
    sig2noise = np.zeros(correlation.shape[0])
    corr_max1 = np.zeros(correlation.shape[0])
    corr_max2 = np.zeros(correlation.shape[0])
    if sig2noise_method == "peak2peak":
        for i, corr in enumerate(correlation):
            # compute first peak position
            (peak1_i, peak1_j), corr_max1[i] = find_first_peak(corr)

            condition = (
                corr_max1[i] < 1e-3
                or peak1_i == 0
                or peak1_i == corr.shape[0] - 1
                or peak1_j == 0
                or peak1_j == corr.shape[1] - 1
            )

            if condition:
                # return zero, since we have no signal.
                # no point to get the second peak, save time
                sig2noise[i] = 0.0
            else:
                # find second peak height
                (peak2_i, peak2_j), corr_max2 = find_second_peak(
                    corr, peak1_i, peak1_j, width=width
                )

                condition = (
                    corr_max2 == 0
                    or peak2_i == 0
                    or peak2_i == corr.shape[0] - 1
                    or peak2_j == 0
                    or peak2_j == corr.shape[1] - 1
                )
                if condition:  # mark failed peak2
                    corr_max2 = np.nan

                sig2noise[i] = corr_max1[i] / corr_max2

    elif sig2noise_method == "peak2mean":  # only one loop
        for i, corr in enumerate(correlation):
            # compute first peak position
            (peak1_i, peak1_j), corr_max1[i] = find_first_peak(corr)

            condition = (
                corr_max1[i] < 1e-3
                or peak1_i == 0
                or peak1_i == corr.shape[0] - 1
                or peak1_j == 0
                or peak1_j == corr.shape[1] - 1
            )

            if condition:
                # return zero, since we have no signal.
                # no point to get the second peak, save time
                corr_max1[i] = 0.0

        # find means of all the correlation maps
        corr_max2 = np.abs(correlation.mean(axis=(-2, -1)))
        corr_max2[corr_max2 == 0] = np.nan  # mark failed ones

        sig2noise = corr_max1 / corr_max2

    else:
        raise ValueError("wrong sig2noise_method")

    # sig2noise is zero for all failed ones
    sig2noise[np.isnan(sig2noise)] = 0.0

    return sig2noise


def vectorized_sig2noise_ratio(correlation, 
                               sig2noise_method = 'peak2peak',
                               width = 2):
    '''
    Computes the signal to noise ratio from the correlation map in a
    mostly vectorized approach, thus much faster.

    The signal to noise ratio is computed from the correlation map with
    one of two available method. It is a measure of the quality of the
    matching between to interrogation windows.

    Parameters
    ----------
    corr : 3d np.ndarray
        the correlation maps of the image pair, concatenated along 0th axis

    sig2noise_method: string
        the method for evaluating the signal to noise ratio value from
        the correlation map. Can be `peak2peak`, `peak2mean` or None
        if no evaluation should be made.

    width : int, optional
        the half size of the region around the first
        correlation peak to ignore for finding the second
        peak. [default: 2]. Only used if sig2noise_method==peak2peak.

    Returns
    -------
    sig2noise : np.array
        the signal to noise ratios from the correlation maps.
    '''
    if sig2noise_method == "peak2peak":
        ind1, peaks1 = find_all_first_peaks(correlation)
        ind2, peaks2 = find_all_second_peaks(correlation, width = width)
        peaks1_i, peaks1_j = ind1[:, 1], ind1[:, 2]
        peaks2_i, peaks2_j = ind2[:, 1], ind2[:, 2]
        # peak checking
        flag = np.zeros(peaks1.shape).astype(bool)
        flag[peaks1 < 1e-3] = True
        flag[peaks1_i == 0] = True
        flag[peaks1_i == correlation.shape[1]-1] = True
        flag[peaks1_j == 0] = True
        flag[peaks1_j == correlation.shape[2]-1] = True
        flag[peaks2 < 1e-3] = True
        flag[peaks2_i == 0] = True
        flag[peaks2_i == correlation.shape[1]-1] = True
        flag[peaks2_j == 0] = True
        flag[peaks2_j == correlation.shape[2]-1] = True
        # peak-to-peak calculation
        peak2peak = np.divide(
            peaks1, peaks2,
            out=np.zeros_like(peaks1),
            where=(peaks2 > 0.0)
        )
        peak2peak[flag==True] = 0 # replace invalid values
        return peak2peak
    
    elif sig2noise_method == "peak2mean":
        peaks, peaks1max = find_all_first_peaks(correlation)
        peaks = np.array(peaks)
        peaks1_i, peaks1_j = peaks[:,1], peaks[:, 2]
        peaks2mean = np.abs(np.nanmean(correlation, axis = (-2, -1)))
        # peak checking        
        flag = np.zeros(peaks1max.shape).astype(bool)
        flag[peaks1max < 1e-3] = True
        flag[peaks1_i == 0] = True
        flag[peaks1_i == correlation.shape[1]-1] = True
        flag[peaks1_j == 0] = True
        flag[peaks1_j == correlation.shape[2]-1] = True
        # peak-to-mean calculation
        peak2mean = np.divide(
            peaks1max, peaks2mean,
            out=np.zeros_like(peaks1max),
            where=(peaks2mean > 0.0)
        )
        peak2mean[flag == True] = 0 # replace invalid values
        return peak2mean
    else:
        raise ValueError(f"sig2noise_method not supported: {sig2noise_method}")
        
        
def fft_correlate_images(image_a, image_b,
                         correlation_method="circular",
                         normalized_correlation=True,
                         conj = np.conj,
                         rfft2 = rfft2_,
                         irfft2 = irfft2_,
                         fftshift = fftshift_):
    """ FFT based cross correlation
    of two images with multiple views of np.stride_tricks()
    The 2D FFT should be applied to the last two axes (-2,-1) and the
    zero axis is the number of the interrogation window
    This should also work out of the box for rectangular windows.
    Parameters
    ----------
    image_a : 3d np.ndarray, first dimension is the number of windows,
        and two last dimensions are interrogation windows of the first image

    image_b : similar

    correlation_method : string
        one of the three methods implemented: 'circular' or 'linear'
        [default: 'circular].

    normalized_correlation : bool
        decides wether normalized correlation is done or not: True or False
        [default: True].
    
    conj : function
        function used for complex conjugate
    
    rfft2 : function
        function used for rfft2
    
    irfft2 : function
        function used for irfft2
    
    fftshift : function
        function used for fftshift
        
    """

    if normalized_correlation:
        # remove the effect of stronger laser or
        # longer exposure for frame B
        # image_a = match_histograms(image_a, image_b)

        # remove mean background, normalize to 0..1 range
        image_a = normalize_intensity(image_a)
        image_b = normalize_intensity(image_b)

    s1 = np.array(image_a.shape[-2:])
    s2 = np.array(image_b.shape[-2:])

    if correlation_method == "linear":
        # have to be normalized, mainly because of zero padding
        size = s1 + s2 - 1
        fsize = 2 ** np.ceil(np.log2(size)).astype(int)
        fslice = (slice(0, image_a.shape[0]),
                  slice((fsize[0]-s1[0])//2, (fsize[0]+s1[0])//2),
                  slice((fsize[1]-s1[1])//2, (fsize[1]+s1[1])//2))
        f2a = conj(rfft2(image_a, fsize, axes=(-2, -1)))
        f2b = rfft2(image_b, fsize, axes=(-2, -1))
        corr = fftshift(irfft2(f2a * f2b).real, axes=(-2, -1))[fslice]
    elif correlation_method == "circular":
        f2a = conj(rfft2(image_a))
        f2b = rfft2(image_b)
        corr = fftshift(irfft2(f2a * f2b).real, axes=(-2, -1))
    else:
        print("method is not implemented!")

    if normalized_correlation:
        corr = corr/(s2[0]*s2[1])  # for extended search area
        corr = np.clip(corr, 0, 1)
    return corr
    
    
def normalize_intensity(window):
    """Normalize interrogation window or strided image of many windows,
       by removing the mean intensity value per window and clipping the
       negative values to zero

    Parameters
    ----------
    window :  2d np.ndarray
        the interrogation window array

    Returns
    -------
    window :  2d np.ndarray
        the interrogation window array, with mean value equal to zero and
        intensity normalized to -1 +1 and clipped if some pixels are
        extra low/high
    """
    window = window.astype(np.float32)
    window -= window.mean(axis=(-2, -1),
                          keepdims=True, dtype=np.float32)
    tmp = window.std(axis=(-2, -1), keepdims=True)
    window = np.divide(window, tmp, out=np.zeros_like(window),
                       where=(tmp != 0))
    return np.clip(window, 0, window.max())


def fft_norm_correlate_images(image_a, image_b,
                              correlation_method="circular",
                              conj = np.conj,
                              rfft2 = rfft2_,
                              irfft2 = irfft2_,
                              fftshift = fftshift_):
    """ FFT based normalized cross correlation
    of two images with multiple views of np.stride_tricks()
    The 2D FFT should be applied to the last two axes (-2,-1) and the
    zero axis is the number of the interrogation window
    This should also work out of the box for rectangular windows.
    Parameters
    ----------
    image_a : 3d np.ndarray, first dimension is the number of windows,
        and two last dimensions are interrogation windows of the first image

    image_b : similar

    correlation_method : string
        one of the three methods implemented: 'circular' or 'linear'
        [default: 'circular].
    
    conj : function
        function used for complex conjugate
    
    rfft2 : function
        function used for rfft2
    
    irfft2 : function
        function used for irfft2
    
    fftshift : function
        function used for fftshift
        
    """
    return fft_correlate_images(
        image_a - image_a.mean(axis = (-2,-1))[:,np.newaxis, np.newaxis].astype(np.int16),
        image_b - image_b.mean(axis = (-2,-1))[:,np.newaxis, np.newaxis].astype(np.int16),
        correlation_method=correlation_method,
        normalized_correlation=False,
        rfft2 = rfft2,
        irfft2 = irfft2,
        conj = conj,
        fftshift = fftshift,
    ) / (image_a.shape[1] * image_a.std(axis = (-2,-1))[:, np.newaxis, np.newaxis] * 
         image_b.shape[2] * image_b.std(axis = (-2,-1))[:, np.newaxis, np.newaxis])
    

def phase_correlation(image_a, image_b,
                      correlation_method = 'circular',
                      normalized_correlation = True,
                      conj = np.conj,
                      rfft2 = rfft2_,
                      irfft2 = irfft2_,
                      fftshift = fftshift_):
    '''
    Phase filtering to produce a phase-only correlation. Two methods
    are implemented here: Phase-only correlation and "symmetric" phase 
    correlation, which is supposedly more robust.
    Parameters
    ----------
    image_a : 3d np.ndarray, first dimension is the number of windows,
        and two last dimensions are interrogation windows of the first image

    image_b : similar

    correlation_method : string
        one of the two methods implemented: 'circular' or 'linear'
        [default: 'circular].
        
    normalized_correlation : bool
        decides wether normalized correlation is done or not: True or False
        [default: True].
        
    Returns
    -------
    corr : 3d np.ndarray
        a three dimensions array for the correlation function.
    '''
    if correlation_method not in ['circular', 'linear']:
        raise ValueError(f'Correlation method not supported {correlation_method}')

    s1 = np.array(image_a.shape[-2:])
    s2 = np.array(image_b.shape[-2:])
    if normalized_correlation == True:
        norm = (image_a.shape[-2] * image_a.std(axis = (-2,-1))[:, np.newaxis, np.newaxis] * 
                image_b.shape[-1] * image_b.std(axis = (-2,-1))[:, np.newaxis, np.newaxis])
        image_a -= image_a.mean(axis = (-2,-1))[:,np.newaxis, np.newaxis].astype(np.int16)
        image_b -= image_b.mean(axis = (-2,-1))[:,np.newaxis, np.newaxis].astype(np.int16)
    else:
        norm = 1
        
    if correlation_method == 'circular':
        f2a = conj(rfft2(image_a))
        f2b = rfft2(image_b)
        r = f2a * f2b
        r /= (np.sqrt(np.absolute(f2a) * np.absolute(f2b)) + 1e-10)
        corr = fftshift(irfft2(r).real, axes=(-2, -1))
    else:
        size = s1 + s2 - 1
        fsize = 2 ** np.ceil(np.log2(size)).astype(int)
        fslice = (slice(0, image_a.shape[0]),
                  slice((fsize[0]-s1[0])//2, (fsize[0]+s1[0])//2),
                  slice((fsize[1]-s1[1])//2, (fsize[1]+s1[1])//2))
        f2a = conj(rfft2(image_a, fsize, axes=(-2, -1)))
        f2b = rfft2(image_b, fsize, axes=(-2, -1))
        r = f2a * f2b
        r /= (np.sqrt(np.absolute(f2a) * np.absolute(f2b)) + 1e-10)
        corr = fftshift(irfft2(r), axes=(-2, -1)).real[fslice]
    return corr / norm


def correlate_windows(window_a, window_b, correlation_method="fft",
                      convolve2d = conv_, rfft2 = rfft2_, irfft2 = irfft2_):
    """Compute correlation function between two interrogation windows.
    The correlation function can be computed by using the correlation
    theorem to speed up the computation.
    Parameters
    ----------
    window_a : 2d np.ndarray
        a two dimensions array for the first interrogation window
        
    window_b : 2d np.ndarray
        a two dimensions array for the second interrogation window
        
    correlation_method : string, methods currently implemented:
            'circular' - FFT based without zero-padding
            'linear' -  FFT based with zero-padding
            'direct' -  linear convolution based
            Default is 'fft', which is much faster.

    convolve2d : function
        function used for 2d convolutions
    
    rfft2 : function
        function used for rfft2
    
    irfft2 : function
        function used for irfft2
        
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

    # first we remove the mean to normalize contrast and intensity
    # the background level which is take as a mean of the image
    # is subtracted
    # import pdb; pdb.set_trace()
    window_a = normalize_intensity(window_a)
    window_b = normalize_intensity(window_b)

    # this is not really circular one, as we pad a bit to get fast 2D FFT,
    # see fft_correlate for implementation
    if correlation_method in ("circular", "fft"):
        corr = fft_correlate_windows(window_a, window_b, rfft2 = rfft2, irfft2 = irfft2)
    elif correlation_method == "linear":
        # save the original size:
        s1 = np.array(window_a.shape)
        s2 = np.array(window_b.shape)
        size = s1 + s2 - 1
        fslice = tuple([slice(0, int(sz)) for sz in size])
        # and slice only the relevant part
        corr = fft_correlate_windows(window_a, window_b, rfft2 = rfft2, irfft2 = irfft2)[fslice]
    elif correlation_method == "direct":
        corr = convolve2d(window_a, window_b[::-1, ::-1], "full")
    else:
        raise ValueError("method is not implemented")

    return corr


def fft_correlate_windows(window_a, window_b,
                          rfft2 = rfft2_,
                          irfft2 = irfft2_):
    """ FFT based cross correlation
    it is a so-called linear convolution based,
    since we increase the size of the FFT to
    reduce the edge effects.
    This should also work out of the box for rectangular windows.
    
    Parameters
    ----------
    window_a : 2d np.ndarray
        a two dimensions array for the first interrogation window
        
    window_b : 2d np.ndarray
        a two dimensions array for the second interrogation window
        
    rfft2 : function
        function used for rfft2
    
    irfft2 : function
        function used for irfft2
        
    # from Stackoverflow:
    from scipy import linalg
    import numpy as np
    # works for rectangular windows as well
    x = [[1 , 0 , 0 , 0] , [0 , -1 , 0 , 0] , [0 , 0 , 3 , 0] ,
        [0 , 0 , 0 , 1], [0 , 0 , 0 , 1]]
    x = np.array(x,dtype=np.float)
    y = [[4 , 5] , [3 , 4]]
    y = np.array(y)
    print ("conv:" ,  signal.convolve2d(x , y , 'full'))
    s1 = np.array(x.shape)
    s2 = np.array(y.shape)
    size = s1 + s2 - 1
    fsize = 2 ** np.ceil(np.log2(size)).astype(int)
    fslice = tuple([slice(0, int(sz)) for sz in size])
    new_x = np.fft.fft2(x , fsize)
    new_y = np.fft.fft2(y , fsize)
    result = np.fft.ifft2(new_x*new_y)[fslice].copy()
    print("fft for my method:" , np.array(result.real, np.int32))
    """
    s1 = np.array(window_a.shape)
    s2 = np.array(window_b.shape)
    size = s1 + s2 - 1
    fsize = 2 ** np.ceil(np.log2(size)).astype(int)
    fslice = tuple([slice(0, int(sz)) for sz in size])
    f2a = rfft2(window_a, fsize)
    f2b = rfft2(window_b[::-1, ::-1], fsize)
    corr = irfft2(f2a * f2b).real[fslice]
    return corr


def extended_search_area_piv(
    frame_a,
    frame_b,
    window_size,
    overlap=0,
    dt=1.0,
    search_area_size=None,
    correlation_method="circular",
    subpixel_method="gaussian",
    sig2noise_method='peak2mean',
    width=2,
    normalized_correlation=False,
    use_vectorized = False,
):
    """Standard PIV cross-correlation algorithm, with an option for
    extended area search that increased dynamic range. The search region
    in the second frame is larger than the interrogation window size in the
    first frame. For Cython implementation see
    openpiv.process.extended_search_area_piv

    This is a pure python implementation of the standard PIV cross-correlation
    algorithm. It is a zero order displacement predictor, and no iterative
    process is performed.

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

    correlation_method : string
        one of the two methods implemented: 'circular' or 'linear',
        default: 'circular', it's faster, without zero-padding
        'linear' requires also normalized_correlation = True (see below)

    subpixel_method : string
         one of the following methods to estimate subpixel location of the
         peak:
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

    search_area_size : int
       the size of the interrogation window in the second frame,
       default is the same interrogation window size and it is a
       fallback to the simplest FFT based PIV

    normalized_correlation: bool
        if True, then the image intensity will be modified by removing
        the mean, dividing by the standard deviation and
        the correlation map will be normalized. It's slower but could be
        more robust

    Returns
    -------
    u : 2d np.ndarray
        a two dimensional array containing the u velocity component,
        in pixels/seconds.

    v : 2d np.ndarray
        a two dimensional array containing the v velocity component,
        in pixels/seconds.

    sig2noise : 2d np.ndarray, ( optional: only if sig2noise_method != None )
        a two dimensional array the signal to noise ratio for each
        window pair.


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

    originally implemented in process.pyx in Cython and converted to
    a NumPy vectorized solution in pyprocess.py

    """
    if search_area_size is not None:
        if isinstance(search_area_size, tuple) == False and isinstance(search_area_size, list) == False:
            search_area_size = [search_area_size, search_area_size]
    if isinstance(window_size, tuple) == False and isinstance(window_size, list) == False:
        window_size = [window_size, window_size]
    if isinstance(overlap, tuple) == False and isinstance(overlap, list) == False:
        overlap = [overlap, overlap]
        
    # check the inputs for validity
    if search_area_size is None:
        search_area_size = window_size

    if overlap[0] >= window_size[0] or overlap[1] >= window_size[1]:
        raise ValueError("Overlap has to be smaller than the window_size")

    if search_area_size[0] < window_size[0] or search_area_size[1] < window_size[1]:
        raise ValueError("Search size cannot be smaller than the window_size")

    if (window_size[1] > frame_a.shape[0]) or (window_size[0] > frame_a.shape[1]):
        raise ValueError("window size cannot be larger than the image")

    # get field shape
    n_rows, n_cols = get_field_shape(frame_a.shape, search_area_size, overlap)

    # We implement the new vectorized code
    aa = sliding_window_array(frame_a, search_area_size, overlap)
    bb = sliding_window_array(frame_b, search_area_size, overlap)

    # for the case of extended seearch, the window size is smaller than
    # the search_area_size. In order to keep it all vectorized the
    # approach is to use the interrogation window in both
    # frames of the same size of search_area_asize,
    # but mask out the region around
    # the interrogation window in the frame A

    if search_area_size > window_size:
        # before masking with zeros we need to remove
        # edges

        aa = normalize_intensity(aa)
        bb = normalize_intensity(bb)

        mask = np.zeros((search_area_size[0], search_area_size[1])).astype(aa.dtype)
        pady = int((search_area_size[0] - window_size[0]) / 2)
        padx = int((search_area_size[1] - window_size[1]) / 2)
        mask[slice(pady, search_area_size[0] - pady),
             slice(padx, search_area_size[1] - padx)] = 1
        mask = np.broadcast_to(mask, aa.shape)
        aa *= mask

    corr = fft_correlate_images(aa, bb,
                                correlation_method=correlation_method,
                                normalized_correlation=normalized_correlation)
    if use_vectorized == True:
        u, v = vectorized_correlation_to_displacements(corr, n_rows, n_cols,
                                           subpixel_method=subpixel_method)
    else:
        u, v = correlation_to_displacement(corr, n_rows, n_cols,
                                           subpixel_method=subpixel_method)

    # return output depending if user wanted sig2noise information
    if sig2noise_method is not None:
        if use_vectorized == True:
            sig2noise = vectorized_sig2noise_ratio(
                corr, sig2noise_method=sig2noise_method, width=width
            )
        else:
            sig2noise = sig2noise_ratio(
                corr, sig2noise_method=sig2noise_method, width=width
            )
    else:
        sig2noise = np.zeros_like(u)*np.nan

    sig2noise = sig2noise.reshape(n_rows, n_cols)

    return u/dt, v/dt, sig2noise


def correlation_to_displacement(corr, n_rows, n_cols,
                                subpixel_method="gaussian"):
    """
    Correlation maps are converted to displacement for each interrogation
    window using the convention that the size of the correlation map
    is 2N -1 where N is the size of the largest interrogation window
    (in frame B) that is called search_area_size
    Inputs:
        corr : 3D nd.array
            contains output of the fft_correlate_images
        n_rows, n_cols : number of interrogation windows, output of the
            get_field_shape
    """
    # iterate through interrogation widows and search areas
    u = np.zeros((n_rows, n_cols))
    v = np.zeros((n_rows, n_cols))

    # center point of the correlation map
    default_peak_position = np.floor(np.array(corr[0, :, :].shape)/2)
    for k in range(n_rows):
        for m in range(n_cols):
            # look at studying_correlations.ipynb
            # the find_subpixel_peak_position returns
            peak = np.array(find_subpixel_peak_position(corr[k*n_cols+m, :, :],
                            subpixel_method=subpixel_method)) -\
                            default_peak_position

        # the horizontal shift from left to right is the u
        # the vertical displacement from top to bottom (increasing row) is v
        # x the vertical shift from top to bottom is row-wise shift is now
        # a negative vertical
            u[k, m], v[k, m] = peak[1], peak[0]

    return (u, v)


def vectorized_correlation_to_displacements(corr, 
                                            n_rows = None,
                                            n_cols = None,
                                            subpixel_method = 'gaussian', 
                                            offset_minimum = False,
                                            eps = 1e-7
):
    """
    Correlation maps are converted to displacement for each interrogation
    window using the convention that the size of the correlation map
    is 2N -1 where N is the size of the largest interrogation window
    (in frame B) that is called search_area_size
    
    Parameters
    ----------
    corr : 3D nd.array
        contains output of the fft_correlate_images
        
    n_rows, n_cols : 
        number of interrogation windows, output of the get_field_shape
        
    mask_width: int
        distance, in pixels, from the interrogation window in which 
        correlation peaks would be flagged as invalid
    Returns
    -------
    u, v: 2D nd.array
        2d array of displacements in pixels/dt
    """
    if subpixel_method not in ("gaussian", "centroid", "parabolic"):
        raise ValueError(f"Method not implemented {subpixel_method}")
    if offset_minimum == True:
        corr_min = corr.min(axis = (-2,-1)) # avoid negative peaks 
        corr_min[corr_min > 0] = 0
        corr -= corr_min[:, np.newaxis, np.newaxis]
    corr = corr.astype(np.float32) + eps # avoids division by zero
    peaks = find_all_first_peaks(corr)[0]
    ind, peaks_x, peaks_y = peaks[:,0], peaks[:,1], peaks[:,2]
    peaks1_i, peaks1_j = peaks_x, peaks_y
    
    # peak checking
    if subpixel_method in ("gaussian", "centroid", "parabolic"):
        mask_width = 1
    invalid = list(np.where(peaks1_i < mask_width)[0])
    invalid += list(np.where(peaks1_i > corr.shape[1] - mask_width - 1)[0])
    invalid += list(np.where(peaks1_j < mask_width - 0)[0])
    invalid += list(np.where(peaks1_j > corr.shape[2] - mask_width - 1)[0])
    peaks1_i[invalid] = corr.shape[1] // 2 # temp. so no errors would be produced
    peaks1_j[invalid] = corr.shape[2] // 2
    
    print(f"Found {len(invalid)} bad peak(s)")
    if len(invalid) == corr.shape[0]: # in case something goes horribly wrong 
        return np.zeros((np.size(corr, 0), 2))*np.nan
    
    #points
    c = corr[ind, peaks1_i, peaks1_j]
    cl = corr[ind, peaks1_i - 1, peaks1_j]
    cr = corr[ind, peaks1_i + 1, peaks1_j]
    cd = corr[ind, peaks1_i, peaks1_j - 1]
    cu = corr[ind, peaks1_i, peaks1_j + 1]
    
    if subpixel_method == "centroid":
        shift_i = ((peaks1_i - 1) * cl + peaks1_i * c + (peaks1_i + 1) * cr) / (cl + c + cr)
        shift_j = ((peaks1_j - 1) * cd + peaks1_j * c + (peaks1_j + 1) * cu) / (cd + c + cu)
        
    elif subpixel_method == "gaussian":
        inv = list(np.where(c <= 0)[0]) # get rid of any pesky NaNs
        inv += list(np.where(cl <= 0)[0])
        inv += list(np.where(cr <= 0)[0])
        inv += list(np.where(cu <= 0)[0])
        inv += list(np.where(cd <= 0)[0])
        
        #cl_, cr_ = np.delete(cl, inv), np.delete(cr, inv)
        #c_ = np.delete(c, inv)
        #cu_, cd_ = np.delete(cu, inv), np.delete(cd, inv)
        
        nom1 = log(cl) - log(cr)
        den1 = 2 * log(cl) - 4 * log(c) + 2 * log(cr)
        nom2 = log(cd) - log(cu)
        den2 = 2 * log(cd) - 4 * log(c) + 2 * log(cu)
        shift_i = np.divide(
            nom1, den1,
            out=np.zeros_like(nom1),
            where=(den1 != 0.0)
        )
        shift_j = np.divide(
            nom2, den2,
            out=np.zeros_like(nom2),
            where=(den2 != 0.0)
        )
        
        if len(inv) >= 1: 
            print(f'Found {len(inv)} negative correlation indices resulting in NaNs\n'+
                   'Fallback for negative indices is a 3 point parabolic curve method')
            shift_i[inv] = (cl[inv] - cr[inv]) / (2 * cl[inv] - 4 * c[inv] + 2 * cr[inv])
            shift_j[inv] = (cd[inv] - cu[inv]) / (2 * cd[inv] - 4 * c[inv] + 2 * cu[inv])
            
    elif subpixel_method == "parabolic":
        shift_i = (cl - cr) / (2 * cl - 4 * c + 2 * cr)
        shift_j = (cd - cu) / (2 * cd - 4 * c + 2 * cu)
        
    if subpixel_method != "centroid":
        disp_vy = (peaks1_i.astype(np.float64) + shift_i) - np.floor(np.array(corr.shape[1])/2)
        disp_vx = (peaks1_j.astype(np.float64) + shift_j) - np.floor(np.array(corr.shape[2])/2)
    else:
        disp_vy = shift_i - np.floor(np.array(corr.shape[1])/2)
        disp_vx = shift_j - np.floor(np.array(corr.shape[2])/2)
        
    disp_vx[invalid] = peaks_x[invalid]*np.nan
    disp_vy[invalid] = peaks_y[invalid]*np.nan
    #disp[ind, :] = np.vstack((disp_vx, disp_vy)).T
    #return disp[:,0].reshape((n_rows, n_cols)), disp[:,1].reshape((n_rows, n_cols))
    if n_rows == None or n_cols == None:
        return disp_vx, disp_vy
    else:
        return disp_vx.reshape((n_rows, n_cols)), disp_vy.reshape((n_rows, n_cols))
    
    
def nextpower2(i):
    """ Find 2^n that is equal to or greater than. """
    n = 1
    while n < i:
        n *= 2
    return n
