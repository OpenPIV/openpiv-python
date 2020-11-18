import numpy.lib.stride_tricks
import numpy as np
from numpy.fft import rfft2, irfft2
from numpy import ma
from scipy.signal import convolve2d
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


def get_coordinates(image_size, search_area_size, overlap):
    """Compute the x, y coordinates of the centers of the interrogation windows.

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

    """

    # get shape of the resulting flow field
    field_shape = get_field_shape(image_size,
                                  search_area_size,
                                  overlap)

    # compute grid coordinates of the search area window centers
    # compute grid coordinates of the search area window centers
    x = (
        np.arange(field_shape[1]) * (search_area_size - overlap)
        + (search_area_size - 1) / 2.0
    )
    y = (
        np.arange(field_shape[0]) * (search_area_size - overlap)
        + (search_area_size - 1) / 2.0
    )

    # moving coordinates further to the center, so that the points at the
    # extreme left/right or top/bottom
    # have the same distance to the window edges. For simplicity only integer
    # movements are allowed.
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

    return np.meshgrid(x, y)


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
        the correlation map

    Returns
    -------

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
    """

    # initialization
    # default_peak_position = (np.floor(corr.shape[0] / 2.),
    # np.floor(corr.shape[1] / 2.))
    # default_peak_position = np.array([0,0])
    eps = 1e-7
    subp_peak_position = (0.0, 0.0)

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
            subp_peak_position = (
                peak1_i + ((log(cl) - log(cr)) / (2 * log(cl) - 4 * log(c) +
                           2 * log(cr))),
                peak1_j + ((log(cd) - log(cu)) / (2 * log(cd) - 4 * log(c) +
                           2 * log(cu))),
            )

        elif subpixel_method == "parabolic":
            subp_peak_position = (
                peak1_i + (cl - cr) / (2 * cl - 4 * c + 2 * cr),
                peak1_j + (cd - cu) / (2 * cd - 4 * c + 2 * cu),
            )

    #     except BaseException:
    #         subp_peak_position = default_peak_position

    #     except IndexError:
    #         subp_peak_position = default_peak_position

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
                or peak1_j == corr.shape[0]
                or peak1_j == 0
                or peak1_j == corr.shape[1]
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
                    or peak2_j == corr.shape[0]
                    or peak2_j == 0
                    or peak2_j == corr.shape[1]
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
                or peak1_j == corr.shape[0]
                or peak1_j == 0
                or peak1_j == corr.shape[1]
            )

            if condition:
                # return zero, since we have no signal.
                # no point to get the second peak, save time
                sig2noise[i] = 0.0

        # find means of all the correlation maps
        corr_max2 = np.abs(correlation.mean(axis=(-2, -1)))
        corr_max2[corr_max2 == 0] = np.nan  # mark failed ones

        sig2noise = corr_max1 / corr_max2

    else:
        raise ValueError("wrong sig2noise_method")

    # sig2noise is zero for all failed ones
    sig2noise[np.isnan(sig2noise)] = 0.0

    return sig2noise


def fft_correlate_windows(window_a, window_b):
    """ FFT based cross correlation
    it is a so-called linear convolution based,
    since we increase the size of the FFT to
    reduce the edge effects.

    This should also work out of the box for rectangular windows.

    Parameters
    ----------
    window_a : 2d np.ndarray
        a two dimensions array for the first interrogation window,

    window_b : 2d np.ndarray
        a two dimensions array for the second interrogation window.

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


def fft_correlate_strided_images(image_a, image_b):
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
    """
    s1 = np.array(image_a.shape[-2:])
    s2 = np.array(image_b.shape[-2:])
    size = s1 + s2 - 1
    fsize = 2 ** np.ceil(np.log2(size)).astype(int)
    fslice = tuple([slice(0, image_a.shape[0])] +
                   [slice(0, int(sz)) for sz in size])
    f2a = rfft2(image_a, fsize, axes=(-2, -1))
    f2b = rfft2(image_b[:, ::-1, ::-1], fsize, axes=(-2, -1))
    corr = irfft2(f2a * f2b, axes=(-2, -1)).real[fslice]
    return corr


def zero_pad(window):
    """ Zero pads the interrogation window to double size
    Inputs:
        window: numpy array

    Outpus:
        window with zeros padded on all sides up to the double size of the
        original window

    Example:
        zero_pad(np.ones((2,2)))

        array( [[0., 0., 0., 0.],
                [0., 1., 1., 0.],
                [0., 1., 1., 0.],
                [0., 0., 0., 0.]])
    """
    return np.pad(window, np.round(np.array(window.shape) / 2).astype(np.int))


def correlate_windows(window_a, window_b, correlation_method="fft"):
    """Compute correlation function between two interrogation windows.

    The correlation function can be computed by using the correlation
    theorem to speed up the computation.

    Parameters
    ----------
    window_a : 2d np.ndarray
        a two dimensions array for the first interrogation window,

    window_b : 2d np.ndarray
        a two dimensions array for the second interrogation window.

    correlation_method : string, methods currently implemented:
            'circular' - FFT based without zero-padding
            'linear' -  FFT based with zero-padding
            'direct' -  linear convolution based
            Default is 'fft', which is much faster.

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
        corr = fft_correlate_windows(window_a, window_b)
    elif correlation_method == "linear":
        # save the original size:
        s1 = np.array(window_a.shape)
        s2 = np.array(window_b.shape)
        size = s1 + s2 - 1
        fslice = tuple([slice(0, int(sz)) for sz in size])
        # and slice only the relevant part
        corr = fft_correlate_windows(zero_pad(window_a),
                                     zero_pad(window_b))[fslice]
    elif correlation_method == "direct":
        corr = convolve2d(window_a, window_b[::-1, ::-1], "full")
    else:
        raise ValueError("method is not implemented")

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
    window = window - window.mean(axis=(-2, -1),
                                  keepdims=True, dtype=np.float32)
    window = window / (1.96 * np.std(window, dtype=np.float32))
    return np.clip(window, -1, 1)


def extended_search_area_piv(
    frame_a,
    frame_b,
    window_size,
    overlap=0,
    dt=1.0,
    search_area_size=0,
    correlation_method="fft",
    subpixel_method="gaussian",
    sig2noise_method=None,
    width=2,
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
        one of the two methods implemented: 'fft' or 'direct',
        [default: 'fft'].

    subpixel_method : string
         one of the following methods to estimate subpixel location of the
         peak:
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

    # check the inputs for validity
    if search_area_size == 0:
        search_area_size = window_size

    if overlap >= window_size:
        raise ValueError("Overlap has to be smaller than the window_size")

    if search_area_size < window_size:
        raise ValueError("Search size cannot be smaller than the window_size")

    if (window_size > frame_a.shape[0]) or (window_size > frame_a.shape[1]):
        raise ValueError("window size cannot be larger than the image")

    # get field shape
    n_rows, n_cols = get_field_shape(frame_a.shape, search_area_size, overlap)

    # We implement the new vectorized code
    frame_a = normalize_intensity(frame_a)
    frame_b = normalize_intensity(frame_b)

    aa = moving_window_array(frame_a, search_area_size, overlap)
    bb = moving_window_array(frame_b, search_area_size, overlap)

    # for the case of extended seearch, the window size is smaller than
    # the search_area_size. In order to keep it all vectorized the
    # approach is to use the interrogation window in both
    # frames of the same size of search_area_asize,
    # but mask out the region around
    # the interrogation window in the frame A

    if search_area_size > window_size:
        mask = np.zeros((search_area_size, search_area_size))
        pad = np.int((search_area_size - window_size) / 2)
        mask[slice(pad, search_area_size - pad),
             slice(pad, search_area_size - pad)] = 1
        mask = np.broadcast_to(mask, aa.shape)
        aa *= mask

    corr = fft_correlate_strided_images(aa, bb)
    u, v = correlation_to_displacement(corr, n_rows, n_cols, search_area_size)

    # return output depending if user wanted sig2noise information
    if sig2noise_method is not None:
        sig2noise = sig2noise_ratio(
            corr, sig2noise_method=sig2noise_method, width=width
        )
    else:
        sig2noise = np.full_like(u, np.nan)

    sig2noise = sig2noise.reshape(n_rows, n_cols)

    return u / dt, v / dt, sig2noise


def correlation_to_displacement(corr, n_rows, n_cols, search_area_size=32):
    """
    Correlation maps are converted to displacement for each interrogation
    window using the convention that the size of the correlation map
    is 2N -1 where N is the size of the largest interrogation window
    (in frame B) that is called search_area_size
    Inputs:
        corr : 3D nd.array
            contains output of the fft_correlate_strided_images
        n_rows, n_cols : number of interrogation windows, output of the
            get_field_shape
        search_area_size : int
            size of the interrogation window in frame B (>= IW in frame A)
    """
    # iterate through interrogation widows and search areas
    u = np.zeros((n_rows, n_cols))
    v = np.zeros((n_rows, n_cols))

    for k in range(n_rows):
        for m in range(n_cols):
            row, col = find_subpixel_peak_position(corr[k * n_cols + m, :, :])
            row -= (2 * search_area_size - 1) // 2
            col -= (2 * search_area_size - 1) // 2

            # get displacements, apply coordinate system definition
            u[k, m], v[k, m] = -col, row

    return (u, v)


def nextpower2(i):
    """ Find 2^n that is equal to or greater than. """
    n = 1
    while n < i:
        n *= 2
    return n
