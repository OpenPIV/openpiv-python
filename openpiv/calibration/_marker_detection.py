import numpy as np
from typing import Tuple

from openpiv.pyprocess import get_field_shape, get_coordinates,\
                              sliding_window_array, fft_correlate_images,\
                              find_all_first_peaks, find_first_peak

from skimage.feature import match_template


__all__ = [
    "preprocess_image",
    "get_circular_template",
    "get_cross_template",
    "get_new_template",
    "detect_markers_template",
    "detect_markers_blobs"
]


def preprocess_image(
    image: np.ndarray,
    threshold: float,
    mask: np.ndarray=None,
    roi: list=None,
    invert: bool=False,
    highpass_sigma: float=None,
    lowpass_sigma: float=None,
    variance_sigma1: float=None,
    variance_sigma2: float=None,
    morph_size: int=None,
    morph_iter: int=2,
    median_size: int=None
):
    """Preprocess calibration image.
    
    Preprocess calibration image for feature extraction. The final image is a
    boolean 2D np.ndarray of shame(n, m). To avoid unwarranted mutation of the
    original calibration image, the passed image is explicitly copied.
    
    Parameters
    ----------
    image : 2D np.ndarray
        A two dimensional array of pixel intensities of the shape (n, m).
    threshold : float
        The threshold used for image binarization. It is a good idea to start with
        a low threshold and slowly increase it in order to get an optimal threshold.
    mask : 2D np.ndarray, optional
        A 2D boolean np.ndarray where True elements are kept and False elements are
        set to zero.
    roi : list, optional
        A four element list containing min x, y and max x, y in pixels. If a mask
        exists, the roi is concatenated with the mask.
    invert : bool, optional
        If True, invert the image.
    highpass_sigma : float. optional
        If not None, perform a high pass filter. Pixel intensities below zero are 
        clipped.
    lowpass_sigma : float, optional
        If not None, perform a low pass filter on the calibration image.
    variance_sigma1, variance_sigma2 : float, optional
        If not None, perform a local variance normalization filter. I is best to
        start with a sigma around 1 to 5 and increase them until most of the 
        background is removed.
    morph_size : int, optional
        If not None, perform erosion and dilation morph_iter amount of times.
        This helps remove noise and non-marker elements.
    morph_iter : int, optional
        The number of iterations to perform the erosion and dilation morphological
        operations.
    median_size : int, odd, optional
        If not None, perform a median filter.
        
    returns
    -------
    bool_image : 2D np.ndarray
        The binarized boolean calibration image of shape (n, m).
        
    """
    from openpiv.preprocess import high_pass, local_variance_normalization
    from scipy.ndimage import median_filter, gaussian_filter, grey_dilation, grey_erosion
    from scipy.signal import convolve2d
    
    cal_img = image.copy()
    cal_img = cal_img.astype(float, copy=False)
    cal_img /= cal_img.max()
    
    if roi is not None:
        if mask is None:
            mask = np.ones_like(cal_img, dtype=bool)
        
        con_mask = np.zeros_like(mask, dtype=mask.dtype)
        
        con_mask[
            roi[1] : roi[3], # y-axis
            roi[0] : roi[2]  # x-axis
        ] = 1
        
        np.multiply(
            mask,
            con_mask,
            out=mask
        )
    
    if invert == True:
        cal_img = 1.0 - cal_img
    
    if highpass_sigma is not None:
        cal_img = high_pass(
            cal_img,
            sigma=highpass_sigma,
            clip=True
        )
        
    if lowpass_sigma is not None:
        cal_img = gaussian_filter(
            cal_img,
            sigma=lowpass_sigma,
            truncate=4
        )
        
    if variance_sigma1 is not None and\
        variance_sigma2 is not None:
        cal_img = local_variance_normalization(
            cal_img,
            variance_sigma1,
            variance_sigma2,
            clip=True
        )
        
    if morph_size is not None:
        if morph_size % 2 != 1:
            raise ValueError(
                "Morphology size must be odd"
            )
        
        for _ in range(morph_iter):
            cal_img = grey_erosion(
                cal_img,
                size=morph_size,
                mode="constant",
                cval=0.
            )
        
        for _ in range(morph_iter):
            cal_img = grey_dilation(
                cal_img,
                size=morph_size,
                mode="constant",
                cval=0.
            )
            
    if median_size is not None:
        if median_size % 2 != 1:
            raise ValueError(
                "Median width must be odd"
            )
        
        cal_img = median_filter(
            cal_img,
            size=median_size,
            mode="mirror"
        )
            
    # binarization
    cal_img = np.where(cal_img > threshold, 1, 0).astype(bool, copy=False)
    
    # if a mask is supplied, copy it to minimize unwarranted mutations
    if mask is not None:
        if roi is not None:
            copy = False
        else:
            copy = True
        
        mask = mask.astype(bool, copy=copy)
        
        np.multiply(
            cal_img,
            mask,
            out=cal_img
        )
            
    return cal_img


def get_circular_template(
    template_radius,
    val=1
):
    """Create a circle template.
    
    Create a circle template based on the template radius and window size.
    This template can be correlated with an image to find features such as
    marker detection on calibration plates.
    
    Parameters
    ----------
    template_radius : int
        The radius of the circle in the template.
    val : float, optional
        The value set to the circular elements in the template.
        
    Returns
    -------
    template : 2D np.ndarray
        A 2D np.ndarray of dtype np.float64 containing a centralized circular
        element.
    
    Examples
    --------
    from openpiv.calib_utils import get_circular_template
    
    >>> get_circular_template(
            template_radius=5,
            val=255
        )
    
    """
    # make sure input is integer
    dot_radius = int(template_radius)
    
    # get template size
    dot_size = 2*template_radius + 1
        
    disk = np.zeros(
        (dot_size, dot_size),
        dtype="float64"
    )

    ys, xs = np.indices([dot_size, dot_size])

    dist = np.sqrt(
        (ys - dot_radius)**2 +
        (xs - dot_radius)**2
    )

    disk[dist <= dot_radius] = val
    
    

    return disk


def get_cross_template(
    template_radius: int,
    line_width: int=None,
    val: float=1
):
    """Create a cross template.
    
    Create a cross template based on the template radius and window size. The
    line width of the cross is found by int(template_radius / 6) + 1. This 
    template can be correlated with an image to find features such as marker 
    detection on calibration plates.
    
    Parameters
    ----------
    template_radius : int
        The radius of the cross in the template.
    line_width : int, optional
        The width of the cross in the template.
    val : float, optional
        The value set to the cross elements in the template.
        
    Returns
    -------
    template : 2D np.ndarray
        A 2D np.ndarray of dtype np.float64 containing a centralized cross
        element.
    
    Examples
    --------
    from openpiv.calib_utils import get_cross_template
    
    >>> get_cross_template(
            template_radius=11,
            val=255
        )
    
    """
    # make sure input is integer
    template_radius = int(template_radius)
    
    # get template size
    template_size = 2*template_radius + 1
        
    cross = np.zeros(
        (template_size, template_size),
        dtype="float64"
    )
    
    ys, xs = np.abs(
        np.indices([template_size, template_size]) - template_radius
    )
    
    if line_width == None:
        line_width = int(template_radius / 6) + 1
        
    cross[ys < line_width] = val
    cross[xs < line_width] = val
    
    return cross


def get_new_template(
    image: np.ndarray,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    template_radius: int,
    dtype: str="float64"
):
    numel = pos_x.shape[0]
        
    min_x = template_radius
    min_y = template_radius
    max_x = image.shape[1] - template_radius - 1
    max_y = image.shape[0] - template_radius - 1

    template = np.zeros((template_radius*2 + 1, template_radius*2 + 1), dtype=dtype)
    denom = 0
    
     # Note: this loop can easily be optimized
    for i in range(numel):
        x, y = int(pos_x[i]), int(pos_y[i])
                
        if not ((min_x < x < max_x) and (min_y < y < max_y)):
            continue
            
        lx = x - template_radius
        rx = x + template_radius + 1
        ly = y - template_radius
        ry = y + template_radius + 1
        
        img_cut = image[ly:ry, lx:rx]
                        
        template += img_cut
        denom += 1
    
    if denom != 0:
        template /= denom
    
    return template


def _find_local_max(
    corr: np.ndarray,
    window_size: int=64,
    overlap: int=32,
):
    """Find a local maximum from correlation matrix.
    
    Find all local maximums from the correlation matrix by subdividing the
    correlation matrix into multiple subwindows and locating the largest value.
    
    Parameters
    ----------
    corr : np.ndarray
        A two dimensional array of marker response values of the shape (n, m).
    window_size : int, optional
        The size of the window used to search for the local maximum. Must be even 
        and smaller than the distance between two markers in pixels. A good
        rule of thumb is to set the window size to slightly smaller than the
        mean marker spacing.
    overlap : int, optional
        The amount of overlaping pixels for each window. The higher the overlap, 
        the better local maximums are registered but at the expense of performance 
        and memory. 
        
    Returns
    -------
    max_ind_x, max_ind_y : np.ndarray
        A two dimensional array containing local peak indexes of the shape (n, m).
    peaks : np.ndarray
        A two dimensional array containing local maximums of the shape (n, m).
    
    """
    # get field shape
    field_shape = get_field_shape(
        corr.shape,
        window_size,
        overlap
    )
        
    # get sub-windows of correlation matrix
    corr_windows = sliding_window_array(
        corr,
        [window_size, window_size],
        [overlap, overlap]
    )    

    # get location of peaks
    max_ind, peaks = find_all_first_peaks(
        corr_windows
    )

    max_ind_x = max_ind[:, 2]
    max_ind_y = max_ind[:, 1]
    
    # reshape field (this is not actually needed)
    max_ind_x = max_ind_x.reshape(field_shape)
    max_ind_y = max_ind_y.reshape(field_shape)
    peaks = peaks.reshape(field_shape)
    
    return max_ind_x, max_ind_y, peaks


def _merge_points(
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    merge_radius: int=10,
    merge_iter: int=5,
    min_count: int=5
):
    """Merge nearby points.
    
    Merge nearby points iteratively and return positions that have a
    minimum amount of merge counts.
    
    Parameters
    ----------
    pos_x, pos_y : np.ndarray
        A two dimensional array containing local peak indexes of the shape (n, m).
    merge_radius : int, optional
        The merge radius when detecting the number of times a marker is found.
        Typically, the merge radius should be 5 to 10.
    merge_iter : int, optional
        The number of iterations to merge neighboring points inside the
        merge radius threshold.
    min_count : float, optional
        The minimum amount a marker is detected. Helps to remove false
        positives on marker detection.
    
    Return
    ------
    pos_x, pos_y : np.ndarray
        An one dimensional array containing local peak indexes of the shape (n,).
    count : np.ndarray
        An one dimensional array containing local peak counts of the shape (n,).
    
    """
    # create 2D array of coordinates
    pos = np.array([pos_x, pos_y], dtype="float64").T

    # find clusters
    clusters = np.hypot(
        pos_x.reshape(-1, 1) - pos_x.reshape(1, -1),
        pos_y.reshape(-1, 1) - pos_y.reshape(1, -1)
    ) <= merge_radius

    # get mean of clusters iteratively
    for _ in range(merge_iter):
        for ind in range(pos.shape[0]):
            pos[ind, :] = np.mean(
                pos[clusters[ind, :], :].reshape(-1, 2),
                axis = 0
            )
    
    # convert to integers by rounding everything down
    new_pos = np.floor(pos)

    # count the number of copies
    new_pos, count = np.unique(
        new_pos,
        return_counts=True,
        axis=0
    )
    
    # remove positions that are not detected enough times
    good_ind = count >= min_count
    
    new_pos = new_pos[good_ind, :]
    count = count[good_ind]
    
    pos_x = new_pos[:, 0]
    pos_y = new_pos[:, 1]
    
    return pos_x, pos_y, count


def _detect_markers(
    image: np.ndarray,
    template: np.ndarray,
    roi: tuple,
    window_size: int,
    overlap: int,
    min_peak_height: float,
    merge_radius: float,
    merge_iter: int,
    min_count: int
):
    """Detect marker point candidates.
    
    Detect marker point candidates by template correlation and thresholding.
    The template is correlated globally for faster image processing and uses
    a normalized correlation technique using sum of squared differences.
    
    Parameters
    ----------
    image : 2D np.ndarray
        A two dimensional array of pixel intensities of the shape (n, m).
    template : 2D np.ndarray
        A square two dimensional array of the shape (n, m) which is to be correlated
        with the image to extract features. Must be of odd shape (e.g., [5, 5]) and
        elements must be either 0 or 1.
    roi : list, optional
        A four element list containing min x, y and max x, y in pixels.
    window_size : int, optional
        The size of the window used to search for the marker. Must be even 
        and smaller than the distance between two markers in pixels. A good
        rule of thumb is to set the window size to slightly smaller than the
        mean marker spacing.
    overlap : int, optional
        The amount of overlaping pixels for each window. If None, overlap is
        automatically set to 75% of the window size. Step size can be
        calculated as window_size - overlap. The higher the overlap, the better
        markers are registered but at the expense of performance and memory. 
    min_peak_height : float, optional
        Reject correlation peaks below threshold to help remove false positives.
    merge_radius : int, optional
        The merge radius when detecting the number of times a marker is found.
        Typically, the merge radius should be 5 to 10.
    merge_iter : int, optional
        The number of iterations to merge neighboring points inside the
        merge radius threshold.
        
    Returns
    -------.
    pos_x, pos_y : np.ndarray
        An one dimensional array containing local peak indexes of the shape (n,).
    peaks : np.ndarray
        An one dimensional array containing local peak heights of the shape (n,).
    count : np.ndarray
        An one dimensional array containing local peak counts of the shape (n,).
    corr : np.ndarray
        A two dimensional array containing the tempalte-image correlation field.
    
    """
    # get roi 
    off_x = off_y = 0
    
    if roi is not None:
        off_x = roi[0]
        off_y = roi[1]

        corr_slice = (
            slice(roi[1], roi[3]),
            slice(roi[0], roi[2])
        )
        
    corr = match_template(
        image,
        template,
        pad_input=True
    )

    if roi is not None:
        corr_cut = corr[corr_slice]
    else:
        corr_cut = corr

    corr_padded = np.pad(
        corr_cut, 
        window_size,
        mode="constant"
    )

    corr_field_shape = corr_padded.shape
    pad_off = window_size

    max_ind_x, max_ind_y, peaks = _find_local_max(
        corr_padded,
        window_size,
        overlap
    )

    # create a grid
    grid_x, grid_y = get_coordinates(
        corr_field_shape,
        window_size,
        overlap,
        center_on_field=False
    ) - np.array([window_size // 2])

    # add grid to peak indexes to get estimated location
    pos_x = grid_x + max_ind_x
    pos_y = grid_y + max_ind_y

    # find points near sub window borders and with low peak heights
    flags = np.zeros_like(pos_x).astype(bool, copy=False)
    p_exclude = 3

    flags[max_ind_x < p_exclude] = True
    flags[max_ind_y < p_exclude] = True
    flags[max_ind_x > window_size - p_exclude - 1] = True
    flags[max_ind_y > window_size - p_exclude - 1] = True
    flags[peaks < min_peak_height] = True

    # remove flagged elements
    pos_x = pos_x[~flags]
    pos_y = pos_y[~flags]

    # add offsets from roi
    pos_x += off_x
    pos_y += off_y

    pos_x, pos_y, count = _merge_points(
        pos_x,
        pos_y,
        merge_radius,
        merge_iter,
        min_count
    )

    # remove padding offsets
    pos_x -= pad_off
    pos_y -= pad_off

    # find points outside of image
    flags = np.zeros_like(pos_x).astype(bool, copy=False)

    n_exclude = 8
    flags[pos_x < n_exclude] = True
    flags[pos_y < n_exclude] = True
    flags[pos_x > image.shape[1] - n_exclude - 1] = True
    flags[pos_y > image.shape[0] - n_exclude - 1] = True
    flags[np.isnan(pos_x)] = True
    flags[np.isnan(pos_y)] = True

    # remove points outside of image
    pos_x = pos_x[~flags]
    pos_y = pos_y[~flags]
    count = count[~flags]
    
    return pos_x, pos_y, count, corr


def _subpixel_approximation(
    corr: np.ndarray,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    refine_radius: int=3,
    refine_iter: int=5,
    refine_cutoff: float=0.5
):
    """Iterative subpixel marker refinement.
    
    Subpixel iterative marker refinement using least squares 2D gaussian
    peak fitting. Least squares minimization is performed by taking the 
    pseudo inverse of a matrix and multiplying it to the logarithm of the 
    correlation matrix.
    
    Paramters
    ---------
    corr : np.ndarray
        A two dimensional array of marker response values of the shape (n, m).
    pos_x, pos_y : np.ndarray
        An one dimensional array containing local peak indexes of the shape (n,).
    refine_radius : int, optional
        The radius of the gaussian kernel. The radius should be similar to the 
        radius of the marker radius. However, if the radius is greater than
        n_exclude, which is predetermined to be 8, the radius is set to 7.
    refine_iter : int, optional
        The amount of iterations to perform the gaussian subpixel estimation.
    refine_cutoff : float, optional
        The cutoff number to stop iterating. Should be between 0.25 to 0.5.
        
    Returns
    -------
    new_pos_x, new_pos_y : np.ndarray
        An one dimensional array containing local peak indexes of the shape (n,).
        
    """
    new_pos_x = pos_x.copy()
    new_pos_y = pos_y.copy()

    _sx, _sy = np.meshgrid(
        np.arange(
            -refine_radius,
            refine_radius + 1,
            dtype=float
        ),
        np.arange(
            -refine_radius,
            refine_radius + 1,
            dtype=float
        )
    )

    nx, ny = _sx.shape

    _sx = np.ravel(_sx)
    _sy = np.ravel(_sy)

    s_arr = np.array(
        [_sx, _sy, _sx**2, _sy**2, np.ones_like(_sx)], 
        dtype=float
    )

    s_arr = np.reshape(
        np.concatenate(s_arr),
        (5, nx * ny)
    ).T

    # we use a psuedo-inverse via SVD so we can solve a system of equations.
    # using iterative nonlinear methods is preferable here, though.
    s_inv = np.linalg.pinv(s_arr)

    # TODO: optimize this loop
    for ind in range(pos_x.shape[0]): 
        x, y = pos_x[ind], pos_y[ind]

        for _ in range(refine_iter):
            x = int(x)
            y = int(y)

            slices = (
                slice(
                    y - refine_radius,
                    y + refine_radius + 1),
                slice(
                    x - refine_radius,
                    x + refine_radius + 1
                )
            )

            corr_sec = corr[slices]
            corr_sec[corr_sec <= 0] = 1e-6

            coef = np.dot(
                s_inv, 
                np.log(np.ravel(corr_sec))
            )

            if coef[2] < 0.0 and coef[3] < 0.0:
                sx = 1 / np.sqrt(-2 * coef[2])
                sy = 1 / np.sqrt(-2 * coef[3])

                shift_x = coef[0] * np.square(sx)
                shift_y = coef[1] * np.square(sy)
            else:
                shift_x = 0
                shift_y = 0

            new_x = x + shift_x
            new_y = y + shift_y

            d = np.sqrt((x - new_x)**2 + (y - new_y)**2)

            x = new_x
            y = new_y

            if d < refine_cutoff:
                break

        new_pos_x[ind] = x
        new_pos_y[ind] = y
        
    return new_pos_x, new_pos_y


def detect_markers_template(
    image: np.ndarray,
    template: np.ndarray,
    roi: list=None,
    window_size: int=32,
    overlap: int=None,
    min_peak_height: float=0.25,
    merge_radius: int=10,
    merge_iter: int=3,
    min_count: float=2,
    refine_pos: bool=False,
    refine_radius: int=3,
    refine_iter: int=5,
    refine_cutoff: float=0.5,
    return_count: bool=False,
    return_corr: bool=False,
    return_template: bool=False
):
    """Detect calibration markers.
    
    Detect circular and cross calibration markers. The markers are first detected
    based on correlating the image with a template. Then, the correlation matrix
    is split into sub-windows where the position of the markers in the local windows 
    is found by locating the maximum of that window, which correlates to the best
    match with the template. Next, false positives are removed by specifying the 
    minimum count a marker is detected and the minimum correlation coefficient of 
    that marker's peak. Finally, a gaussian peak fit based on pseudo-inverse via
    singular value decomposition is performed.
    
    Parameters
    ----------
    image : 2D np.ndarray
        A two dimensional array of pixel intensities of the shape (n, m).
    template : 2D np.ndarray
        A square two dimensional array of the shape (n, m) which is to be correlated
        with the image to extract features. Must be of odd shape (e.g., [5, 5]) and
        elements must be either 0 or 1.
    roi : list, optional
        A four element list containing min x, y and max x, y in pixels.
    window_size : int, optional
        The size of the window used to search for the marker. Must be even 
        and smaller than the distance between two markers in pixels. A good
        rule of thumb is to set the window size to slightly smaller than the
        mean marker spacing.
    overlap : int, optional
        The amount of overlapping pixels for each window. If None, overlap is
        automatically set to 62.5% of the window size. Step size can be
        calculated as window_size - overlap. The higher the overlap, the better
        markers are registered but at the expense of performance and memory. 
    min_peak_height : float, optional
        Reject correlation peaks below threshold to help remove false positives.
    merge_radius : int, optional
        The merge radius when detecting the number of times a marker is found.
        Typically, the merge radius should be 5 to 10.
    merge_iter : int, optional
        The number of iterations to merge neighboring points inside the
        merge radius threshold.
    refine_pos : bool, optional
        Refine the position of the markers with a gaussian peak fit.
    refine_radius : int, optional
        The radius of the gaussian kernel. The radius should be similar to the 
        radius of the marker radius. However, if the radius is greater than
        n_exclude, which is predetermined to be 8, the radius is set to 7.
    refine_temp_radius : int, optional
        If set, the new template radius will be 2*refine_temp_radius + 1.
        Otherwise, the radius would be 1.25 * template radius.
    refine_iter : int, optional
        The amount of iterations to perform the gaussian subpixel estimation.
    refine_cutoff : float, optional
        The cutoff number to stop iterating. Should be between 0.25 to 0.5.
    return_count : bool, optional
        Return the number of times a marker gets counted. This can be used to
        find the ideal threshold to find the correct markers.
    return_corr : bool, optional
        Return the correlation of the image and template. This can be used to
        determine the template radius.
    
    Returns
    -------
    markers : 2D np.ndarray
        Marker positions in [x, y]' image coordinates.
    counts : 2D np.ndarray, optional
        Marker counts in [x, y]'. Returned if return_count is True.
    
    Notes
    -----
    The gaussian subpixel fitting algorithm is basically a hit or miss when it comes
    to improving the marker locations. Because of this, it is not recommended to use
    this parameter unless there is a noticable decrease in RMS errors.
    
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv import calib_utils
    >>> from openpiv.data.test5 import cal_image
    
    >>> cal_img = cal_image(preproc = True)
    
    >>> template = calib_utils.get_circular_template(radius = 7)
    
    >>> marks_pos, counts = calib_utils.detect_markers_template(
            cal_img,
            template,
            window_size = 64,
            min_peak_height = 0.5,
            merge_radius = 10,
            merge_iter=5,
            min_count=5,
            return_count=True
        )
        
    >>> marks_pos
    
    >>> counts
    
    """     
    # @ErichZimmer
    # Note to developers, this function was originally written as a prototype
    # for the OpenPIV c++ version. However, it was refined in order to be useful
    # for the Python version of OpenPIV.
    
    min_count = 2
    
    # make sure template size is smaller than window size
    if template.shape[0] >= window_size:
        raise ValueError(
            "template_radius is too large for given window_size."
        )
    
    # if overlap is None, set overlap to 75% of window size
    if overlap is None:
        overlap = window_size - window_size * 0.325
    
    # make sure window_size and overlap are integers
    window_size = int(window_size)
    overlap = int(overlap)
    
    # data type conversion to float64
    image = image.astype("float64")
    template = template.astype("float64")
    
    # scale the image to [0, 1]
    image[image < 0] = 0. # cut negative pixel intensities
    image /= image.max() # normalize
#    image *= 255. # rescale
    
    pos_x, pos_y, counts, corr = _detect_markers(
        image,
        template,
        roi,
        window_size,
        overlap,
        min_peak_height,
        merge_radius,
        merge_iter,
        min_count
    )

    max_radius = 8
    if refine_pos == True:
        if refine_radius > max_radius:
            refine_radius = max_radius
        
        pos_x, pos_y = _subpixel_approximation(
            corr,
            pos_x,
            pos_y,
            refine_radius,
            refine_iter,
            refine_cutoff
        )
    
    return_list = [
        np.array([pos_x, pos_y], dtype="float64")
    ]
    
    if return_count == True:
        return_list.append(counts)
    
    if return_corr == True:
        return_list.append(corr)
        
    if return_template == True and refine_template == True:
        return_list.append(new_template)
    
    return return_list


def detect_markers_blobs(
    image: np.ndarray,
    roi: list=None,
    min_area: int=None,
    max_area: int=None
):
    """Detect blob markers.
    
    Detect blob markers by labeling an image, removing outliers by thresholding, and
    finding the center of mass of the labels.
    
    Parameters
    ----------
    image : 2D np.ndarray
        A two dimensional array of pixel intensities of the shape (n, m).
    roi : list, optional
        A four element list containing min x, y and max x, y in pixels.
    min_area : int, optional
        The minimum amount of pixels a labeled marker can have.
    max_area : int, optional
        The maximum amount of pixels a labeled marker can have.
        
    Returns
    -------
    markers : 2D np.ndarray
        Marker positions in [x, y]' image coordinates.
    
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv import calib_utils
    >>> from openpiv.data.test5 import cal_image
    
    >>> cal_img = cal_image(z=0)
    
    >>> marks_poss = calib_utils.detect_markers_blobs(
            cal_img,
            roi=[0, 0, None, 950],
            min_area=50
        )
        
    >>> marks_pos
    
    """
    from scipy.ndimage import label, center_of_mass
    
    image = image.astype("float64")
    
    # set ROI if needed
    off_x = off_y = 0
    
    if roi is not None:
        off_x = roi[0]
        off_y = roi[1]
        
        image = image[
            roi[1] : roi[3], # y-axis
            roi[0] : roi[2]  # x-axis
        ]
    
    # scale the image to [0, 1]
    image[image < 0] = 0. # cut negative pixel intensities
    image /= image.max() # normalize
    image = image > image.mean() # binarize
    
    # label possible markers
    labels, n_labels = label(image)
    
    labels_ind = np.arange(1, n_labels + 1)
    
    # get label area
    label_area = np.zeros(n_labels)
    for i in labels_ind:
        label_area[i-1] = np.sum(labels == i)
    
    # remove invalid areas
    flag = np.zeros(n_labels, dtype=bool)

    if min_area is not None:
        flag[label_area < min_area] = True
    
    if max_area is not None:
        flag[label_area > max_area] = True

    valid_labels_ind = labels_ind[~flag]
    
    # get center of mass of valid labels
    _pos = center_of_mass(
        image,
        labels,
        valid_labels_ind
    )
    
    _pos = np.array(_pos, dtype="float64")
    
    # rearrange x and y coordinates and apply roi offsets
    pos = np.empty_like(_pos, dtype="float64")
    
    pos[:, 0] = _pos[:, 1] + off_x
    pos[:, 1] = _pos[:, 0] + off_y
    
    # sort so the results behave like detect_markers_template
    order = np.lexsort(
        (pos[:, 1], pos[:, 0])
    )
    
    return pos[order].T