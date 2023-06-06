import numpy as np
from typing import Tuple


__all__ = [
    "preprocess_image",
    "get_circular_template",
    "get_cross_template",
    "detect_markers_local",
    "detect_markers_blobs",
    "show_calibration_image",
    "get_pairs_anal",
    "get_reprojection_error",
    "get_los_error",
    "get_image_mapping"
]


def preprocess_image(
    image: np.ndarray,
    threshold: float,
    mask: np.ndarray=None,
    roi: list=None,
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
    highpass_sigma : float. optional
        If not None, perform a high pass filter on the calibration image. Pixel
        intensities below zero are clipped.
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
        If not None, perform a median filter on the calibration image.
        
    returns
    -------
    bool_image : 2D np.ndarray
        The binariazed boolean calibration image of shape (n, m).
        
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
    
    Create a circle template based on the temnplate radius and window size.
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
            window_size=32,
            template_radius=5,
            val=255
        )
    
    """
    # make sure input is integer
    template_radius = int(template_radius)
    
    # get template size
    template_size = 2*template_radius + 1
        
    disk = np.zeros(
        (template_size, template_size),
        dtype="float64"
    )

    ys, xs = np.indices([template_size, template_size])

    dist = np.sqrt(
        (ys - template_radius)**2 +
        (xs - template_radius)**2
    )

    disk[dist <= template_radius] = 255.

    return disk


def get_cross_template(
    template_radius,
    val=1
):
    """Create a cross template.
    
    Create a cross template based on the temnplate radius and window size. The
    line width of the cross is found by int(template_radius / 6) + 1. This 
    template can be correlated with an image to find features such as marker 
    detection on calibration plates.
    
    Parameters
    ----------
    template_radius : int
        The radius of the cross in the template.
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
    
    line_width = int(template_radius / 6) + 1
        
    cross[ys < line_width] = val
    cross[xs < line_width] = val
    
    return cross


def detect_markers_local(
    image: np.ndarray,
    template: np.ndarray,
    roi: list=None,
    window_size: int=64,
    overlap = None,
    min_peak_height: float=0.025,
    merge_radius: int=10,
    merge_iter: int=5,
    min_count: float=2,
    refine_pos: bool=False,
    return_count=False
):
    """Detect calibration markers.
    
    Detect circular and cross calibration markers. The markers are first detected
    based on correlating the image with a template. Then, the correlation matrix
    is split into sub-windows where the position of the markers in the local windows 
    is found by locating the maximum of that window, which correlates to the best
    match with the template. Next, false positives are removed by specifying the 
    minimum count a marker is detected and the minimum correlation coefficient of 
    that marker's peak. Finally, a gaussian peak fit based on psuedo-inverse via
    singular value decomposition is performed.
    
    Parameters
    ----------
    image : 2D np.ndarray
        A two dimensional array of pixel intensities of the shape (n, m).
    template : 2D np.ndarray
        A square two dimensional array of the shape (n, m) which is to be correlated
        with the image to extract features. Must be of odd shape (e.g., [5, 5]).
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
    min_count : float, optional
        The minimum amount a marker is detected. Helps to remove false
        positives on marker detection.
    refine_pos : bool, optional
        Refine the position of the markers with a gaussian peak fit. The gaussian
        peak fitting algorithm is a hit or miss, so checking the RMS error for
        improments is important.
    return_count : bool, optional
        Return the number of times a marker gets counted. This can be used to
        find the ideal threshold to find the correct markers.
    
    Returns
    -------
    markers : 2D np.ndarray
        Marker positions in [x, y]' image coordinates.
    
    counts : 2D np.ndarray, optional
        Marker counts in [x, y]. Returned if return_count is True.
    
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
    
    >>> marks_pos, counts = calib_utils.detect_markers_local(
            cal_img,
            template,
            window_size = 64,
            min_peak_height = 0.03,
            merge_radius = 10,
            merge_iter=5,
            min_count=5,
            return_count=True
        )
        
    >>> marks_pos
    
    >>> counts
    
    """
    from openpiv.pyprocess import get_field_shape, get_coordinates,\
                                  sliding_window_array, fft_correlate_images,\
                                  find_all_first_peaks  
    
    # @ErichZimmer
    # Note to developers, this function was originally written as a prototype
    # for the OpenPIV c++ version. However, it was refined in order to be useful
    # for the Python version of OpenPIV.
    
    # make sure template size is smaller than window size
    if template.shape[0] >= window_size:
        raise ValueError(
            "template_radius is too large for given window_size."
        )
    
    # if overlap is None, set overlap to 75% of window size
    if overlap is None:
        overlap = window_size - window_size * 0.25
    
    # make sure window_size and overlap are integers
    window_size = int(window_size)
    overlap = int(overlap)
    
    # data type conversion to float64
    image = image.astype("float64")
    
    # scale the image to [0, 255]
    image[image < 0] = 0. # cut negative pixel intensities
    image /= image.max() # normalize
    image *= 255. # scale
    
    # now pad tempalte to window size
    template_padded = np.zeros(
        (image.shape[0], image.shape[1]), 
        dtype= "float64"
    )
    
    temp_half_x = template.shape[1] // 2
    temp_half_y = template.shape[0] // 2
    
    template_padded[
        template_padded.shape[0] // 2 - temp_half_y - 1 : template_padded.shape[0] // 2 + temp_half_y - 0,
        template_padded.shape[1] // 2 - temp_half_x - 1 : template_padded.shape[1] // 2 + temp_half_x - 0] =\
        template
    
    # normalized cross correlation
    corr = fft_correlate_images(
        image[np.newaxis, :, :],
        template_padded[np.newaxis, :, :],
        normalized_correlation = True,
        correlation_method = "linear"
    )[0, ::-1, ::-1] # flipped due to convolution theroem
    
    # set ROI if needed
    off_x = off_y = 0
    
    if roi is not None:
        off_x = roi[0]
        off_y = roi[1]
        
        corr_cut = corr[
            roi[1] : roi[3], # y-axis
            roi[0] : roi[2]  # x-axis
        ]
    
    # now pad by window size
    corr_padded = np.pad(corr, window_size, mode="constant")
    pad_off = window_size
    
    # get sub-windows of correlation matrix
    corr_windows = sliding_window_array(
        corr_padded,
        [window_size, window_size],
        [overlap, overlap]
    )    
   
    # get field shape
    field_shape = get_field_shape(
        corr_padded.shape,
        window_size,
        overlap
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
    
    # create a grid
    grid_x, grid_y = get_coordinates(
        corr_padded.shape,
        window_size,
        overlap,
        center_on_field=False
    ) - np.array([window_size // 2])
    
    # add grid to peak indexes to get estimated location
    pos_x = grid_x + max_ind_x
    pos_y = grid_y + max_ind_y
    
    # find points near sub window borders and with low peak heights
    flags = np.zeros_like(pos_x).astype(bool, copy=False)
    n_exclude = 3
    
    flags[max_ind_x < n_exclude] = True
    flags[max_ind_y < n_exclude] = True
    flags[max_ind_x > window_size - n_exclude - 1] = True
    flags[max_ind_y > window_size - n_exclude - 1] = True
    flags[peaks < min_peak_height] = True
    
    # remove flagged elements
    pos_x = pos_x[~flags]
    pos_y = pos_y[~flags]
    
    # add offsets from roi
    pos_x += off_x
    pos_y += off_y
    
    # create 2D array of coordinates
    pos = np.array([pos_x, pos_y], dtype="float64").T

    # find clusters
    clusters = np.sqrt(
            np.square(
                pos[:, 0].reshape(-1, 1) - pos[:, 0].reshape(1,-1)
            ) + 
            np.square(
                pos[:, 1].reshape(-1, 1) - pos[:, 1].reshape(1,-1)
            )
        ) <= merge_radius
    
    # get mean of clusters iteratively
    for _ in range(merge_iter):
        for ind in range(pos.shape[0]):
            pos[ind, :] = np.mean(
                pos[clusters[ind, :], :].reshape(-1, 2),
                axis = 0
            )
    
    # convert to integers by rounding everything down
    pos = np.floor(pos)

    # count the number of copies
    pos, count = np.unique(
        pos,
        return_counts=True,
        axis=0
    )
    
    # remove positions that are not detected enough times
    pos = pos[count >= min_count, :]
    count = count[count >= min_count]
    
    # remove padding offsets
    pos -= pad_off
    
    # find points outside of image
    flags = np.zeros_like(pos[:, 0]).astype(bool, copy=False)
    
    n_exclude = 8
    flags[pos[:, 0] < n_exclude] = True
    flags[pos[:, 1] < n_exclude] = True
    flags[pos[:, 0] > image.shape[1] - n_exclude - 1] = True
    flags[pos[:, 1] > image.shape[0] - n_exclude - 1] = True
    
    # remove points outside of image
    pos = pos[~flags]
    count = count[~flags]
    
    if refine_pos == True:
        # kernel size for gaussian estimator
        n_halfwidth = 3

        _sx, _sy = np.meshgrid(
            np.arange(
                -n_halfwidth,
                n_halfwidth + 1,
                dtype=float
            ),
            np.arange(
                -n_halfwidth,
                n_halfwidth + 1,
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
        
        # we use a psuedo-inverse via SVD so wi can solve a system of equations.
        # using the least squares methods is preferable here, though.
        s_inv = np.linalg.pinv(s_arr)
                        
        # TODO: optimize this loop
        for ind in range(pos.shape[0]): 
            x, y = pos[ind, :].astype(int)
            
            slices = (
                slice(
                    y - n_halfwidth,
                    y + n_halfwidth + 1),
                slice(
                    x - n_halfwidth,
                    x + n_halfwidth + 1
                )
            )

            corr_sec = np.ravel(corr[slices])
            corr_sec[corr_sec <= 0] = 1e-6
            
            coef = np.dot(s_inv, np.log(corr_sec))
            
            sx = 1 / np.sqrt(-2 * coef[2])
            sy = 1 / np.sqrt(-2 * coef[3])
            
            shift_x = coef[0] * np.square(sx)
            shift_y = coef[1] * np.square(sy)
            
            pos[ind, 0] = x + shift_x
            pos[ind, 1] = y + shift_y
        
    if return_count == True:
        return pos, count
    else:
        return pos


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
    from scipy.ndimage import label, labeled_comprehension, center_of_mass, find_objects
    
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
    
    # scale the image to [0, 255]
    image[image < 0] = 0. # cut negative pixel intensities
    image /= image.max() # normalize
    image *= 255. # scale
    
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
    
    # sort so the results behave like detect_markers_local
    order = np.lexsort(
        (pos[:, 1], pos[:, 0])
    )
    
    return pos[order]


# @author: Theo
# Created on Thu Mar 25 21:03:47 2021

# @ErichZimemr - Changes (June 2, 2023):
# Revised function
def show_calibration_image(
    image: np.ndarray, 
    markers: np.ndarray, 
    radius: int=30
):
    """Plot markers on image.
    
    Plot markers on image and their associated index. This allows one to find the
    origin, x-axis, and y-axis point indexes for object-image point matching.
    
    Parameters
    ----------
    image : 2D np.ndarray
        A 2D array containing grayscale pixel intensities.
    markers : 2D np.ndarray
        A 2D array containing image marker coordinates in [x, y]` image coordinates.
    radius : int, optional
        The radius of the circle drawn around the marker point.
    
    Returns
    -------
    None
    
    Examples
    --------
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv import calib_utils
    >>> from openpiv.data.test5 import cal_image
    
    >>> cal_img = cal_image(z=0)
    
    >>> marks_pos = calib_utils.detect_markers_local(
            cal_img,
            window_size = 64,
            template_radius=5,
            min_peak_height = 0.2,
            merge_radius = 10,
            merge_iter=5,
            min_count=8,
        )
    
    >>> calib_utils.show_calibration_image(
        cal_img,
        marks_pos
    )
    
    """
    from PIL import Image, ImageFont, ImageDraw
    from matplotlib import pyplot as plt
    
    # funtction to show th clalibration iamge with numbers and circles
    plt.close('all')
    
    marker_numbers=np.arange(0,np.size(markers[:,0]))
    
    image_p = Image.fromarray(np.uint8((image/np.max(image[::]))*255))
    
    draw = ImageDraw.Draw(image_p)
    font = ImageFont.truetype("arial.ttf", 35)
    
    for i in range(0, np.size(markers, 0)):
        x, y=markers[i,:]
        draw.text((x, y), str(marker_numbers[i]), fill=(255),
                  anchor='mb',font=font)
        
    plt.figure(1)
    fig, ax = plt.subplots(1)
    ax.imshow(image_p)
    
    for marker in markers:
        x, y = marker
        c = plt.Circle((x, y), radius, color='red', linewidth=2, fill=False)
        ax.add_patch(c)
        
    plt.show()
    plt.pause(1)
    
    
# @author: Theo
# Created on Thu Mar 25 21:03:47 2021

# @ErichZimemr - Changes (June 2, 2023):
# Revised function
def get_pairs_anal(
    image_points: np.ndarray,
    origin_ind: int,
    x_ind: int,
    y_ind: int,
    grid_size: Tuple[int, int],
    spacing: float,
    z: float=0
): 
    """Match object points to image points analytically.
    
    Match object points to image points. This is only applicable for planar 
    calibration plates that are relatively perpendicular to the camera position
    (e.g., no rotation) and non-severe distortion.
    
    Parameters
    ----------
    image_points : 2D np.ndarray
        2D np.ndarray of [x, y]` image coordinates.
    origin_ind : int
        Index of the point to define the origin.
    x_ind : int
        Index of the point to define the x-axis.
    y_ind : int
        Index of the point to define the y-axis.
    grid_size : tuple[int, int]
        Grid size for the x- and y-axis.
    spacing : float
        Grid spacing in millimeters.
    z : float, optional
        The z plane where the calibration plate is located.
    
    Returns
    -------
    image_points : 2D np.ndarray
        2D matched image points of [x, y]` in image coordinates.
    object_points : 2D np.ndarray
        2D matched object points of [X, Y, Z]` in world coordinates.
    
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv import calib_utils
    >>> from openpiv.data.test5 import cal_image
    
    >>> cal_img = cal_image(z=0)
    
    >>> marks_pos = calib_utils.detect_markers_local(
            cal_img,
            window_size = 64,
            template_radius=5,
            min_peak_height = 0.2,
            merge_radius = 10,
            merge_iter=5,
            min_count=8,
        )
    
    Here, we get the indexes for the origin, x-axis, and y-axis. Index 118
    corresponds to the selected origin, while index 132 and 119 defines the 
    x-axis and y-axis respectively.
    
    >>> calib_utils.show_calibration_image(
        cal_img,
        marks_pos
    )
    
    >>> img_points, obj_points = calib_utils.get_pairs_anal(
            marks_pos,
            orig_ind=118,
            x_ind=132,
            y_ind=119,
            grid_size=[15, 15],
            spacing=30,
            z=0
        )
    
    """
    from scipy.spatial.distance import cdist

    # rearrange image coordinates
    coords = np.zeros_like(image_points)
    coords[:, 0] = image_points[:, 1] # y
    coords[:, 1] = image_points[:, 0] # x
    
    # get and set origin
    origin  = coords[origin_ind, :]
    search_x = coords[x_ind, :] - origin
    search_y = coords[y_ind, :] - origin
    
    # build meshgrid large enough to contain any possible origin position
    range_y = np.arange(-grid_size[1] + 1, grid_size[1]) 
    range_x = np.arange(-grid_size[0] + 1, grid_size[0])
    
    y_0, x_0 = np.meshgrid(
        range_y*search_y[0],
        range_x*search_x[0], 
        indexing='ij'
    )
    y_1, x_1 = np.meshgrid(
        range_y*search_y[1],
        range_x*search_x[1],
        indexing='ij'
    )

    # new meshgrid with origin at [0,0]
    y_s = y_0 + y_1 + origin[0]
    y_s = y_s
    x_s = x_0 + x_1 + origin[1]

    # search points to find the nearest points on
    y_s = np.reshape(y_s, (-1,1))
    x_s = np.reshape(x_s, (-1,1))
    search_points = np.concatenate((y_s,x_s), axis=-1)

    # create a mask to reduce the size of the search grid
    min_y = np.min(coords[:,0])
    max_y = np.max(coords[:,0])
    min_x = np.min(coords[:,1])
    max_x = np.max(coords[:,1])
    
    tol_y = np.floor(abs((search_y[0] + search_x[0])/2))
    tol_x = np.floor(abs((search_y[1] + search_x[1])/2))
    
    con_1 = ((min_y - tol_y) <= search_points[:,0]) &\
            ((max_y + tol_y) >= search_points[:,0])
    con_2 = ((min_x - tol_x) <= search_points[:,1]) &\
            ((max_x + tol_x) >= search_points[:,1])
    
    con_comb = con_1 & con_2
    search_points = search_points[np.where(con_comb)]

    # calculate the euclidean distance between the search grid points and the marker coordinates
    # to determine the nearest neighbours
    dist = cdist(search_points, coords)
    val_object_grid_index = np.argmin(dist, axis=0)
    search_points = search_points[val_object_grid_index, :]
    
    # get the markers in the right oder
    dist_2 = cdist(search_points, coords)
    right_order_index = np.argmin(dist_2, axis=1)
    image_points = image_points[right_order_index, :]
    
    if np.size(val_object_grid_index) != np.size(right_order_index):
        raise('A problem related to the point matching occured')

    # create the object grid
    object_mesh_y, object_mesh_x = np.meshgrid(
        range_x*spacing,
        range_y*spacing,
        indexing='ij'
    )
    object_mesh_y = object_mesh_y
    object_mesh_y = np.reshape(object_mesh_y, (-1,1))
    object_mesh_x = np.reshape(object_mesh_x, (-1,1))
    
    object_points = np.concatenate(
        (object_mesh_x, object_mesh_y, np.zeros_like(object_mesh_x) + z),
        axis=-1
    )
    
    object_points = object_points[np.where(con_comb)]
    object_points = object_points[val_object_grid_index]

    return image_points, object_points


def get_reprojection_error(
    cam_struct: dict,
    proj_func: "function",
    object_points: np.ndarray,
    image_points: np.ndarray
):
    """Calculate camera calibration error.
    
    Calculate the camera calibration error by projecting object points into image
    points and calculating the root mean square (RMS) error.
    
    Parameters
    ----------
    cam_struct : dict
        A dictionary structure of camera parameters.
    proj_func : function
        Projection function with the following signiture:
        res = func(cam_struct, object_points).
    object_points: 2D np.ndarray
        A numpy array containing [X, Y, Z] object points.
    image_points: 2D np.ndarray
        A numpy array containing [x, y] image points.
        
    Returns
    -------
    RMSE : float
        Root mean square (RMS) error of camera paramerters.
        
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv import calib_utils, calib_polynomial
    >>> from openpiv.data.test5 import cal_points
    
    >>> obj_x, obj_y, obj_z, img_x, img_y, img_size_x, img_size_y = cal_points()
    
    >>> camera_parameters = calib_polynomial.generate_camera_params(
            name="cam1", 
            [img_size_x, img_size_y]
        )
        
    >>> camera_parameters = calib_polynomial.minimize_polynomial(
            camera_parameters,
            np.array([obj_x, obj_y, obj_z]),
            np.array([img_x, img_y])
        )
    
    >>> calib_utils.get_reprojection_error(
            camera_parameters2, 
            calib_polynomial.project_points,
            [obj_x, obj_y, obj_z],
            [img_x, img_y]
        )
    
    """ 
    res = proj_func(
        cam_struct,
        object_points
    )
        
    error = res - image_points
    
    RMSE = np.mean(
        np.sqrt(
            np.sum(
                np.square(error),
                axis=0
            )
        )
    )
    
    return RMSE


def get_los_error(
    cam_struct,
    project_to_z_func: "function",
    project_points_func: "function",
    z
):
    """Calculate camera LOS error.
    
    Calculate camera line of sight error at the selected volume depth.
    
    Parameters
    ----------
    cam_struct : dict
        A dictionary structure of camera parameters.
    project_to_z_func : function
        Projection function with the following signiture:
        res = func(cam_struct, image_points, Z).
    project_points_func : function
        Projection function with the following signiture:
        res = func(cam_struct, object_points).
    z : float
        A float specifying the Z (depth) value to project to.
    
    Returns
    -------
    RMSE : float
        Root mean square (RMS) error of camera paramerters.
    
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv import calib_utils, calib_polynomial
    >>> from openpiv.data.test5 import cal_points
    
    >>> obj_x, obj_y, obj_z, img_x, img_y, img_size_x, img_size_y = cal_points()
    
    >>> camera_parameters = calib_polynomial.generate_camera_params(
            name="cam1", 
            [img_size_x, img_size_y]
        )
        
    >>> camera_parameters = calib_polynomial.minimize_polynomial(
            camera_parameters,
            np.array([obj_x, obj_y, obj_z]),
            np.array([img_x, img_y])
        )
    
    >>> calib_utils.get_los_error(
            camera_parameters2,
            calib_polynomial.project_to_z,
            calib_polynomial.project_points,
            z = 0
        )
        
    """
    
    # create a meshgrid for every x and y pixel for back projection.
    py, px = np.meshgrid(
        np.arange(0, cam_struct["resolution"][1]),
        np.arange(0, cam_struct["resolution"][0]),
        indexing="ij"
    )
    
    image_grid = np.concatenate(
        [py.reshape(-1, 1), px.reshape(-1, 1)],
        axis=1
    )
    
    x = image_grid[:, 1]
    y = image_grid[:, 0]
    
    # get depth
    Z = np.zeros_like(x) + z
    
    # project image coordinates to world points
    X, Y, Z = project_to_z_func(
        cam_struct,
        [x, y],
        Z
    )
    
    # project world points back to image coordinates
    res = project_points_func(
        cam_struct,
        [X, Y, Z]
    )
    
    error = res - np.array([x, y])
    
    RMSE = np.mean(
        np.sqrt(
            np.sum(
                np.square(error),
                axis=0
            )
        )
    )
    
    return RMSE


# This script was originally from Theo's polynomial calibration repository.
def get_image_mapping(
    cam_struct: dict,
    project_to_z_func: "function",
    project_points_func: "function"
):
    """Get image Mapping.
    
    Get image mapping for rectifying 2D images.
    
    Parameters
    ----------
    cam_struct : dict
        A dictionary structure of camera parameters.
    project_to_z_func : function
        Projection function with the following signiture:
        res = func(cam_struct, image_points, Z).
    project_points_func : function
        Projection function with the following signiture:
        res = func(cam_struct, object_points).
    
    Returns
    -------
    x : 2D np.ndarray
        Mappings for x-coordinates.
    y : 2D np.ndarray
        Mappings for y-coordinates.
    scale : float
        Image to world scale factor.
        
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv import calib_utils, calib_polynomial
    >>> from openpiv.data.test5 import cal_points
    
    >>> obj_x, obj_y, obj_z, img_x, img_y, img_size_x, img_size_y = cal_points()
    
    >>> camera_parameters = calib_polynomial.generate_camera_params(
            name="cam1", 
            [img_size_x, img_size_y]
        )
        
    >>> camera_parameters = calib_polynomial.minimize_polynomial(
            camera_parameters,
            np.array([obj_x, obj_y, obj_z]),
            np.array([img_x, img_y])
        )
    
    >>> mappings, scale = calib_utils.get_image_mapping(
            camera_parameters,
            calib_polynomial.project_to_z,
            calib_polynomial.project_points
        )
        
    >>> mappings
    
    >>> scale
    
    """
    
    # create a meshgrid for every x and y pixel for back projection.
    py, px = np.meshgrid(
        np.arange(0, cam_struct["resolution"][1]),
        np.arange(0, cam_struct["resolution"][0]),
        indexing="ij"
    )
    
    image_grid = np.concatenate(
        [py.reshape(-1, 1), px.reshape(-1, 1)],
        axis=1
    )
    
    x = image_grid[:, 1]
    y = image_grid[:, 0]
    
    # We set Z to zero since there is no depth
    Z = np.zeros_like(x)
    
    # project image coordinates to world points
    world_x, world_y, _ = project_to_z_func(
        cam_struct,
        [x, y],
        Z
    )
    
    world_x = world_x.reshape(cam_struct["resolution"], order='C')
    world_y = world_y.reshape(cam_struct["resolution"], order='C')
    
    # get scale
    lower_bound_X = np.min(np.absolute(world_x[:, 0]))
    upper_bound_X = np.min(np.absolute(world_x[:, -1]))
    lower_bound_Y = np.min(np.absolute(world_x[0, :]))
    upper_bound_Y = np.min(np.absolute(world_x[-1, :]))
    
    scale_X = (lower_bound_X + upper_bound_X) / np.size(world_x, 1)
    scale_Y = (lower_bound_Y + upper_bound_Y) / np.size(world_x, 0)
    
    Scale = min(scale_X, scale_Y)
    
    # get border limits
    min_X = np.min(world_x)
    max_X = np.max(world_x)
    min_Y = np.min(world_y)
    max_Y = np.max(world_y)
    
    # create a meshgrid for every x and y point for forward projection.
    X, Y = np.meshgrid(
        np.linspace(
            min_X + scale_X,
            max_X, 
            num=cam_struct["resolution"][0], 
            endpoint=True
        ),
        np.linspace(
            min_Y + scale_Y,
            max_Y, 
            num=cam_struct["resolution"][1], 
            endpoint=True
        )
    )
    
    X = np.squeeze(X.reshape(-1, 1))
    Y = np.squeeze(Y.reshape(-1, 1))
    
    # project world points to image coordinates
    mapped_grid = project_points_func(
        cam_struct,
        [X, Y, Z]
    )
    
    mapped_grid_x = mapped_grid[0].reshape(cam_struct["resolution"])
    mapped_grid_y = mapped_grid[1].reshape(cam_struct["resolution"])
    
    return np.array([mapped_grid_x, mapped_grid_y]), Scale