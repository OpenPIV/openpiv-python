import numpy as np

from .. import _cal_doc_utils


__all__ = [
    'show_calibration_image',
    'find_nearest_points',
    'find_middle_point',
    'find_median_point'
]


# @author: Theo
# Created on Thu Mar 25 21:03:47 2021

# @ErichZimmer - Changes (June 2, 2023):
# Revised function

# @ErichZimmer - Changes (Decemmber 2, 2023):
# Revised function
def show_calibration_image(
    image: np.ndarray, 
    markers: np.ndarray,
    figsize=(8,11),
    radius: int=30,
    fontsize=20
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
    
    Let cal_img be a calibration image.
    
    >>> marks_pos = calib_utils.detect_markers_template(
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
    
    markers = markers.T
    
    # funtction to show th clalibration iamge with numbers and circles
    plt.close('all')
    
    marker_numbers=np.arange(0,np.size(markers[:,0]))
    
    image_p = Image.fromarray(np.uint8((image/np.max(image[::]))*255))
    
    draw = ImageDraw.Draw(image_p)
    font = ImageFont.truetype("arial.ttf", fontsize)
    
    for i in range(0, np.size(markers, 0)):
        x, y=markers[i,:]
        draw.text((x, y), str(marker_numbers[i]), fill=(255),
                  anchor='mb',font=font)
        
    plt.figure(1)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image_p)
    
    for marker in markers:
        x, y = marker
        c = plt.Circle((x, y), radius, color='red', linewidth=2, fill=False)
        ax.add_patch(c)
        
    plt.show()
    plt.pause(1)


@_cal_doc_utils.docfiller
def find_nearest_points(
    image_points: np.ndarray,
    point: np.ndarray,
    threshold: float=None,
    flag_nans: bool=False,
    return_indices: bool=False
):
    """Locate the nearest image point.
    
    Locate the closest image point to the user selected image point.
    This function is implemented by find the minimum distance to the
    point of interest, so it will always return a point no matter how
    (un)realistic that point is.
    
    Parameters
    ----------
    %(image_points)s
    point : 2D np.ndarray
        A 2D array of points of interest in [x, y]' image coordinates.
    threshold : float, optional
        If set, distances that are greater than the threshold are ignored.
    flag_nans : bool, optional
        If enabled, set flagged points that exceed the threshold to nan.
        Otherwise, the flagged points are not returned. Rejected indices
        are represented by -1 when return_indices is enabled.
    return_indices : bool, optional
        Return input column indices instead of matched coordinates.
        
    Returns
    -------
    points : np.ndarray
        Matched coordinates of shape (2, m), or a one-dimensional array
        of input column indices when return_indices is enabled.
    
    """
    dist = np.linalg.norm(
        [   
            image_points[0][:, np.newaxis] - point[0],
            image_points[1][:, np.newaxis] - point[1]
        ],
        axis=0
    )
        
    index = np.argmin(dist, axis = 0)
    
    set_to_nan = np.zeros_like(index).astype("bool")
    
    if threshold is not None:
        min_dist = np.min(dist, axis=0)
        
        if flag_nans == True:
            set_to_nan[min_dist > threshold] = True
            
        else:
            index = index[min_dist < threshold]
            set_to_nan = np.zeros_like(index).astype("bool")
        
    if return_indices:
        index[set_to_nan] = -1
        return index

    candidate_points = image_points[:, index]

    if flag_nans:
        candidate_points = candidate_points.astype(np.float64, copy=False)
        candidate_points[:, set_to_nan] = np.nan
    
    return candidate_points


@_cal_doc_utils.docfiller
def find_middle_point(
    image_points: np.ndarray,
    return_index: bool=False
):
    """Find the detection nearest the coordinate-wise field mean.

    Calculate the mean x and y coordinates of the detection field, then
    select the nearest detected point.

    Parameters
    ----------
    %(image_points)s
    return_index : bool, optional
        Return the original input column index instead of its coordinates.
        Default is False.

    Returns
    -------
    point : np.ndarray or int
        Coordinates of shape (2,), or an integer input column index when
        return_index is True.

    Raises
    ------
    ValueError
        If the input detection field is empty.

    """
    if image_points.shape[1] == 0:
        msg = 'The point field is empty'
        raise ValueError(msg)

    nearest = find_nearest_points(
        image_points,
        np.mean(image_points, axis=1),
        return_indices=return_index
    )

    if return_index:
        return int(nearest[0])

    return nearest[:, 0]


@_cal_doc_utils.docfiller
def find_median_point(
    image_points: np.ndarray,
    return_index: bool=False
):
    """Find the detection nearest the coordinate-wise field median.

    Calculate the median x and y coordinates of the detection field, then
    select the nearest detected point.

    Parameters
    ----------
    %(image_points)s
    return_index : bool, optional
        Return the original input column index instead of its coordinates.
        Default is False.

    Returns
    -------
    point : np.ndarray or int
        Coordinates of shape (2,), or an integer input column index when
        return_index is True.

    Raises
    ------
    ValueError
        If the input detection field is empty.

    """
    if image_points.shape[1] == 0:
        msg = 'The point field is empty'
        raise ValueError(msg)

    nearest = find_nearest_points(
        image_points,
        np.median(image_points, axis=1),
        return_indices=return_index
    )

    if return_index:
        return int(nearest[0])

    return nearest[:, 0]
