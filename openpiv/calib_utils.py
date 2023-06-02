import numpy as np
from typing import Tuple


def get_circular_template(
    window_size,
    template_radius,
    val=1
):
    """Create a circle template.
    
    Create a circle template based on the temnplate radius and window size.
    This template can be correlated with an image to find features such as
    marker detection on calibration plates.
    
    Parameters
    ----------
    window_size : int
        A square window size.
    template_radius : int
        The radius of the circle in the template.
    val : float, optional
        The value set to the circular elements in the template.
        
    Returns
    -------
    template : 2D np.ndarray
        A 2D np.ndarray of dtype np.float32 containing a centralized circular
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
    # make sure input is integers
    window_size = int(window_size)
    template_radius = int(template_radius)
    
    # get template size
    template_size = 2*template_radius + 1
    
    # make sure template size is smaller than window size
    if template_size >= window_size:
        raise ValueError(
            "template_radius is too large for given window_size."
        )
        
    disk = np.zeros(
        (template_size, template_size),
        dtype="float32"
    )


    ys, xs = np.indices([template_size, template_size])

    dist = np.sqrt(
        (ys - template_radius)**2 +
        (xs - template_radius)**2
    )

    disk[dist <= template_radius] = 255.

    template = np.zeros(
        (window_size, window_size), 
        dtype= "float32"
    )
    
    template[
        window_size // 2 - template_radius - 1 : window_size // 2 + template_radius,
        window_size // 2 - template_radius - 1 : window_size // 2 + template_radius] =\
        disk
    
    return template


def get_cross_template(
    window_size,
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
    window_size : int
        A square window size.
    template_radius : int
        The radius of the cross in the template.
    val : float, optional
        The value set to the cross elements in the template.
        
    Returns
    -------
    template : 2D np.ndarray
        A 2D np.ndarray of dtype np.float32 containing a centralized cross
        element.
    
    Examples
    --------
    from openpiv.calib_utils import get_cross_template
    
    >>> get_cross_template(
            window_size=48,
            template_radius=11,
            val=255
        )
    
    """
    # make sure input is integers
    window_size = int(window_size)
    template_radius = int(template_radius)
    
    # get template size
    template_size = 2*template_radius + 1
    
    # make sure template size is smaller than window size
    if template_size >= window_size:
        raise ValueError(
            "template_radius is too large for given window_size."
        )
        
    cross = np.zeros(
        (template_size, template_size),
        dtype="float32"
    )
    
    ys, xs = np.abs(
        np.indices([template_size, template_size]) - template_radius
    )
    
    line_width = int(template_radius / 6) + 1
        
    cross[ys < line_width] = val
    cross[xs < line_width] = val
    
    template = np.zeros(
        (window_size, window_size), 
        dtype= "float32"
    )
    
    template[
        window_size // 2 - template_radius - 1 : window_size // 2 + template_radius,
        window_size // 2 - template_radius - 1 : window_size // 2 + template_radius] =\
        cross
    
    return template


def detect_markers_local(
    image: np.ndarray,
    roi: list=None,
    window_size: int=64,
    overlap = None,
    template_type: str="dot",
    template_radius: int=7,
    correlation_method: str="circular",
    min_peak_height: float=0.25,
    merge_radius: int=10,
    merge_iter: int=5,
    min_count: float=2,
    return_count=False
):
    """Detect calibration markers.
    
    Detect circular and cross calibration markers. The markers are first detected
    based on correlating a local window with a template. The position of the
    marker in the local windows is found by locating the maximum of that window, 
    which correlates to the best match with the template. Finally, false positives
    are removed by specifying the minimum count a marker is detected and the 
    minimum correlation coefficient of that marker's peak.
    
    Parameters
    ----------
    image : 2D np.ndarray
        A two dimensional array of pixel intensities of the shape (n, m).
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
    template_type : str {'dot', 'cross'}, optional
        The template type used for matching. Currently, there are two options:
        dot and cross. It is important to select the right one when template
        matching.
    template_radius : int, optional
        The radius of the dot or cross in the template. Should be roughly
        the same radius as the markers in pixels, if not larger. Improperly set
        template radius will reduce quality of marker detection and the subpixel 
        fitting.
    correlation_method : str {'circular'. 'linear'}, optional
        A string that indicates what type of correlation to perform.
        
        ``circular``
        The input is not padded and thus is faster and less memory intesive than
        its padded counterpart.
        
        ``linear``
        The input is padded by the next power of 2 to remove periodic frequencies.
        This requires substantially more computational resources and time.
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
    return_count : bool, optional
        Return the number of times a marker gets counted. This can be used to
        find the ideal threshold to find the correct markers.
    
    Returns
    -------
    markers : 2D np.ndarray
        Marker positions in [x, y]'.
        
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv import calib_utils, calib_pinhole
    >>> from openpiv.data.test5 import cal_image
    
    >>> cal_img = cal_image(z=0)
    
    >>> marks_pos, counts = calib_utils.detect_markers_local(
            cal_img,
            window_size = 64,
            template_radius=5,
            min_peak_height = 0.2,
            merge_radius = 10,
            merge_iter=5,
            min_count=8,
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
    
    # data type conversion to float32
    image = image.astype("float32")
    
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
    
    # pad image so markers near the border are detected better
    image_padded =  np.pad(image, window_size, mode="constant")
    pad_off = window_size
    
    # create template
    if template_type == "dot":
        template = get_circular_template(
            window_size,
            template_radius
        )
    
    elif template_type == "cross":
        template = get_cross_template(
            window_size,
            template_radius
        )
        
    else:
        raise ValueError(
            "template_type must be either 'dot' or 'cross'"
        )
    
    # if overlap is None, set overlap to 80% of window size
    if overlap is None:
        overlap = window_size - window_size * 0.2
    
    # make sure window_size and overlap are integers
    window_size = int(window_size)
    overlap = int(overlap)
                   
    sub_windows = sliding_window_array(
        image_padded,
        [window_size, window_size],
        [overlap, overlap]
    )
    
    # flip the sub_windows so the cross-correlation places the best fit
    # of the template in the right location
    sub_windows = sub_windows[:, ::-1, ::-1]
    
    # broadcast template to same shape as sub_windows
    template = np.broadcast_to(template, sub_windows.shape)
    
    # normalized cross correlation
    corr = fft_correlate_images(
        sub_windows,
        template,
        normalized_correlation = True,
        correlation_method = correlation_method
    )
        
    # get field shape
    field_shape = get_field_shape(
        image_padded.shape,
        window_size,
        overlap
    )
    
    # get location of peaks
    max_ind, peaks = find_all_first_peaks(
        corr,
    )
        
    max_ind_x = max_ind[:, 2]
    max_ind_y = max_ind[:, 1]
    
    # reshape field (this is not actually needed)
    max_ind_x = max_ind_x.reshape(field_shape)
    max_ind_y = max_ind_y.reshape(field_shape)
    peaks = peaks.reshape(field_shape)
    
    # create a grid
    grid_x, grid_y = get_coordinates(
        image_padded.shape,
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
    pos = np.array([pos_x, pos_y], dtype="float32").T

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

    if return_count == True:
        return pos, count
    else:
        return pos


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
        A 2D array containing image marker coordinates in [x, y]`.
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
    >>> from openpiv import calib_utils, calib_pinhole
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
    
    >>> show_calibration_image(
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
def get_obj_img_pairs(
    img_points: np.ndarray,
    origin_ind: int,
    x_ind: int,
    y_ind: int,
    grid_size: Tuple[int, int],
    spacing: float,
    z: float
):
    """ Match object and image points.
    
    Match object and image points. 
    
    Parameters
    ----------
    img_points : 2D np.ndarray
        2D np.ndarray of [x, y]` coordinates.
    origin_ind : int
        Index to define the origin.
    x_ind : int
        Index for the point to define the x-axis.
    y_ind : int
        Index for the point to define the y-axis.
    grid_size : tuple[int, int]
        Grid size for the x- and y-axis.
    spacing : float
        Grid spacing in millimeters.
    z : float
        The z plane where the calibration plate is located.
    
    Returns
    -------
    img_points : 2D np.ndarray
        2D matched image points of [x, y]` coordinates.
    obj_points : 2D np.ndarray
        2D matched object points of [x, y, z]` coordinates.
    
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv import calib_utils, calib_pinhole
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
    
    >>> show_calibration_image(
        cal_img,
        marks_pos
    )
    
    >>> img_points, obj_points = get_obj_img_pairs(
            marks_pos,
            orig_ind=118,
            x_ind=132,
            y_ind=119,
            grid_size=[15, 15],
            spacing=30,
            z=0
        )
        
    
    
    """
    # rearrange image coordinates
    coords = np.zeros_like(img_points)
    coords[:, 0] = img_points[:, 1] # y
    coords[:, 1] = img_points[:, 0] # x
    
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
    image_points = img_points[right_order_index, :]
    
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