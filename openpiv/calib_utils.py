import numpy as np


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
    from openpiv.calib_utils import get_cross_template
    
    >>> get_cross_template(
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
    
    >>> marks_pos, counts = detect_markers2(
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
    pos = np.array([pos_x, pos_y], dtype=float).T

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
    Z = x*0 + z
    
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
    
    >>> mappings, scale = get_image_mapping(
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
    Z = x*0.
    
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