import numpy as np


def detect_markers(
    image: np.ndarray,
    roi: list=None,
    window_size: int=64,
    overlap: int=None,
    template_type: str="dot",
    template_radius: int=5,
    merge_radius: int=10,
    min_count: float=10
):
    """Detect calibration markers.
    
    Detect circular and cross calibration markers. The markers are first detected
    based on correlating a local window with a template. The position of the
    marker in the local windows is found by finding the maximum of that window, 
    which correlates to the best match with the template. Then, false positives
    are removed by specifying the minimum count a marker is detected. Once the
    estimated target points are found, the local windows around the targets are
    correlated with the template and the final peak is fitted with a centroid
    estimator to get subpixel accuracy.
    
    Parameters
    ----------
    image : np.ndarray of shape (n, m)
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
    template_type : str, optional
        The template type used for matching. Currently, there are two options:
        dot and cross. It is important to select the right one when template
        matching.
    template_radius : int, optional
        The radius of the dot or cross in the template. Should be roughly
        the same radius as the markers in pixels, if not larger. Improperly set
        template radius will reduce quality of marker detection and the subpixel 
        fitting.
    merge_radius : ing, optional
        The merge radius when detecting the number of times a marker is found.
        Typically, the merge radius should be 5 to 10.
    min_count : float, optional
        The minimum amount a marker is detected. Helps to remove false
        positives on marker detection.
    
    Returns
    -------
    markers : np.ndarray of shape (n, 2)
        Marker positions in [x, y]'.
    
    Notes
    -----
    The subpixel fitting is performed by a 3 point gaussian or centroid estimator 
    In the future, a gaussian peak fitting algorithm based on the pseudo-inverse
    (computed by SVD) could be used as a generalized least-squares solution.
    As the pseudo-inverse only has to be computed once, the algorithm would
    theroetically be fast while enabling n x n pixels, where n is an integer that 
    is the half-width of the fitting kernel, to be used when peak fitting. This 
    may be more accurate for wider peaks as more information is used in the subpixel 
    estimation.
    
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
    
    # create template
    if template_type == "dot":
        template_size = 2*template_radius + 1
        
        disk = np.zeros(
            (template_size, template_size),
            dtype="float32"
        )
        
        template = np.zeros(
            (window_size, window_size), 
            dtype= "float32"
        )
        
        xs, ys = np.indices([template_size, template_size])
        
        dist = np.sqrt(
            (ys - template_radius)**2 +
            (xs - template_radius)**2
        )
        
        disk[dist <= template_radius] = 255.
             
        template[
            window_size // 2 - template_radius - 1 : window_size // 2 + template_radius,
            window_size // 2 - template_radius - 1 : window_size // 2 + template_radius] =\
             disk
    
    elif template_type == "cross":
        raise ValueError(
            "Lost due to an unexpected microsoft update :("
        )
        
    else:
        raise ValueError(
            "template_type must be either 'dot' or 'cross'"
        )
    
    # if overlap is None, set overlap to 75% of window size
    if overlap is None:
        overlap = window_size - window_size * 0.25
    
    # make sure window_size and overlap are integers
    window_size = int(window_size)
    overlap = int(overlap)
    
    # get local sub-windows for template matching
    sub_windows = sliding_window_array(
        image,
        [window_size, window_size],
        [overlap, overlap]
    )
    
    # flip the sub_windows so the cross-correlation places the best fit
    # of the template in the right location
    sub_windows = sub_windows[:, ::-1, ::-1]
    
    # broadcast template to came shape as sub_windows
    template = np.broadcast_to(template, sub_windows.shape)
    
    # normalized linear cross correlation
    corr = fft_correlate_images(
        sub_windows,
        template,
        normalized_correlation = True,
        correlation_method = "linear"
    )

    # reshape field
    field_shape = get_field_shape(
        image.shape,
        window_size,
        overlap
    )
    
    # get location of peaks
    max_ind = find_all_first_peaks(
        corr
    )[0][:, 1:]
    
    max_ind_x = max_ind[:, 1]
    max_ind_y = max_ind[:, 0]
    
    max_ind_x = max_ind_x.reshape(field_shape)
    max_ind_y = max_ind_y.reshape(field_shape)
    
    # delete corr and sub_windows as we no longer need it
    del corr, sub_windows
    
    # create a grid
    grid_x, grid_y = get_coordinates(
        image.shape,
        window_size,
        overlap,
        center_on_field=False
    ) - np.array([window_size // 2])
    
    # add grid to peak indexes to get estimated location
    pos_x = grid_x + max_ind_x
    pos_y = grid_y + max_ind_y
    
    # find points near sub window borders
    flags = np.zeros_like(pos_x).astype(bool, copy=False)
    n_exclude = 3
    
    flags[max_ind_x < n_exclude] = True
    flags[max_ind_y < n_exclude] = True
    flags[max_ind_x > window_size - n_exclude - 1] = True
    flags[max_ind_y > window_size - n_exclude - 1] = True
    
    # set flagged elements to nan
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
    for _ in range(8):
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
    pos = pos[count >= min_count]
    
    # create new windows at specific locations
    new_grid_x = pos[:, 0] - window_size // 2
    new_grid_y = pos[:, 1] - window_size // 2
    
    new_grid_x[new_grid_x < 0] = 0
    new_grid_y[new_grid_y < 0] = 0
    
    new_grid_x_e = np.reshape(new_grid_x, (-1, 1, 1)).astype(int)
    new_grid_y_e = np.reshape(new_grid_y, (-1, 1, 1)).astype(int)

    win_x, win_y = np.meshgrid(
        np.arange(0, window_size), 
        np.arange(0, window_size)
    )
    
    win_x = win_x[np.newaxis,:,:] + new_grid_x_e
    win_y = win_y[np.newaxis,:,:] + new_grid_y_e
    
    sub_windows = image[win_y, win_x][:, ::-1, ::-1]
    
    # broadcast template to new stack size
    template = np.broadcast_to(template[0, :, :], sub_windows.shape)
    
    # normalized linear cross correlation
    corr = fft_correlate_images(
        sub_windows,
        template,
        normalized_correlation = True,
        correlation_method = "linear"
    )
    
    # get location of peaks
    new_max_ind = find_all_first_peaks(
        corr
    )[0][:, 1:]
    
    new_max_ind_x = new_max_ind[:, 1]
    new_max_ind_y = new_max_ind[:, 0]
    
    # add grid to peak indexes to get estimated location
    new_pos_x = new_grid_x + new_max_ind_x + off_x
    new_pos_y = new_grid_y + new_max_ind_y + off_y
    
    # now get subpixel offset
    n_halfwidth = 5
    
    centroid_grid_x = centroid_grid_y = np.arange(
        -n_halfwidth,
        n_halfwidth + 1,
        dtype=float
    )
    
    centroid_x, centroid_y = np.meshgrid(
        centroid_grid_x,
        centroid_grid_y
    )
    
    # TODO: optimize this loop and add protection from peaks near edges
    for ind in range(corr.shape[0]):
        corr_crop_slice = (
            ind,
            slice(
                new_max_ind_x[ind] - n_halfwidth,
                new_max_ind_x[ind] + n_halfwidth + 1,
            ),        
            slice(
                new_max_ind_y[ind] - n_halfwidth,
                new_max_ind_y[ind] + n_halfwidth + 1,
            )
        )
        
        corr_sec = corr[corr_crop_slice]
        
        corr_sum = np.sum(corr_sec)
        
        shift_x = np.sum(centroid_x * corr_sec) / corr_sum
        shift_y = np.sum(centroid_y * corr_sec) / corr_sum
        
        new_pos_x[ind] = new_pos_x[ind] + shift_x
        new_pos_y[ind] = new_pos_y[ind] + shift_y
    
    return np.array([new_pos_x, new_pos_y], dtype=float).T


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
    
    >>> path_to_calib = "../openpiv/data/test5/test_cal.csv"
    
    >>> obj_x, obj_y, obj_z, img_x, img_y = np.loadtxt(
            path_to_calib,
            unpack = True,
            skiprows=1,
            delimiter = ','
        )
        
    >>> camera_parameters = calib_polynomial.generate_camera_params(
            "cam1", 
            [1024, 1024]
        )
        
    >>> camera_parameters = calib_polynomial.minimize_polynomial(
            camera_parameters,
            np.array([obj_x, obj_y, obj_z]),
            np.array([img_x0, img_y0])
        )
    
    >>> calib_utils.get_reprojection_error(
            camera_parameters2, 
            calib_polynomial.project_points,
            [obj_x, obj_y, obj_z],
            [img_x0, img_y0]
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
    
    >>> path_to_calib = "../openpiv/data/test5/test_cal.csv"
    
    >>> obj_x, obj_y, obj_z, img_x, img_y = np.loadtxt(
            path_to_calib,
            unpack = True,
            skiprows=1,
            delimiter = ','
        )
        
    >>> camera_parameters = calib_polynomial.generate_camera_params(
            "cam1", 
            [1024, 1024]
        )
        
    >>> camera_parameters = calib_polynomial.minimize_polynomial(
            camera_parameters,
            np.array([obj_x, obj_y, obj_z]),
            np.array([img_x0, img_y0])
        )
    
    >>> calib_utils.get_los_error(
            camera_parameters2,
            calib_polynomial.project_to_z,
            calib_polynomial.project_points,
            z = -5
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
    X : 2D np.ndarray
        Mappings for x-coordinates.
    Y : 2D np.ndarray
        Mappings for y-coordinates.
    scale : float
        Image to world scale factor.
        
    Notes
    -----
    Scale is only applicable if the image and object points are
    not normalized.
    
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