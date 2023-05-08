import numpy as np


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
        RMSE = func(cam_struct, object_points).
    object_points: 2D np.ndarray
        A numpy array containing [x, y, z] object points.
    image_points: 2D np.ndarray
        A numpy array containing [x, y] image points.
        
    Returns
    -------
    RMSE : float
        Root mean square (RMS) error of camera paramerters.
    """
    if not isinstance(object_points, np.ndarray):
        object_points = np.array(object_points).T
    
    if not isinstance(image_points, np.ndarray):
        image_points = np.array(image_points).T
        
    res = proj_func(
        cam_struct,
        object_points
    )
        
    error = res - image_points
    
    RMSE = np.mean(
        np.sqrt(
            np.sum(
                np.square(error),
                axis=1
            )
        )
    )
    
    return RMSE