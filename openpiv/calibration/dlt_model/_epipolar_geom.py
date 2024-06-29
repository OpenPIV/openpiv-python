import numpy as np

from .._epipolar_utils import _line_intersect, _multi_line_intersect


__all__ = [
    "line_intersect",
    "multi_line_intersect"
]


def line_intersect(
    cam_1: class,
    cam_2: class,
    img_points_1: np.ndarray,
    img_points_2: np.ndarray
):
    """Calculate where two rays intersect.
    
    Using two cameras, calculate the world coordinates where two rays
    intersect. This is done through an analytical solution based on 
    direction vectors and camera origins/translations. 
    
    Parameters
    ----------
    cam_1, cam_2 : class
        An instance of a DLT camera.
    img_points_1, img_points_2 : np.ndarray
        Image coordinates stored in a ndarray structured like [x, y]'.
        
    Returns
    -------
    coords : np.ndarray
        The world coordinate that is nearest to the two rays intersecting.
    dist : float
        The minimum dinstance between the two rays.
    
    """
    cam_1._check_parameters()
    cam_2._check_parameters()
    
    dtype1 = cam_1.dtype
    dtype2 = cam_2.dtype
    
    # all cameras should have the same dtype
    if dtype1 != dtype2:
        raise ValueError(
            "Dtypes between camera structures must match"
        )
    
    img_points_1 = np.array(img_points_1, dtype=dtype1)
    img_points_2 = np.array(img_points_2, dtype=dtype2)
    
    t1, r1 = cam_1._get_inverse_vector(
        img_points_1
    )
    
    t2, r2 = cam_2._get_inverse_vector(
        img_points_2
    )
    
    return _line_intersect(
        t1[:, np.newaxis], 
        r1, 
        t2[:, np.newaxis], 
        r2
    )


def multi_line_intersect(
    cameras: list,
    img_points: list,
):
    """Calculate where multiple rays intersect.
    
    Using at least two cameras, calculate the world coordinates where the
    rays intersect. This is done through an least squares solution based on 
    direction vectors and camera origins/translations. 
    
    Parameters
    ----------
    cameras : list
        A list of instances of DLT cameras.
    img_points : list
        A list of image coordinates for each canera structure.
        
    Returns
    -------
    coords : np.ndarray
        The world coordinate that is nearest to the intersection of all rays.
    
    """
    n_cams = len(cameras)
    n_imgs = len(img_points)
    
    # make sure each camera has a set of images
    if n_cams != n_imgs:
        raise ValueError(
            f"Camera - image size mismatch. Got {n_cams} cameras and " +
            f"{n_imgs} images"
        )
    
    # check each camera structure
    for cam in range(n_cams):
        cameras[cam]._check_parameters()
        
    # all cameras should have the same dtype
    dtype1 = cameras[0].dtype
    for cam in range(1, n_cams):
        dtype2 = cameras[cam].dtype
        
        if dtype1 != dtype2:
            raise ValueError(
                "Dtypes between camera structures must match"
            )
    
    lines = []
    for cam in range(n_cams):
        points = np.array(img_points[cam], dtype=dtype1)
    
        t, r = cameras[cam]._get_inverse_vector(
            points
        )
        
        lines.append([t, r])
    
    return _multi_line_intersect(lines, dtype1)