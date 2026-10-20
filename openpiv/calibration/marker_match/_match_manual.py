import numpy as np


__all__ = [
    'get_pairs_proj'
]


def get_pairs_proj(
    cam: object,
    object_points: np.ndarray,
    image_points: np.ndarray,
    tolerance: float=10
): 
    """Match object points to image points via projection.
    
    Match object points to image points by projection with a rough
    calibration of at least 6 points. This method is more reliable than
    matching image points to object points based on homographies since
    image points and object points are being projected and compared to 
    each other to find a best match. This allows non-planar calibration
    plates to be used.
    
    Parameters
    ----------
    cam : object
        An instance of a camera.
    proj_func : function
        Projection function with the following signiture:
        res = func(cam_struct, object_points).
    image_points : 2D np.ndarray
        2D np.ndarray of [x, y]` image coordinates.
    object_points : 2D np.ndarray
        2D np.ndarray of [X, Y, Z]` world coordinates.
    tolerance : float, optional
        The maximum RMS error between the image point and an object point.
    
    Returns
    -------
    img_points : 2D np.ndarray
        2D matched image points of [x, y] in image coordinates.
    obj_points : 2D np.ndarray
        2D matched object points of [x, y, z] in world coordinates.
    
    Notes
    -----
    This function is used when a rough calibration is performed over some points of
    the calibration plate. These points are usually manually selected and given
    world point coordinates. At least 9 points for a pinhole model or 19 points for
    a polynomial model are needed since this gives a good enough calibration to pair
    the correct object points to image points.
    
    """
    object_points = np.array(object_points, dtype=np.float64)
    image_points = np.array(image_points, dtype=np.float64)
    
    image_points_proj = cam.project_points(
        object_points
    )
    
    obj_pairs = []
    img_pairs = []
    
    for i in range(image_points.shape[1]):
        min_j = -1
        min_rmse = 1000
        
        for j in range(image_points_proj.shape[1]):
            rmse = np.mean(
                np.sqrt(
                    np.square(
                        image_points[:, i] - image_points_proj[:, j]
                    ),
                )
            )
            
            if rmse < min_rmse:
                min_rmse = rmse
                min_j = j
        
        if min_rmse < tolerance:
            if min_j == -1:
                continue
                
            obj_pairs.append(object_points[:, min_j])
            img_pairs.append(image_points[:, i])
    
    return (
        np.array(img_pairs, dtype=np.float64).T,
        np.array(obj_pairs, dtype=np.float64).T
    )
