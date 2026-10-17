import numpy as np


def _standardize_points_2d(
    x: np.ndarray,
    y: np.ndarray
):
    """Normalize 2-D points with an isotropic similarity transform.
    
    Normalize the x and y coordinates through standardization. This allows
    for a better fit when calibrating the direct linear transformation.
    
    Parameters
    ==========
    x, y : np.ndarray
        The coordinates to normalize through standardization.
        
    Returns
    =======
    x_n, y_n : np.ndarray
        Normalized x-y coordinates.
    norm_matrix : np.ndarray
        The normalization matrix for normalization and denormalization.
    
    """
    centroid = np.array(
        [np.mean(x), np.mean(y)],
        dtype="float64"
    )
    centered = np.array(
        [x - centroid[0], y - centroid[1]],
        dtype="float64"
    )
    mean_distance = np.mean(np.linalg.norm(centered, axis=0))

    if not np.isfinite(mean_distance):
        raise ValueError("Cannot normalize coincident 2D points")

    scale = np.sqrt(2.0) / mean_distance
    xy_norm = scale * centered

    norm_mat = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0]
        ], 
        dtype="float64"
    )
    
    return xy_norm, norm_mat
    

def _standardize_points_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray
):
    """Normalize 3-D points with an isotropic similarity transform.
    
    Normalize the x, y and z coordinates through standardization. This allows
    for a better fit when calibrating the direct linear transformation.
    
    Parameters
    ==========
    x, y, z : np.ndarray
        The coordinates to normalize through standardization.
        
    Returns
    =======
    x_n, y_n, z_n : np.ndarray
        Normalized x, y, z coordinates.
    norm_matrix : np.ndarray
        The normalization matrix for further normalization and denormalization.
    
    """
    centroid = np.array(
        [np.mean(x), np.mean(y), np.mean(z)],
        dtype="float64"
    )
    centered = np.array(
        [
            x - centroid[0],
            y - centroid[1],
            z - centroid[2]
        ],
        dtype="float64"
    )
    mean_distance = np.mean(np.linalg.norm(centered, axis=0))

    if not np.isfinite(mean_distance):
        raise ValueError("Cannot normalize coincident 3D points")

    scale = np.sqrt(3.0) / mean_distance
    xyz_norm = scale * centered

    norm_mat = np.array(
        [
            [scale, 0.0, 0.0, -scale * centroid[0]],
            [0.0, scale, 0.0, -scale * centroid[1]],
            [0.0, 0.0, scale, -scale * centroid[2]],
            [0.0, 0.0, 0.0, 1.0]
        ], 
        dtype="float64"
    )
    
    return xyz_norm, norm_mat