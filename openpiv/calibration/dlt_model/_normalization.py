import numpy as np


def _standardize_points_2d(
    x: np.ndarray,
    y: np.ndarray
):
    """Standardize x and y coordinates.
    
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
    x_a = np.mean(x)
    y_a = np.mean(y)
    
    x_s = np.sqrt(2 / np.std(x))
    y_s = np.sqrt(2 / np.std(y))
    
    x_n = x_s * x + (-x_s * x_a)
    y_n = y_s * y + (-y_s * y_a)
    
    xy_norm = np.array([x_n, y_n], dtype="float64")

    norm_mat = np.array(
        [
            [x_s, 0,   -x_s * x_a],
            [0,   y_s, -y_s * y_a],
            [0,   0,    1]
        ], 
        dtype="float64"
    )
    
    return xy_norm, norm_mat
    

def _standardize_points_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray
):
    """Standardize x, y and z coordinates.
    
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
    x_a = np.mean(x)
    y_a = np.mean(y)
    z_a = np.mean(z)
    
    x_s = np.sqrt(2 / np.std(x))
    y_s = np.sqrt(2 / np.std(y))
    z_s = np.sqrt(2 / np.std(z))
    
    x_n = x_s * x + (-x_s * x_a)
    y_n = y_s * y + (-y_s * y_a)
    z_n = z_s * z + (-z_s * z_a)
    
    xyz_norm = np.array([x_n, y_n, z_n], dtype="float64")

    norm_mat = np.array(
        [
            [x_s, 0,   0,   -x_s * x_a],
            [0,   y_s, 0,   -y_s * y_a],
            [0,   0,   z_s, -z_s * z_a],
            [0,   0,   0,   1]
        ], 
        dtype="float64"
    )
    
    return xyz_norm, norm_mat