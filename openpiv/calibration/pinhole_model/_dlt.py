import numpy as np


__all__ = [
    "_standardize_points_2d",
    "_standardize_points_3d",
    "calibrate_dlt"
]


def _standardize_points_2d(
    x: np.ndarray,
    y: np.ndarray
):
    """Standardize x and y coordinates.
    
    Normalize the x and y coordinates through standardization. This allows for a
    better fit when calibrating the direct linear transformation.
    
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
    
    Normalize the x, y and z coordinates through standardization. This allows for a
    better fit when calibrating the direct linear transformation.
    
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


def calibrate_dlt(
    object_points: np.ndarray,
    image_points: np.ndarray,
    enforce_coplanar: bool=False
):
    """Coplanar DLT for homography.
    
    Compute a homography matrix using direct linear transformation. For 2D lab
    coordinates, a 2D DLT is performed. For lab coordinates that include a Z-axis,
    a 3D DLT is performed. For 3D DLT's, an option to error if the Z-axis is not
    co-planar is available.
    
    Parameters
    ==========
    object_points : 2D np.ndarray
        Real world coordinates. The ndarray is structured like [X, Y].
    image_points : 2D np.ndarray
        Image coordinates. The ndarray is structured like [x, y].
    enforce_coplanar : bool
        If a Z plane is supplied in the object points, check whether or not the Z
        planes are co-planar.
        
    Returns
    =======
    H : 2D np.ndarray
        A 3x3 matrix containing a homography fit for the object and image points.
    error : float
        The RMSE error of the DLT fit.
    
    """
    object_points = np.array(object_points, dtype="float64")
    image_points = np.array(image_points, dtype="float64")
    
    ndims = object_points.shape[0]
    
    min_points = 4
    
    if ndims == 3 and  enforce_coplanar != True:
        min_points += 2
    
    if object_points.shape[1] < min_points:
        raise ValueError(
            f"Too little points to calibrate. Need at least {min_points} points"
        )
    
    if object_points.shape[1] != image_points.shape[1]:
        raise ValueError(
            "Object point image point size mismatch"
        )

    if ndims not in [2, 3]:
        raise ValueError(
            "Object points must be in either [X, Y] (shape = [N, 2]) or [X, Y, Z] "+
            "format (shape = [N, 3]). Recieved shape = [N, {}]".format(ndims)
        )
    
    if enforce_coplanar == True:
        if ndims == 3:
            if np.std(object_points[3]) > 0.00001:
                raise ValueError(
                    "Object points must be co-planar"
                )
            ndims = 2
            object_points = object_points[:2, :]
    
    if ndims == 2:
        X_raw, Y_raw = object_points
        x_raw, y_raw = image_points
        
        # normalize for better dlt results
        [X, Y], lab_norm_mat = _standardize_points_2d(X_raw, Y_raw)
        [x, y], img_norm_mat = _standardize_points_2d(x_raw, y_raw)
            
        # mount constraints
        A = np.zeros([x.shape[0] * 2, 9], dtype="float64")
        A[0::2, 0] = X
        A[0::2, 1] = Y
        A[0::2, 2] = 1
        A[0::2, 6] = -x * X
        A[0::2, 7] = -x * Y
        A[0::2, 8] = -x

        A[1::2, 3] = X
        A[1::2, 4] = Y
        A[1::2, 5] = 1
        A[1::2, 6] = -y * X
        A[1::2, 7] = -y * Y
        A[1::2, 8] = -y
        
    else:
        X_raw, Y_raw, Z_raw = object_points
        x_raw, y_raw = image_points

        # normalize for better dlt results
        [X, Y, Z], lab_norm_mat = _standardize_points_3d(X_raw, Y_raw, Z_raw)
        [x, y], img_norm_mat = _standardize_points_2d(x_raw, y_raw)

        # mount constraints
        A = np.zeros([x.shape[0] * 2, 12], dtype="float64")
        A[0::2, 0] = X
        A[0::2, 1] = Y
        A[0::2, 2] = Z
        A[0::2, 3] = 1
        A[0::2, 8]  = -x * X
        A[0::2, 9]  = -x * Y
        A[0::2, 10] = -x * Z
        A[0::2, 11] = -x

        A[1::2, 4] = X
        A[1::2, 5] = Y
        A[1::2, 6] = Z
        A[1::2, 7] = 1
        A[1::2, 8]  = -y * X
        A[1::2, 9]  = -y * Y
        A[1::2, 10] = -y * Z
        A[1::2, 11] = -y
    
    # solve
    U, E, V = np.linalg.svd(A, full_matrices=True)
    
    H = V[-1, :].reshape([3, ndims+1])
    
    # un-normalize DLT matrix
    H = np.matmul(
        np.matmul(
            np.linalg.inv(img_norm_mat),
            H
        ),
        lab_norm_mat
    )
    
    H = H / H[-1,-1]
    
    # compute RMSE error
    xy2 = np.dot(
        H, 
        np.concatenate((object_points, np.ones((1, object_points.shape[1]))))
    )
    
    res = xy2 / xy2[2, :]
    res = res[:2, :]
    
    error = res - image_points
    
    RMSE = np.sqrt(
        np.mean(
            np.sum(
                np.square(error),
                axis=0
            )
        )
    )
    
    return H, RMSE