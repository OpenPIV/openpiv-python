import numpy as np
from scipy.optimize import curve_fit

from ._check_params import _check_parameters
from ._normalization import _standardize_points_2d, _standardize_points_3d
from .._calib_utils import homogenize, get_rmse
from .. import _cal_doc_utils


__all__ = [
    "calibrate_dlt",
    "minimize_params"
]


@_cal_doc_utils.docfiller
def calibrate_dlt(
    object_points: np.ndarray,
    image_points: np.ndarray,
    enforce_coplanar: bool=False
):
    """Coplanar DLT for homography.
    
    Compute a homography matrix using direct linear transformation. For 2D
    lab coordinates, a 2D DLT is performed. For lab coordinates that include
    a Z-axis, a 3D DLT is performed. For 3D DLTs, an option to error if the
    Z-axis is not co-planar is available.
    
    Parameters
    ----------
    %(object_points)s
    %(image_points)s
    enforce_coplanar : bool
        If a Z plane is supplied in the object points, check whether or not the Z
        planes are co-planar.
        
    Returns
    -------
    H : 2D np.ndarray
        A 3x3 matrix containing a homography fit for the object and image points.
    error : float
        The RMSE error of the DLT fit.
    
    Raises
    ------
    ValueError
        If the object coordinates contain non-planar z-coordinates and
        enforce_coplanar is enabled.
    ValueError
        If there are not enough points to calculate the DLT coefficients.
    
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
        A[0::2, 0] = -X
        A[0::2, 1] = -Y
        A[0::2, 2] = -1
        A[0::2, 6] = x * X
        A[0::2, 7] = x * Y
        A[0::2, 8] = x

        A[1::2, 3] = -X
        A[1::2, 4] = -Y
        A[1::2, 5] = -1
        A[1::2, 6] = y * X
        A[1::2, 7] = y * Y
        A[1::2, 8] = y
        
    else:
        X_raw, Y_raw, Z_raw = object_points
        x_raw, y_raw = image_points

        # normalize for better dlt results
        [X, Y, Z], lab_norm_mat = _standardize_points_3d(X_raw, Y_raw, Z_raw)
        [x, y], img_norm_mat = _standardize_points_2d(x_raw, y_raw)

        # mount constraints
        A = np.zeros([x.shape[0] * 2, 12], dtype="float64")
        A[0::2, 0] = -X
        A[0::2, 1] = -Y
        A[0::2, 2] = -Z
        A[0::2, 3] = -1
        A[0::2, 8]  = x * X
        A[0::2, 9]  = x * Y
        A[0::2, 10] = x * Z
        A[0::2, 11] = x

        A[1::2, 4] = -X
        A[1::2, 5] = -Y
        A[1::2, 6] = -Z
        A[1::2, 7] = -1
        A[1::2, 8]  = y * X
        A[1::2, 9]  = y * Y
        A[1::2, 10] = y * Z
        A[1::2, 11] = y
    
    # solve
    U, E, V = np.linalg.svd(A, full_matrices=True)
    
    H = V[-1, :]
    H /= H[-1]
    H = H.reshape([3, ndims+1])
    
    # denormalize DLT matrix
    H = np.matmul(
        np.matmul(
            np.linalg.inv(img_norm_mat),
            H
        ),
        lab_norm_mat
    )
        
    # compute RMSE error
    xy2 = np.dot(
        H,
        homogenize(object_points)
    )
    
    res = xy2 / xy2[2, :]
    res = res[:2, :]
    
    error = res - image_points
    
    RMSE = get_rmse(error)
    
    return H, RMSE


#@_cal_doc_utils.docfiller
#def calibrate_dlt_robust(
#    object_points: np.ndarray,
#    image_points: np.ndarray,
#    enforce_coplanar: bool=False
#):
#    """Coplanar DLT for homography.
#    
#    Compute a homography matrix using direct linear transformation. For 2D
#    lab coordinates, a 2D DLT is performed. For lab coordinates that include
#    a Z-axis, a 3D DLT is performed. For 3D DLTs, an option to error if the
#    Z-axis is not co-planar is available.
#    
#    Parameters
#    ----------
#    %(object_points)s
#    %(image_points)s
#    enforce_coplanar : bool
#        If a Z plane is supplied in the object points, check whether or not the Z
#        planes are co-planar.
#        
#    Returns
#    -------
#    H : 2D np.ndarray
#        A 3x3 matrix containing a homography fit for the object and image points.
#    error : float
#        The RMSE error of the DLT fit.
#
#    Raises
#    ------
#    ValueError
#        If the object coordinates contain non-planar z-coordinates and
#        enforce_coplanar is enabled.
#    ValueError
#        If there are not enough points to calculate the DLT coefficients.
#    
#    """
#    object_points = np.array(object_points, dtype="float64")
#    image_points = np.array(image_points, dtype="float64")
#    
#    ndims = object_points.shape[0]
#    
#    min_points = 10
#    
#    if ndims == 3 and  enforce_coplanar != True:
#        min_points += 8
#    
#    if object_points.shape[1] < min_points:
#        raise ValueError(
#            f"Too little points to calibrate. Need at least {min_points} points"
#        )
#    
#    if object_points.shape[1] != image_points.shape[1]:
#        raise ValueError(
#            "Object point image point size mismatch"
#        )
#
#    if ndims not in [2, 3]:
#        raise ValueError(
#            "Object points must be in either [X, Y] (shape = [2, N]) or [X, Y, Z] "+
#            "format (shape = [3, N]). Recieved shape = [{}, N]".format(ndims)
#        )
#    
#    if enforce_coplanar == True:
#        if ndims == 3:
#            if np.std(object_points[3]) > 0.00001:
#                raise ValueError(
#                    "Object points must be co-planar"
#                )
#            ndims = 2
#            object_points = object_points[:2, :]
#    
#    if ndims == 2:
#        X_raw, Y_raw = object_points
#        x_raw, y_raw = image_points
#        
#        # normalize for better dlt results
#        [X, Y], lab_norm_mat = _standardize_points_2d(X_raw, Y_raw)
#        [x, y], img_norm_mat = _standardize_points_2d(x_raw, y_raw)
#            
#        # mount constraints
#        A = np.zeros([x.shape[0] * 2, 21], dtype="float64")
#        r2 = X*Z + Y*Y
#        r4 = r2 * r2
#        r6 = r4 * r2
#        
#        A[0::2, 0] = -X
#        A[0::2, 1] = -Y
#        A[0::2, 2] = -1
#        
#        A[0::2, 3] = -X * r2
#        A[0::2, 4] = -Y * r2
#        A[0::2, 5] = -1 * r2
#        
#        A[0::2, 6] = -X * r4
#        A[0::2, 7] = -Y * r4
#        A[0::2, 8] = -1 * r4
#        
#        A[0::2, 18] = x * X
#        A[0::2, 19] = x * Y
#        A[0::2, 20] = x
#
#        A[1::2, 9] =  -X
#        A[1::2, 10] = -Y
#        A[1::2, 11] = -1
#        
#        A[1::2, 12] = -X * r2
#        A[1::2, 13] = -Y * r2
#        A[1::2, 14] = -1 * r2
#        
#        A[1::2, 15] = -X * r4
#        A[1::2, 16] = -Y * r4
#        A[1::2, 17] = -1 * r4
#        
#        A[1::2, 18] = y * X
#        A[1::2, 19] = y * Y
#        A[1::2, 20] = y
#        
#    else:
#        raise ValueError(
#            "3D robust DLT is not supported"
#        )
#    
#    # solve
#    U, E, V = np.linalg.svd(A, full_matrices=True)
#    
#    H = V[-1, :]
#    H /= H[-1]
#    H = H.reshape([3, 7])
#    
#    # denormalize DLT matrix
#    H = np.matmul(
#        np.matmul(
#            np.linalg.inv(img_norm_mat),
#            H
#        ),
#        lab_norm_mat
#    )
#    
#    # compute RMSE error
#    xy2 = np.dot(
#        H, 
#        homogenize(object_points)
#    )
#    
#    res = xy2 / xy2[2, :]
#    res = res[:2, :]
#    
#    error = res - image_points
#    
#    RMSE = get_rmse(error)
#    
#    return H, RMSE


@_cal_doc_utils.docfiller
def minimize_params(
    cam_struct: dict,
    object_points: np.ndarray,
    image_points: np.ndarray,
    enforce_coplanar: bool=False
):
    """Least squares wrapper for DLT calibration.
    
    A wrapper around the function 'calibrate_dlt' for use with camera structures.
    In the future, a robust DLT calibration would be implemented and its
    interface would be linked here too.
    
    Parameters
    ----------
    %(cam_struct)s
    %(object_points)s
    %(image_points)s
    enforce_coplanar : bool
        If a Z plane is supplied in the object points, check whether or not
        the Z planes are co-planar.
        
    Returns
    -------
    %(cam_struct)s
        
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv.calibration import calib_utils, dlt_model
    >>> from openpiv.data.test5 import cal_points
    
    >>> obj_x, obj_y, obj_z, img_x, img_y, img_size_x, img_size_y = cal_points()
    
    >>> camera_parameters = dlt_model.get_cam_params(
            name="cam1", 
            [img_size_x, img_size_y]
        )
    
    >>> camera_parameters = dlt_model.minimize_params(
            camera_parameters,
            [obj_x, obj_y, obj_z],
            [img_x, img_y]
        )
        
    >>> calib_utils.get_reprojection_error(
            camera_parameters, 
            dlt_model.project_points,
            [obj_x, obj_y, obj_z],
            [img_x, img_y]
        )
        
    """
    
    _check_parameters(cam_struct)
    
    dtype = cam_struct["dtype"]
    
    object_points = np.array(object_points, dtype=dtype)
    image_points = np.array(image_points, dtype=dtype)

    H, error = calibrate_dlt(
        object_points,
        image_points,
        enforce_coplanar
    )
    
    cam_struct["coefficients"] = H
    
    return cam_struct


def _refine_func2D(
    data,
    *H
):
    h11, h12, h13, h21, h22, h23, h31, h32, h33 = H

    X = data[0::2]
    Y = data[1::2]

    x = (h11 * X + h12 * Y + h13) / (h31 * X + h32 * Y + h33)
    y = (h21 * X + h22 * Y + h23) / (h31 * X + h32 * Y + h33)

    res = np.zeros_like(data)
    
    res[0::2] = x
    res[1::2] = y

    return res


def _refine_jac2D(
    data,
    *H
):
    h11, h12, h13, h21, h22, h23, h31, h32, h33 = H

    X = data[0::2]
    Y = data[1::2]

    N = data.shape[0]
    
    jac = np.zeros((N, 9), dtype="float64")

    s_x = h11 * X + h12 * Y + h13
    s_y = h21 * X + h22 * Y + h23
    w   = h31 * X + h32 * Y + h33
    w_sq = w**2

    jac[0::2, 0] = X / w
    jac[0::2, 1] = Y / w
    jac[0::2, 2] = 1. / w
    jac[0::2, 6] = (-s_x * X) / w_sq
    jac[0::2, 7] = (-s_x * Y) / w_sq
    jac[0::2, 8] = -s_x / w_sq

    jac[1::2, 3] = X / w
    jac[1::2, 4] = Y / w
    jac[1::2, 5] = 1. / w
    jac[1::2, 6] = (-s_y * X) / w_sq
    jac[1::2, 7] = (-s_y * Y) / w_sq
    jac[1::2, 8] = -s_y / w_sq
    
    return jac


def _refine2D(
    H,
    object_points,
    image_points
):
    
    X, Y = object_points
    x, y = image_points

    N = X.shape[0]

    init_guess = H.ravel()

    independent = np.zeros(N * 2)
    independent[0::2] = X
    independent[1::2] = Y

    dependent = np.zeros(N * 2)
    dependent[0::2] = x
    dependent[1::2] = y

    # curve_fit uses damped Newton-Raphson optimization (aka LM) by default
    new_h, _ = curve_fit(
        _refine_func2D, 
        independent, 
        dependent, 
        p0=init_guess, 
        jac=_refine_jac2D
    )
    
    new_h /= new_h[-1]
    new_h = new_h.reshape((3,3))

    return new_h