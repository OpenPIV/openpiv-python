import numpy as np
from scipy.optimize import curve_fit, least_squares

from ._normalization import _standardize_points_2d, _standardize_points_3d
from .._calib_utils import homogenize, get_rmse
from .. import _cal_doc_utils


__all__ = [
    'calibrate_dlt'
]


@_cal_doc_utils.docfiller
def calibrate_dlt(
    object_points: np.ndarray,
    image_points: np.ndarray,
    enforce_coplanar: bool=False,
    nonlinear: bool=False
):
    """Normalized DLT with optional nonlinear refinement.

    Compute a 3x3 planar homography from 2D lab coordinates or a 3x4 camera
    projection matrix from 3D lab coordinates using normalized direct linear
    transformation. When enabled, nonlinear refinement minimizes image reprojection error
    using Levenberg-Marquardt before denormalization. 

    Parameters
    ----------
    %(object_points)s
    %(image_points)s
    enforce_coplanar : bool, optional
        If True, require the standard deviation of supplied Z coordinates
        to be no greater than 1e-5 in lab-coordinate units.
        Default is False.
    nonlinear : bool, optional
        If True, refine the initial DLT matrix by minimizing image
        reprojection error. If False, return the normalized linear DLT
        solution after denormalization. Default is False.

    Returns
    -------
    H : 2D np.ndarray
        A 3x3 planar homography or 3x4 camera projection matrix mapping
        lab coordinates to homogeneous image coordinates.
    error : float
        RMSE of the image reprojection residuals for the returned matrix.

    Raises
    ------
    ValueError
        If the input shapes are invalid, correspondence counts differ,
        inputs contain non-finite values, or there are insufficient points.
    ValueError
        If Z coordinates fail the constant-Z check when enforce_coplanar
        is True.
    ValueError
        If normalization fails, the DLT configuration is degenerate, or
        the fitted matrix is non-finite or rank deficient.
    ValueError
        If the fitted matrix produces non-finite homogeneous image
        coordinates when computing RMSE.
    RuntimeError
        If nonlinear refinement fails to converge.

    Notes
    -----
    Linear DLT requires at least four planar correspondences or six
    noncoplanar 3D correspondences. The existing planar nonlinear refinement
    requires at least five correspondences because it optimizes all nine
    homography coefficients.

    """
    object_points = np.array(object_points, dtype=np.float64)
    image_points = np.array(image_points, dtype=np.float64)

    if object_points.ndim != 2:
        raise ValueError('Object points must be a two-dimensional array')
    if image_points.ndim != 2 or image_points.shape[0] != 2:
        raise ValueError('Image points must have shape (2, N)')

    ndims = object_points.shape[0]

    if ndims not in (2, 3):
        raise ValueError('Object points must have shape (2, N) or (3, N)')
    if object_points.shape[1] != image_points.shape[1]:
        raise ValueError('Object point image point size mismatch')
    
    if not (
        np.all(np.isfinite(object_points))
        and np.all(np.isfinite(image_points))
    ):
        raise ValueError('Calibration points must contain only finite values')

    min_points = 4

    if ndims == 3 and enforce_coplanar != True:
        min_points += 2

    if object_points.shape[1] < min_points:
        raise ValueError(
            f'Too little points to calibrate. Need at least {min_points} points'
        )

    if enforce_coplanar == True:
        if ndims == 3:
            z = object_points[2]

            if np.std(z) > 1e-5:
                raise ValueError(
                    'Object points must be co-planar'
                )
            
            ndims = 2
            object_points = object_points[:2, :]
    
    if ndims == 2:
        X_raw, Y_raw = object_points
        x_raw, y_raw = image_points
        
        # normalize for better dlt results
        [X, Y], obj_norm_mat = _standardize_points_2d(X_raw, Y_raw)
        [x, y], img_norm_mat = _standardize_points_2d(x_raw, y_raw)
            
        # mount constraints
        A = np.zeros([x.shape[0] * 2, 9], dtype=np.float64)
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
        [X, Y, Z], obj_norm_mat = _standardize_points_3d(X_raw, Y_raw, Z_raw)
        [x, y], img_norm_mat = _standardize_points_2d(x_raw, y_raw)

        # mount constraints
        A = np.zeros([x.shape[0] * 2, 12], dtype=np.float64)
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
    
    # Solve Ah = 0. full_matrices=True is required for the 8x9 matrix
    # produced by the minimal four-point planar case. The null-space vector
    # is the final row of the full Vh matrix and has our linear estimate.
    _, singular_values, vh = np.linalg.svd(A, full_matrices=True)

    rank_tolerance = (
        singular_values[0]
        * max(A.shape)
        * np.finfo(A.dtype).eps
    )
    rank = np.count_nonzero(singular_values > rank_tolerance)

    if rank < A.shape[1] - 1:
        raise ValueError(
            'Calibration points form a degenerate DLT configuration'
        )

    H = vh[-1, :]

    coefficient_tolerance = (
        np.finfo(H.dtype).eps * max(A.shape) * np.linalg.norm(H)
    )
    homogeneous_scale = H[-1]

    if abs(homogeneous_scale) <= coefficient_tolerance:
        homogeneous_scale = np.linalg.norm(H)
    if abs(homogeneous_scale) <= np.finfo(H.dtype).tiny:
        raise ValueError('Unable to determine finite DLT coefficients')

    H /= homogeneous_scale
    H = H.reshape([3, ndims+1])

    if nonlinear:
        if ndims == 2:
            H = _refine2D(
                H,
                np.array([X, Y]),
                np.array([x, y])
            )
        else:
            H = _refine3D(
                H,
                np.array([X, Y, Z]),
                np.array([x, y])
            )

    # denormalize DLT matrix
    H = np.dot(
        np.linalg.solve(img_norm_mat, H),
        obj_norm_mat
    )

    if not np.all(np.isfinite(H)) or np.linalg.matrix_rank(H) < 3:
        raise ValueError('The fitted DLT matrix is rank deficient')
        
    # compute RMSE error
    homogeneous_object_points = homogenize(object_points)
    xy2 = np.dot(H, homogeneous_object_points)
    
    if not np.all(np.isfinite(xy2)):
        raise ValueError('The fitted DLT produced non-finite coordinates')

    denominator = xy2[2, :]

    res = xy2[:2, :] / denominator
    
    error = res - image_points
    
    RMSE = get_rmse(error)
    
    return H, RMSE


@_cal_doc_utils.docfiller
def _minimize_params(
    self,
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
    %(object_points)s
    %(image_points)s
    enforce_coplanar : bool
        If a Z plane is supplied in the object points, check whether or not
        the Z planes are co-planar.
        
    Examples
    --------
    >>> import numpy as np
    >>> from importlib_resources import files
    >>> from openpiv.calibration import dlt_model, calib_utils

    >>> path_to_calib = files('openpiv.data').joinpath('test7/D_Cal.csv')

    >>> obj_x, obj_y, obj_z, img_x, img_y = np.loadtxt(
        path_to_calib,
        unpack=True,
        skiprows=1,
        usecols=range(5),
        delimiter=','
    )

    >>> obj_points = np.array([obj_x[0:3], obj_y[0:3], obj_z[0:3]], dtype=np.float64)
    >>> img_points = np.array([img_x[0:3], img_y[0:3]], dtype=np.float64)

    >>> cam = dlt_model.camera(
        'cam1', 
        [4512, 800]
    )

    >>> cam.minimize_params(
        [obj_x, obj_y, obj_z],
        [img_x, img_y]
    )

    >>> calib_utils.get_reprojection_error(
        cam,
        [obj_x, obj_y, obj_z],
        [img_x, img_y]
    )
    2.6181007833551034e-07
    
    """
    
    self._check_parameters()
    
    dtype = self.dtype
    
    object_points = np.array(object_points, dtype=dtype)
    image_points = np.array(image_points, dtype=dtype)

    H, error = calibrate_dlt(
        object_points,
        image_points,
        enforce_coplanar
    )
    
    self.coeffs = H


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
    
    jac = np.zeros((N, 9), dtype=np.float64)

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


def _unpack3D(
    parameters: np.ndarray,
    gauge_index: int
):
    coefficients = np.empty(12, dtype=np.float64)

    free_parameters = np.ones(12, dtype=bool)
    free_parameters[gauge_index] = False

    coefficients[free_parameters] = parameters
    coefficients[gauge_index] = 1.0

    return coefficients.reshape((3, 4))


def _refine_func3D(
    parameters: np.ndarray,
    homogeneous_object_points: np.ndarray,
    image_points: np.ndarray,
    gauge_index: int
):
    H = _unpack3D(parameters, gauge_index)

    projected_points = np.dot(
        H,
        homogeneous_object_points
    )

    predicted_image_points = (
        projected_points[:2, :] / projected_points[2, :]
    )

    # Interleaved residual order:
    # [x0, y0, x1, y1, ...]
    residuals = predicted_image_points - image_points

    return residuals.T.ravel()


def _refine3D(
    H: np.ndarray,
    object_points: np.ndarray,
    image_points: np.ndarray
):
    initial_coefficients = H.ravel().copy()

    # Select a stable local projective gauge.
    gauge_index = int(
        np.argmax(np.abs(initial_coefficients))
    )

    initial_coefficients /= initial_coefficients[gauge_index]

    initial_parameters = np.delete(
        initial_coefficients,
        gauge_index
    )

    result = least_squares(
        _refine_func3D,
        x0=initial_parameters,
        jac='2-point',
        method='lm',
        x_scale='jac',
        args=(
            homogenize(object_points),
            image_points,
            gauge_index
        )
    )

    if not result.success:
        raise RuntimeError(
            f'3D DLT refinement failed: {result.message}'
        )

    return _unpack3D(
        result.x,
        gauge_index
    )