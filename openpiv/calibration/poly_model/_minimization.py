import numpy as np
from scipy.optimize import least_squares

from ._check_params import _check_parameters
# from ..poly_model import calibrate_dlt
from openpiv.calibration import dlt_model as calib_dlt
from .. import _cal_doc_utils


__all__ = [
    "minimize_params"
]


def _refine_poly(
    poly_coeffs: np.ndarray,
    polynomial: np.ndarray,
    expected: np.ndarray
):     
    # use lambdas since it keeps the function definitions local
    def refine_func(coeffs):
        projected = np.dot(polynomial, coeffs)
        return projected - expected

    return least_squares(
            refine_func, 
            poly_coeffs,
            method="trf" # more modern lm algorithm
        ).x


@_cal_doc_utils.docfiller
def minimize_params(
    cam_struct: dict,
    object_points: list,
    image_points: list,
):
    """Minimize polynomials.
    
    Minimize polynomials using Least Squares minimization.
    
    Parameters
    ----------
    %(cam_struct)s
    %(object_points)s
    %(image_points)s

        
    Returns
    -------
    %(cam_struct)s
        
    Examples
    --------
    >>> import numpy as np
    >>> from importlib_resources import files
    >>> from openpiv.calibration import poly_model, calib_utils

    >>> path_to_calib = files('openpiv.data').joinpath('test7/D_Cal.csv')

    >>> obj_x, obj_y, obj_z, img_x, img_y = np.loadtxt(
        path_to_calib,
        unpack=True,
        skiprows=1,
        usecols=range(5),
        delimiter=','
    )

    >>> obj_points = np.array([obj_x[0:3], obj_y[0:3], obj_z[0:3]], dtype="float64")
    >>> img_points = np.array([img_x[0:3], img_y[0:3]], dtype="float64")

    >>> cam_params = poly_model.get_cam_params(
        'cam1', 
        [4512, 800]
    )

    >>> cam_params = poly_model.minimize_params(
        cam_params,
        [obj_x, obj_y, obj_z],
        [img_x, img_y]
    )

    >>> calib_utils.get_reprojection_error(
        cam_params, 
        poly_model.project_points,
        [obj_x, obj_y, obj_z],
        [img_x, img_y]
    )
    0.16553632335727653
        
    """
    _check_parameters(cam_struct)
    
    dtype = cam_struct["dtype"]
    
    object_points = np.array(object_points, dtype=dtype)
    image_points = np.array(image_points, dtype=dtype)
    
    if object_points.shape[1] < 19:
        raise ValueError(
            "Too little points to calibrate"
        )
    
    if object_points.shape[1] != image_points.shape[1]:
        raise ValueError(
            "Object point image point size mismatch"
        )
        
    x = image_points[0]
    y = image_points[1]
    
    X = object_points[0]
    Y = object_points[1]
    Z = object_points[2]

    polynomial_wi = np.array(
        [
            np.ones_like(X),
            X,     Y,     Z, 
            X*Y,   X*Z,   Y*Z,
            X**2,  Y**2,  Z**2,
            X**3,  X*X*Y, X*X*Z,
            Y**3,  X*Y*Y, Y*Y*Z,
            X*Z*Z, Y*Z*Z, X*Y*Z
        ],
        dtype=dtype
    ).T
    
    # in the future, break this into three Z subvolumes to further reduce errors.
    polynomial_iw = np.array(
        [
            np.ones_like(x),
            x,     y,     Z, 
            x*y,   x*Z,   y*Z,
            x**2,  y**2,  Z**2,
            x**3,  x*x*y, x*x*Z,
            y**3,  x*y*y, y*y*Z,
            x*Z*Z, y*Z*Z, x*y*Z
        ],
        dtype=dtype
    ).T
    

    # world to image (forward projection)
    coeff_wi = np.zeros([19, 2], dtype=dtype)
    
    for i in range(2):
        coeff_wi[:, i] = np.linalg.lstsq(
            polynomial_wi,
            image_points[i], 
            rcond=None
        )[0]
        
        coeff_wi[:, i] = _refine_poly(
            coeff_wi[:, i],
            polynomial_wi,
            image_points[i]
        )
    
    # image to world (back projection)
    coeff_iw = np.zeros([19, 3], dtype=dtype)
    
    for i in range(3):
        coeff_iw[:, i] = np.linalg.lstsq(
            polynomial_iw,
            object_points[i], 
            rcond=None
        )[0]
        
        coeff_iw[:, i] = _refine_poly(
            coeff_iw[:, i],
            polynomial_iw,
            object_points[i]
        )
    
    # DLT estimator
    dlt_matrix, _residual = calibrate_dlt(
        object_points,
        image_points
    )

    cam_struct["poly_wi"] = coeff_wi
    cam_struct["poly_iw"] = coeff_iw
    cam_struct["dlt"] = dlt_matrix
    
    return cam_struct