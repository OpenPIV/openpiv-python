import numpy as np
from scipy.optimize import least_squares

from .. import _cal_doc_utils


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
def _minimize_params(
    self,
    object_points: list,
    image_points: list,
):
    """Minimize polynomials.
    
    Minimize polynomials using Least Squares minimization.
    
    Parameters
    ----------
    %(object_points)s
    %(image_points)s
        
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

    >>> cam = poly_model.camera(
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
    0.16553632335727653
        
    """
    self._check_parameters()
    
    dtype = self.dtype
    
    object_points = np.array(object_points, dtype=dtype)
    image_points = np.array(image_points, dtype=dtype)
    
    if object_points.shape[1] < 20:
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
    
    # X,Y are cubic in image coordinates and linear in world Z.
    polynomial_iw = np.array(
        [
            np.ones_like(x),
            x,       y,
            x*x,     x*y,     y*y,
            x*x*x,   x*x*y,   x*y*y,   y*y*y,
            Z,
            x*Z,     y*Z,
            x*x*Z,   x*y*Z,   y*y*Z,
            x*x*x*Z, x*x*y*Z, x*y*y*Z, y*y*y*Z
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
    
    # image to world (ray origins and direction slopes)
    coeff_iw = np.zeros([20, 2], dtype=dtype)

    # Scale columns for fitting; stored coefficients use raw image coordinates.
    scale = np.linalg.norm(polynomial_iw.astype("float64"), axis=0)
    scale[scale == 0] = 1.0
    polynomial_iw_scaled = polynomial_iw / scale

    for i in range(2):
        result = np.linalg.lstsq(
            polynomial_iw_scaled,
            object_points[i],
            rcond=None
        )
        if result[2] != 20:
            raise ValueError("Volume ray calibration requires full-rank volumetric points")
        coeff_iw[:, i] = result[0] / scale

    self.poly_wi = coeff_wi
    self.poly_iw_o = coeff_iw[:10, :]
    self.poly_iw_v = coeff_iw[10:, :]