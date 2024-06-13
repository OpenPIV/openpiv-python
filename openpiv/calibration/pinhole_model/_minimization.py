import numpy as np
from scipy.optimize import minimize

from ._check_params import _check_parameters
from ._projection import project_points, _normalize_image_points
from ._utils import get_rotation_matrix
from ..calib_utils import get_reprojection_error, get_los_error
from .. import _cal_doc_utils


__all__ = [
    "minimize_params"
]


@_cal_doc_utils.docfiller
def minimize_params(
    cam_struct: dict,
    object_points: list,
    image_points: list,
    correct_focal: bool = False,
    correct_distortion: bool = False,
    max_iter: int = 1000,
    iterations: int = 3
):
    """Minimize camera parameters.
    
    Minimize camera parameters using BFGS optimization. To do this, the
    root mean square error (RMS error) is calculated for each iteration.
    The set of parameters with the lowest RMS error is returned (which is
    hopefully correct the minimum).
    
    Parameters
    ----------
    %(cam_struct)s
    %(object_points)s
    %(image_points)s
    correct_focal : bool
        If true, minimize the focal point.
    correct_distortion : bool
        If true, minimize the distortion model.
    max_iter : int
        Maximum amount of iterations in Nelder-Mead minimization.
    iterations : int
        Number of iterations to perform.
        
    Returns
    -------
    %(cam_struct)s
    
    Notes
    -----
    When minimizing the camera parameters, it is important that the
    parameters are estimated first before distortion correction. This allows
    a better estimation of the camera parameters and distortion coefficients.
    For instance, if one were to calibrate the camera intrinsic and distortion
    coefficients before moving the camera to the lab apparatus, it would be
    important to calibrate the camera parameters before the distortion model
    to ensure a better convergence and thus, lower root mean square (RMS) errors.
    This can be done in the following procedure:
    
    1. Place the camera directly in front of a planar calibration plate.
    
    2. Estimate camera parameters with out distortion correction.
    
    3. Estimate distortion model coefficients and refine camera parameters.
        Note: It may be best to use least squares minimization first.
    
    4. Place camera in lab apparatus.
    
    5. Calibrate camera again without distortion or intrinsic correction.
    
    A brief example is shown below. More can be found in the example PIV lab
    experiments.
    
    On a side note, for a decent calibration to occur, at least 20 points are
    needed. For attaining a rough estimate for marker detection purposes, at
    least 9 points are needed (of course, this is excluding distortion correction).
    
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv.calibration import calib_utils, pinhole_model
    >>> from openpiv.data.test5 import cal_points
    
    >>> obj_x, obj_y, obj_z, img_x, img_y, img_size_x, img_size_y = cal_points()
    
    >>> camera_parameters = pinhole_model.get_cam_params(
            name="cam1", 
            [img_size_x, img_size_y]
        )
    
     >>> camera_parameters = pinhole_model.minimize_params(
            camera_parameters, 
            [obj_x, obj_y, obj_z],
            [img_x, img_y],
            correct_focal = True,
            correct_distortion = False,
            iterations=5
        )
    
    >>> camera_parameters = pinhole_model.minimize_params(
            camera_parameters, 
            [obj_x, obj_y, obj_z],
            [img_x, img_y],
            correct_focal = True,
            correct_distortion = True,
            iterations=5
        )
    
    >>> calib_utils.get_reprojection_error(
            camera_parameters, 
            pinhole_model.project_points,
            [obj_x, obj_y, obj_z],
            [img_x, img_y]
        )
    
    """
    _check_parameters(cam_struct)
    
    dtype = cam_struct["dtype"]
    
    object_points = np.array(object_points, dtype=dtype)
    image_points = np.array(image_points, dtype=dtype)
    
    if object_points.shape[1] < 9:
        raise ValueError(
            "Too little points to calibrate"
        )
    
    if object_points.shape[1] != image_points.shape[1]:
        raise ValueError(
            "Object point image point size mismatch"
        )
    
    # For each iteration, calculate the RMS error of this function. The input is a numpy
    # array to meet the requirements of scipy's minimization functions.
    def func_to_minimize(x):
        cam_struct["translation"] = x[0:3]
        cam_struct["orientation"] = x[3:6]
        cam_struct["principal"] = x[6:8]
        
        if correct_focal == True:
            cam_struct["focal"] = x[8:10]
        
        cam_struct["rotation"] = get_rotation_matrix(
            cam_struct
        )
        
        if correct_distortion == True:
            if cam_struct["distortion_model"].lower() == "brown":
                cam_struct["distortion1"] = x[10:18]
            else:
                cam_struct["distortion2"][0, :] = x[18:24]
                cam_struct["distortion2"][1, :] = x[24:30]
                
        RMS_error = get_reprojection_error(
            cam_struct,
            project_points,
            object_points,
            image_points
        )
        
        return RMS_error
    
    # Create a numpy array since we cannot pass a dictionary to scipy's minimize function.
    params_to_minimize = [
        cam_struct["translation"],
        cam_struct["orientation"],
        cam_struct["principal"],
        cam_struct["focal"],
        cam_struct["distortion1"],
        cam_struct["distortion2"].ravel()
    ]
    
    params_to_minimize = np.hstack(
        params_to_minimize
    )
    
    # Peform multiple iterations to hopefully attain a better calibration.
    for _ in range(iterations):
        # Discard output of minimization as we are interested in the camera params dict.
        res = minimize(
            func_to_minimize,
            params_to_minimize,
            method="bfgs",
            options={"maxiter": max_iter},
            jac = "2-point"
        )
    
    if correct_distortion == True:
        if cam_struct["distortion_model"].lower() == "polynomial": 
            # Since I couldn't get an inverse model to work using the linearization
            # of the error terms via Taylor Series expansion, so I decided to explicitly
            # compute it like in the polynomial camera model.
            obj_img_points = project_points(
                cam_struct,
                object_points,
                correct_distortion=False
            )
            
            x1, y1 = _normalize_image_points(
                cam_struct,
                image_points
            )
            
            x2, y2 = _normalize_image_points(
                cam_struct,
                obj_img_points
            )
            
            # create polynomials
            poly1 = np.array([np.ones_like(x1), x1, y1, x1**2, y1**2, x1*y1])
#            poly2 = np.array([np.ones_like(x2), x2, y2, x2**2, y2**2, x2*y2])
            
            # minimize the polynomials
#            coeff1 = np.linalg.lstsq(
#                poly2.T,
#                np.array([x1, y1], dtype="float64").T, 
#               rcond=None
#            )[0].T
            
            coeff2 = np.linalg.lstsq(
                poly1.T,
                np.array([x2, y2], dtype=dtype).T, 
                rcond=None
            )[0].T
            
#            cam_struct["distortion2"][0, :] = coeff1[0, :]
#            cam_struct["distortion2"][1, :] = coeff1[1, :]
            cam_struct["distortion2"][2, :] = coeff2[0, :]
            cam_struct["distortion2"][3, :] = coeff2[1, :]
    
    return cam_struct