import numpy as np
from scipy.optimize import minimize

from ..calib_utils import get_reprojection_error, get_los_error
from .. import _cal_doc_utils


@_cal_doc_utils.docfiller
def _minimize_params(
    self,
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
    None
    
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
    >>> from importlib_resources import files
    >>> from openpiv.calibration import pinhole_model, calib_utils
    
    >> path_to_calib = files('openpiv.data').joinpath('test7/D_Cal.csv')

    >>> obj_x, obj_y, obj_z, img_x, img_y = np.loadtxt(
        path_to_calib,
        unpack=True,
        skiprows=1,
        usecols=range(5),
        delimiter=','
    )
    
    >>> cam = pinhole_model.camera(
        'cam1', 
        [4512, 800],
        translation = [-340, 125, 554],
        orientation = [0., 0, np.pi],
        focal = [15310, 15310],
    )
    
     >>> cam.minimize_params(
            [obj_x, obj_y, obj_z],
            [img_x, img_y],
            correct_focal = True,
            correct_distortion = False,
            iterations=5
        )
    
    >>> cam.minimize_params(
            [obj_x, obj_y, obj_z],
            [img_x, img_y],
            correct_focal = True,
            correct_distortion = True,
            iterations=5
        )
        
    >> calib_utils.get_reprojection_error(
        cam,
        [obj_x, obj_y, obj_z],
        [img_x0, img_y0]
    )
    
    >> 0.041367701972229325

    """
    self._check_parameters()
    
    dtype = self.dtype
    
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
        self.translation = x[0:3]
        self.orientation = x[3:6]
        self.principal = x[6:8]
        
        if correct_focal == True:
            self.focal = x[8:10]
        
        self._get_rotation_matrix()
        
        if correct_distortion == True:
            if self.distortion_model.lower() == "brown":
                self.distortion1 = x[10:18]
            else:
                self.distortion2[0, :] = x[18:24]
                self.distortion2[1, :] = x[24:30]
                
        RMS_error = get_reprojection_error(
            self,
            object_points,
            image_points
        )
        
        return RMS_error
    
    # Create a numpy array since we cannot pass a dictionary to scipy's minimize function.
    params_to_minimize = [
        self.translation,
        self.orientation,
        self.principal,
        self.focal,
        self.distortion1,
        self.distortion2.ravel()
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
        if self.distortion_model.lower() == "polynomial": 
            # Since I couldn't get an inverse model to work using the linearization
            # of the error terms via Taylor Series expansion, so I decided to explicitly
            # compute it like in the polynomial camera model.
            obj_img_points = self.project_points(
                object_points,
                correct_distortion=False
            )
            
            x1, y1 = self._normalize_image_points(
                image_points
            )
            
            x2, y2 = self._normalize_image_points(
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
            
#            self.distortion2[0, :] = coeff1[0, :]
#            self.distortion2[1, :] = coeff1[1, :]
            self.distortion2[2, :] = coeff2[0, :]
            self.distortion2[3, :] = coeff2[1, :]