import numpy as np
from typing import Tuple

from ._check_params import _check_parameters
from ._distortion import (_undistort_points_brown, _distort_points_brown,
                          _undistort_points_poly, _distort_points_poly)
from ._minimization import _minimize_params
from ._projection import (_normalize_world_points, _normalize_image_points,
                          _project_points, _project_to_z, _get_inverse_vector)
from ._utils import _get_rotation_matrix, _save_parameters, _load_parameters


__all__ = ["camera"]


class camera(object):
    """An instance of a DLT camera model.
    
    Create an instance of a pinhole camera. The pinhole camera model is 
    based on a physically constrained model where image points and object
    points are mapped through an extrinsic and intrinsic matrix with a
    distortion compensation model.
    
    Attributes
    ----------
    name : str
        Name of camera.
    resolution : tuple[int, int]
        Resolution of camera in x and y axes respectively.
    translation : 1D np.ndarray-like
        Location of camera origin/center in x, y, and z axes respectively.
    orientation : 1D np.ndarray-like
        Orientation of camera in x, y, z axes respectively.
    rotation : 2D np.ndarray
        Rotational camera parameter for camera system.
    distortion_model : str
        The type of distortion model to use.
        
        ``brown``
        The Brown model follows the distortion model incorporated by OpenCV.
        It consists of a radial and tangential model to compensate for
        distortion.
        
        ``polynomial``
        The polynomial model is used for general distortion compensation. It
        consists of a 2nd order polynomial in the x and y axes. This
        distortion model relies on the theory put forth by MyPTV and copies
        the model in whole.
        
        Both models do not attempt to correct distortions along the z-plane.
        
    distortion1 : 2D np.ndarray
       Radial and tangential distortion compensation matrix for a camera.
    distortion2 : 2D np.ndarray
       2nd order polynomial distortion compensation matrix for a camera.
    focal : tuple[float, float]
        Focal distance/magnification of camera-lens system for x any y axis
        respectively.
    principal : tuple[float, float]
        Principal point offset for x any y axis respectively.
    dtype : str
        The dtype used in the projections. All data is copied if the dtype is
        different. It is highly unadvisable to change this parameter.
        
    Methods
    -------
    minimize_params
    project_points
    project_to_z
    save_parameters
    load_parameters
    
    References
    ----------
    .. [1] Shnapp, R. (2022). MyPTV: A Python Package for 3D Particle
        Tracking. J. Open Source Softw., 7, 4398.
        
    .. [2] Čuljak, I., Abram, D., Pribanić, T., Džapo, H., & Cifrek, M.
        (2012). A brief introduction to OpenCV. 2012 Proceedings of the
        35th International Convention MIPRO, 1725-1730.
        
    """
    def __init__(
        self,
        name: str,
        resolution: Tuple[int, int],
        translation: np.ndarray=[0, 0, 1],
        orientation: np.ndarray=np.zeros(3, dtype="float64"),
        distortion_model: str="polynomial",
        distortion1: np.ndarray=np.zeros(8, dtype="float64"),
        distortion2: np.ndarray=np.zeros([2, 5], dtype="float64"),
        focal: Tuple[float, float]=[1.0, 1.0],
        principal: Tuple[float, float]=None,
        dtype: str="float64"
    ):
        translation = np.array(translation, dtype=dtype)
        orientation = np.array(orientation, dtype=dtype)
        distortion1 = np.array(distortion1, dtype=dtype)
        distortion2 = np.array(distortion2, dtype=dtype)
    
        self.name = name
        self.resolution = resolution
        self.translation = np.array(translation)
        self.orientation = np.array(orientation)
        self.distortion_model = distortion_model
        self.distortion1 = distortion1
        self.distortion2 = distortion2
        self.focal = focal
        self.dtype = dtype
        
        if principal is not None:
            self.principal = principal
        else:
            # temporary place holder
            self.principal = [0, 0]
        
        # method definitions
        camera._check_parameters = _check_parameters
        camera._get_inverse_vector = _get_inverse_vector
        camera._get_rotation_matrix = _get_rotation_matrix
        camera._normalize_world_points = _normalize_world_points
        camera._normalize_image_points = _normalize_image_points
        camera._undistort_points_brown = _undistort_points_brown
        camera._distort_points_brown = _distort_points_brown
        camera._undistort_points_poly = _undistort_points_poly
        camera._distort_points_poly = _distort_points_poly
        camera.minimize_params = _minimize_params
        camera.project_points = _project_points
        camera.project_to_z = _project_to_z
        camera.save_parameters = _save_parameters
        camera.load_parameters = _load_parameters
        
        # check parameters
        self._check_parameters()
        
        # fix principal point if necessary (placed here due to error checking)
        if principal is None:
            self.principal = [self.resolution[0] / 2, self.resolution[1] / 2]
            
        # get roation matrix
        self._get_rotation_matrix()