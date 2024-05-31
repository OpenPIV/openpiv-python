import numpy as np

from ._check_params import _check_parameters
from .._doc_utils import (docstring_decorator,
                          doc_obj_coords, doc_cam_struct)


__all__ = [
    "project_points"
]


@docstring_decorator(doc_cam_struct, doc_obj_coords)
def project_points(
    cam_struct: dict,
    object_points: np.ndarray
):
    """Project lab coordinates to image points.
        
    Parameters
    ----------
    cam_struct : dict
        {0}
    object_points : 2D np.ndarray
        {1}
        
    Returns
    -------
    x : 1D np.ndarray
        Projected image x-coordinates.
    y : 1D np.ndarray
        Projected image y-coordinates.
        
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv import calib_utils, dlt_model
    >>> from openpiv.data.test5 import cal_points
    
    >>> obj_x, obj_y, obj_z, img_x, img_y, img_size_x, img_size_y = cal_points()
    
    >>> obj_points = np.array([obj_x[0:2], obj_y[0:2], obj_z[0:2]], dtype="float64")
    >>> img_points = np.array([img_x[0:2], img_y[0:2]], dtype="float64")
    
    >>> camera_parameters = dlt_model.get_cam_params(
            name="cam1", 
            [img_size_x, img_size_y]
        )
    
    >>> camera_parameters = dlt_model.minimize_params(
            camera_parameters,
            [obj_x, obj_y, obj_z],
            [img_x, img_y]
        )
        
    >>> dlt_model.project_points(
            camera_parameters,
            obj_points
        )
        
    >>> img_points
    
    """ 
    _check_parameters(cam_struct)
    
    object_points = np.array(object_points, dtype=dtype)
    
    dtype = cam_struct["dtype"]
    
    H = cam_struct["coefficients"]
    
    ndim = H.shape[1] - 1
    
    if ndim == 2:        
        object_points = object_points[:2, :]
    else:
        object_points = object_points[:3, :]
        
    # compute RMSE error
    xy = np.dot(
        H, 
        np.concatenate([
            object_points, 
            np.ones((1, object_points.shape[1]))
        ])
    )
    
    img_points = xy / xy[2, :]
    img_points = res[:2, :]
    
    return img_points.astype(dtype, copy=False)