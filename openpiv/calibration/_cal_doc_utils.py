from scipy._lib import doccer 


__all__ = ['docfiller']


# typing the same doc string multiple times gets annoying and error prone...
_obj_coords = (
"""object_points : 2D np.ndarray
    Real world coordinates stored in a ndarray structured like [X, Y, Z]'.""")

_img_coords = (
"""image_points : 2D np.ndarray
    Image coordinates stored in a ndarray structured like [x, y].""")

_cam_struct = (
"""cam_struct : dict
    A dictionary structure of camera parameters.""")

_project_points_func = (
"""project_points_func : function
    Projection function with the following signature:
    res = func(cam_struct, object_points).""")

_project_to_z_func = (
"""project_to_z_func : function
    Projection function with the following signiture:
    res = func(cam_struct, image_points, Z).""")

_x_lab_coord = (
"""X : 1D np.ndarray
        Projected world x-coordinates.""")

_y_lab_coord = (
"""Y : 1D np.ndarray
        Projected world y-coordinates.""")

_z_lab_coord = (
"""Z : 1D np.ndarray
        Projected world z-coordinates.""")

_x_img_coord = (
"""x : 1D np.ndarray
        Projected image x-coordinates.""")

_y_img_coord = (
"""y : 1D np.ndarray
        Projected image y-coordinates.""")

_project_z = (
"""z : float
    A float specifying the Z (depth) value to project to.""")

docdict = {
    "object_points": _obj_coords,
    "image_points": _img_coords,
    "cam_struct": _cam_struct,
    "project_points_func": _project_points_func,
    "project_to_z_func": _project_to_z_func,
    "x_lab_coord": _x_lab_coord,
    "y_lab_coord": _y_lab_coord,
    "z_lab_coord": _z_lab_coord,
    "x_img_coord": _x_img_coord,
    "y_img_coord": _y_img_coord,
    "project_z": _project_z
}

# SciPy's nifty decorator (works better than my simple implementation)
docfiller = doccer.filldoc(docdict)