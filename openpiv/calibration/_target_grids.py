import numpy as np
from typing import Tuple


__all__ = [
    "get_simple_grid",
    "get_asymmetric_grid"
]


def get_simple_grid(
    nx: int,
    ny: int,
    z: int,
    x_ind: int,
    y_ind: int,
    flip_y: bool=True,
    spacing: float=1.0
):
    """Make a simple rectangular grid.
    
    Create a simple rectangular grid with the origin based on the index of
    the selected point. This allows an arbitrary size grid and location of
    origin.
    
    Parameters
    ----------
    nx : int
        Number of markers in the x-axis.
    ny : int
        Number of markers in the y-axis.
    z : float, optional
        The z plane where the calibration plate is located.
    x_ind : int
        Index of the point to define the x-axis.
    y_ind : int
        Index of the point to define the y-axis.
    flip_y : bool, optional
        Flip the signs of the y-axis. This is use enabled by default, but
        if the grid has an origin in the top-left corner, this should be
        disabled.
    spacing : float, optional
        Grid spacing in millimeters.
    
    Returns 
    -------
    object_points : 2D np.ndarray
        2D object points of [X, Y, Z] in world coordinates.
        
    Examples
    --------
    >> from openpiv.calib_utils import get_simple_grid
    
    >>> object_points = get_simple_grid(
        nx = 9,
        ny = 9,
        z = 0,
        x_ind = 4,
        y_ind = 4   
        
    >>> object_points
    
    """
    range_x = np.arange(nx).astype("float64")
    range_y = np.arange(ny).astype("float64")
    
    origin_x = range_x[x_ind]
    origin_y = range_y[y_ind]
    
    range_x -= origin_x
    range_y -= origin_y
    
    range_x *= spacing
    range_y *= spacing
    
    x, y = np.meshgrid(range_x, range_y)
    
    if flip_y == True:
        y = -y
    
    object_points = np.array(
        [x.ravel(), y.ravel(), np.zeros_like(x).ravel() + z],
        dtype="float64"
    )
    
    return object_points


def get_asymmetric_grid(
    nx: int,
    ny: int,
    z: int,
    x_ind: int,
    y_ind: int,
    flip_y: bool=True,
    spacing: float=1.0
):
    """Make an asymmetrical grid.
    
    Create an asymmetrical grid with the origin based on the index of the
    selected point. This allows an arbitrary size grid and location of
    origin.
    
    Parameters
    ----------
    nx : int
        Number of markers in the x-axis.
    ny : int
        Number of markers in the y-axis.
    z : float, optional
        The z plane where the calibration plate is located.
    x_ind : int
        Index of the point to define the x-axis.
    y_ind : int
        Index of the point to define the y-axis.
    flip_y : bool, optional
        Flip the signs of the y-axis. This is use enabled by default, but
        if the grid has an origin in the top-left corner, this should be
        disabled.
    spacing : float, optional
        Grid spacing in millimeters.
    
    Returns 
    -------
    object_points : 2D np.ndarray
        2D object points of [X, Y, Z] in world coordinates.
        
    Examples
    --------
    >> from openpiv.calib_utils import get_sasymmetric_grid
    
    >>> object_points = get_sasymmetric_grid(
        nx = 9,
        ny = 9,
        z = 0,
        x_ind = 4,
        y_ind = 4   
        
    >>> object_points
    
    """
    
    simple_grid = get_simple_grid(
        nx,
        ny,
        z,
        x_ind,
        y_ind,
        flip_y,
        spacing
    )
    
    grid = simple_grid.reshape([3, ny, nx])
    
    # TODO: optimize this loop
    _grid_x = []
    _grid_y = []
    _grid_z = []
    
    for i in range(ny):
        _grid_x += (grid[0, i, i%2::2].tolist())
        _grid_y += (grid[1, i, i%2::2].tolist()) 
        _grid_z += (grid[2, i, i%2::2].tolist())  
    
    return np.array(
        [
            _grid_x,
            _grid_y, 
            _grid_z
        ],
        dtype="float64"
    )