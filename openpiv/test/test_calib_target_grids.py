import os
import numpy as np
import pytest

from numpy.testing import (assert_equal, assert_allclose,
                           assert_almost_equal, assert_array_almost_equal,
                           assert_array_equal, assert_)

from openpiv.calibration import poly_model
from openpiv.calibration.calib_utils import get_simple_grid, get_asymmetric_grid


def test_simple_grid_01():
    grid_shape = (9, 9)
    origin = np.array(grid_shape) // 2
    z_plane = 0
    spacing = 10
    
    grid = get_simple_grid(
        grid_shape[0], grid_shape[1],
        z_plane,
        origin[0], origin[1], 
        spacing=spacing
    )
    
    # check horizontal and vertical spacing
    assert_((grid[0, 1] - grid[0, 0]) == spacing)
    assert_((grid[1, 1] - grid[1, grid_shape[1]]) == spacing)
    
    # make sure z-plane is correct
    assert_(np.all(grid[2, :] == z_plane))
    
    # check grid shape
    assert_equal(
        grid.shape, 
        (3, np.prod(grid_shape))
    )
    
    # make sure origin is 0
    grid = grid.reshape((3, grid_shape[0], grid_shape[1]))
    
    origin = grid[:2, origin[0], origin[1]]
    
    assert_(origin[0] == 0)
    assert_(origin[1] == 0)
    
    
def test_asymmetric_grid_01():
    grid_shape = (9, 9)
    origin = np.array(grid_shape) // 2
    
    z_plane = 0
    spacing = 10
    
    grid = get_asymmetric_grid(
        grid_shape[0], grid_shape[1],
        z_plane,
        origin[0], origin[1], 
        spacing=spacing
    )
    
    # check horizontal and vertical spacing
    assert_((grid[0, 1] - grid[0, 0]) == (spacing * 2))
    assert_((grid[1, 1] - grid[1, grid_shape[1]]) == (spacing * 2))
    
    # make sure z-plane is correct
    assert_(np.all(grid[2, :] == z_plane))
    
    # check grid shape
    grid_length = int(np.prod(grid_shape) / 2 + 0.5)
    
    assert_equal(
        grid.shape, 
        (3, grid_length)
    )
    
    # make sure origin is 0 
    origin_ind = int(grid_length / 2)
    
    origin = grid[:2, origin_ind]
    
    assert_(origin[0] == 0)
    assert_(origin[1] == 0)