import os
import numpy as np
import pytest

from numpy.testing import (assert_equal, assert_allclose,
                           assert_almost_equal, assert_array_almost_equal,
                           assert_array_equal, assert_)

from openpiv.calibration import calib_utils


def test_find_nearest_points_01():
    grid = calib_utils.get_simple_grid(
        10, 10,
        0, 0, 0,
        spacing=10,
        flip_y=False
    )[:2]
    
    selected_points = np.array(
        [
            [1, 1],
            [2, 2],
            [4, 4],
            [6, 6],
            [93, 93],
            [110, 110]
        ],
        dtype="float64"
    ).T
    
    expected_points = np.array(
        [
            [0, 0],
            [0, 0],
            [0, 0],
            [10, 10],
            [90, 90],
            [90, 90]
        ],
        dtype="float64"
    ).T
    
    # without NaNs threshold
    found_points = calib_utils.find_nearest_points(
        grid, 
        selected_points,
        threshold=None, 
        flag_nans=False
    )
    
    assert_array_equal(
        expected_points,
        found_points
    )
    
    # with NaNs threshold
    found_points = calib_utils.find_nearest_points(
        grid, 
        selected_points,
        threshold=5, 
        flag_nans=True
    )
    
    assert_(np.isnan(found_points[0, 2]))
    assert_(np.isnan(found_points[0, 3]))
    assert_(np.isnan(found_points[0, 5]))
    
    assert_(np.isnan(found_points[1, 2]))
    assert_(np.isnan(found_points[1, 3]))
    assert_(np.isnan(found_points[1, 5]))
    
    nan_mask = ~np.isnan(found_points[0,:])
    
    found_points = found_points[:, nan_mask]
    expected_points = expected_points[:, nan_mask]
    
    assert_array_equal(
        expected_points,
        found_points
    )


def test_find_nearest_points_02():
    grid = calib_utils.get_simple_grid(
        10, 10,
        0, 0, 0,
        spacing=10,
        flip_y=False
    )[:2]
    
    selected_points = np.array(
        [
            [1, 1],
            [2, 2],
            [4, 4],
            [6, 6],
            [93, 93],
            [110, 110]
        ],
        dtype="float64"
    ).T
    
    expected_points = np.array(
        [
            [0, 0],
            [0, 0],
#            [0, 0], # out of bound of threshold
#            [10, 10], # out of bound of threshold
            [90, 90],
#            [90, 90] # out of bound of threshold
        ],
        dtype="float64"
    ).T
    
    found_points = calib_utils.find_nearest_points(
        grid, 
        selected_points,
        threshold=5.5, 
        flag_nans=False
    )
    
    assert_array_equal(
        expected_points,
        found_points
    )