import numpy as np
import pytest

from numpy.testing import (assert_equal, assert_allclose,
                           assert_almost_equal, assert_array_almost_equal,
                           assert_)

from openpiv.calibration import calib_utils

def test_homogenize_01():
    shape = (2, 24)
    expected = (shape[0] + 1, shape[1])
    
    img_points = np.random.rand(shape[0], shape[1])
    
    homogenized = calib_utils.homogenize(img_points)
    
    assert_equal(
        homogenized.shape, 
        expected
    )
    assert_(np.all(homogenized[2, :] == 1))
    
    
def test_get_rmse_01():
    shape = 24
    offset = 1.5
    
    test_array = np.zeros(shape, dtype = "float64")
    
    error = (test_array + offset) - test_array
    
    rmse = calib_utils.get_rmse(error)
    
    assert_equal(
        rmse,
        offset
    )
    
    
def test_get_rmse_02():
    shape = (2, 24)
    offset = 2
    expected = np.sqrt(np.power(2, 3))
    
    test_array = np.zeros(shape, dtype = "float64")
    
    error = (test_array + offset) - test_array
    
    rmse = calib_utils.get_rmse(error)
    
    assert_equal(
        rmse, 
        expected
    )
    
    
def test_get_rmse_03():
    shape = (2, 24, 5)
    offset = 2
    expected = np.sqrt(np.power(2, 3))
    
    test_array = np.zeros(shape, dtype = "float64")
    
    error = (test_array + offset) - test_array
    
    with pytest.raises(ValueError):
        # invalid shape, get_rmse expects a 1D or 2D array
        rmse = calib_utils.get_rmse(error)


