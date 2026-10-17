import numpy as np
import os
import pytest

from numpy.testing import (assert_equal, assert_allclose,
                           assert_almost_equal, assert_array_almost_equal,
                           assert_)

from openpiv.calibration import calib_utils
from openpiv.calibration import dlt_model


file = os.path.join(os.path.dirname(__file__),"test_calibration_points.npz")


# this function is not used yet, but will be when more utility functions are tested
def get_camera_instance():
    cal_data = np.load(file)
    
    case = 1
    
    cal_obj_points = cal_data[f"obj_points_{case}"]
    cal_img_points = cal_data[f"img_points_{case}"]
    
    cam = dlt_model.camera(
        "dummy",
        resolution = [512, 512],
    )
    
    cam.minimize_params(
        cal_obj_points,
        cal_img_points
    )
    
    return cam

    
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


def test_reprojection_error_01():
    cam = get_camera_instance()
    
    offset = 0.4
    
    test_object_points = np.array(
        [
            [0, 0, 0],
            [1, 2, 3],
            [3, 2, 1],
            [15, 15, -15],
            [15, 15, 0],
            [15, 15, 15]
        ],
        dtype=cam.dtype
    ).T
    
    test_image_points = cam.project_points(test_object_points)
    
    # Our definition of RMSE
    error = test_image_points - (test_image_points + offset)
    
    expected = np.sqrt(np.mean(np.sum(np.square(error), 0)))
    
    # Now use the error function
    results = calib_utils.get_reprojection_error(
        cam,
        test_object_points,
        test_image_points + offset
    )
    
    assert_array_almost_equal(
        expected,
        results,
        decimal=4
    )