import numpy as np
import pytest

from numpy.testing import (assert_equal, assert_allclose,
                           assert_almost_equal, assert_array_almost_equal,
                           assert_array_equal, assert_)

from openpiv.calibration import poly_model as calib_polynomial
from openpiv.calibration.calib_utils import get_reprojection_error, get_los_error


def test_parameters_input():
    with pytest.raises(TypeError):
         # missing camera name
        calib_polynomial.generate_camera_params()
        
        # missing resolution
        calib_polynomial.generate_camera_params(
            "name"
        )
                                 
    with pytest.raises(ValueError):
        # name is not a string
        calib_polynomial.generate_camera_params(
            0,
            resolution=[0, 0]
        )
        
        # not two element tuple
        calib_polynomial.generate_camera_params(
            "name",
            resolution=[0]
        )
        
        # not 2D
        calib_polynomial.generate_camera_params(
            "name",
            resolution=[0, 0],
            poly_wi = np.zeros(19)
        )
        
        # not 2D
        calib_polynomial.generate_camera_params(
            "name",
            resolution=[0, 0],
            poly_iw = np.zeros(19)
        )
        
        # not correct shape
        calib_polynomial.generate_camera_params(
            "name",
            resolution=[0, 0],
            poly_wi = np.zeros((10, 10))
        )
        
        # not correct shape
        calib_polynomial.generate_camera_params(
            "name",
            resolution=[0, 0],
            poly_iw = np.zeros((10, 10))
        )
        

def test_parameters_initialization():
    params = calib_polynomial.generate_camera_params(
            "name",
            resolution=[0, 0]
        )
    
    assert_("name" in params)
    assert_("resolution" in params)
    assert_("poly_wi" in params)
    assert_("poly_iw" in params)
    
    assert_(len(params["resolution"]) == 2)
    
    assert_equal(
        params["poly_wi"].shape,
        [19, 2]
    )
    
    assert_equal(
        params["poly_iw"].shape,
        [19, 3]
    )


def test_minimization_01():    
    cal_data = np.load("./test_calibration_points.npz")
    
    cal_obj_points = cal_data["obj_points"]
    cal_img_points = cal_data["img_points"]
    
    params = calib_polynomial.generate_camera_params(
        "poly",
        resolution = [512, 512],
    )
    
    params = calib_polynomial.minimize_polynomial(
        params,
        cal_obj_points,
        cal_img_points
    )
    
    RMSE = get_reprojection_error(
        params,
        calib_polynomial.project_points,
        cal_obj_points,
        cal_img_points
    )
    
    assert_(RMSE < 1e-2)


def test_projection_01():
    cal_data = np.load("./test_calibration_points.npz")
    
    cal_obj_points = cal_data["obj_points"]
    cal_img_points = cal_data["img_points"]
    
    params = calib_polynomial.generate_camera_params(
        "poly",
        resolution = [512, 512],
    )
    
    params = calib_polynomial.minimize_polynomial(
        params,
        cal_obj_points,
        cal_img_points
    )
    
    obj_points = np.random.rand(3, 32)
    obj_points[0, :] = np.int32(obj_points[0, :] * 50)
    obj_points[1, :] = np.int32(obj_points[1, :] * 50)
    obj_points[2, :] = np.int32(obj_points[2, :] * 10)
    
    obj_points = obj_points.astype("float64", copy=False)
    
    img_points = calib_polynomial.project_points(
        params,
        obj_points
    )
    
    recon_obj_points = calib_polynomial.project_to_z(
        params,
        img_points,
        obj_points[2]
    )
    
    assert_array_almost_equal(
        obj_points,
        recon_obj_points,
        decimal=2
    )

    
def test_projection_02():
    cal_data = np.load("./test_calibration_points.npz")
    
    cal_obj_points = cal_data["obj_points"]
    cal_img_points = cal_data["img_points"]
    
    params = calib_polynomial.generate_camera_params(
        "poly",
        resolution = [512, 512],
    )
    
    params = calib_polynomial.minimize_polynomial(
        params,
        cal_obj_points,
        cal_img_points
    )
    
    x, y = calib_polynomial.project_points(
        params,
        cal_obj_points
    )
    
    assert_array_almost_equal(
        x, cal_img_points[0],
        decimal=2
    )
    
    assert_array_almost_equal(
        y, cal_img_points[1],
        decimal=2
    )
    
    
def test_projection_03():    
    cal_data = np.load("./test_calibration_points.npz")
    
    cal_obj_points = cal_data["obj_points"]
    cal_img_points = cal_data["img_points"]
    
    params = calib_polynomial.generate_camera_params(
        "poly",
        resolution = [512, 512],
    )
    
    params = calib_polynomial.minimize_polynomial(
        params,
        cal_obj_points,
        cal_img_points
    )
    
    RMSE_0 = get_los_error(
        params,
        calib_polynomial.project_to_z,
        calib_polynomial.project_points,
        z = -10
    )
    
    RMSE_1 = get_los_error(
        params,
        calib_polynomial.project_to_z,
        calib_polynomial.project_points,
        z = 0
    )
    
    RMSE_2 = get_los_error(
        params,
        calib_polynomial.project_to_z,
        calib_polynomial.project_points,
        z = 10
    )
    
    assert_(RMSE_0 < 1e-2)
    assert_(RMSE_1 < 1e-2)
    assert_(RMSE_2 < 1e-2)
    
    
def test_save_parameters_1():
    params = calib_polynomial.generate_camera_params(
        "dummy",
        resolution = [512, 512]
    )
        
    calib_polynomial.save_parameters(
        params,
        "."
    )


def test_save_parameters_2():
    params = calib_polynomial.generate_camera_params(
        "dummy",
        resolution = [512, 512]
    )

        
    calib_polynomial.save_parameters(
        params,
        ".", "saved_params"
    )

    
def test_load_parameters_1():
    with pytest.raises(FileNotFoundError):
        params_loaded = calib_polynomial.load_parameters(
            ".",
            "does not exist (hopefully)"
        )
    

def test_load_parameters_2():
    params_orig = calib_polynomial.generate_camera_params(
        "dummy",
        resolution = [512, 512]
    )
        
    calib_polynomial.save_parameters(
        params_orig,
        ".",
        "dummy"
    )
    
    params_new = calib_polynomial.load_parameters(
        ".",
        "dummy"
    )
    assert_array_equal(
        params_orig["name"], 
        params_new["name"]
    )
    
    assert_array_equal(
        params_orig["resolution"], 
        params_new["resolution"]
    )
    
    assert_array_equal(
        params_orig["poly_wi"], 
        params_new["poly_wi"]
    )
    
    assert_array_equal(
        params_orig["poly_iw"], 
        params_new["poly_iw"]
    )
    
    assert_array_equal(
        params_orig["dtype"], 
        params_new["dtype"]
    )