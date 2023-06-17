import numpy as np
import pytest

from numpy.testing import (assert_equal, assert_allclose,
                           assert_almost_equal, assert_array_almost_equal,
                           assert_)

from openpiv import calib_pinhole
from openpiv.calib_utils import get_reprojection_error, get_los_error


def get_test_camera_params():
    with open("test_calibration_points_pinhole.txt", 'r') as f:
        name = f.readline()
        
        line = f.readline()[:]
        resolution = [float(num) for num in line.split()]
        
        line = f.readline()[:]
        translation = [float(num) for num in line.split()]
        
        line = f.readline()[:]
        orientation = [float(num) for num in line.split()]
        
        line = f.readline()[:]
        distortion = [float(num) for num in line.split()]
        
        line = f.readline()[:]
        focal = [float(num) for num in line.split()]
        
        line = f.readline()[:]
        principal = [float(num) for num in line.split()]
    
    params = calib_pinhole.generate_camera_params(
        name = name,
        resolution = resolution,
        translation = translation,
        orientation = orientation,
        distortion = distortion,
        focal = focal,
        principal = principal
    )
    
    params["rotation"] = calib_pinhole.get_rotation_matrix(params)
    
    return params


def test_parameters_input():
    with pytest.raises(TypeError):
         # missing camera name
        calib_pinhole.generate_camera_params()
        
        # missing resolution
        calib_pinhole.generate_camera_params(
            "name"
        )
                                 
    with pytest.raises(ValueError):
        # name is not a string
        calib_pinhole.generate_camera_params(
            0,
            resolution=[0, 0]
        )
        
        # not two element tuple
        calib_pinhole.generate_camera_params(
            "name",
            resolution=[0]
        )
        
        # not three element vector
        calib_pinhole.generate_camera_params(
            "name",
            resolution=[0, 0],
            translation=[0, 0]
        )
        
        # not three element vector
        calib_pinhole.generate_camera_params(
            "name",
            resolution=[0, 0],
            orientation=[0, 0]
        )
        
        # not 3x3 matrix
        calib_pinhole.generate_camera_params(
            "name",
            resolution=[0, 0],
            rotation=np.zeros([3,2])
        )
        
        # not 8 element vector 
        calib_pinhole.generate_camera_params(
            "name",
            resolution=[0, 0],
            distortion=np.zeros(7)
        )
        
        # not 2 element list-like
        calib_pinhole.generate_camera_params(
            "name",
            resolution=[0, 0],
            focal="[2, 2]"
        )
        
        # not 2 element vector 
        calib_pinhole.generate_camera_params(
            "name",
            resolution=[0, 0],
            focal=[1]
        )
        
        # not 2 element list-like
        calib_pinhole.generate_camera_params(
            "name",
            resolution=[0, 0],
            principal="[2, 2]"
        )
        
        # not 2 element vector 
        calib_pinhole.generate_camera_params(
            "name",
            resolution=[0, 0],
            principal=[1]
        )
        

def test_parameters_initialization():
    params = calib_pinhole.generate_camera_params(
            "name",
            resolution=[0, 0]
        )
    
    assert_("name" in params)
    assert_("resolution" in params)
    assert_("translation" in params)
    assert_("orientation" in params)
    assert_("rotation" in params)
    assert_("distortion" in params)
    assert_("focal" in params)
    assert_("principal" in params)
    

def test_rotation_matrix_01():
    params = calib_pinhole.generate_camera_params(
        "name",
        resolution = [1024, 1024]
    )
        
    assert_allclose(
        params["rotation"],
        np.eye(3,3)
    )
    
    
def test_rotation_matrix_02():
    params = calib_pinhole.generate_camera_params(
        "name",
        resolution = [1024, 1024],
        translation = [0, 0, 1],
        orientation = [0, 0, 0]
    )
        
    assert_allclose(
        params["rotation"],
        np.eye(3,3)
    )


@pytest.mark.parametrize("dist_coefs", (np.zeros(8), np.random.rand(8)))
def test_projection_01(
    dist_coefs: np.ndarray
):
    params = calib_pinhole.generate_camera_params(
        "name",
        resolution = [1024, 1024]
    )
    
    X, Y, Z = np.random.rand(3, 32) * 100.0
    
    x, y = calib_pinhole.project_points(
        params,
        [X, Y, Z]
    )
    
    X_new, Y_new, Z_new = calib_pinhole.project_to_z(
        params,
        [x, y],
        Z
    )
    
    assert_array_almost_equal(
        X, X_new,
        decimal=4
    )
    
    assert_array_almost_equal(
        Y, Y_new,
        decimal=4
    )
    
    assert_array_almost_equal(
        Z, Z_new,
        decimal=4
    )


def test_projection_02():
    params = get_test_camera_params()
    
    cal_data = np.load("./test_calibration_points.npz")
    
    cal_obj_points = cal_data["obj_points"]
    cal_img_points = cal_data["img_points"]
    
    x, y = calib_pinhole.project_points(
        params,
        cal_obj_points
    )
    
    assert_array_almost_equal(
        x, cal_img_points[0],
        decimal=4
    )
    
    assert_array_almost_equal(
        y, cal_img_points[1],
        decimal=4
    )
    

def test_projection_03():    
    params = get_test_camera_params()
    
    cal_data = np.load("./test_calibration_points.npz")
    
    cal_obj_points = cal_data["obj_points"]
    cal_img_points = cal_data["img_points"]
    
    RMSE = get_reprojection_error(
        params,
        calib_pinhole.project_points,
        cal_obj_points,
        cal_img_points
    )
    
    assert_(RMSE < 1e-2)
    
    
def test_projection_04():    
    params = get_test_camera_params()
    
    cal_data = np.load("./test_calibration_points.npz")
    
    cal_obj_points = cal_data["obj_points"]
    cal_img_points = cal_data["img_points"]
    
    RMSE_0 = get_los_error(
        params,
        calib_pinhole.project_to_z,
        calib_pinhole.project_points,
        z = -10
    )
    
    RMSE_1 = get_los_error(
        params,
        calib_pinhole.project_to_z,
        calib_pinhole.project_points,
        z = 0
    )
    
    RMSE_2 = get_los_error(
        params,
        calib_pinhole.project_to_z,
        calib_pinhole.project_points,
        z = 10
    )
    
    assert_(RMSE_0 < 1e-2)
    assert_(RMSE_1 < 1e-2)
    assert_(RMSE_2 < 1e-2)  
    
    
def test_projection_05():
    params = get_test_camera_params()
    
    cal_data = np.load("./test_calibration_points.npz")
    
    cal_obj_points = cal_data["obj_points"]
    cal_img_points = cal_data["img_points"]
    
    X, Y, Z = calib_pinhole.project_to_z(
        params,
        cal_img_points,
        cal_obj_points[2]
    )
    
    assert_array_almost_equal(
        X, cal_obj_points[0],
        decimal=4
    )
    
    assert_array_almost_equal(
        Y, cal_obj_points[1],
        decimal=4
    )
    
    assert_array_almost_equal(
        Z, cal_obj_points[2],
        decimal=4
    )


def test_minimization_01():        
    cal_data = np.load("./test_calibration_points.npz")
    
    cal_obj_points = cal_data["obj_points"]
    cal_img_points = cal_data["img_points"]
    
    params = calib_pinhole.generate_camera_params(
        "minimized",
        resolution = [512, 512],
        translation = [1, 1, 520], # initial guess
        orientation = [-3, -0.01, 0.01], # initial guess
        focal = [1000, 1000]
    )
    
    params = calib_pinhole.minimize_camera_params(
        params,
        cal_obj_points,
        cal_img_points,
        correct_focal = False,
        correct_distortion = False,
        iterations = 3
    )
    
    RMSE = get_reprojection_error(
        params,
        calib_pinhole.project_points,
        cal_obj_points,
        cal_img_points
    )
        
    assert_(RMSE < 1e-2)


def test_minimization_02():
    params_orig = get_test_camera_params()
    
    cal_data = np.load("./test_calibration_points.npz")
    
    cal_obj_points = cal_data["obj_points"]
    cal_img_points = cal_data["img_points"]
    
    params_new = calib_pinhole.generate_camera_params(
        "minimized",
        resolution = [512, 512],
        translation = [1, 1, 520], # initial guess
        orientation = [-3, -0.01, 0.01], # initial guess
        focal = [1000, 1000]
    )
    
    params_new = calib_pinhole.minimize_camera_params(
        params_new,
        cal_obj_points,
        cal_img_points,
        correct_focal = False,
        correct_distortion = False,
        iterations = 3
    )
    
    assert_array_almost_equal(
        params_orig["translation"], 
        params_new["translation"],
        decimal = 0
    )
    
    assert_array_almost_equal(
        params_orig["orientation"], 
        params_new["orientation"],
        decimal = 0
    )
    
    assert_array_almost_equal(
        params_orig["rotation"], 
        params_new["rotation"],
        decimal = 0
    )
    
    assert_array_almost_equal(
        params_orig["focal"], 
        params_new["focal"],
        decimal = 0
    )
    
    assert_array_almost_equal(
        params_orig["principal"], 
        params_new["principal"],
        decimal = 0
    )