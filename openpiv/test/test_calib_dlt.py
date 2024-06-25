import numpy as np
import pytest

from numpy.testing import (assert_equal, assert_allclose,
                           assert_almost_equal, assert_array_almost_equal,
                           assert_array_equal, assert_)

from .calibration import dlt_model as calib_dlt
from .calibration.calib_utils import get_reprojection_error, get_los_error


def test_parameters_input():
    with pytest.raises(TypeError):
         # missing camera name
        calib_dlt.get_cam_params()
        
        # missing resolution
        calib_dlt.get_cam_params(
            "name"
        )
                                 
    with pytest.raises(ValueError):
        # name is not a string
        calib_dlt.get_cam_params(
            0,
            resolution=[0, 0]
        )
        
        # not two element tuple
        calib_dlt.get_cam_params(
            "name",
            resolution=[0]
        )
        
        # not 2D or 3D
        calib_dlt.get_cam_params(
            "name",
            resolution=[0, 0],
            ndim = 4
        )
        
        # coefficients not correct dimension
        calib_dlt.get_cam_params(
            "name",
            resolution=[0, 0],
            coefficients = np.zeros((10, 10, 2))
        )
        
        # coefficients not correct shape
        calib_dlt.get_cam_params(
            "name",
            resolution=[0, 0],
            coefficients = np.zeros((10, 10))
        )
        

def test_parameters_initialization():
    params = calib_dlt.get_cam_params(
            "name",
            resolution=[0, 0]
        )
    
    assert_("name" in params)
    assert_("resolution" in params)
    assert_("coefficients" in params)
    assert_("dtype" in params)
    
    assert_(len(params["resolution"]) == 2)
    
    assert_(params["coefficients"].shape in [(3, 3), (3, 4)])
    
    assert_(params["dtype"] in ["float32", "float64"])


@pytest.mark.parametrize("case", (1, 2))
def test_minimization_01(
    case: int
):    
    cal_data = np.load("./test_calibration_points.npz")
    
    cal_obj_points = cal_data[f"obj_points_{case}"]
    cal_img_points = cal_data[f"img_points_{case}"]
    
    params = calib_dlt.get_cam_params(
        "poly",
        resolution = [512, 512],
    )
    
    params = calib_dlt.minimize_params(
        params,
        cal_obj_points,
        cal_img_points
    )
    
    RMSE = get_reprojection_error(
        params,
        calib_dlt.project_points,
        cal_obj_points,
        cal_img_points
    )
    
    assert_(RMSE < 1e-2)


@pytest.mark.parametrize("case", (1, 2))
def test_projection_01(
    case: int
):
    cal_data = np.load("./test_calibration_points.npz")
    
    cal_obj_points = cal_data[f"obj_points_{case}"]
    cal_img_points = cal_data[f"img_points_{case}"]
    
    params = calib_dlt.get_cam_params(
        "poly",
        resolution = [512, 512],
    )
    
    params = calib_dlt.minimize_params(
        params,
        cal_obj_points,
        cal_img_points
    )
    
    obj_points = np.random.rand(3, 32)
    obj_points[0, :] = np.int32(obj_points[0, :] * 50)
    obj_points[1, :] = np.int32(obj_points[1, :] * 50)
    obj_points[2, :] = np.int32(obj_points[2, :] * 10)
    
    obj_points = obj_points.astype("float64", copy=False)
    
    img_points = calib_dlt.project_points(
        params,
        obj_points
    )
    
    recon_obj_points = calib_dlt.project_to_z(
        params,
        img_points,
        obj_points[2]
    )
    
    assert_array_almost_equal(
        obj_points,
        recon_obj_points,
        decimal=2
    )


@pytest.mark.parametrize("case", (1, 2))
def test_projection_02(
    case: int
):
    cal_data = np.load("./test_calibration_points.npz")
    
    cal_obj_points = cal_data[f"obj_points_{case}"]
    cal_img_points = cal_data[f"img_points_{case}"]
    
    params = calib_dlt.get_cam_params(
        "poly",
        resolution = [512, 512],
    )
    
    params = calib_dlt.minimize_params(
        params,
        cal_obj_points,
        cal_img_points
    )
    
    x, y = calib_dlt.project_points(
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
    

@pytest.mark.parametrize("case", (1, 2))
def test_projection_03(
    case: int
):
    cal_data = np.load("./test_calibration_points.npz")
    
    cal_obj_points = cal_data[f"obj_points_{case}"]
    cal_img_points = cal_data[f"img_points_{case}"]
    
    params = calib_dlt.get_cam_params(
        "poly",
        resolution = [512, 512],
    )
    
    params = calib_dlt.minimize_params(
        params,
        cal_obj_points,
        cal_img_points
    )
    
    RMSE_0 = get_los_error(
        params,
        calib_dlt.project_to_z,
        calib_dlt.project_points,
        z = -10
    )
    
    RMSE_1 = get_los_error(
        params,
        calib_dlt.project_to_z,
        calib_dlt.project_points,
        z = 0
    )
    
    RMSE_2 = get_los_error(
        params,
        calib_dlt.project_to_z,
        calib_dlt.project_points,
        z = 10
    )
        
    assert_(RMSE_0 < 1e-2)
    assert_(RMSE_1 < 1e-2)
    assert_(RMSE_2 < 1e-2)
    

# TODO: Add more camera views
def test_line_intersect_01():
    case = (1, 2)
    cal_data = np.load("./test_calibration_points.npz")
    
    cal_obj_points_1 = cal_data[f"obj_points_{case[0]}"]
    cal_img_points_1 = cal_data[f"img_points_{case[0]}"]
    
    cal_obj_points_2 = cal_data[f"obj_points_{case[1]}"]
    cal_img_points_2 = cal_data[f"img_points_{case[1]}"]
    
    cam_1 = calib_dlt.get_cam_params(
        "poly",
        resolution = [512, 512]
    )
    
    cam_2 = calib_dlt.get_cam_params(
        "poly",
        resolution = [512, 512]
    )
    
    cam_1 = calib_dlt.minimize_params(
        cam_1,
        cal_obj_points_1,
        cal_img_points_1
    )
    
    cam_2 = calib_dlt.minimize_params(
        cam_2,
        cal_obj_points_2,
        cal_img_points_2
    )
    
    test_object_points = np.array(
        [
            [0, 0, 0],
            [1, 2, 3],
            [3, 2, 1],
            [15, 15, -15],
            [15, 15, 0],
            [15, 15, 15]
        ],
        dtype=cam_1["dtype"]
    ).T
    
    test_image_points_1 = calib_dlt.project_points(
        cam_1,
        test_object_points
    )
    
    test_image_points_2 = calib_dlt.project_points(
        cam_2,
        test_object_points
    )
    
    recon_obj_points, _ = calib_dlt.line_intersect(
        cam_1,
        cam_2,
        test_image_points_1,
        test_image_points_2
    )
    
    assert_array_almost_equal(
        test_object_points,
        recon_obj_points,
        decimal=2
    )
    

# TODO: Add more camera views
def test_line_intersect_02():
    case = (1, 2)
    cal_data = np.load("./test_calibration_points.npz")
    
    cal_obj_points_1 = cal_data[f"obj_points_{case[0]}"]
    cal_img_points_1 = cal_data[f"img_points_{case[0]}"]
    
    cal_obj_points_2 = cal_data[f"obj_points_{case[1]}"]
    cal_img_points_2 = cal_data[f"img_points_{case[1]}"]
    
    cam_1 = calib_dlt.get_cam_params(
        "poly",
        resolution = [512, 512]
    )
    
    cam_2 = calib_dlt.get_cam_params(
        "poly",
        resolution = [512, 512]
    )
    
    cam_1 = calib_dlt.minimize_params(
        cam_1,
        cal_obj_points_1,
        cal_img_points_1
    )
    
    cam_2 = calib_dlt.minimize_params(
        cam_2,
        cal_obj_points_2,
        cal_img_points_2
    )
    
    test_object_points = np.array(
        [
            [0, 0, 0],
            [1, 2, 3],
            [3, 2, 1],
            [15, 15, -15],
            [15, 15, 0],
            [15, 15, 15]
        ],
        dtype=cam_1["dtype"]
    ).T
    
    test_image_points_1 = calib_dlt.project_points(
        cam_1,
        test_object_points
    )
    
    test_image_points_2 = calib_dlt.project_points(
        cam_2,
        test_object_points
    )
    
    recon_obj_points = calib_dlt.multi_line_intersect(
        [cam_1, cam_2],
        [test_image_points_1, test_image_points_2]
    )
    
    assert_array_almost_equal(
        test_object_points,
        recon_obj_points,
        decimal=0
    )
   
    
def test_save_parameters_1():
    params = calib_dlt.get_cam_params(
        "dummy",
        resolution = [512, 512]
    )
        
    calib_dlt.save_parameters(
        params,
        "."
    )


def test_save_parameters_2():
    params = calib_dlt.get_cam_params(
        "dummy",
        resolution = [512, 512]
    )

        
    calib_dlt.save_parameters(
        params,
        ".", "saved_params"
    )

    
def test_load_parameters_1():
    with pytest.raises(FileNotFoundError):
        params_loaded = calib_dlt.load_parameters(
            ".",
            "does not exist (hopefully)"
        )
    

def test_load_parameters_2():
    params_orig = calib_dlt.get_cam_params(
        "dummy",
        resolution = [512, 512]
    )
        
    calib_dlt.save_parameters(
        params_orig,
        ".",
        "dummy"
    )
    
    params_new = calib_dlt.load_parameters(
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
        params_orig["coefficients"], 
        params_new["coefficients"]
    )
    
    assert_array_equal(
        params_orig["dtype"], 
        params_new["dtype"]
    )