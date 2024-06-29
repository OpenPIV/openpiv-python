import numpy as np
import pytest

from numpy.testing import (assert_equal, assert_allclose,
                           assert_almost_equal, assert_array_almost_equal,
                           assert_array_equal, assert_)

from openpiv.calibration import dlt_model
from openpiv.calibration.calib_utils import get_reprojection_error, get_los_error


def test_parameters_input():
    with pytest.raises(TypeError):
         # missing camera name
        dlt_model.camera()
        
        # missing resolution
        dlt_model.camera(
            "name"
        )
                                 
    with pytest.raises(ValueError):
        # name is not a string
        dlt_model.camera(
            0,
            resolution=[0, 0]
        )
        
        # not two element tuple
        dlt_model.camera(
            "name",
            resolution=[0]
        )
        
        # not 2D or 3D
        dlt_model.camera(
            "name",
            resolution=[0, 0],
            ndim = 4
        )
        
        # coefficients not correct dimension
        dlt_model.camera(
            "name",
            resolution=[0, 0],
            coeffs = np.zeros((10, 10, 2))
        )
        
        # coefficients not correct shape
        dlt_model.camera(
            "name",
            resolution=[0, 0],
            coeffs = np.zeros((10, 10))
        )
        

def test_parameters_initialization():
    cam = dlt_model.camera(
            "name",
            resolution=[0, 0]
        )
    
    assert_(hasattr(cam, "name"))
    assert_(hasattr(cam, "resolution"))
    assert_(hasattr(cam, "coeffs"))
    assert_(hasattr(cam, "dtype"))
    
    assert_(len(cam.resolution) == 2)
    
    assert_(cam.coeffs.shape in [(3, 3), (3, 4)])
    
    assert_(cam.dtype in ["float32", "float64"])


@pytest.mark.parametrize("case", (1, 2))
def test_minimization_01(
    case: int
):    
    cal_data = np.load("./test_calibration_points.npz")
    
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
    
    RMSE = get_reprojection_error(
        cam,
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
    
    cam = dlt_model.camera(
        "dummy",
        resolution = [512, 512],
    )
    
    cam.minimize_params(
        cal_obj_points,
        cal_img_points
    )
    
    obj_points = np.random.rand(3, 32)
    obj_points[0, :] = np.int32(obj_points[0, :] * 50)
    obj_points[1, :] = np.int32(obj_points[1, :] * 50)
    obj_points[2, :] = np.int32(obj_points[2, :] * 10)
    
    obj_points = obj_points.astype("float64", copy=False)
    
    img_points = cam.project_points(
        obj_points
    )
    
    recon_obj_points = cam.project_to_z(
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
    
    cam = dlt_model.camera(
        "dummy",
        resolution = [512, 512],
    )
    
    cam.minimize_params(
        cal_obj_points,
        cal_img_points
    )
    
    x, y = cam.project_points(
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
    
    cam = dlt_model.camera(
        "dummy",
        resolution = [512, 512],
    )
    
    cam.minimize_params(
        cal_obj_points,
        cal_img_points
    )
    
    RMSE_0 = get_los_error(
        cam,
        z = -10
    )
    
    RMSE_1 = get_los_error(
        cam,
        z = 0
    )
    
    RMSE_2 = get_los_error(
        cam,
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
    
    cam_1 = dlt_model.camera(
        "dummy",
        resolution = [512, 512]
    )
    
    cam_2 = dlt_model.camera(
        "dummy",
        resolution = [512, 512]
    )
    
    cam_1.minimize_params(
        cal_obj_points_1,
        cal_img_points_1
    )
    
    cam_2.minimize_params(
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
        dtype=cam_1.dtype
    ).T
    
    test_image_points_1 = cam_1.project_points(
        test_object_points
    )
    
    test_image_points_2 = cam_2.project_points(
        test_object_points
    )
    
    recon_obj_points, _ = dlt_model.line_intersect(
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
    
    cam_1 = dlt_model.camera(
        "dummy",
        resolution = [512, 512]
    )
    
    cam_2 = dlt_model.camera(
        "dummy",
        resolution = [512, 512]
    )
    
    cam_1.minimize_params(
        cal_obj_points_1,
        cal_img_points_1
    )
    
    cam_2.minimize_params(
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
        dtype=cam_1.dtype
    ).T
    
    test_image_points_1 = cam_1.project_points(
        test_object_points
    )
    
    test_image_points_2 = cam_2.project_points(
        test_object_points
    )
    
    recon_obj_points = dlt_model.multi_line_intersect(
        [cam_1, cam_2],
        [test_image_points_1, test_image_points_2]
    )
    
    assert_array_almost_equal(
        test_object_points,
        recon_obj_points,
        decimal=0
    )
   
    
def test_save_parameters_1():
    cam = dlt_model.camera(
        "dummy",
        resolution = [512, 512]
    )
        
    cam.save_parameters(
        "."
    )


def test_save_parameters_2():
    cam = dlt_model.camera(
        "dummy",
        resolution = [512, 512]
    )

        
    cam.save_parameters(
        ".", "saved_params"
    )

    
def test_load_parameters_1():
    cam = dlt_model.camera(
        "dummy",
        resolution = [512, 512]
    )
    
    with pytest.raises(FileNotFoundError):
        cam.load_parameters(
            ".",
            "does not exist (hopefully)"
        )
    

def test_load_parameters_2():
    cam_orig = dlt_model.camera(
        "dummy",
        resolution = [512, 512]
    )
    
    cam_new = dlt_model.camera(
        "dummy",
        resolution = [512, 512]
    )
        
    cam_orig.save_parameters(
        ".",
        "dummy"
    )
    
    cam_new.load_parameters(
        ".",
        "dummy"
    )
    
    assert_array_equal(
        cam_orig.name, 
        cam_new.name
    )
    
    assert_array_equal(
        cam_orig.resolution, 
        cam_new.resolution
    )
    
    assert_array_equal(
        cam_orig.coeffs, 
        cam_new.coeffs
    )
    
    assert_array_equal(
        cam_orig.dtype, 
        cam_new.dtype
    )