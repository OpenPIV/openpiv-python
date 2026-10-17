import os
import numpy as np
import pytest

from numpy.testing import (assert_equal, assert_allclose,
                           assert_almost_equal, assert_array_almost_equal,
                           assert_array_equal, assert_)

from openpiv.calibration import poly_model
from openpiv.calibration.calib_utils import get_reprojection_error, get_los_error

file = os.path.join(os.path.dirname(__file__),"test_calibration_points.npz")

@pytest.mark.parametrize("args", ((), ("name",)))
def test_required_parameters_input(args):
    with pytest.raises(TypeError):
        poly_model.camera(*args)


@pytest.mark.parametrize("parameters", [
    {"name": 0},
    {"resolution": [0]},
    {"poly_wi": np.zeros(19)},
    {"poly_wi": np.zeros((10, 10))},
    {"poly_iw_o": np.zeros(10)},
    {"poly_iw_o": np.zeros((19, 2))},
    {"poly_iw_o": np.zeros((10, 10))},
    {"poly_iw_v": np.zeros(10)},
    {"poly_iw_v": np.zeros((19, 2))},
    {"poly_iw_v": np.zeros((10, 10))},
])
def test_parameters_input(parameters):
    kwargs = {"name": "name", "resolution": [0, 0]}
    kwargs.update(parameters)
    with pytest.raises(ValueError):
        poly_model.camera(**kwargs)
        

def test_parameters_initialization():
    cam = poly_model.camera(
        "name",
        resolution=[0, 0]
    )
    
    assert_(hasattr(cam, "name"))
    assert_(hasattr(cam, "resolution"))
    assert_(hasattr(cam, "poly_wi"))
    assert_(hasattr(cam, "poly_iw_o"))
    assert_(hasattr(cam, "poly_iw_v"))
    assert_(hasattr(cam, "dtype"))
    
    assert_(len(cam.resolution) == 2)
            
    assert_equal(
        cam.poly_wi.shape,
        [19, 2]
    )
    
    assert_equal(
        cam.poly_iw_o.shape,
        [10, 2]
    )

    assert_equal(
        cam.poly_iw_v.shape,
        [10, 2]
    )
    
    assert_(cam.dtype in ["float32", "float64"])


@pytest.mark.parametrize("case", (1, 2, 3))
def test_minimization_01(
    case: int
):    
    cal_data = np.load(file)
    
    cal_obj_points = cal_data[f"obj_points_{case}"]
    cal_img_points = cal_data[f"img_points_{case}"]
    
    cam = poly_model.camera(
        "poly",
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
    cal_data = np.load(file)
    
    cal_obj_points = cal_data[f"obj_points_{case}"]
    cal_img_points = cal_data[f"img_points_{case}"]
    
    cam = poly_model.camera(
        "poly",
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


@pytest.mark.parametrize("case", (1, 2, 3))
def test_projection_02(
    case: int
):
    cal_data = np.load(file)
    
    cal_obj_points = cal_data[f"obj_points_{case}"]
    cal_img_points = cal_data[f"img_points_{case}"]
    
    cam = poly_model.camera(
        "poly",
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
    
# Test case 1 and 3 needs higher thresholds due to camera malignment
@pytest.mark.parametrize("case", (1, 2, 3))
def test_projection_03(
    case: int
):    
    cal_data = np.load(file)
    
    cal_obj_points = cal_data[f"obj_points_{case}"]
    cal_img_points = cal_data[f"img_points_{case}"]
    
    cam = poly_model.camera(
        "poly",
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
    
    assert_(RMSE_0 < 0.5)
    assert_(RMSE_1 < 0.5)
    assert_(RMSE_2 < 0.5)
    

@pytest.mark.parametrize("case", ((1, 2), (1, 3), (2, 3), (1, 2, 3)))
def test_line_intersect_01(
    case: tuple
):
    cal_data = np.load(file)
    n_cams = len(case)
    
    cal_obj_points = []
    cal_img_points = []
    for cam in case:
        cal_obj_points.append(cal_data[f"obj_points_{cam}"])
        cal_img_points.append(cal_data[f"img_points_{cam}"])
    
    cams = []
    for cam in range(n_cams):
        cam_1 = poly_model.camera(
            "dummy",
            resolution = [512, 512]
        )

        cam_1.minimize_params(
            cal_obj_points[cam],
            cal_img_points[cam]
        )
        
        cams.append(cam_1)

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
    
    test_image_points = []
    for cam in range(n_cams):
        test_image_points.append(
            cams[cam].project_points(
                test_object_points
            )
        )

    recon_obj_points = poly_model.multi_line_intersect(
        cams,
        test_image_points
    )
    
    assert_array_almost_equal(
        test_object_points,
        recon_obj_points,
        decimal=0
    )
    
    
def test_save_parameters_1():
    cam = poly_model.camera(
        "dummy",
        resolution = [512, 512]
    )
        
    cam.save_parameters(
        "."
    )


def test_save_parameters_2():
    cam = poly_model.camera(
        "dummy",
        resolution = [512, 512]
    )

        
    cam.save_parameters(
        ".", "saved_params"
    )

    
def test_load_parameters_1():
    cam = poly_model.camera(
        "dummy",
        resolution = [512, 512]
    )
    
    with pytest.raises(FileNotFoundError):
        cam.load_parameters(
            ".",
            "does not exist (hopefully)"
        )
    

def test_load_parameters_2():
    cam_orig = poly_model.camera(
        "dummy",
        resolution = [512, 512],
        poly_iw_o = np.arange(20, dtype="float64").reshape(10, 2),
        poly_iw_v = np.arange(20, 40, dtype="float64").reshape(10, 2)
    )
    
    cam_new = poly_model.camera(
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
        cam_orig.poly_wi, 
        cam_new.poly_wi
    )
    
    assert_array_equal(
        cam_orig.poly_iw_o,
        cam_new.poly_iw_o
    )

    assert_array_equal(
        cam_orig.poly_iw_v,
        cam_new.poly_iw_v
    )
    
    assert_array_equal(
        cam_orig.dtype, 
        cam_new.dtype
    )
