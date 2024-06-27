import numpy as np
import pytest

from numpy.testing import (assert_equal, assert_allclose,
                           assert_almost_equal, assert_array_almost_equal,
                           assert_array_equal, assert_)

from openpiv.calibration import pinhole_model as calib_pinhole
from openpiv.calibration.calib_utils import get_reprojection_error, get_los_error


def get_test_camera_params(
    case: int=1
):
    if case == 1:
        pinhole_cam = "test_calibration_pinhole_1.txt"
    else:
        pinhole_cam = "test_calibration_pinhole_2.txt"
        
    with open(pinhole_cam, 'r') as f:
        name = f.readline()[:-1]
        
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
    
    params = calib_pinhole.get_cam_params(
        name = name,
        resolution = resolution,
        translation = translation,
        orientation = orientation,
        distortion_model = "brown",
        distortion1 = distortion,
        focal = focal,
        principal = principal
    )
    
    params["rotation"] = calib_pinhole.get_rotation_matrix(params)
    
    return params


def test_parameters_input():
    with pytest.raises(TypeError):
         # missing camera name
        calib_pinhole.get_cam_params()
        
        # missing resolution
        calib_pinhole.get_cam_params(
            "name"
        )
                                 
    with pytest.raises(ValueError):
        # name is not a string
        calib_pinhole.get_cam_params(
            0,
            resolution=[0, 0]
        )
        
        # not two element tuple
        calib_pinhole.get_cam_params(
            "name",
            resolution=[0]
        )
        
        # not three element vector
        calib_pinhole.get_cam_params(
            "name",
            resolution=[0, 0],
            translation=[0, 0]
        )
        
        # not three element vector
        calib_pinhole.get_cam_params(
            "name",
            resolution=[0, 0],
            orientation=[0, 0]
        )
        
        # not 3x3 matrix
        calib_pinhole.get_cam_params(
            "name",
            resolution=[0, 0],
            rotation=np.zeros([3,2])
        )
        
        # wrong distortion model
        calib_pinhole.get_cam_params(
            "name",
            resolution=[0, 0],
            distortion_model="non-existent <random symbols here>",
        )
        
        # not 8 element vector for brown model 
        calib_pinhole.get_cam_params(
            "name",
            resolution=[0, 0],
            distortion_model="brown",
            distortion1=np.zeros(7)
        )
        
        # not 4 x 6 matrix for polynomial model 
        calib_pinhole.get_cam_params(
            "name",
            resolution=[0, 0],
            distortion_model="polynomial",
            distortion2=np.zeros([4, 3])
        )
        
        # not 2 element list-like
        calib_pinhole.get_cam_params(
            "name",
            resolution=[0, 0],
            focal="[2, 2]"
        )
        
        # not 2 element vector 
        calib_pinhole.get_cam_params(
            "name",
            resolution=[0, 0],
            focal=[1]
        )
        
        # not 2 element list-like
        calib_pinhole.get_cam_params(
            "name",
            resolution=[0, 0],
            principal="[2, 2]"
        )
        
        # not 2 element vector 
        calib_pinhole.get_cam_params(
            "name",
            resolution=[0, 0],
            principal=[1]
        )
        
        # not a support dtype (supported dtypes are float32 and float64)
        calib_pinhole.get_cam_params(
            "name",
            resolution=[0, 0],
            dtype=int
        )
        

def test_parameters_initialization():
    params = calib_pinhole.get_cam_params(
            "name",
            resolution=[0, 0]
        )
    
    assert_("name" in params)
    assert_("resolution" in params)
    assert_("translation" in params)
    assert_("orientation" in params)
    assert_("rotation" in params)
    assert_("distortion_model" in params)
    assert_("distortion1" in params)
    assert_("distortion2" in params)
    assert_("focal" in params)
    assert_("principal" in params)
    assert_("dtype" in params)
    
    assert_(len(params["resolution"]) == 2)
    
    assert_equal(
        params["translation"].shape,
        [3, ]
    )
    
    assert_equal(
        params["orientation"].shape,
        [3, ]
    )
    
    assert_equal(
        params["rotation"].shape,
        [3, 3]
    )
    
    assert_equal(
        params["distortion1"].shape,
        [8, ]
    )
    
    assert_equal(
        params["distortion2"].shape,
        [4, 6]
    )
    
    assert_(len(params["focal"]) == 2)
    
    assert_(len(params["principal"]) == 2)
    
    # float32 does not work well with the pinhole model, should it be deprecated?
    assert_(params["dtype"] in ["float32", "float64"])
        

def test_rotation_matrix_01():
    params = calib_pinhole.get_cam_params(
        "name",
        resolution = [1024, 1024]
    )
        
    assert_allclose(
        params["rotation"],
        np.eye(3,3)
    )
    
    
def test_rotation_matrix_02():
    params = calib_pinhole.get_cam_params(
        "name",
        resolution = [1024, 1024],
        translation = [0, 0, 1],
        orientation = [0, 0, 0]
    )
        
    assert_allclose(
        params["rotation"],
        np.eye(3,3)
    )


def test_projection_01():
    params = calib_pinhole.get_cam_params(
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


@pytest.mark.parametrize("case", (1, 2))
def test_projection_02(
    case: int
):
    params = get_test_camera_params(case)
    
    cal_data = np.load("./test_calibration_points.npz")
    
    cal_obj_points = cal_data[f"obj_points_{case}"]
    cal_img_points = cal_data[f"img_points_{case}"]
    
    x, y = calib_pinhole.project_points(
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
    params = get_test_camera_params(case)
    
    cal_data = np.load("./test_calibration_points.npz")
    
    cal_obj_points = cal_data[f"obj_points_{case}"]
    cal_img_points = cal_data[f"img_points_{case}"]
    
    RMSE = get_reprojection_error(
        params,
        calib_pinhole.project_points,
        cal_obj_points,
        cal_img_points
    )
    
    assert_(RMSE < 1e-2)
    

@pytest.mark.parametrize("case", (1, 2))
def test_projection_04(
    case: int
):    
    params = get_test_camera_params(case)
    
    cal_data = np.load("./test_calibration_points.npz")
    
    cal_obj_points = cal_data[f"obj_points_{case}"]
    cal_img_points = cal_data[f"img_points_{case}"]
    
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
    
    
@pytest.mark.parametrize("case", (1, 2))
def test_projection_05(
    case: int
):
    params = get_test_camera_params(case)
    
    cal_data = np.load("./test_calibration_points.npz")
    
    cal_obj_points = cal_data[f"obj_points_{case}"]
    cal_img_points = cal_data[f"img_points_{case}"]
    
    X, Y, Z = calib_pinhole.project_to_z(
        params,
        cal_img_points,
        cal_obj_points[2]
    )
    
    assert_array_almost_equal(
        X, cal_obj_points[0],
        decimal=2
    )
    
    assert_array_almost_equal(
        Y, cal_obj_points[1],
        decimal=2
    )
    
    assert_array_almost_equal(
        Z, cal_obj_points[2],
        decimal=2
    )


@pytest.mark.parametrize("model", ("brown", "polynomial"))
def test_minimization_01(
    model: str
):        
    cal_data = np.load("./test_calibration_points.npz")
    
    cal_obj_points = cal_data["obj_points_1"]
    cal_img_points = cal_data["img_points_1"]
    
    params = calib_pinhole.get_cam_params(
        "minimized",
        resolution = [512, 512],
        translation = [1, 1, 520], # initial guess
        orientation = [0, np.pi, np.pi], # initial guess
        focal = [1000, 1000],
        distortion_model = model
    )
    
    params = calib_pinhole.minimize_params(
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


@pytest.mark.parametrize("model", ("brown", "polynomial"))
def test_minimization_02(
    model: str
):    
    case = 1
    params_orig = get_test_camera_params(case)
    
    cal_data = np.load("./test_calibration_points.npz")
    
    cal_obj_points = cal_data[f"obj_points_{case}"]
    cal_img_points = cal_data[f"img_points_{case}"]
    
    params_new = calib_pinhole.get_cam_params(
        "minimized",
        resolution = [512, 512],
        translation = [1, 1, 520], # initial guess
        orientation = [0, np.pi, np.pi], # initial guess
        focal = [1000, 1000],
        distortion_model = model
    )
    
    params_new = calib_pinhole.minimize_params(
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
        decimal = 1
    )
    
    assert_array_almost_equal(
        params_orig["orientation"], 
        params_new["orientation"],
        decimal = 1
    )
    
    assert_array_almost_equal(
        params_orig["rotation"], 
        params_new["rotation"],
        decimal = 1
    )
    
    assert_array_almost_equal(
        params_orig["focal"], 
        params_new["focal"],
        decimal = 1
    )
    
    assert_array_almost_equal(
        params_orig["principal"], 
        params_new["principal"],
        decimal = 1
    )

    
# TODO: Add more camera views
def test_line_intersect_01():
    case = (1, 2)
    cal_data = np.load("./test_calibration_points.npz")
    
    cal_obj_points_1 = cal_data[f"obj_points_{case[0]}"]
    cal_img_points_1 = cal_data[f"img_points_{case[1]}"]
    
    cal_obj_points_2 = cal_data[f"obj_points_{case[0]}"]
    cal_img_points_2 = cal_data[f"img_points_{case[1]}"]
    
    cam_1 = get_test_camera_params(case[0])
    
    cam_2 = get_test_camera_params(case[1])

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
    
    test_image_points_1 = calib_pinhole.project_points(
        cam_1,
        test_object_points
    )
    
    test_image_points_2 = calib_pinhole.project_points(
        cam_2,
        test_object_points
    )
    
    recon_obj_points, _ = calib_pinhole.line_intersect(
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
    cal_img_points_1 = cal_data[f"img_points_{case[1]}"]
    
    cal_obj_points_2 = cal_data[f"obj_points_{case[0]}"]
    cal_img_points_2 = cal_data[f"img_points_{case[1]}"]
    
    cam_1 = get_test_camera_params(case[0])
    
    cam_2 = get_test_camera_params(case[1])

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
    
    test_image_points_1 = calib_pinhole.project_points(
        cam_1,
        test_object_points
    )
    
    test_image_points_2 = calib_pinhole.project_points(
        cam_2,
        test_object_points
    )
    
    recon_obj_points = calib_pinhole.multi_line_intersect(
        [cam_1, cam_2],
        [test_image_points_1, test_image_points_2]
    )
    
    assert_array_almost_equal(
        test_object_points,
        recon_obj_points,
        decimal=2
    )
    
    
def test_save_parameters_1():
    params = get_test_camera_params()
        
    calib_pinhole.save_parameters(
        params,
        "."
    )


@pytest.mark.parametrize("default", (True, False))
def test_save_parameters_2(
    default: bool
):
    
    if default:
        params = calib_pinhole.get_cam_params(
            "dummy",
            resolution = [512, 512]
        )
    else:
        params = get_test_camera_params()
        
    calib_pinhole.save_parameters(
        params,
        ".", "saved_params"
    )

    
def test_load_parameters_1():
    with pytest.raises(FileNotFoundError):
        params_loaded = calib_pinhole.load_parameters(
            ".",
            "does not exist (hopefully)"
        )
    

def test_load_parameters_2():
    params_orig = get_test_camera_params()
        
    calib_pinhole.save_parameters(
        params_orig,
        ".",
        "dummy"
    )
    
    params_new = calib_pinhole.load_parameters(
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
        params_orig["translation"], 
        params_new["translation"]
    )
    
    assert_array_equal(
        params_orig["orientation"], 
        params_new["orientation"]
    )
    
    assert_array_equal(
        params_orig["rotation"], 
        params_new["rotation"]
    )
    
    assert_array_equal(
        params_orig["distortion_model"], 
        params_new["distortion_model"]
    )
    
    assert_array_equal(
        params_orig["distortion1"], 
        params_new["distortion1"]
    )
    
    assert_array_equal(
        params_orig["distortion2"], 
        params_new["distortion2"]
    )
    
    assert_array_equal(
        params_orig["focal"], 
        params_new["focal"]
    )
    
    assert_array_equal(
        params_orig["principal"], 
        params_new["principal"]
    )
    
    assert_array_equal(
        params_orig["dtype"], 
        params_new["dtype"]
    )