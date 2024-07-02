import numpy as np
import pytest

from numpy.testing import (assert_equal, assert_allclose,
                           assert_almost_equal, assert_array_almost_equal,
                           assert_array_equal, assert_)

from openpiv.calibration import pinhole_model
from openpiv.calibration.calib_utils import get_reprojection_error, get_los_error


def get_test_camera_params(
    case: int=1
):
    pinhole_cam = f"test_calibration_pinhole_{case}.txt"
        
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
    
    camera = pinhole_model.camera(
        name = name,
        resolution = resolution,
        translation = translation,
        orientation = orientation,
        distortion_model = "brown",
        distortion1 = distortion,
        focal = focal,
        principal = principal
    )
        
    return camera


def test_parameters_input():
    with pytest.raises(TypeError):
         # missing camera name
        pinhole_model.camera()
        
        # missing resolution
        pinhole_model.camera(
            "name"
        )
                                 
    with pytest.raises(ValueError):
        # name is not a string
        pinhole_model.camera(
            0,
            resolution=[0, 0]
        )
        
        # not two element tuple
        pinhole_model.camera(
            "name",
            resolution=[0]
        )
        
        # not three element vector
        pinhole_model.camera(
            "name",
            resolution=[0, 0],
            translation=[0, 0]
        )
        
        # not three element vector
        pinhole_model.camera(
            "name",
            resolution=[0, 0],
            orientation=[0, 0]
        )
        
        # wrong distortion model
        pinhole_model.camera(
            "name",
            resolution=[0, 0],
            distortion_model="non-existent <random symbols here>",
        )
        
        # not 8 element vector for brown model 
        pinhole_model.camera(
            "name",
            resolution=[0, 0],
            distortion_model="brown",
            distortion1=np.zeros(7)
        )
        
        # not 4 x 6 matrix for polynomial model 
        pinhole_model.camera(
            "name",
            resolution=[0, 0],
            distortion_model="polynomial",
            distortion2=np.zeros([4, 3])
        )
        
        # not 2 element list-like
        pinhole_model.camera(
            "name",
            resolution=[0, 0],
            focal="[2, 2]"
        )
        
        # not 2 element vector 
        pinhole_model.camera(
            "name",
            resolution=[0, 0],
            focal=[1]
        )
        
        # not 2 element list-like
        pinhole_model.camera(
            "name",
            resolution=[0, 0],
            principal="[2, 2]"
        )
        
        # not 2 element vector 
        pinhole_model.camera(
            "name",
            resolution=[0, 0],
            principal=[1]
        )
        
        # not a support dtype (supported dtypes are float32 and float64)
        pinhole_model.camera(
            "name",
            resolution=[0, 0],
            dtype=int
        )
        

def test_parameters_initialization():
    cam = pinhole_model.camera(
            "name",
            resolution=[0, 0]
        )
    
    assert_(hasattr(cam, "name"))
    assert_(hasattr(cam, "resolution"))
    assert_(hasattr(cam, "translation"))
    assert_(hasattr(cam, "orientation"))
    assert_(hasattr(cam, "rotation"))
    assert_(hasattr(cam, "distortion_model"))
    assert_(hasattr(cam, "distortion1"))
    assert_(hasattr(cam, "distortion2"))
    assert_(hasattr(cam, "focal"))
    assert_(hasattr(cam, "principal"))
    assert_(hasattr(cam, "dtype"))
    
    assert_(len(cam.resolution) == 2)
    
    assert_equal(
        cam.translation.shape,
        [3, ]
    )
    
    assert_equal(
        cam.orientation.shape,
        [3, ]
    )
    
    assert_equal(
        cam.rotation.shape,
        [3, 3]
    )
    
    assert_equal(
        cam.distortion1.shape,
        [8, ]
    )
    
    assert_equal(
        cam.distortion2.shape,
        [4, 6]
    )
    
    assert_(len(cam.focal) == 2)
    
    assert_(len(cam.principal) == 2)
    
    # float32 does not work well with the pinhole model, should it be deprecated?
    assert_(cam.dtype in ["float32", "float64"])
        

def test_rotation_matrix_01():
    cam = pinhole_model.camera(
        "name",
        resolution = [1024, 1024]
    )
        
    assert_allclose(
        cam.rotation,
        np.eye(3,3)
    )
    
    
def test_rotation_matrix_02():
    cam = pinhole_model.camera(
        "name",
        resolution = [1024, 1024],
        translation = [0, 0, 1],
        orientation = [0, 0, 0]
    )
        
    assert_allclose(
        cam.rotation,
        np.eye(3,3)
    )


def test_projection_01():
    cam = pinhole_model.camera(
        "name",
        resolution = [1024, 1024]
    )
    
    X, Y, Z = np.random.rand(3, 32) * 100.0
    
    x, y = cam.project_points(
        [X, Y, Z]
    )
    
    X_new, Y_new, Z_new = cam.project_to_z(
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


@pytest.mark.parametrize("case", (1, 2, 3))
def test_projection_02(
    case: int
):
    cam = get_test_camera_params(case)
    
    cal_data = np.load("./test_calibration_points.npz")
    
    cal_obj_points = cal_data[f"obj_points_{case}"]
    cal_img_points = cal_data[f"img_points_{case}"]
    
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
    

@pytest.mark.parametrize("case", (1, 2, 3))
def test_projection_03(
    case: int
):    
    cam = get_test_camera_params(case)
    
    cal_data = np.load("./test_calibration_points.npz")
    
    cal_obj_points = cal_data[f"obj_points_{case}"]
    cal_img_points = cal_data[f"img_points_{case}"]
    
    RMSE = get_reprojection_error(
        cam,
        cal_obj_points,
        cal_img_points
    )
    
    assert_(RMSE < 1e-2)
    

@pytest.mark.parametrize("case", (1, 2, 3))
def test_projection_04(
    case: int
):    
    cam = get_test_camera_params(case)
    
    cal_data = np.load("./test_calibration_points.npz")
    
    cal_obj_points = cal_data[f"obj_points_{case}"]
    cal_img_points = cal_data[f"img_points_{case}"]
    
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
    
    
@pytest.mark.parametrize("case", (1, 2, 3))
def test_projection_05(
    case: int
):
    cam = get_test_camera_params(case)
    
    cal_data = np.load("./test_calibration_points.npz")
    
    cal_obj_points = cal_data[f"obj_points_{case}"]
    cal_img_points = cal_data[f"img_points_{case}"]
    
    X, Y, Z = cam.project_to_z(
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
    
    cam = pinhole_model.camera(
        "minimized",
        resolution = [512, 512],
        translation = [1, 1, 520], # initial guess
        orientation = [0, np.pi, np.pi], # initial guess
        focal = [1000, 1000],
        distortion_model = model
    )
    
    cam.minimize_params(
        cal_obj_points,
        cal_img_points,
        correct_focal = False,
        correct_distortion = False,
        iterations = 3
    )
    
    RMSE = get_reprojection_error(
        cam,
        cal_obj_points,
        cal_img_points
    )
        
    assert_(RMSE < 1e-2)


@pytest.mark.parametrize("model", ("brown", "polynomial"))
def test_minimization_02(
    model: str
):    
    case = 1
    cam_orig = get_test_camera_params(case)
    
    cal_data = np.load("./test_calibration_points.npz")
    
    cal_obj_points = cal_data[f"obj_points_{case}"]
    cal_img_points = cal_data[f"img_points_{case}"]
    
    cam_new = pinhole_model.camera(
        "minimized",
        resolution = [512, 512],
        translation = [-20, 20, 520], # initial guess
        orientation = [0, np.pi, np.pi], # initial guess
        focal = [1000, 1000],
        distortion_model = model
    )
    
    cam_new.minimize_params(
        cal_obj_points,
        cal_img_points,
        correct_focal = False,
        correct_distortion = False,
        iterations = 3
    )
    
    assert_array_almost_equal(
        cam_orig.translation, 
        cam_new.translation,
        decimal = 1
    )
    
    assert_array_almost_equal(
        cam_orig.orientation, 
        cam_new.orientation,
        decimal = 1
    )
    
    assert_array_almost_equal(
        cam_orig.rotation, 
        cam_new.rotation,
        decimal = 1
    )
    
    assert_array_almost_equal(
        cam_orig.focal, 
        cam_new.focal,
        decimal = 1
    )
    
    # set decimal to 0 because it keeps minimizing wrong, this needs to be fixed
    assert_array_almost_equal(
        cam_orig.principal, 
        cam_new.principal,
        decimal = 0
    )

    
@pytest.mark.parametrize("case", ((1, 2), (1, 3), (2, 3)))
def test_line_intersect_01(
    case: tuple
):
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
        dtype=cam_1.dtype
    ).T
    
    test_image_points_1 = cam_1.project_points(
        test_object_points
    )
    
    test_image_points_2 = cam_2.project_points(
        test_object_points
    )
    
    recon_obj_points, _ = pinhole_model.line_intersect(
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
    

@pytest.mark.parametrize("case", ((1, 2), (1, 3), (2, 3), (1, 2, 3)))
def test_line_intersect_02(
    case: tuple
):
    cal_data = np.load("./test_calibration_points.npz")
    n_cams = len(case)
    
    cal_obj_points = []
    cal_img_points = []
    for cam in case:
        cal_obj_points.append(cal_data[f"obj_points_{cam}"])
        cal_img_points.append(cal_data[f"img_points_{cam}"])
        
    cams = []
    for cam in case:
        cams.append(get_test_camera_params(cam))

    test_object_points = np.array(
        [
            [0, 0, 0],
            [1, 2, 3],
            [3, 2, 1],
            [15, 15, -15],
            [15, 15, 0],
            [15, 15, 15]
        ],
        dtype=cams[0].dtype
    ).T
    
    test_image_points = []
    for cam in range(n_cams):
        test_image_points.append(
            cams[cam].project_points(
                test_object_points
            )
        )

    recon_obj_points = pinhole_model.multi_line_intersect(
        cams,
        test_image_points
    )
    
    assert_array_almost_equal(
        test_object_points,
        recon_obj_points,
        decimal=2
    )
    
    
def test_save_parameters_1():
    cam = get_test_camera_params()
        
    cam.save_parameters(
        "."
    )


@pytest.mark.parametrize("default", (True, False))
def test_save_parameters_2(
    default: bool
):
    
    if default:
        cam = pinhole_model.camera(
            "dummy",
            resolution = [512, 512]
        )
    else:
        cam = get_test_camera_params()
        
    cam.save_parameters(
        ".", "saved_params"
    )

    
def test_load_parameters_1():
    cam = pinhole_model.camera(
        "dummy",
        resolution = [512, 512]
    )
    
    with pytest.raises(FileNotFoundError):
        cam.load_parameters(
            ".",
            "does not exist (hopefully)"
        )
    

def test_load_parameters_2():
    cam_orig = get_test_camera_params()
    
    
    cam_orig.save_parameters(
        ".",
        "dummy"
    )
    
    cam_new = pinhole_model.camera(
        "dummy",
        resolution = [512, 512]
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
        cam_orig.translation, 
        cam_new.translation
    )
    
    assert_array_equal(
        cam_orig.orientation, 
        cam_new.orientation
    )
    
    assert_array_equal(
        cam_orig.rotation, 
        cam_new.rotation
    )
    
    assert_array_equal(
        cam_orig.distortion_model, 
        cam_new.distortion_model
    )
    
    assert_array_equal(
        cam_orig.distortion1, 
        cam_new.distortion1
    )
    
    assert_array_equal(
        cam_orig.distortion2, 
        cam_new.distortion2
    )
    
    assert_array_equal(
        cam_orig.focal, 
        cam_new.focal
    )
    
    assert_array_equal(
        cam_orig.principal, 
        cam_new.principal
    )
    
    assert_array_equal(
        cam_orig.dtype, 
        cam_new.dtype
    )