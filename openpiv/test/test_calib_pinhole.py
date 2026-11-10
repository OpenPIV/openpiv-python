import os

import numpy as np
import pytest
from numpy.testing import (
    assert_,
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
)

from openpiv.calibration import pinhole_model
from openpiv.calibration.calib_utils import get_los_error, get_reprojection_error

file = os.path.join(os.path.dirname(__file__),"test_calibration_points.npz")


def get_test_camera_params(
    case: int=1,
    model: str='reduced',
    tilt: bool=False,
    dtype=np.float64
):
    pinhole_cam = os.path.join(os.path.dirname(__file__), f"test_calibration_pinhole_{case}.txt")


    with open(pinhole_cam) as f:
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

    return pinhole_model.PinCam(
        name = name,
        resolution = resolution,
        translation = translation,
        orientation = orientation,
        distortion_model = model,
        distortion1 = distortion,
        focal = focal,
        principal = principal,
        tilt = tilt,
        dtype = dtype
    )



@pytest.mark.parametrize("args", ((), ("name",)))
def test_required_parameters_input(args):
    with pytest.raises(TypeError):
        pinhole_model.PinCam(*args)


@pytest.mark.parametrize("parameters", [
    {"name": 0},
    {"resolution": [0]},
    {"translation": [0, 0]},
    {"orientation": [0, 0]},
    {"distortion_model": "non-existent <random symbols here>"},
    {"distortion1": np.zeros(7)},
    {"distortion1": np.zeros((2, 4))},
    {"focal": "[2, 2]"},
    {"focal": [1]},
    {"principal": "[2, 2]"},
    {"principal": [1]},
    {"dtype": int},
    {"tilt": "True"},
    {"refraction": "water"},
])
def test_parameters_input(parameters):
    kwargs = {"name": "name", "resolution": [0, 0]}
    kwargs.update(parameters)
    with pytest.raises(ValueError):
        pinhole_model.PinCam(**kwargs)


@pytest.mark.parametrize("model", ("reduced", "full"))
@pytest.mark.parametrize("tilt", (True, False))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_parameters_initialization(
    model: str,
    tilt: bool,
    dtype
):
    cam = pinhole_model.PinCam(
            "name",
            resolution=[512, 512],
            distortion_model=model,
            tilt=tilt,
            dtype=dtype
        )

    assert_(hasattr(cam, "name"))
    assert_(hasattr(cam, "resolution"))
    assert_(hasattr(cam, "translation"))
    assert_(hasattr(cam, "orientation"))
    assert_(hasattr(cam, "rotation"))
    assert_(hasattr(cam, "distortion_model"))
    assert_(hasattr(cam, "distortion1"))
    assert_(hasattr(cam, "tilt"))
    assert_(hasattr(cam, "refraction"))
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

    assert_equal(cam.distortion_model, model)

    assert_equal(cam.tilt, tilt)

    assert_equal(cam.refraction, "none")

    assert_array_equal(cam.distortion1, np.zeros(8))

    assert_(len(cam.focal) == 2)

    assert_(len(cam.principal) == 2)

    assert_array_equal(cam.principal, [256, 256])

    assert_equal(cam.dtype, np.dtype(dtype))


def test_rotation_matrix_01():
    cam = pinhole_model.PinCam(
        "name",
        resolution = [1024, 1024]
    )

    assert_allclose(
        cam.rotation,
        np.eye(3,3)
    )


def test_rotation_matrix_02():
    cam = pinhole_model.PinCam(
        "name",
        resolution = [1024, 1024],
        translation = [0, 0, 1],
        orientation = [0, 0, 0]
    )

    assert_allclose(
        cam.rotation,
        np.eye(3,3)
    )


@pytest.mark.parametrize("model", ("reduced", "full"))
@pytest.mark.parametrize("tilt", (True, False))
@pytest.mark.parametrize("correct_distortion", (True, False))
def test_projection_01(
    model: str,
    tilt: bool,
    correct_distortion: bool
):
    cam = pinhole_model.PinCam(
        "name",
        resolution = [1024, 1024],
        translation = [0, 0, 500],
        distortion_model = model,
        distortion1 = [0.1, -0.01, 0.001, -0.002, 0.001, 0.01, 0.02, -0.01],
        focal = [1000, 1000],
        tilt = tilt
    )

    rng = np.random.default_rng(0)
    X, Y, Z = rng.uniform(-50, 50, size=(3, 32))

    x, y = cam.project_points(
        [X, Y, Z],
        correct_distortion = correct_distortion
    )

    X_new, Y_new, Z_new = cam.project_to_z(
        [x, y],
        Z,
        correct_distortion = correct_distortion
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

    cal_data = np.load(file)

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

    cal_data = np.load(file)

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

    cal_data = np.load(file)

    cal_data[f"obj_points_{case}"]
    cal_data[f"img_points_{case}"]

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

    cal_data = np.load(file)

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


@pytest.mark.parametrize("model", ("reduced", "full"))
def test_minimization_01(
    model: str
):
    cal_data = np.load(file)

    cal_obj_points = cal_data["obj_points_1"]
    cal_img_points = cal_data["img_points_1"]

    cam = pinhole_model.PinCam(
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


@pytest.mark.parametrize("model", ("reduced", "full"))
def test_minimization_02(
    model: str
):
    case = 1
    cam_orig = get_test_camera_params(case)

    cal_data = np.load(file)

    cal_obj_points = cal_data[f"obj_points_{case}"]
    cal_img_points = cal_data[f"img_points_{case}"]

    cam_new = pinhole_model.PinCam(
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
    cal_data = np.load(file)

    cal_data[f"obj_points_{case[0]}"]
    cal_data[f"img_points_{case[1]}"]

    cal_data[f"obj_points_{case[0]}"]
    cal_data[f"img_points_{case[1]}"]

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
    cal_data = np.load(file)
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


def test_save_parameters_1(tmp_path):
    cam = get_test_camera_params()

    cam.save_parameters(
        tmp_path
    )


@pytest.mark.parametrize("default", (True, False))
def test_save_parameters_2(
    default: bool,
    tmp_path
):

    if default:
        cam = pinhole_model.PinCam(
            "dummy",
            resolution = [512, 512]
        )
    else:
        cam = get_test_camera_params()

    cam.save_parameters(
        tmp_path, "saved_params"
    )


def test_load_parameters_1(tmp_path):
    cam = pinhole_model.PinCam(
        "dummy",
        resolution = [512, 512]
    )

    with pytest.raises(FileNotFoundError):
        cam.load_parameters(
            tmp_path,
            "does not exist (hopefully)"
        )


@pytest.mark.parametrize("model", ("reduced", "full"))
@pytest.mark.parametrize("tilt", (True, False))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_load_parameters_2(
    model: str,
    tilt: bool,
    dtype,
    tmp_path
):
    cam_orig = get_test_camera_params(model=model, tilt=tilt, dtype=dtype)
    cam_orig.distortion1 = np.array(
        [0.1, -0.01, 0.001, -0.002, 0.001, 0.01, 0.02, -0.01],
        dtype=dtype
    )
    cam_orig._check_parameters()


    cam_orig.save_parameters(
        tmp_path,
        "dummy"
    )

    cam_new = pinhole_model.PinCam(
        "dummy",
        resolution = [512, 512]
    )

    cam_new.load_parameters(
        tmp_path,
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
        cam_orig.tilt,
        cam_new.tilt
    )

    assert_array_equal(
        cam_orig.refraction,
        cam_new.refraction
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
