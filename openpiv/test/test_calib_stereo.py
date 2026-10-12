"""RED tests for stereo-PIV 1-4: calib, dewarp, disparity, 3C reconstruction."""
import os
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_

file = os.path.join(os.path.dirname(__file__), "test_calibration_points.npz")


def _get_stereo_cams():
    from openpiv.calibration import poly_model
    cal = np.load(file)
    cams = []
    for case in (1, 2):
        obj = cal[f"obj_points_{case}"]
        img = cal[f"img_points_{case}"]
        cam = poly_model.camera("cam", resolution=[512, 512])
        cam.minimize_params(obj, img)
        cams.append(cam)
    return cams


def test_01_stereo_module_importable():
    # 1: stereo pair calibration helpers exist
    from openpiv.calibration import stereo
    for fn in (
        "get_common_world_grid",
        "get_dewarp_mapping",
        "dewarp_image",
        "estimate_disparity",
        "self_calibration",
        "reconstruct_3c",
        "triangulate",
    ):
        assert_(hasattr(stereo, fn), f"missing stereo.{fn}")


def test_02_common_grid_and_dewarp_roundtrip():
    # 2: dewarp to common world grid
    from openpiv.calibration import stereo
    cams = _get_stereo_cams()
    Xw, Yw, info = stereo.get_common_world_grid(cams, z=0.0)
    assert_(Xw.shape == Yw.shape)
    assert_(Xw.size > 0)
    # each cam must provide a finite mapping on that grid
    for cam in cams:
        map_x, map_y = stereo.get_dewarp_mapping(cam, Xw, Yw, z=0.0)
        assert_(np.all(np.isfinite(map_x)))
        assert_(np.all(np.isfinite(map_y)))
        # center of common grid must fall inside both sensors
        # (borders may legitimately extrapolate -> masked with cval)
        ny, nx = Xw.shape
        cx, cy = map_x[ny // 2, nx // 2], map_y[ny // 2, nx // 2]
        assert_(0 <= cx <= 512 and 0 <= cy <= 512, f"center maps to {cx},{cy}")
        inside = (
            (map_x >= 0) & (map_x <= 512) & (map_y >= 0) & (map_y <= 512)
        ).mean()
        assert_(inside > 0.5, f"only {inside:.2f} of common grid inside sensor")
    # roundtrip: world -> image -> world at z=0 within 1px reprojection
    x0, y0 = Xw.ravel()[:32], Yw.ravel()[:32]
    z0 = np.zeros_like(x0)
    for cam in cams:
        img = cam.project_points(np.array([x0, y0, z0]))
        back = cam.project_to_z(img, z0)
        assert_array_almost_equal(back[0], x0, decimal=1)
        assert_array_almost_equal(back[1], y0, decimal=1)


def test_03_disparity_estimation_recovers_known_shift():
    # 3: disparity correction after calibration
    from openpiv.calibration import stereo
    rng = np.random.default_rng(0)
    # synthetic particle-like dewarped pair with known integer shift
    H, W = 256, 256
    imgL = np.zeros((H, W), dtype=np.float32)
    ys = rng.integers(0, H, 400)
    xs = rng.integers(0, W, 400)
    imgL[ys, xs] = 255.0
    from scipy.ndimage import gaussian_filter
    imgL = gaussian_filter(imgL, 1.0)
    dx_true, dy_true = 3.0, -2.0
    from scipy.ndimage import shift as ndshift
    imgR = ndshift(imgL, shift=(dy_true, dx_true), order=1, mode="constant")
    x, y, dx, dy, _ = stereo.estimate_disparity(
        imgL, imgR, window_size=32, overlap=16
    )
    # median over valid vectors must recover shift to ~0.5px (nanmedian:
    # isolated interrogation windows may legitimately return nan)
    assert_(abs(np.nanmedian(dx) - dx_true) < 0.7, f"dx median {np.nanmedian(dx)}")
    assert_(abs(np.nanmedian(dy) - dy_true) < 0.7, f"dy median {np.nanmedian(dy)}")


def test_04_reconstruct_3c_recovers_known_displacement():
    # 4: higher-quality 3C result from two 2C views
    from openpiv.calibration import stereo
    cams = _get_stereo_cams()
    # known world points + known world displacement
    rng = np.random.default_rng(1)
    X = rng.uniform(-20, 20, 12)
    Y = rng.uniform(-20, 20, 12)
    Z = np.zeros_like(X)
    U_true = np.full_like(X, 2.0)
    V_true = np.full_like(X, -1.0)
    W_true = np.full_like(X, 0.5)
    dt = 1.0
    # forward-project to pixel displacements per camera
    u_list, v_list, x_list, y_list = [], [], [], []
    for cam in cams:
        img0 = cam.project_points(np.array([X, Y, Z]))
        img1 = cam.project_points(np.array([X + U_true * dt, Y + V_true * dt, Z + W_true * dt]))
        u_list.append(img1[0] - img0[0])
        v_list.append(img1[1] - img0[1])
        x_list.append(img0[0])
        y_list.append(img0[1])
    Xw = np.array([X, Y, Z])
    U, V, W = stereo.reconstruct_3c(cams, x_list, y_list, u_list, v_list, Xw, dt=dt)
    assert_array_almost_equal(U, U_true, decimal=1)
    assert_array_almost_equal(V, V_true, decimal=1)
    assert_array_almost_equal(W, W_true, decimal=1)


def test_05_self_calibration_reduces_residual():
    from openpiv.calibration import stereo
    cams = _get_stereo_cams()
    rng = np.random.default_rng(2)
    H, W = 256, 256
    img = np.zeros((H, W), dtype=np.float32)
    ys = rng.integers(0, H, 500)
    xs = rng.integers(0, W, 500)
    img[ys, xs] = 255.0
    from scipy.ndimage import gaussian_filter, shift as ndshift
    img = gaussian_filter(img, 1.0)
    # simulate mis-registered right camera by 2px
    imgL, imgR = img, ndshift(img, shift=(1.0, 2.0), order=1, mode="constant")
    out = stereo.self_calibration(
        cams, imgL, imgR, z=0.0, window_size=32, overlap=16, max_iter=2
    )
    assert_("residual_before" in out and "residual_after" in out)
    assert_(out["residual_after"] <= out["residual_before"])


def test_06_triangulate_and_image_mapping():
    from openpiv.calibration import stereo, calib_utils
    cams = _get_stereo_cams()
    # legacy single-cam mapping still works and is finite
    for cam in cams:
        m, scale = calib_utils.get_image_mapping(cam)
        assert_(m.shape == (2, 512, 512))
        assert_(np.all(np.isfinite(m)) and scale > 0)
    # generic triangulation recovers known world point
    Xtrue = np.array([[5.0], [3.0], [0.0]])
    imgs = [c.project_points(Xtrue) for c in cams]
    Xt = stereo.triangulate(cams, imgs)
    assert_array_almost_equal(Xt, Xtrue, decimal=3)
    errs = stereo.stereo_reprojection_error(cams, Xtrue, imgs)
    assert_(errs["triangulation_rmse"] < 1e-2)
