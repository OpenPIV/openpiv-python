"""Tests for the stereo-PIV example package (``openpiv.examples.stereo``)."""

import os

import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_array_almost_equal

from openpiv.examples.stereo import (
    StereoConfig,
    build_dewarp_maps,
    build_world_grid,
    common_marks,
    correct_disparity,
    davis,
    fit_cameras,
    make_camera,
    pipeline,
    reconstruct_3c,
    reprojection_report,
    run_2c_piv,
    triangulation_report,
)

file = os.path.join(os.path.dirname(__file__), "test_calibration_points.npz")

MARK_TABLE_XML = """<?xml version="1.0"?>
<MarkTable Version="2.0" DistanceMarkToMarkInMm="15" PlateThicknessMm="14.3"
           PlaneToPlaneDistanceMm="3" MarkSizeMm="3.2"
           CalibrationPlateIdentifier="204-15">
  <Camera CameraNumber="1" Width="2432" Height="2048">
    <View isFrontSide="true">
      <Mark>
        <WorldPos x="-90" y="30" z="-0"/>
        <RawPos x="294.379" y="173.073"/>
      </Mark>
      <Mark>
        <WorldPos x="-75" y="30" z="-0"/>
        <RawPos x="484.536" y="165.519"/>
      </Mark>
      <Mark>
        <WorldPos x="-90" y="15" z="-3"/>
        <RawPos x="300.100" y="200.500"/>
      </Mark>
    </View>
  </Camera>
  <Camera CameraNumber="2" Width="2432" Height="2048">
    <View isFrontSide="true">
      <Mark>
        <WorldPos x="-90" y="30" z="-0"/>
        <RawPos x="410.000" y="160.000"/>
      </Mark>
      <Mark>
        <WorldPos x="-75" y="30" z="-0"/>
        <RawPos x="600.000" y="155.000"/>
      </Mark>
    </View>
  </Camera>
</MarkTable>
"""

CALIBRATION_XML = """<?xml version="1.0"?>
<Calibration Version="2">
  <CoordinateSystemsForEachView CoordinateSystemIdentifier="0">
    <CoordinateSystem FieldOfView="SideBySideStereoVolume">
      <CoordinateMapper CameraIdentifier="1" Type="Polynomial3rdOrder">
        <CommonParameters>
          <FitError RMS="0.39570367680180918"/>
        </CommonParameters>
      </CoordinateMapper>
      <CoordinateMapper CameraIdentifier="2" Type="Polynomial3rdOrder">
        <CommonParameters>
          <FitError RMS="0.36118708410534223"/>
        </CommonParameters>
      </CoordinateMapper>
    </CoordinateSystem>
  </CoordinateSystemsForEachView>
</Calibration>
"""


@pytest.fixture
def mark_table(tmp_path):
    path = tmp_path / "MarkPositionTable.xml"
    path.write_text(MARK_TABLE_XML)
    return path


@pytest.fixture
def calibration_xml(tmp_path):
    path = tmp_path / "Calibration.xml"
    path.write_text(CALIBRATION_XML)
    return path


@pytest.fixture
def stereo_cams():
    """Two cameras fitted to the bundled synthetic calibration points."""
    data = np.load(file)
    marks = {
        1: davis.CameraMarks(1, data["obj_points_1"], data["img_points_1"]),
        2: davis.CameraMarks(2, data["obj_points_2"], data["img_points_2"]),
    }
    cams = fit_cameras("poly", (512, 512), marks, (1, 2))
    return cams, marks


# ----------------------------------------------------------------------
# DaVis XML readers
# ----------------------------------------------------------------------
def test_parse_mark_position_table(mark_table):
    marks = davis.parse_mark_position_table(mark_table)
    assert_(set(marks) == {1, 2})
    assert_(marks[1].count == 3)
    assert_(marks[2].count == 2)
    assert_allclose(marks[1].object_points[:, 0], [-90.0, 30.0, 0.0])
    assert_allclose(marks[1].image_points[:, 0], [294.379, 173.073])
    planes = marks[1].planes()
    assert_(set(planes) == {0.0, -3.0})
    assert_(planes[-3.0].tolist() == [2])


def test_parse_mark_position_table_missing_file(tmp_path):
    with pytest.raises(OSError):
        davis.parse_mark_position_table(tmp_path / "nope.xml")


def test_parse_calibration_rms(calibration_xml):
    rms = davis.parse_calibration_rms(calibration_xml)
    assert_(set(rms) == {1, 2})
    assert_allclose(rms[1], 0.39570367680180918)
    assert_allclose(rms[2], 0.36118708410534223)


def test_parse_plate_info(mark_table):
    info = davis.parse_plate_info(mark_table)
    assert_(info.identifier == "204-15")
    assert_allclose(info.mark_spacing_mm, 15.0)
    assert_allclose(info.plane_to_plane_distance_mm, 3.0)


def test_camera_marks_validates_shapes():
    with pytest.raises(ValueError):
        davis.CameraMarks(1, np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        davis.CameraMarks(1, np.zeros((3, 3)), np.zeros((2, 2)))


# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
def test_config_defaults(tmp_path):
    config = StereoConfig(data_root=tmp_path)
    assert_(config.cameras == (1, 2))
    assert_(config.sensor_size == (2432, 2048))
    assert_(config.sheet_z == 0.0)
    assert_(config.model == "poly")
    assert_(config.out_dir == tmp_path / "stereo_out")


def test_config_rejects_single_camera(tmp_path):
    with pytest.raises(ValueError):
        StereoConfig(data_root=tmp_path, cameras=(1,))
    with pytest.raises(ValueError):
        StereoConfig(data_root=tmp_path, model="nope")


def test_config_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENPIV_STEREO_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENPIV_STEREO_CAMERAS", "3,4")
    monkeypatch.setenv("OPENPIV_STEREO_SENSOR_SIZE", "1024,768")
    monkeypatch.setenv("OPENPIV_STEREO_SHEET_Z", "-3")
    monkeypatch.setenv("OPENPIV_STEREO_PAIR_INDEX", "7")
    config = StereoConfig.from_env()
    assert_(config.data_root == tmp_path)
    assert_(config.cameras == (3, 4))
    assert_(config.sensor_size == (1024, 768))
    assert_(config.sheet_z == -3.0)
    assert_(config.pair_index == 7)


def test_config_from_args(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENPIV_STEREO_DATA_ROOT", raising=False)
    config = StereoConfig.from_args(
        [
            "--data-root", str(tmp_path),
            "--cameras", "2,1",
            "--sensor-size", "640,480",
            "--sheet-z", "1.5",
            "--model", "dlt",
        ]
    )
    assert_(config.data_root == tmp_path)
    assert_(config.cameras == (2, 1))
    assert_(config.sensor_size == (640, 480))
    assert_(config.sheet_z == 1.5)
    assert_(config.model == "dlt")


def test_config_ensure_out(tmp_path):
    config = StereoConfig(data_root=tmp_path)
    out = config.ensure_out("previews")
    assert_(out.is_dir())
    assert_(out.name == "previews")


# ----------------------------------------------------------------------
# pipeline
# ----------------------------------------------------------------------
def test_make_camera_unknown_model():
    with pytest.raises(ValueError):
        make_camera("nope", "cam", (512, 512))


def test_fit_cameras_and_reprojection_report(stereo_cams):
    cams, marks = stereo_cams
    assert_(len(cams) == 2)
    report = reprojection_report(cams, marks, (1, 2))
    assert_(len(report) == 2)
    for entry in report:
        assert_(entry["rms"] < 1.0, f"rms {entry['rms']}")
        assert_(entry["marks"] == 605)


def test_fit_cameras_missing_marks(stereo_cams):
    _, marks = stereo_cams
    with pytest.raises(KeyError):
        fit_cameras("poly", (512, 512), marks, (1, 3))


def test_build_world_grid_and_dewarp_maps(stereo_cams):
    cams, _ = stereo_cams
    Xw, Yw, info = build_world_grid(cams, z=0.0)
    assert_(Xw.shape == Yw.shape)
    assert_(Xw.size > 0)
    assert_(info["scale"] > 0)
    maps = build_dewarp_maps(cams, Xw, Yw, z=0.0)
    assert_(set(maps) == {"mLx", "mLy", "mRx", "mRy"})
    for value in maps.values():
        assert_(value.shape == Xw.shape)
        assert_(np.all(np.isfinite(value)))


def test_rectify_is_inverse_of_projection(stereo_cams):
    cams, _ = stereo_cams
    Xw, Yw, _ = build_world_grid(cams, z=0.0)
    maps = build_dewarp_maps(cams, Xw, Yw, z=0.0)
    # pick a world node inside both sensors, project it and sample the
    # rectified image at the projected position
    X, Y = Xw[10, 10], Yw[10, 10]
    projected = cams[0].project_points(np.array([[X], [Y], [0.0]]))
    image = np.zeros((512, 512), dtype=np.float32)
    px, py = round(projected[0][0]), round(projected[1][0])
    if 0 <= px < 512 and 0 <= py < 512:
        image[py, px] = 1.0
        rect = pipeline.rectify(image, maps["mLx"], maps["mLy"])
        assert_(rect[10, 10] > 0.5)


def test_common_marks_and_triangulation_report(stereo_cams):
    cams, marks = stereo_cams
    world, image_first, image_second = common_marks(marks[1], marks[2])
    assert_(world.shape == (3, 605))
    assert_(image_first.shape == (2, 605))
    assert_(image_second.shape == (2, 605))
    report = triangulation_report(cams, marks[1], marks[2])
    assert_(report["count"] == 605)
    assert_(report["median"] < 0.5, f"triangulation median {report['median']}")
    assert_(report["planes"])


def test_run_2c_piv_recovers_shift():
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import shift as ndshift

    rng = np.random.default_rng(3)
    H, W = 256, 256
    image = np.zeros((H, W), dtype=np.float32)
    image[rng.integers(0, H, 400), rng.integers(0, W, 400)] = 255.0
    image = gaussian_filter(image, 1.0)
    moved = ndshift(image, shift=(1.0, 2.0), order=1, mode="constant")
    x, y, u, v, s2n = run_2c_piv(image, moved, window_size=32, overlap=16)
    assert_(u.shape == v.shape == s2n.shape)
    assert_(abs(np.nanmedian(u) - 2.0) < 0.7, f"u median {np.nanmedian(u)}")
    assert_(abs(np.nanmedian(v) - 1.0) < 0.7, f"v median {np.nanmedian(v)}")


def test_correct_disparity_reports_residual():
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import shift as ndshift

    rng = np.random.default_rng(4)
    H, W = 256, 256
    image = np.zeros((H, W), dtype=np.float32)
    image[rng.integers(0, H, 500), rng.integers(0, W, 500)] = 255.0
    image = gaussian_filter(image, 1.0)
    right = ndshift(image, shift=(1.0, 2.0), order=1, mode="constant")
    identity = {
        "mLx": np.tile(np.arange(W, dtype=float), (H, 1)),
        "mLy": np.tile(np.arange(H, dtype=float)[:, None], (1, W)),
        "mRx": np.tile(np.arange(W, dtype=float), (H, 1)),
        "mRy": np.tile(np.arange(H, dtype=float)[:, None], (1, W)),
    }
    maps, report = correct_disparity(
        [image, right], identity, disparity_passes=((64, 32), (32, 16)), iterations=1
    )
    assert_(set(maps) == {"mLx", "mLy", "mRx", "mRy"})
    assert_(report["residual_after"] <= report["residual_before"])
    assert_(report["bulk_shift"][0] != 0.0)


def test_reconstruct_3c_recovers_known_velocity(stereo_cams):
    """The 3C reconstruction must recover a known world velocity.

    The 2C fields are built analytically from the calibration: for every
    interrogation node the pixel displacement of a rigid world motion.
    """
    from openpiv import pyprocess

    cams, _ = stereo_cams
    window, overlap = 32, 16
    shape = tuple(int(s) for s in pyprocess.get_field_shape((512, 512), window, overlap))
    x, y = pyprocess.get_rect_coordinates((512, 512), window, overlap)
    velocity = (0.5, -0.25, 0.1)

    fields = []
    valid = None
    for cam in cams:
        world_x, world_y = pipeline.backproject_to_plane(cam, x.ravel(), y.ravel(), z=0.0)
        world = np.array([world_x, world_y, np.zeros_like(world_x)])
        jacobian = np.stack(
            [pipeline.projection_jacobian(cam, world[:, i]) for i in range(world.shape[1])]
        )
        displacement = np.einsum('nki,k->ni', jacobian, np.array(velocity))
        fields.append((displacement[:, 0].reshape(shape), displacement[:, 1].reshape(shape)))
        # the polynomial model is only valid inside the calibrated volume
        forward = np.asarray(cam.project_points(world), dtype=np.float64)
        node_valid = np.all(np.isfinite(forward), axis=0).reshape(shape)
        valid = node_valid if valid is None else (valid & node_valid)

    U, V, W = pipeline.reconstruct_3c(
        cams,
        [x, x],
        [y, y],
        [fields[0][0], fields[1][0]],
        [fields[0][1], fields[1][1]],
        z=0.0,
    )
    mask = valid.ravel() & np.isfinite(U) & np.isfinite(V) & np.isfinite(W)
    assert_(mask.mean() > 0.8, f"only {mask.mean():.2f} of the nodes are usable")
    assert_allclose(U[mask], velocity[0], atol=0.05)
    assert_allclose(V[mask], velocity[1], atol=0.05)
    assert_allclose(W[mask], velocity[2], atol=0.05)


# ----------------------------------------------------------------------
# command line interface
# ----------------------------------------------------------------------
def test_cli_requires_a_step(capsys):
    from openpiv.examples.stereo.__main__ import main

    assert_(main([]) == 2)


def test_cli_reports_missing_files(tmp_path, capsys):
    from openpiv.examples.stereo.__main__ import main

    code = main(["--data-root", str(tmp_path), "--cameras", "1,2", "info"])
    assert_(code == 1)
    assert_("error" in capsys.readouterr().err)
