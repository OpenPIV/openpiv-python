import os
import pathlib
import tempfile
import numpy as np
import pytest
from imageio.v3 import imwrite

openpiv_rust = pytest.importorskip("openpiv_rust")

from openpiv import windef
from openpiv.settings import PIVSettings
from openpiv.test import test_process


def test_multigrid_windef_rust_vs_scipy_accuracy():
    """Verify that multigrid window deformation gives identical results with Rust vs SciPy backend."""
    frame_a, frame_b = test_process.create_pair(image_size=128)

    # 1. Run SciPy multigrid
    settings_scipy = PIVSettings()
    settings_scipy.windowsizes = (32, 16)
    settings_scipy.overlap = (16, 8)
    settings_scipy.num_iterations = 2
    settings_scipy.backend = "scipy"
    settings_scipy.sig2noise_validate = False
    settings_scipy.show_all_plots = False
    settings_scipy.show_plot = False

    x_s, y_s, u_s, v_s, flags_s = windef.multigrid_windef(frame_a, frame_b, settings_scipy)

    # 2. Run Rust multigrid
    settings_rust = PIVSettings()
    settings_rust.windowsizes = (32, 16)
    settings_rust.overlap = (16, 8)
    settings_rust.num_iterations = 2
    settings_rust.backend = "rust"
    settings_rust.sig2noise_validate = False
    settings_rust.show_all_plots = False
    settings_rust.show_plot = False

    x_r, y_r, u_r, v_r, flags_r = windef.multigrid_windef(frame_a, frame_b, settings_rust)

    # Validate grids and shapes
    assert np.array_equal(x_s, x_r)
    assert np.array_equal(y_s, y_r)
    assert u_s.shape == u_r.shape
    assert v_s.shape == v_r.shape

    # Displacements must match within subpixel interpolation tolerance
    diff_u = np.nanmax(np.abs(u_s - u_r))
    diff_v = np.nanmax(np.abs(v_s - v_r))
    assert diff_u < 1e-4, f"Displacement u difference too high: {diff_u}"
    assert diff_v < 1e-4, f"Displacement v difference too high: {diff_v}"
    assert np.array_equal(flags_s, flags_r)


def test_first_pass_and_multipass_deform_with_rust():
    """Verify individual first_pass and multipass_img_deform functions with backend='rust'."""
    frame_a, frame_b = test_process.create_pair(image_size=128)

    settings = PIVSettings()
    settings.windowsizes = (32, 16)
    settings.overlap = (16, 8)
    settings.num_iterations = 2
    settings.backend = "rust"
    settings.sig2noise_validate = False
    settings.show_all_plots = False
    settings.show_plot = False

    x, y, u, v, s2n = windef.first_pass(frame_a, frame_b, settings)
    assert np.allclose(u, test_process.SHIFT_U, atol=test_process.THRESHOLD)
    assert np.allclose(v, test_process.SHIFT_V, atol=test_process.THRESHOLD)

    u_m = np.ma.masked_array(u, mask=np.ma.nomask)
    v_m = np.ma.masked_array(v, mask=np.ma.nomask)

    x2, y2, u2, v2, grid_mask, flags = windef.multipass_img_deform(
        frame_a, frame_b, 1, x, y, u_m, v_m, settings
    )
    assert np.allclose(u2, test_process.SHIFT_U, atol=test_process.THRESHOLD)
    assert np.allclose(v2, test_process.SHIFT_V, atol=test_process.THRESHOLD)


def test_multiprocessing_piv_with_rust_backend():
    """Verify windef.piv with multiprocessing n_cpus=2 and backend='rust'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        out_dir = tmp_path / "output"
        out_dir.mkdir()

        frame_a, frame_b = test_process.create_pair(image_size=128)
        for i in range(2):
            imwrite(img_dir / f"test_{i:02d}_a.tif", frame_a.astype(np.uint8))
            imwrite(img_dir / f"test_{i:02d}_b.tif", frame_b.astype(np.uint8))

        settings = PIVSettings()
        settings.filepath_images = img_dir
        settings.save_path = out_dir
        settings.save_folder_suffix = "rust_test"
        settings.frame_pattern_a = "test_*_a.tif"
        settings.frame_pattern_b = "test_*_b.tif"
        settings.windowsizes = (32, 16)
        settings.overlap = (16, 8)
        settings.num_iterations = 2
        settings.backend = "rust"
        settings.n_cpus = 2
        settings.show_plot = False
        settings.save_plot = False
        settings.show_all_plots = False
        settings.sig2noise_validate = False

        windef.piv(settings)

        result_folders = list(out_dir.glob("OpenPIV_results_*"))
        assert len(result_folders) == 1
        txt_files = sorted(list(result_folders[0].glob("*.txt")))
        assert len(txt_files) == 2
        for f in txt_files:
            assert f.stat().st_size > 0
