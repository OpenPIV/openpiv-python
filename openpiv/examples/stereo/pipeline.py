"""The stereo-PIV pipeline.

The pipeline is a thin, well documented layer on top of
:mod:`openpiv.calibration.stereo` and :mod:`openpiv.pyprocess`.  It is
deliberately split into small functions that only depend on plain arrays,
so that every step can be tested without the raw camera data:

1. :func:`fit_cameras` -- fit the stereo pair to the DaVis marks;
2. :func:`build_world_grid` + :func:`build_dewarp_maps` -- rectify both
   cameras onto a common world grid at the light-sheet plane;
3. :func:`correct_disparity` -- estimate the particle disparity and
   apply the (Wieneke style) self-calibration correction;
4. :func:`run_2c_piv` + :func:`reconstruct_3c` -- 2C PIV per camera and
   the 3C reconstruction of the out-of-plane velocity component.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from openpiv.calibration import stereo

from .davis import CameraMarks

__all__ = [
    "backproject_to_plane",
    "build_dewarp_maps",
    "build_world_grid",
    "common_marks",
    "correct_disparity",
    "fit_cameras",
    "make_camera",
    "projection_jacobian",
    "reconstruct_3c",
    "rectify",
    "reprojection_report",
    "run_2c_piv",
    "sample_velocity_field",
    "triangulation_report",
]

#: Names of the four dewarp maps of a stereo pair.
MAP_NAMES = ("mLx", "mLy", "mRx", "mRy")


def make_camera(model: str, name: str, resolution: tuple[int, int]) -> Any:
    """Create an empty camera of the requested model.

    Parameters
    ----------
    model:
        ``"poly"`` (Soloff), ``"dlt"`` or ``"pinhole"``.
    name:
        Name of the camera instance.
    resolution:
        ``(nx, ny)`` sensor size in pixels.
    """
    if model == "poly":
        from openpiv.calibration import poly_model

        return poly_model.camera(name, resolution=resolution)
    if model == "dlt":
        from openpiv.calibration import dlt_model

        return dlt_model.camera(name, resolution=resolution)
    if model == "pinhole":
        from openpiv.calibration import pinhole_model

        return pinhole_model.camera(name, resolution=resolution)
    raise ValueError(f"unknown camera model {model!r}")


def fit_cameras(
    model: str,
    resolution: tuple[int, int],
    marks: Mapping[int, CameraMarks],
    cameras: Sequence[int],
) -> list[Any]:
    """Fit one camera per stereo-pair member to its calibration marks.

    Parameters
    ----------
    model:
        Camera model, see :func:`make_camera`.
    resolution:
        ``(nx, ny)`` sensor size in pixels.
    marks:
        ``camera number -> CameraMarks``.
    cameras:
        The stereo pair, e.g. ``(1, 2)``.

    Returns
    -------
    list
        The fitted cameras, in the order of ``cameras``.
    """
    fitted = []
    for number in cameras:
        if number not in marks:
            raise KeyError(f"no calibration marks for camera {number}")
        camera_marks = marks[number]
        cam = make_camera(model, f"cam{number}", resolution)
        cam.minimize_params(camera_marks.object_points, camera_marks.image_points)
        fitted.append(cam)
    return fitted


def reprojection_report(
    cams: Sequence[Any], marks: Mapping[int, CameraMarks], cameras: Sequence[int]
) -> list[dict]:
    """Reprojection statistics of the fitted cameras.

    Each entry holds the RMS error, the median/p95/max residual and the
    number of marks, so that an OpenPIV fit can be compared with the
    DaVis reference.
    """
    from openpiv.calibration import calib_utils

    report = []
    for number, cam in zip(cameras, cams):
        camera_marks = marks[number]
        rms = float(calib_utils.get_reprojection_error(cam, camera_marks.object_points, camera_marks.image_points))
        projected = cam.project_points(camera_marks.object_points)
        residual = np.hypot(
            np.asarray(projected[0]) - camera_marks.image_points[0],
            np.asarray(projected[1]) - camera_marks.image_points[1],
        )
        report.append(
            {
                "camera": number,
                "rms": rms,
                "median": float(np.median(residual)),
                "p95": float(np.percentile(residual, 95)),
                "max": float(residual.max()),
                "marks": camera_marks.count,
            }
        )
    return report


def build_world_grid(
    cams: Sequence[Any], z: float = 0.0, stride: int = 4
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Common world grid of the stereo pair at the light-sheet plane.

    Returns
    -------
    (Xw, Yw, info)
        World coordinates of the grid nodes (mm) and a dictionary with the
        grid bounds, the mm-per-pixel scale and the plane.
    """
    return stereo.get_common_world_grid(list(cams), z=float(z), stride=int(stride))


def build_dewarp_maps(
    cams: Sequence[Any], Xw: np.ndarray, Yw: np.ndarray, z: float = 0.0
) -> dict[str, np.ndarray]:
    """Dewarp mappings of the stereo pair for every node of the world grid.

    Returns
    -------
    dict
        ``{"mLx": ..., "mLy": ..., "mRx": ..., "mRy": ...}`` -- the image
        coordinates of every world node, per camera.
    """
    maps: dict[str, np.ndarray] = {}
    for prefix, cam in zip(("mL", "mR"), cams):
        map_x, map_y = stereo.get_dewarp_mapping(cam, Xw, Yw, z=float(z))
        maps[f"{prefix}x"] = map_x
        maps[f"{prefix}y"] = map_y
    return maps


def rectify(frame: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Rectify one camera frame onto the world grid."""
    return stereo.dewarp_image(np.asarray(frame, dtype=np.float32), map_x, map_y)


def _rectified_pair(
    frames: Sequence[np.ndarray], maps: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    left = rectify(frames[0], maps["mLx"], maps["mLy"])
    right = rectify(frames[1], maps["mRx"], maps["mRy"])
    return left, right


def _median_residual(dx: np.ndarray, dy: np.ndarray) -> float:
    mask = np.isfinite(dx) & np.isfinite(dy)
    if not np.any(mask):
        return float("inf")
    return float(np.median(np.hypot(dx[mask], dy[mask])))


def correct_disparity(
    frames: Sequence[np.ndarray],
    maps: Mapping[str, np.ndarray],
    disparity_passes: Sequence[tuple[int, int]] = ((128, 64), (64, 32)),
    iterations: int = 3,
) -> tuple[dict[str, np.ndarray], dict]:
    """Estimate the particle disparity and self-calibrate the mappings.

    The disparity of the dewarped pair is measured with the finest pass
    (this is the reference residual) and with a coarse pass, whose median
    gives a robust bulk misregistration.  The bulk part is corrected
    first, then ``iterations`` symmetric half-corrections are applied on
    the finest pass.  The map *fields* are resampled, never shifted by a
    constant.

    Parameters
    ----------
    frames:
        The two raw camera frames of the same instant.
    maps:
        The dewarp mappings, see :func:`build_dewarp_maps`.
    disparity_passes:
        ``(window, overlap)`` pairs, coarse to fine.
    iterations:
        Number of refinement iterations on the finest pass.

    Returns
    -------
    (maps, report)
        The corrected mappings and a report with the residuals before and
        after the correction.
    """
    current = {key: np.asarray(value, dtype=np.float64) for key, value in maps.items()}
    fine_window, fine_overlap = disparity_passes[-1]

    # reference residual: measured with the finest pass, before any correction
    left, right = _rectified_pair(frames, current)
    x, y, dx, dy, s2n = stereo.estimate_disparity(
        left, right, window_size=fine_window, overlap=fine_overlap
    )
    residual_before = _median_residual(dx, dy)

    # a coarse pass gives a robust bulk shift (larger window, more particles)
    coarse_window, coarse_overlap = disparity_passes[0]
    left, right = _rectified_pair(frames, current)
    x_coarse, y_coarse, dx_coarse, dy_coarse, _ = stereo.estimate_disparity(
        left, right, window_size=coarse_window, overlap=coarse_overlap
    )
    bulk_x = float(np.nanmedian(dx_coarse))
    bulk_y = float(np.nanmedian(dy_coarse))
    current = dict(
        zip(
            MAP_NAMES,
            stereo.correct_maps(
                (current["mLx"], current["mLy"], current["mRx"], current["mRy"]),
                x_coarse,
                y_coarse,
                np.full_like(dx_coarse, bulk_x),
                np.full_like(dy_coarse, bulk_y),
                factor=0.5,
            ),
        )
    )

    history = []
    for iteration in range(max(0, iterations)):
        left, right = _rectified_pair(frames, current)
        x, y, dx, dy, s2n = stereo.estimate_disparity(
            left, right, window_size=fine_window, overlap=fine_overlap
        )
        history.append({"iteration": iteration, "residual": _median_residual(dx, dy)})
        current = dict(
            zip(
                MAP_NAMES,
                stereo.correct_maps(
                    (current["mLx"], current["mLy"], current["mRx"], current["mRy"]),
                    x,
                    y,
                    dx,
                    dy,
                    factor=0.5,
                ),
            )
        )

    left, right = _rectified_pair(frames, current)
    x, y, dx, dy, s2n = stereo.estimate_disparity(
        left, right, window_size=fine_window, overlap=fine_overlap
    )
    report = {
        "residual_before": residual_before,
        "residual_after": _median_residual(dx, dy),
        "bulk_shift": (bulk_x, bulk_y),
        "history": history,
        "x": x,
        "y": y,
        "dx": dx,
        "dy": dy,
        "s2n": s2n,
    }
    return current, report


def run_2c_piv(
    frame_a: np.ndarray, frame_b: np.ndarray, window_size: int = 64, overlap: int = 32
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """2C PIV of a rectified frame pair.

    Returns
    -------
    (x, y, u, v, s2n)
        The interrogation grid and the measured displacements (pixels) and
        signal-to-noise ratios.
    """
    from openpiv import pyprocess

    image_a = np.ascontiguousarray(frame_a, dtype=np.float32)
    image_b = np.ascontiguousarray(frame_b, dtype=np.float32)
    u, v, s2n = pyprocess.extended_search_area_piv(
        image_a,
        image_b,
        window_size=int(window_size),
        overlap=int(overlap),
        search_area_size=int(window_size),
    )
    shape = tuple(int(s) for s in pyprocess.get_field_shape(image_a.shape, int(window_size), int(overlap)))
    x, y = pyprocess.get_rect_coordinates(image_a.shape, int(window_size), int(overlap))
    return (
        x,
        y,
        np.asarray(u).reshape(shape),
        np.asarray(v).reshape(shape),
        np.asarray(s2n).reshape(shape),
    )


def backproject_to_plane(
    cam: Any, x: np.ndarray, y: np.ndarray, z: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """World position of the image pixels ``(x, y)`` on the plane ``z``.

    Returns
    -------
    (X, Y)
        The world coordinates (mm) of the back-projected pixels.
    """
    pixels = np.asarray(x, dtype=np.float64).ravel()
    rows = np.asarray(y, dtype=np.float64).ravel()
    world_x, world_y, _ = cam.project_to_z(
        [pixels, rows], np.full(pixels.size, float(z), dtype=np.float64)
    )
    return np.asarray(world_x, dtype=np.float64), np.asarray(world_y, dtype=np.float64)


def projection_jacobian(cam: Any, world: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Derivative ``d(image)/d(world)`` of a camera, shape ``(3, 2)``.

    Computed with central differences; the result is the linearisation
    used to turn a world velocity into a pixel displacement.
    """
    world = np.asarray(world, dtype=np.float64).ravel()
    jacobian = np.zeros((3, 2))
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = eps * (1.0 + abs(world[axis]))
        forward = np.asarray(cam.project_points((world + step)[:, None]), dtype=np.float64)
        backward = np.asarray(cam.project_points((world - step)[:, None]), dtype=np.float64)
        jacobian[axis] = ((forward[0, 0] - backward[0, 0]) / (2 * step[axis]),
                          (forward[1, 0] - backward[1, 0]) / (2 * step[axis]))
    return jacobian


def sample_velocity_field(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a PIV velocity field at arbitrary image positions.

    Parameters
    ----------
    x, y:
        The (2D) interrogation grid of the field.
    u, v:
        The velocity components on that grid.
    points:
        ``(n, 2)`` array of ``(x, y)`` image positions.

    Returns
    -------
    (u_at_points, v_at_points)
        The interpolated velocity components.
    """
    from scipy.interpolate import RegularGridInterpolator

    x_axis = np.asarray(x)[0, :]
    y_axis = np.asarray(y)[:, 0]
    interpolator_u = RegularGridInterpolator(
        (y_axis, x_axis), np.asarray(u, dtype=np.float64), bounds_error=False, fill_value=np.nan
    )
    interpolator_v = RegularGridInterpolator(
        (y_axis, x_axis), np.asarray(v, dtype=np.float64), bounds_error=False, fill_value=np.nan
    )
    sample = np.column_stack([np.asarray(points)[:, 1], np.asarray(points)[:, 0]])
    return interpolator_u(sample), interpolator_v(sample)


def reconstruct_3c(
    cams: Sequence[Any],
    x_list: Sequence[np.ndarray],
    y_list: Sequence[np.ndarray],
    u_list: Sequence[np.ndarray],
    v_list: Sequence[np.ndarray],
    z: float = 0.0,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine the 2C views of a stereo pair into the 3C velocity field.

    The 2C PIV is run on the *raw* camera images, so ``u`` and ``v`` are
    pixel displacements in the camera frame.  The particles of the second
    camera are found by projecting the world position of every particle of
    the first camera (back-projected onto the plane ``z``) into the second
    camera and sampling its velocity field there.

    Parameters
    ----------
    cams:
        The stereo pair.
    x_list, y_list:
        Interrogation grids of the two cameras (raw image pixels).
    u_list, v_list:
        The measured pixel displacements of the two cameras.
    z:
        Light-sheet plane in world coordinates (mm).
    dt:
        Time between the two frames.

    Returns
    -------
    (U, V, W)
        The world velocity components (mm per time unit) on the grid of
        the first camera.  Nodes whose counterpart in another camera falls
        outside the interrogation grid are returned as ``nan``.
    """
    if not (len(cams) == len(x_list) == len(y_list) == len(u_list) == len(v_list)):
        raise ValueError("cams, x, y, u and v must have the same length")
    world_x, world_y = backproject_to_plane(cams[0], x_list[0], y_list[0], z=z)
    world = np.array([world_x, world_y, np.full(world_x.size, float(z))])

    # the same particle, seen by the other cameras
    matched_u: list[np.ndarray] = [np.asarray(u_list[0], dtype=np.float64).ravel()]
    matched_v: list[np.ndarray] = [np.asarray(v_list[0], dtype=np.float64).ravel()]
    valid = np.isfinite(matched_u[0]) & np.isfinite(matched_v[0])
    for index in range(1, len(cams)):
        projected = cams[index].project_points(world)
        points = np.column_stack(
            [np.asarray(projected[0], dtype=np.float64), np.asarray(projected[1], dtype=np.float64)]
        )
        sampled_u, sampled_v = sample_velocity_field(
            x_list[index], y_list[index], u_list[index], v_list[index], points
        )
        matched_u.append(sampled_u)
        matched_v.append(sampled_v)
        valid &= np.isfinite(sampled_u) & np.isfinite(sampled_v)

    u, v, w = stereo.reconstruct_3c(
        list(cams),
        list(x_list),
        list(y_list),
        matched_u,
        matched_v,
        world,
        dt=float(dt),
    )
    return (
        np.where(valid, u, np.nan),
        np.where(valid, v, np.nan),
        np.where(valid, w, np.nan),
    )


def common_marks(
    first: CameraMarks, second: CameraMarks
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Marks seen by both cameras, matched by their world position.

    Returns
    -------
    (world, image_first, image_second)
        ``(3, n)`` world positions and the two ``(2, n)`` image positions.
    """
    lookup = {
        (round(float(x), 6), round(float(y), 6), round(float(z), 6)): index
        for index, (x, y, z) in enumerate(first.object_points.T)
    }
    shared: list[tuple[int, int]] = []
    for index, (x, y, z) in enumerate(second.object_points.T):
        key = (round(float(x), 6), round(float(y), 6), round(float(z), 6))
        if key in lookup:
            shared.append((lookup[key], index))
    if not shared:
        raise ValueError("the two cameras have no marks in common")
    first_idx = np.array([i for i, _ in shared], dtype=int)
    second_idx = np.array([j for _, j in shared], dtype=int)
    world = first.object_points[:, first_idx]
    return world, first.image_points[:, first_idx], second.image_points[:, second_idx]


def triangulation_report(
    cams: Sequence[Any], first: CameraMarks, second: CameraMarks
) -> dict:
    """Triangulate the common marks and report the 3D error.

    This is the decisive stereo check: the marks are known in world
    coordinates, so the deviation of the triangulated positions from them
    quantifies the stereo geometry.

    Returns
    -------
    dict
        ``median``, ``p95`` and ``max`` error in mm plus the per-plane
        breakdown and the number of common marks.
    """
    world, image_first, image_second = common_marks(first, second)
    triangulated = stereo.triangulate(list(cams), [image_first, image_second])
    error = np.linalg.norm(triangulated - world, axis=0)
    planes = {}
    for z_value in np.unique(world[2]):
        mask = np.isclose(world[2], z_value)
        planes[f"z={float(z_value):g}"] = {
            "count": int(mask.sum()),
            "median": float(np.median(error[mask])),
        }
    return {
        "count": int(error.size),
        "median": float(np.median(error)),
        "p95": float(np.percentile(error, 95)),
        "max": float(error.max()),
        "planes": planes,
    }
