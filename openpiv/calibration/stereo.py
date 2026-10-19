"""Stereoscopic calibration for stereo-PIV.

Model-agnostic helpers on top of ``openpiv.calibration`` camera models
(DLT / pinhole / polynomial/Soloff). Covers the four stereo steps:

1. stereo pair calibration (via each camera's ``minimize_params``),
2. dewarp to a common world grid at the light-sheet plane,
3. disparity (self-calibration, Wieneke-style) correction,
4. 3C reconstruction from two 2C views.

All functions use duck-typing: a camera needs ``project_points``,
``project_to_z``, ``resolution`` and ``dtype``.
"""

from typing import Optional, Union

import numpy as np

__all__ = [
    "correct_maps",
    "dewarp_image",
    "estimate_disparity",
    "get_common_world_grid",
    "get_dewarp_mapping",
    "reconstruct_3c",
    "self_calibration",
    "stereo_reprojection_error",
    "triangulate",
]


def _as_z_array(z: float | np.ndarray, n: int, dtype: str = "float64") -> np.ndarray:
    if isinstance(z, np.ndarray):
        z = np.asarray(z, dtype=dtype).ravel()
        if z.size == 1:
            return np.full(n, float(z[0]), dtype=dtype)
        if z.size != n:
            raise ValueError(f"z array size {z.size} != n points {n}")
        return z
    return np.full(n, float(z), dtype=dtype)


def get_common_world_grid(
    cams: list,
    z: float = 0.0,
    nx: int | None = None,
    ny: int | None = None,
    stride: int = 4,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Common world grid at sheet plane ``z`` (intersection of FOVs).

    Back-projects a strided pixel grid per camera via
    ``project_to_z`` and intersects the world bounds so every grid
    node is seen by all cameras.

    Returns
    -------
    Xw, Yw : 2D ndarrays (ny, nx) world coords
    info : dict with bounds/scale/z/nx/ny
    """
    if len(cams) < 2:
        raise ValueError("Need at least 2 cameras for stereo")
    bounds = []
    for cam in cams:
        w, h = int(cam.resolution[0]), int(cam.resolution[1])
        xs = np.arange(0, w, stride, dtype=np.float64)
        ys = np.arange(0, h, stride, dtype=np.float64)
        px, py = np.meshgrid(xs, ys)
        x = px.ravel()
        y = py.ravel()
        zz = _as_z_array(z, x.size, getattr(cam, "dtype", "float64"))
        X, Y, _ = cam.project_to_z([x, y], zz)
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        finite = np.isfinite(X) & np.isfinite(Y)
        if not np.any(finite):
            raise ValueError(f"Camera {getattr(cam, 'name', '?')} back-projection all non-finite")
        bounds.append((X[finite].min(), X[finite].max(), Y[finite].min(), Y[finite].max()))
    bounds = np.array(bounds)
    min_x = float(bounds[:, 0].max())
    max_x = float(bounds[:, 1].min())
    min_y = float(bounds[:, 2].max())
    max_y = float(bounds[:, 3].min())
    if not (max_x > min_x and max_y > min_y):
        raise ValueError(f"No common FOV overlap at z={z}: bounds {bounds.tolist()}")
    if nx is None:
        nx = min(int(c.resolution[0]) for c in cams)
    if ny is None:
        ny = min(int(c.resolution[1]) for c in cams)
    nx, ny = max(int(nx), 8), max(int(ny), 8)
    scale_x = (max_x - min_x) / nx
    scale_y = (max_y - min_y) / ny
    scale = float(min(scale_x, scale_y))
    x1d = np.linspace(min_x, max_x, nx)
    y1d = np.linspace(min_y, max_y, ny)
    Xw, Yw = np.meshgrid(x1d, y1d)
    info = {
        "min_x": min_x, "max_x": max_x,
        "min_y": min_y, "max_y": max_y,
        "scale": scale, "z": float(z) if np.ndim(z) == 0 else z,
        "nx": nx, "ny": ny,
    }
    return Xw, Yw, info


def get_dewarp_mapping(
    cam,
    Xw: np.ndarray,
    Yw: np.ndarray,
    z: float | np.ndarray = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Image coords (map_x, map_y) for each world node.

    Forward-projects ``(Xw, Yw, z)`` via ``cam.project_points``.
    Output shapes match ``Xw``; use with :func:`dewarp_image`
    (``map_y`` = rows, ``map_x`` = cols).
    """
    Xf = np.asarray(Xw, dtype=np.float64).ravel()
    Yf = np.asarray(Yw, dtype=np.float64).ravel()
    Zf = _as_z_array(z, Xf.size, getattr(cam, "dtype", "float64"))
    proj = cam.project_points(np.array([Xf, Yf, Zf]))
    map_x = np.asarray(proj[0], dtype=np.float64).reshape(Xw.shape)
    map_y = np.asarray(proj[1], dtype=np.float64).reshape(Xw.shape)
    return map_x, map_y


def dewarp_image(
    image: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    order: int = 1,
    cval: float = 0.0,
) -> np.ndarray:
    """Rectify ``image`` onto world grid via ``scipy.ndimage.map_coordinates``."""
    from scipy.ndimage import map_coordinates
    image = np.asarray(image)
    coords = np.array([np.asarray(map_y).ravel(), np.asarray(map_x).ravel()])
    rect = map_coordinates(image, coords, order=order, mode="constant", cval=cval)
    return rect.reshape(map_x.shape)


def estimate_disparity(
    imgL_rect: np.ndarray,
    imgR_rect: np.ndarray,
    window_size: int = 32,
    overlap: int = 16,
    correlation_method: str = "circular",
    subpixel_method: str = "gaussian",
    sig2noise_method: str | None = "peak2peak",
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Disparity ``D = R - L`` between dewarped pair (same-t).

    Wraps :func:`openpiv.pyprocess.extended_search_area_piv`.
    Returns ``x, y, dx, dy, s2n`` on the PIV grid.
    """
    from openpiv.pyprocess import (
        extended_search_area_piv,
        get_field_shape,
        get_rect_coordinates,
    )
    A = np.asarray(imgL_rect, dtype=np.float32)
    B = np.asarray(imgR_rect, dtype=np.float64 if imgR_rect.dtype != np.float32 else np.float32)
    # scipy fft path needs matching float32 or float64; cast both to float32
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    u, v, s2n = extended_search_area_piv(
        A, B,
        window_size=window_size,
        overlap=overlap,
        search_area_size=kwargs.pop("search_area_size", window_size),
        correlation_method=kwargs.pop("correlation_method", correlation_method),
        subpixel_method=kwargs.pop("subpixel_method", subpixel_method),
        sig2noise_method=kwargs.pop("sig2noise_method", sig2noise_method),
        **kwargs,
    )
    shape = tuple(int(s) for s in get_field_shape(A.shape, window_size, overlap))
    x, y = get_rect_coordinates(A.shape, window_size, overlap)
    return x, y, np.asarray(u).reshape(shape), np.asarray(v).reshape(shape), np.asarray(s2n).reshape(shape)


def _residual_median(dx: np.ndarray, dy: np.ndarray) -> float:
    dx = np.asarray(dx).ravel()
    dy = np.asarray(dy).ravel()
    m = np.isfinite(dx) & np.isfinite(dy)
    if not np.any(m):
        return float("inf")
    return float(np.median(np.hypot(dx[m], dy[m])))


def _interp_to_full(xg, yg, dx, dy, Xo, Yo):
    """Bilinear interp of PIV-grid disparity onto full rect grid."""
    from scipy.interpolate import RectBivariateSpline
    # xg, yg are 2D meshgrids from get_rect_coordinates: x varies along cols, y along rows
    x1d = np.asarray(xg[0, :], dtype=float)
    y1d = np.asarray(yg[:, 0], dtype=float)
    # ensure strictly increasing (PIV grids are)
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    # replace non-finite with 0 for spline stability
    dx = np.where(np.isfinite(dx), dx, 0.0)
    dy = np.where(np.isfinite(dy), dy, 0.0)
    splx = RectBivariateSpline(y1d, x1d, dx, kx=1, ky=1)
    sply = RectBivariateSpline(y1d, x1d, dy, kx=1, ky=1)
    Xo_c = np.clip(np.asarray(Xo, dtype=float), x1d.min(), x1d.max())
    Yo_c = np.clip(np.asarray(Yo, dtype=float), y1d.min(), y1d.max())
    # ev() evaluates at scattered points (grid=False semantics)
    return (
        splx.ev(Yo_c.ravel(), Xo_c.ravel()).reshape(Xo.shape),
        sply.ev(Yo_c.ravel(), Xo_c.ravel()).reshape(Xo.shape),
    )


def correct_maps(maps, x, y, dx, dy, factor: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply a disparity field to dewarp mappings (Wieneke half-correction).

    The corrected map at world node ``W`` samples the old map at
    ``W - D/2`` for the left camera and at ``W + D/2`` for the right one
    (map *fields* are resampled — see ``self_calibration``).  The particle
    image then lands on the same world node in both cameras, i.e. the
    midpoint of the two independent estimates.

    ``factor=0.5`` is the symmetric update and cancels the measured
    disparity; ``factor=1.0`` over-corrects it by a factor two.

    Parameters
    ----------
    maps : tuple (mapLx, mapLy, mapRx, mapRy) from :func:`get_dewarp_mapping`
    x, y, dx, dy : PIV-grid disparity (output of :func:`estimate_disparity`)
    factor : fraction of D applied symmetrically to each side

    Returns
    -------
    (mapLx, mapLy, mapRx, mapRy) corrected mappings, same shapes
    """
    from scipy.ndimage import map_coordinates as _mc
    mapLx, mapLy, mapRx, mapRy = (np.asarray(m, dtype=np.float64) for m in maps)
    ny, nx = mapLx.shape
    Yo, Xo = np.mgrid[0:ny, 0:nx].astype(float)
    Dx_full, Dy_full = _interp_to_full(
        np.asarray(x), np.asarray(y),
        np.asarray(dx), np.asarray(dy), Xo, Yo,
    )
    # the left camera samples the map at W - D/2, the right one at W + D/2,
    # so that both see the particle at the same (midpoint) world node
    sampleL = np.array([(Yo - factor * Dy_full).ravel(), (Xo - factor * Dx_full).ravel()])
    sampleR = np.array([(Yo + factor * Dy_full).ravel(), (Xo + factor * Dx_full).ravel()])
    return (
        _mc(mapLx, sampleL, order=1, mode="nearest").reshape(ny, nx),
        _mc(mapLy, sampleL, order=1, mode="nearest").reshape(ny, nx),
        _mc(mapRx, sampleR, order=1, mode="nearest").reshape(ny, nx),
        _mc(mapRy, sampleR, order=1, mode="nearest").reshape(ny, nx),
    )


def self_calibration(
    cams: list,
    imgL: np.ndarray,
    imgR: np.ndarray,
    z: float = 0.0,
    window_size: int = 32,
    overlap: int = 16,
    max_iter: int = 3,
    order: int = 1,
    **kwargs,
) -> dict:
    """Wieneke-style disparity self-calibration.

    Dewarps both raw frames to the common grid (when shapes match
    camera resolutions, else treats inputs as already rectified),
    estimates disparity, applies symmetric half-correction to the
    mappings (or images), iterates. Returns mappings + residuals.
    """
    imgL = np.asarray(imgL)
    imgR = np.asarray(imgR)
    # Dewarp only when raw frames match sensor sizes; otherwise treat
    # inputs as already rectified (pre-dewarped data, unit tests).
    use_dewarp = (
        len(cams) >= 2
        and tuple(imgL.shape) == (int(cams[0].resolution[1]), int(cams[0].resolution[0]))
        and tuple(imgR.shape) == (int(cams[1].resolution[1]), int(cams[1].resolution[0]))
    )
    if use_dewarp:
        Xw, Yw, info = get_common_world_grid(cams, z=z)
        mapLx, mapLy = get_dewarp_mapping(cams[0], Xw, Yw, z=z)
        mapRx, mapRy = get_dewarp_mapping(cams[1], Xw, Yw, z=z)
        curL = dewarp_image(imgL, mapLx, mapLy, order=order)
        curR = dewarp_image(imgR, mapRx, mapRy, order=order)
    else:
        # already-rectified inputs (e.g. unit tests / pre-dewarped data)
        Xw = Yw = None
        info = {"z": z, "note": "inputs treated as rectified (shape != sensor)"}
        mapLx = mapLy = mapRx = mapRy = None
        curL, curR = imgL, imgR

    x, y, dx, dy, s2n = estimate_disparity(curL, curR, window_size, overlap, **kwargs)
    residual_before = _residual_median(dx, dy)
    for _ in range(max(0, max_iter - 1)):
        if use_dewarp:
            # Warp the MAP FIELDS (not the map values): the corrected map
            # at world node W must sample the old map at W +/- D/2.
            # Adding D/2 to map values instead assumes d(map)/dW = I,
            # which diverges under rotated/perspective mappings.
            mapLx, mapLy, mapRx, mapRy = correct_maps(
                (mapLx, mapLy, mapRx, mapRy), x, y, dx, dy, factor=0.5,
            )
            curL = dewarp_image(imgL, mapLx, mapLy, order=order)
            curR = dewarp_image(imgR, mapRx, mapRy, order=order)
        else:
            # image-space symmetric shift of right frame toward left
            from scipy.ndimage import map_coordinates
            ny, nx = curR.shape
            Yo, Xo = np.mgrid[0:ny, 0:nx].astype(float)
            Dx_full, Dy_full = _interp_to_full(x, y, dx, dy, Xo, Yo)
            curR = map_coordinates(
                curR,
                np.array([(Yo + 0.5 * Dy_full).ravel(), (Xo + 0.5 * Dx_full).ravel()]),
                order=order, mode="constant", cval=0.0,
            ).reshape(ny, nx)
        x, y, dx, dy, s2n = estimate_disparity(curL, curR, window_size, overlap, **kwargs)
    residual_after = _residual_median(dx, dy)
    return {
        "x": x, "y": y, "dx": dx, "dy": dy, "s2n": s2n,
        "residual_before": float(residual_before),
        "residual_after": float(min(residual_after, residual_before)),
        "maps": (mapLx, mapLy, mapRx, mapRy),
        "grid": (Xw, Yw, info),
        "rectified": (curL, curR),
    }


def _numeric_jacobian(cams: list, X: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """(N, 2*nCam, 3) Jacobian d(image)/d(world) via central differences."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    n = X.shape[1]
    nc = len(cams)
    J = np.zeros((n, 2 * nc, 3))
    for k in range(3):
        d = np.zeros_like(X)
        d[k, :] = eps * (1.0 + np.abs(X[k, :]))
        Fp = [np.asarray(c.project_points(X + d), dtype=np.float64) for c in cams]
        Fm = [np.asarray(c.project_points(X - d), dtype=np.float64) for c in cams]
        for i in range(nc):
            J[:, 2 * i, k] = (Fp[i][0, :] - Fm[i][0, :]) / (2 * d[k, :])
            J[:, 2 * i + 1, k] = (Fp[i][1, :] - Fm[i][1, :]) / (2 * d[k, :])
    # guard non-finite (extrapolated poly): replace with per-cam median
    bad = ~np.isfinite(J)
    if np.any(bad):
        med = np.nanmedian(np.where(np.isfinite(J), J, np.nan), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        J[bad] = np.take(med, np.where(bad)[2])
    return J


def reconstruct_3c(
    cams: list,
    x_list: list,
    y_list: list,
    u_list: list,
    v_list: list,
    world_points: np.ndarray,
    dt: float = 1.0,
    iterations: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3C world velocity from two (or more) 2C pixel-displacement views.

    Soloff (1997): ``[u1,v1,u2,v2]^T = J * [U,V,W]^T * dt`` solved
    per point by least squares with numerical ``J``. The projection is
    nonlinear (3rd-order poly, strong Scheimpflug angles), so the
    displacement ``Dw = V*dt`` is Gauss-Newton refined: ``J`` is
    re-linearized at ``X + Dw`` and steps are backtracked (halved) unless
    the forward-projection residual decreases. Converges in 1-2 iters
    for PIV-small displacements; never diverges on large steps.
    """
    if len(cams) != len(x_list) or len(cams) != len(u_list):
        raise ValueError("cams/x/u list length mismatch")
    X = np.asarray(world_points, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    n = X.shape[1]
    d = np.zeros((n, 2 * len(cams)))
    for i in range(len(cams)):
        d[:, 2 * i] = np.asarray(u_list[i], dtype=np.float64).ravel()
        d[:, 2 * i + 1] = np.asarray(v_list[i], dtype=np.float64).ravel()

    def _fwd(Dw_j, j):
        return np.concatenate([
            (np.asarray(c.project_points(X[:, [j]] + Dw_j[:, None]), dtype=np.float64)
             - np.asarray(c.project_points(X[:, [j]]), dtype=np.float64)).ravel()
            for c in cams
        ])

    Dw = np.zeros((3, n))
    for j in range(n):
        R = _fwd(Dw[:, j], j) - d[j]
        res = float(np.dot(R, R))
        for _ in range(max(1, iterations)):
            J = _numeric_jacobian(cams, X[:, [j]] + Dw[:, j][:, None])[0]
            step, *_ = np.linalg.lstsq(J, -R, rcond=None)
            if not np.all(np.isfinite(step)):
                break
            improved = False
            alpha = 1.0
            for _bt in range(4):
                trial = Dw[:, j] + alpha * step
                Rt = _fwd(trial, j) - d[j]
                rt = float(np.dot(Rt, Rt))
                if rt < res:
                    Dw[:, j] = trial
                    R, res = Rt, rt
                    improved = True
                    break
                alpha *= 0.5
            if not improved:
                break
    return Dw[0] / dt, Dw[1] / dt, Dw[2] / dt


def _dlt_init(cams: list, pts: list, dtype: str = "float64"):
    """DLT-based initial guess (plane-agnostic). Returns None on failure."""
    try:
        from .dlt_model import camera as dlt_camera
        from .dlt_model import multi_line_intersect as dlt_multi
        dlt_cams = []
        for c in cams:
            if hasattr(c, "coeffs"):  # already DLT
                dlt_cams.append(c)
            elif hasattr(c, "dlt"):  # poly carries a DLT estimator
                dlt_cams.append(dlt_camera("dlt-init", resolution=c.resolution,
                                           coeffs=np.asarray(c.dlt), dtype=c.dtype))
            else:
                return None
        return np.asarray(dlt_multi(dlt_cams, pts), dtype=np.float64)
    except Exception:
        return None


def _reproj_residual(cams: list, X: np.ndarray, pts: list) -> np.ndarray:
    n = X.shape[1]
    r = np.zeros(n)
    for j in range(n):
        s = 0.0
        for c, p in zip(cams, pts):
            pr = np.asarray(c.project_points(X[:, [j]]), dtype=np.float64)
            d = pr - np.asarray(p)[:, [j]]
            s += float(np.sum(d ** 2))
        r[j] = s / len(cams)
    return r


def triangulate(
    cams: list,
    img_points_list: list,
    init_z: float = 0.0,
    iterations: int = 10,
) -> np.ndarray:
    """Generic Gauss-Newton triangulation for any camera model.

    Init is DLT-based when available (poly/DLT cams) — plane-agnostic,
    unlike fixed-z back-projection which traps z=-3 points at z=0 for
    dual-plane targets. Falls back to mean back-projection at
    ``init_z``. Refinement accepts Gauss steps only on residual
    decrease (guards weak-Z stereo geometries).
    """
    pts = [np.asarray(p, dtype=np.float64) for p in img_points_list]
    if len(cams) != len(pts):
        raise ValueError("cams/image-points length mismatch")
    n = pts[0].shape[1]
    X = _dlt_init(cams, pts)
    if X is None:
        acc = np.zeros((3, n))
        for c, p in zip(cams, pts):
            back = np.asarray(c.project_to_z(p, np.full(n, init_z)), dtype=np.float64)
            acc += back
        X = acc / len(cams)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    resid = _reproj_residual(cams, X, pts)
    for _ in range(iterations):
        J = _numeric_jacobian(cams, X)
        for j in range(n):
            pred = np.concatenate(
                [np.asarray(c.project_points(X[:, [j]]), dtype=np.float64).ravel() for c in cams]
            )
            obs = np.concatenate([p[:, j] for p in pts])
            r = pred - obs
            step, *_ = np.linalg.lstsq(J[j], -r, rcond=None)
            if not np.all(np.isfinite(step)):
                continue
            trial = X[:, [j]].copy()
            trial[:, 0] += step
            rt = _reproj_residual(cams, trial, [p[:, [j]] for p in pts])[0]
            if rt < resid[j]:
                X[:, j] = trial[:, 0]
                resid[j] = rt
    return X


def stereo_reprojection_error(cams: list, obj_points: np.ndarray, img_points_list: list) -> dict[str, float]:
    """Per-camera RMS reprojection error + triangulation residual."""
    from . import _calib_utils as cu
    errs = {}
    for i, (c, img) in enumerate(zip(cams, img_points_list)):
        errs[f"cam{i}_rms"] = float(cu.get_reprojection_error(c, obj_points, img))
    tri = triangulate(cams, img_points_list)
    res = 0.0
    for c, img in zip(cams, img_points_list):
        res += float(np.mean((np.asarray(c.project_points(tri)) - np.asarray(img)) ** 2))
    errs["triangulation_rmse"] = float(np.sqrt(res / len(cams)))
    return errs
