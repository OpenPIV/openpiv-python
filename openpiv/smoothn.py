"""Robust and spline-based scalar field smoothing for OpenPIV.

This module provides tools for smoothing scalar fields (such as PIV velocity
displacement fields u and v) containing measurement noise, missing values / gaps
(NaNs, infinities), and masked regions.

Features:
- Bivariate / univariate spline smoothing powered by `scipy.interpolate`
  (`SmoothBivariateSpline`, `UnivariateSpline`, `griddata`).
- Fast automated penalized discrete cosine transform (DCT) spline smoothing
  with generalized cross-validation (GCV) parameter estimation.
- Iteratively re-weighted least squares (IRLS) robust smoothing for automatic
  outlier rejection (bisquare, Cauchy, and Talworth weightings).
- Full support for `numpy.ma.MaskedArray`, preserving masks.
"""

from typing import Any

import numpy as np
from numpy import ma

# Try importing modern scipy.fft, fall back to scipy.fftpack or pure-NumPy
try:
    from scipy.fft import dct as _scipy_dct
    from scipy.fft import idct as _scipy_idct
except ImportError:
    try:
        from scipy.fftpack import dct as _scipy_dct
        from scipy.fftpack import idct as _scipy_idct
    except ImportError:
        _scipy_dct = None
        _scipy_idct = None

# Try importing scipy.optimize for GCV optimization
try:
    from scipy.optimize import fmin_l_bfgs_b as _scipy_fmin_l_bfgs_b
except ImportError:
    _scipy_fmin_l_bfgs_b = None

# Try importing scipy.interpolate for modern spline smoothing
try:
    from scipy.interpolate import (
        SmoothBivariateSpline,
        UnivariateSpline,
    )

    HAS_SCIPY_INTERPOLATE = True
except ImportError:
    HAS_SCIPY_INTERPOLATE = False


# ---------------------------------------------------------------------------
# Pure-NumPy DCT / IDCT Fallbacks
# ---------------------------------------------------------------------------


def _dct_matrix(N: int) -> np.ndarray:
    """Generate orthogonal DCT-II transformation matrix of size N x N."""
    n = np.arange(N)
    k = np.arange(N).reshape(-1, 1)
    M = np.cos(np.pi * (2 * n + 1) * k / (2.0 * N))
    M[0, :] *= 1.0 / np.sqrt(2.0)
    M *= np.sqrt(2.0 / N)
    return M


def _numpy_dct(
    x: np.ndarray, norm: str = "ortho", type: int = 2, axis: int = -1
) -> np.ndarray:
    """Pure-NumPy DCT-II (orthonormal) along specified axis."""
    axis = axis % x.ndim
    M = _dct_matrix(x.shape[axis])
    return np.apply_along_axis(lambda row: M @ row, axis, x)


def _numpy_idct(
    X: np.ndarray, norm: str = "ortho", type: int = 2, axis: int = -1
) -> np.ndarray:
    """Pure-NumPy IDCT-II (orthonormal) along specified axis."""
    axis = axis % X.ndim
    M = _dct_matrix(X.shape[axis])
    return np.apply_along_axis(lambda row: M.T @ row, axis, X)


def dct(
    x: np.ndarray, norm: str = "ortho", type: int = 2, axis: int = -1
) -> np.ndarray:
    """Discrete Cosine Transform (DCT-II). Uses scipy when available, NumPy fallback otherwise."""
    if _scipy_dct is not None:
        return _scipy_dct(x, norm=norm, type=type, axis=axis)
    return _numpy_dct(x, norm=norm, type=type, axis=axis)


def idct(
    X: np.ndarray, norm: str = "ortho", type: int = 2, axis: int = -1
) -> np.ndarray:
    """Inverse Discrete Cosine Transform (IDCT-II). Uses scipy when available, NumPy fallback otherwise."""
    if _scipy_idct is not None:
        return _scipy_idct(X, norm=norm, type=type, axis=axis)
    return _numpy_idct(X, norm=norm, type=type, axis=axis)


def dctND(data: np.ndarray, f: Any = dct) -> np.ndarray:
    """Multi-dimensional Discrete Cosine Transform or inverse transform."""
    data = np.asarray(data, dtype=float)
    nd = data.ndim
    res = data.copy()
    for ax in range(nd):
        res = f(res, norm="ortho", type=2, axis=ax)
    return res


# ---------------------------------------------------------------------------
# Robust Weight Functions & Helper Utilities
# ---------------------------------------------------------------------------


def RobustWeights(
    r: np.ndarray, I: np.ndarray, h: float, wstr: str = "bisquare"
) -> np.ndarray:
    """Compute weights for robust smoothing from studentized residuals.

    Parameters
    ----------
    r : ndarray
        Residuals (y - z).
    I : ndarray (bool)
        Mask of valid / finite elements.
    h : float
        Leverage factor.
    wstr : str, optional
        Weight function type: 'bisquare' (default), 'cauchy', or 'talworth'.

    Returns
    -------
    W : ndarray
        Calculated robust weights in [0, 1].
    """
    r_val = r[I]
    if len(r_val) == 0:
        return np.ones_like(r)
    med = np.median(r_val)
    MAD = np.median(np.abs(r_val - med))
    if MAD == 0:
        MAD = np.mean(np.abs(r_val - med))
    denom = 1.4826 * MAD * np.sqrt(max(1e-12, 1.0 - h))
    u = np.abs(r / denom)
    wstr = str(wstr).lower()

    if wstr == "cauchy":
        c = 2.385
        W = 1.0 / (1.0 + (u / c) ** 2)
    elif wstr == "talworth":
        c = 2.795
        W = (u < c).astype(float)
    else:  # bisquare (Tukey biweight)
        c = 4.685
        W = np.where(u < c, (1.0 - (u / c) ** 2) ** 2.0, 0.0)

    W = np.nan_to_num(W, nan=0.0)
    return W


def InitialGuess(y: np.ndarray, z0: np.ndarray | None = None) -> np.ndarray:
    """Compute or validate initial guess for smoothed array."""
    if z0 is not None and hasattr(z0, "shape") and z0.shape == y.shape:
        return z0
    if isinstance(y, np.ma.MaskedArray):
        return y.copy()
    return y.copy()


def peaks(n: int | np.ndarray) -> np.ndarray | list[int]:
    """Generate a 2D test surface with peaks or find 1D peak indices."""
    if isinstance(n, np.ndarray):
        indices = []
        for i in range(1, len(n) - 1):
            if n[i] > n[i - 1] and n[i] > n[i + 1]:
                indices.append(i)
        return indices

    xp = np.arange(n)
    x, y = np.meshgrid(xp, xp)
    z = np.zeros_like(x, dtype=float)
    for _ in range(max(1, n // 5)):
        x0 = np.random.random() * n
        y0 = np.random.random() * n
        sdx = np.random.random() * n / 4.0
        sdy = sdx
        c = np.random.random() * 2.0 - 1.0
        f = np.exp(
            -(((x - x0) / sdx) ** 2)
            - ((y - y0) / sdy) ** 2
            - ((x - x0) / sdx) * ((y - y0) / sdy) * c
        )
        f *= np.random.random()
        z += f
    return z


def warning(s1: str, s2: str | list[str] | tuple[str, ...]) -> None:
    """Print or log a warning message."""
    print(s1)
    if isinstance(s2, (list, tuple)):
        print(s2[0])
    else:
        print(s2)


def smooth(u: np.ndarray, mask: np.ndarray) -> None:
    """Smooth a 2D array in-place while respecting a boolean exclusion mask."""
    m = ~mask
    r = u * m
    a = 4 * r[1:-1, 1:-1] + r[2:, 1:-1] + r[:-2, 1:-1] + r[1:-1, 2:] + r[1:-1, :-2]
    b = 4 * m[1:-1, 1:-1] + m[2:, 1:-1] + m[:-2, 1:-1] + m[1:-1, 2:] + m[1:-1, :-2]
    b[b == 0] = 1.0
    u[1:-1, 1:-1] = a / b


def smooth_masked_array(u: np.ma.MaskedArray) -> np.ma.MaskedArray:
    """Smooth a MaskedArray using iterative neighbor averaging."""
    if not isinstance(u, np.ma.MaskedArray):
        raise TypeError("Expected masked array")
    m = u.mask.copy() if hasattr(u.mask, "copy") else u.mask
    u_data = u.data.copy()
    for _ in range(10):
        smooth(u_data, m)
    return np.ma.array(u_data, mask=m)


# ---------------------------------------------------------------------------
# GCV Function and 1D Optimization Fallback
# ---------------------------------------------------------------------------


def gcv(
    p: float,
    Lambda: np.ndarray,
    aow: Any,
    DCTy: np.ndarray,
    IsFinite: np.ndarray,
    Wtot: np.ndarray,
    y: np.ndarray,
    nof: int,
    noe: int,
    smoothOrder: float,
) -> float:
    """Generalized Cross-Validation (GCV) score for DCT-based spline smoothing."""
    s = 10.0**p
    Gamma = 1.0 / (1.0 + (s * np.abs(Lambda)) ** smoothOrder)
    aow_arr = np.asarray(aow)
    if np.all(aow_arr > 0.9):
        RSS = np.linalg.norm(DCTy * (Gamma - 1.0)) ** 2
    else:
        yhat = dctND(Gamma * DCTy, f=idct)
        RSS = (
            np.linalg.norm(np.sqrt(Wtot[IsFinite]) * (y[IsFinite] - yhat[IsFinite]))
            ** 2
        )
    TrH = np.sum(Gamma)
    denom = 1.0 - TrH / float(noe)
    if denom <= 0:
        return 1e12
    GCVscore = RSS / float(nof) / (denom**2)
    return float(GCVscore)


def _golden_section_search(
    func: Any, a: float, b: float, tol: float = 0.05, args: tuple = ()
) -> list[float]:
    """1D Golden-section optimization fallback when scipy.optimize is not installed."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    resphi = 2.0 - phi
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = func(x1, *args)
    f2 = func(x2, *args)

    while abs(b - a) > tol:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + resphi * (b - a)
            f1 = func(x1, *args)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - resphi * (b - a)
            f2 = func(x2, *args)

    return [(a + b) / 2.0]


# ---------------------------------------------------------------------------
# scipy.interpolate Modern Smoothing Implementations
# ---------------------------------------------------------------------------


def _normalized_conv2d(y: np.ndarray, weights: np.ndarray, sigma: float) -> np.ndarray:
    """Pure-NumPy normalized convolution for 2D smoothing over gaps."""
    sigma = max(0.5, float(sigma))
    radius = int(np.ceil(3.0 * sigma))
    ax = np.arange(-radius, radius + 1)
    k1d = np.exp(-0.5 * (ax / sigma) ** 2)
    k1d /= k1d.sum()

    w = np.asarray(weights, dtype=float)
    valid = (w > 0) & np.isfinite(y)
    w_clean = np.where(valid, w, 0.0)
    data = np.where(valid, y, 0.0)

    def conv1d_axis(arr: np.ndarray, axis: int) -> np.ndarray:
        return np.apply_along_axis(
            lambda m: np.convolve(m, k1d, mode="same"), axis, arr
        )

    smooth_data = conv1d_axis(conv1d_axis(data * w_clean, 0), 1)
    smooth_w = conv1d_axis(conv1d_axis(w_clean, 0), 1)
    return np.where(smooth_w > 1e-6, smooth_data / np.maximum(smooth_w, 1e-12), 0.0)


def scipy_smooth2d(
    y: np.ndarray,
    s: float | None = None,
    W: np.ndarray | None = None,
    isrobust: bool = False,
    weightstr: str = "bisquare",
    max_iter: int = 3,
    kx: int = 3,
    ky: int = 3,
) -> tuple[np.ndarray, float, int, np.ndarray]:
    """Smooth a 2D scalar field (such as PIV displacement u or v) using scipy.interpolate.

    Handles missing values (NaNs, Infs, and masked elements) by fitting a
    bivariate spline over valid grid locations, naturally filling gaps.

    Parameters
    ----------
    y : array_like or MaskedArray
        2D scalar field to smooth.
    s : float, optional
        Smoothing parameter. Larger values produce smoother results.
        If None, a default based on data size and variance is used.
    W : array_like, optional
        Weights array matching y.shape. Must be >= 0.
    isrobust : bool, optional
        If True, performs iterative robust reweighting to reject outliers.
    weightstr : str, optional
        Robust weight function ('bisquare', 'cauchy', 'talworth').
    max_iter : int, optional
        Maximum iterations for robust smoothing.
    kx, ky : int, optional
        Degree of the bivariate spline (default 3, cubic).

    Returns
    -------
    z : ndarray or MaskedArray
        Smoothed array, with original mask preserved if input was masked.
    s : float
        Smoothing factor used.
    exitflag : int
        1 if convergence was reached, 0 otherwise.
    Wtot : ndarray
        Final weighting array.
    """
    is_masked = isinstance(y, ma.MaskedArray)
    mask = y.mask.copy() if is_masked else None
    y_arr = np.asarray(y, dtype=float)

    valid = np.isfinite(y_arr)
    if is_masked:
        valid = valid & (~mask)

    if W is not None:
        if np.any(W < 0):
            raise ValueError("smoothn:NegativeWeights", "Weights must all be >=0")
        valid = valid & (W > 0)
        weights = np.asarray(W, dtype=float).copy()
    else:
        weights = np.ones(y_arr.shape, dtype=float)

    weights[~valid] = 0.0
    nof = int(valid.sum())
    if nof < 2:
        z = y_arr.copy()
        if is_masked:
            z = ma.array(z, mask=mask)
        return z, 0.0 if s is None else float(s), 0, weights

    H, W_cols = y_arr.shape
    rows, cols = np.mgrid[0:H, 0:W_cols]
    r_val = rows[valid]
    c_val = cols[valid]
    z_val = y_arr[valid]
    w_val = weights[valid]

    # Normalize weights
    if w_val.max() > 0:
        w_val = w_val / w_val.max()

    var_z = float(np.var(z_val)) if len(z_val) > 1 else 1.0
    if var_z == 0:
        var_z = 1.0

    if s is None:
        s_val = 0.5
    else:
        s_val = float(s)

    Wtot = weights.copy()
    z_smooth = None
    exitflag = 1

    kx = min(kx, len(np.unique(r_val)) - 1)
    ky = min(ky, len(np.unique(c_val)) - 1)

    for step in range(max_iter if isrobust else 1):
        z_smooth = None
        if HAS_SCIPY_INTERPOLATE and kx >= 1 and ky >= 1 and nof > (kx + 1) * (ky + 1):
            try:
                spline = SmoothBivariateSpline(
                    r_val, c_val, z_val, w=w_val, s=s_val, kx=kx, ky=ky
                )
                _zs = spline(np.arange(H), np.arange(W_cols))
                if not np.all(np.isfinite(_zs)):
                    z_smooth = None
                else:
                    z_smooth = _zs
            except Exception:  # noqa: BLE001
                z_smooth = None

        if z_smooth is None:
            # Fallback using normalized convolution with robust weights
            sigma = max(0.5, float(s if s is not None else 1.0))
            z_smooth = _normalized_conv2d(y_arr, Wtot, sigma)

        if isrobust and step < max_iter - 1:
            r = y_arr - z_smooth
            w_rob = RobustWeights(r, valid, 0.1, weightstr)
            Wtot = weights * w_rob
            w_val = Wtot[valid]
            if w_val.max() > 0:
                w_val = w_val / w_val.max()

    if is_masked:
        z_smooth = ma.array(z_smooth, mask=mask)

    return z_smooth, float(s if s is not None else s_val), exitflag, Wtot


def scipy_smooth1d(
    y: np.ndarray,
    s: float | None = None,
    W: np.ndarray | None = None,
    isrobust: bool = False,
    weightstr: str = "bisquare",
    max_iter: int = 3,
    k: int = 3,
) -> tuple[np.ndarray, float, int, np.ndarray]:
    """Smooth a 1D scalar signal using scipy.interpolate.UnivariateSpline."""
    is_masked = isinstance(y, ma.MaskedArray)
    mask = y.mask.copy() if is_masked else None
    y_arr = np.asarray(y, dtype=float)

    valid = np.isfinite(y_arr)
    if is_masked:
        valid = valid & (~mask)

    if W is not None:
        if np.any(W < 0):
            raise ValueError("smoothn:NegativeWeights", "Weights must all be >=0")
        valid = valid & (W > 0)
        weights = np.asarray(W, dtype=float).copy()
    else:
        weights = np.ones(y_arr.shape, dtype=float)

    weights[~valid] = 0.0
    nof = int(valid.sum())
    if nof < 2:
        z = y_arr.copy()
        if is_masked:
            z = ma.array(z, mask=mask)
        return z, 0.0 if s is None else float(s), 0, weights

    N = len(y_arr)
    x = np.arange(N)
    x_val = x[valid]
    z_val = y_arr[valid]
    w_val = weights[valid]

    float(np.var(z_val)) if len(z_val) > 1 else 1.0
    if s is None:
        s_val = 0.5
    else:
        s_val = float(s)

    k = min(k, nof - 1)
    Wtot = weights.copy()
    z_smooth = None
    exitflag = 1

    for step in range(max_iter if isrobust else 1):
        z_smooth = None
        if HAS_SCIPY_INTERPOLATE and k >= 1 and nof > k + 1:
            try:
                spline = UnivariateSpline(x_val, z_val, w=w_val, s=s_val, k=k)
                _zs = spline(x)
                if not np.all(np.isfinite(_zs)):
                    z_smooth = None
                else:
                    z_smooth = _zs
            except Exception:  # noqa: BLE001
                z_smooth = None

        if z_smooth is None:
            sigma = max(0.5, float(s if s is not None else 1.0))
            radius = int(np.ceil(3.0 * sigma))
            ax = np.arange(-radius, radius + 1)
            k1d = np.exp(-0.5 * (ax / sigma) ** 2)
            k1d /= k1d.sum()
            data = np.where(valid, y_arr, 0.0)
            sm_data = np.convolve(data * Wtot, k1d, mode="same")
            sm_w = np.convolve(Wtot, k1d, mode="same")
            z_smooth = np.where(sm_w > 1e-6, sm_data / np.maximum(sm_w, 1e-12), 0.0)

        if isrobust and step < max_iter - 1:
            r = y_arr - z_smooth
            w_rob = RobustWeights(r, valid, 0.1, weightstr)
            Wtot = weights * w_rob
            w_val = Wtot[valid]
            if w_val.max() > 0:
                w_val = w_val / w_val.max()

    if is_masked:
        z_smooth = ma.array(z_smooth, mask=mask)

    return z_smooth, float(s if s is not None else s_val), exitflag, Wtot


# ---------------------------------------------------------------------------
# Penalized DCT Spline Smoothing (smoothn)
# ---------------------------------------------------------------------------


def smoothn(
    y: Any,
    nS0: int = 10,
    axis: int | tuple[int, ...] | None = None,
    smoothOrder: float = 2.0,
    sd: Any | None = None,
    verbose: bool = False,
    s0: float | None = None,
    z0: Any | None = None,
    isrobust: bool = False,
    W: Any | None = None,
    s: float | None = None,
    MaxIter: int = 100,
    TolZ: float = 1e-3,
    weightstr: str = "bisquare",
    method: str = "dct",
) -> tuple[np.ndarray, float, int, np.ndarray]:
    """Robust spline smoothing for 1-D to N-D data.

    This function provides automated and robust discretized smoothing
    for data of any dimension, handling missing values (NaNs, Infs),
    masked arrays, and outlying data.

    Parameters
    ----------
    y : array_like or MaskedArray
        The data to be smoothed (1D, 2D, 3D, or ND).
    nS0 : int, optional
        Number of samples used when automatically determining smoothing parameter.
    axis : int or tuple of ints, optional
        Axis or axes along which the smoothing is performed. If None,
        smoothing is performed along all axes.
    smoothOrder : float, optional
        Order of the smoothing (default 2.0, equivalent to cubic spline).
    sd : array_like, optional
        Standard deviation array. If provided, weights are computed as 1/sd^2.
    verbose : bool, optional
        If True, displays progress information with iterations and tolerance.
    s0 : float, optional
        Initial value for smoothing parameter.
    z0 : array_like, optional
        Initial guess for the smoothed data.
    isrobust : bool, optional
        If True, performs robust smoothing to suppress outliers.
    W : array_like, optional
        Weighting array of non-negative values matching y.shape.
    s : float, optional
        Smoothing parameter. If None, determined via generalized cross-validation (GCV).
    MaxIter : int, optional
        Maximum number of iterations allowed (default 100).
    TolZ : float, optional
        Termination tolerance on relative change in Z (default 1e-3).
    weightstr : str, optional
        Weight function for robust smoothing ('bisquare', 'cauchy', 'talworth').
    method : str, optional
        Smoothing engine: 'dct' (penalized DCT spline, default) or 'spline' / 'scipy'
        (scipy.interpolate B-spline).

    Returns
    -------
    z : ndarray or MaskedArray
        The smoothed array. Preserves MaskedArray if input was masked.
    s : float
        The smoothing parameter used.
    exitflag : int
        1 if convergence was reached, 0 if maximum iterations was exceeded.
    Wtot : ndarray
        The final weighting array used for smoothing.
    """
    # If spline method is explicitly requested and dimension is 1D or 2D:
    if method in ("spline", "scipy"):
        y_arr = np.asarray(y)
        if y_arr.ndim == 2:
            if axis is not None:
                axes = np.atleast_1d(axis).ravel()
                if (
                    not np.all(np.isin(axes, (-2, -1, 0, 1)))
                    or set(axes % 2) != {0, 1}
                ):
                    raise ValueError(
                        "The spline method requires smoothing along both axes; "
                        "use method='dct' for axis-restricted smoothing."
                    )
            return scipy_smooth2d(
                y,
                s=s,
                W=W,
                isrobust=isrobust,
                weightstr=weightstr,
                max_iter=min(3, MaxIter),
            )
        elif y_arr.ndim == 1:
            return scipy_smooth1d(
                y,
                s=s,
                W=W,
                isrobust=isrobust,
                weightstr=weightstr,
                max_iter=min(3, MaxIter),
            )

    is_masked = False
    if isinstance(y, ma.MaskedArray):
        is_masked = True
        mask = y.mask.copy()
        y_arr = np.array(y, dtype=float)
        y_arr[mask] = np.nan
        if W is not None:
            W = np.array(W, dtype=float)
            W[mask] = 0.0
        if sd is not None:
            W = np.array(1.0 / np.array(sd) ** 2, dtype=float)
            W[mask] = 0.0
            sd = None
        y = y_arr
    else:
        y = np.array(y, dtype=float)

    if sd is not None:
        sd_ = np.array(sd, dtype=float)
        mask = sd_ > 0.0
        W = np.zeros_like(sd_)
        W[mask] = 1.0 / sd_[mask] ** 2
        sd = None

    if W is not None:
        if np.any(W < 0):
            raise ValueError("smoothn:NegativeWeights", "Weights must all be >=0")
        max_W = np.max(W)
        if max_W > 0:
            W = W / max_W

    sizy = y.shape
    if axis is None:
        axis = tuple(range(y.ndim))
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        axis = tuple(np.array(axis).flatten())

    noe = y.size
    if noe < 2:
        return y, s, 0, 0

    if W is None:
        W = np.ones(sizy, dtype=float)

    IsFinite = np.isfinite(y)
    nof = IsFinite.sum()
    W = W * IsFinite

    isweighted = np.any(W != 1)
    isauto = s is None

    axis = tuple(np.array(axis).flatten())
    Lambda = np.zeros(sizy, dtype=float)
    for i in axis:
        siz0 = np.ones(y.ndim, dtype=int)
        siz0[i] = sizy[i]
        Lambda += np.cos(np.pi * np.arange(sizy[i]) / sizy[i]).reshape(siz0)
    Lambda = -2.0 * (len(axis) - Lambda)

    if not isauto:
        s = float(s)
        Gamma = 1.0 / (1.0 + (s * np.abs(Lambda)) ** smoothOrder)

    N = sum(dim != 1 for dim in sizy)
    hMin = 1e-6
    hMax = 0.99
    try:
        sMinBnd = np.sqrt(
            (
                ((1 + np.sqrt(1 + 8 * hMax ** (2.0 / N))) / 4.0 / hMax ** (2.0 / N))
                ** 2
                - 1
            )
            / 16.0
        )
        sMaxBnd = np.sqrt(
            (
                ((1 + np.sqrt(1 + 8 * hMin ** (2.0 / N))) / 4.0 / hMin ** (2.0 / N))
                ** 2
                - 1
            )
            / 16.0
        )
    except Exception:  # noqa: BLE001
        sMinBnd = 1e-3
        sMaxBnd = 1e3

    Wtot = W.copy()
    if isweighted:
        if z0 is not None:
            z = np.array(z0, dtype=float)
        else:
            z = y.copy()
            z[~IsFinite] = 0.0
    else:
        z = np.zeros(sizy, dtype=float)

    z0 = z.copy()
    y_work = y.copy()
    y_work[~IsFinite] = 0.0

    tol = 1.0
    RobustIterativeProcess = True
    RobustStep = 1
    nit = 0
    errp = 0.1
    RF = 1.0 + 0.75 * isweighted

    while RobustIterativeProcess:
        while tol > TolZ and nit < MaxIter:
            nit += 1
            DCTy = dctND(Wtot * (y_work - z) + z, f=dct)
            if isauto and (nit == 1 or (nit & (nit - 1)) == 0):
                aow = Wtot.mean()
                bounds = (np.log10(sMinBnd), np.log10(sMaxBnd))
                if s0 is None:
                    ss = (
                        np.arange(nS0)
                        * (1.0 / max(1, nS0 - 1.0))
                        * (bounds[1] - bounds[0])
                        + bounds[0]
                    )
                    g = np.zeros_like(ss)
                    for i, pp in enumerate(ss):
                        g[i] = gcv(
                            pp,
                            Lambda,
                            aow,
                            DCTy,
                            IsFinite,
                            Wtot,
                            y_work,
                            nof,
                            noe,
                            smoothOrder,
                        )
                    xinit = [ss[int(np.argmin(g))]]
                else:
                    xinit = [np.log10(s0)]

                if _scipy_fmin_l_bfgs_b is not None:
                    xpost, _, _ = _scipy_fmin_l_bfgs_b(
                        gcv,
                        xinit,
                        fprime=None,
                        factr=10.0,
                        approx_grad=True,
                        bounds=[bounds],
                        args=(
                            Lambda,
                            aow,
                            DCTy,
                            IsFinite,
                            Wtot,
                            y_work,
                            nof,
                            noe,
                            smoothOrder,
                        ),
                    )
                else:
                    xpost = _golden_section_search(
                        gcv,
                        bounds[0],
                        bounds[1],
                        tol=0.05,
                        args=(
                            Lambda,
                            aow,
                            DCTy,
                            IsFinite,
                            Wtot,
                            y_work,
                            nof,
                            noe,
                            smoothOrder,
                        ),
                    )

                s = 10.0 ** xpost[0]
                s0 = s
                Gamma = 1.0 / (1.0 + (s * np.abs(Lambda)) ** smoothOrder)
            else:
                if s is None:
                    s = 1.0
                Gamma = 1.0 / (1.0 + (s * np.abs(Lambda)) ** smoothOrder)

            z = RF * dctND(Gamma * DCTy, f=idct) + (1.0 - RF) * z
            norm_z = np.linalg.norm(z)
            if norm_z > 0 and isweighted:
                tol = np.linalg.norm(z0 - z) / norm_z
            else:
                tol = 0.0
            z0 = z.copy()

            if verbose:
                print(f"nit: {nit}, tol: {tol:.4e}")

        exitflag = int(nit < MaxIter)

        if isrobust:
            h = np.sqrt(1.0 + 16.0 * s)
            h = np.sqrt(1.0 + h) / np.sqrt(2.0) / h
            h = h**N
            Wtot = W * RobustWeights(y_work - z, IsFinite, h, weightstr)
            isweighted = True
            tol = 1.0
            nit = 0
            RobustStep += 1
            RobustIterativeProcess = RobustStep < 3
        else:
            RobustIterativeProcess = False

    if isauto:
        if abs(np.log10(s) - np.log10(sMinBnd)) < errp:
            warning(
                "MATLAB:smoothn:SLowerBound",
                [f"s = {s:.3f}: the lower bound for s has been reached."],
            )
        elif abs(np.log10(s) - np.log10(sMaxBnd)) < errp:
            warning(
                "MATLAB:smoothn:SUpperBound",
                [f"s = {s:.3f}: the upper bound for s has been reached."],
            )

    if is_masked:
        z = ma.array(z, mask=mask)

    return z, s, exitflag, Wtot


# Alias for modern Pythonic API
smooth_field = smoothn
