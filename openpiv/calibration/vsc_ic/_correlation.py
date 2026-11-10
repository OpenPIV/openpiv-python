import numpy as np
from scipy.ndimage import map_coordinates, maximum_filter
from scipy.signal import find_peaks

from openpiv.pyprocess import fft_correlate_images


def _correlate_images(
    image_a: np.ndarray,
    image_b: np.ndarray,
    method: str='circular',
    subtract_mean: bool=True,
) -> np.ndarray:
    # Equation 5; correlation of undistorted images
    image_a = np.asarray(image_a, dtype=np.float64)
    image_b = np.asarray(image_b, dtype=np.float64)

    if subtract_mean:
        divisor = image_a.size * np.std(image_a) * np.std(image_b)
        
        image_a = image_a - image_a.mean()
        image_b = image_b - image_b.mean()

    corr = fft_correlate_images(
        image_a[np.newaxis],
        image_b[np.newaxis],
        correlation_method=method,
        normalized_correlation=False,
    )[0]

    if subtract_mean:
        if divisor > 0:
            corr /= divisor
        else:
            corr[:] = 0.0

    return corr


def _get_lag_axes(shape: tuple) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape

    lag_x = np.arange(width) - width // 2
    lag_y = np.arange(height) - height // 2

    return lag_x, lag_y


def _remove_correlation_background(
    corr: np.ndarray,
    window_shape: tuple,
    method: str,
    background: float | None = None,
) -> np.ndarray:
    height, width = window_shape

    if method == 'circular':
        return corr - corr.mean()

    if background is None:
        msg = 'Linear correlation background requires input image means'
        raise ValueError(msg)

    x, y = _get_lag_axes(corr.shape)

    overlap_rows = height - np.abs(y)
    overlap_columns = width - np.abs(x)
    overlap = np.outer(overlap_rows, overlap_columns)

    return corr - background * overlap


def _fit_peak_1d(
    profile: np.ndarray,
    width: int = 3,
    peak_index: int | None = None,
) -> tuple[float, float]:
    profile = np.asarray(profile, dtype=np.float64)

    if peak_index is None:
        index = int(np.argmax(profile))
    else:
        index = peak_index

    peak = profile[index]

    if index in (0, profile.size - 1) or peak <= 0:
        return np.nan, 0.0
    
    left, center, right = profile[index - 1:index + 2]

    if min(left, center, right) > 0:
        left, center, right = np.log([left, center, right])

    denominator = left - 2 * center + right

    if denominator >= 0:
        return np.nan, 0.0
    
    delta = 0.5 * (left - right) / denominator

    if abs(delta) > 1:
        return np.nan, 0.0

    candidates, _ = find_peaks(profile)
    candidates = candidates[np.abs(candidates - index) > width]

    if candidates.size:
        peak2 = max(float(np.max(profile[candidates])), 0.0)
    else:
        peak2 = 0.0

    if peak2 > 0:
        ratio = peak / peak2
    else:
        ratio = np.inf

    return float(index + delta), float(ratio)


def _fit_peak_2d(
    corr: np.ndarray,
    window_shape: tuple,
    width: int = 3,
) -> tuple[np.ndarray, float]:
    lag_x, lag_y = _get_lag_axes(corr.shape)

    window_height, window_width = window_shape

    reliable_x = np.abs(lag_x) < window_width / 4
    reliable_y = np.abs(lag_y) < window_height / 4
    reliable = reliable_y[:, np.newaxis] & reliable_x[np.newaxis, :]

    search = np.where(reliable, corr, -np.inf)
    row, col = np.unravel_index(np.argmax(search), search.shape)

    x, _ = _fit_peak_1d(corr[row], peak_index=col)
    y, _ = _fit_peak_1d(corr[:, col], peak_index=row)

    if not np.isfinite(x + y):
        return np.full(2, np.nan), 0.0

    search[
        max(0, row - width):row + width + 1,
        max(0, col - width):col + width + 1,
    ] = -np.inf

    if not np.any(np.isfinite(search)):
        return np.full(2, np.nan), 0.0

    maxima = corr == maximum_filter(
        corr,
        size=3,
        mode='constant',
        cval=-np.inf,
    )

    candidates = search[maxima & np.isfinite(search)]

    if candidates.size:
        peak2 = max(float(np.max(candidates)), 0.0)
    else:
        peak2 = 0.0

    peak = float(corr[row, col])

    if peak2 > 0:
        ratio = peak / peak2
    else:
        ratio = np.inf

    offset = np.array([
        x - corr.shape[1] // 2,
        y - corr.shape[0] // 2,
    ])

    return offset, ratio


def _get_streak_profile(
    corr: np.ndarray,
    geometry: dict,
    window_shape: tuple,
    depth_interval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Equation 11; somewhat self explanatory
    tangent = geometry['tangent']
    normal = geometry['normal']
    limit = min(window_shape) / 4

    start, stop = np.asarray(depth_interval) * np.linalg.norm(geometry['depth_direction'])
    start = np.ceil(max(start, -limit))
    stop = np.floor(min(stop, limit))

    along = np.arange(start, stop + 1)
    across = np.arange(-int(limit) + 1, int(limit))

    lags = across[:, None, None] * normal + along[None, :, None] * tangent

    coords = [
        lags[..., 1] + corr.shape[0] // 2,
        lags[..., 0] + corr.shape[1] // 2,
    ]

    profile = map_coordinates(
        corr,
        coords,
        order=1,
        mode='constant',
        cval=0,
    )
    profile = profile.sum(axis=1)

    return across, profile


def _correlate_camera_triple(
    image_a: np.ndarray,
    aligned_image_b: np.ndarray,
    target: np.ndarray,
    method: str,
) -> np.ndarray:
    # Equation 23; Triple correlation supports calculating disparities of
    # any arrangement. This solves the lienar camera arrangement issue
    # where the streaks become all the same angles and the disparities
    # becomes impossible to calculate.
    product = image_a * aligned_image_b

    return _correlate_images(
        product,
        target,
        method,
        subtract_mean=False,
    )