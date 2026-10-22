import numpy as np
from scipy import ndimage

__all__ = [
    'checkerboard_find_points'
]


def _rotate_image(
    image: np.ndarray,
    angle: float,
    output_shape: tuple | None=None
) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)

    matrix = np.array(
        [
            [cosine, -sine],
            [sine, cosine]
        ]
    )

    center = (np.asarray(image.shape) - 1) / 2

    if output_shape is None:
        corners = np.array(
            [
                [0, 0],
                [0, image.shape[1] - 1],
                [image.shape[0] - 1, 0],
                [image.shape[0] - 1, image.shape[1] - 1]
            ]
        )

        corners = np.dot(
            corners - center,
            matrix.T
        )

        output_shape = tuple(
            np.ceil(
                np.ptp(corners, axis=0) - 1e-12
            ).astype(int) + 1
        )

    out_center = (np.asarray(output_shape) - 1) / 2

    rows, cols = np.indices(
        output_shape,
        dtype=float
    )

    rows -= out_center[0]
    cols -= out_center[1]

    source = np.array(
        [
            cosine * rows + sine * cols + center[0],
            -sine * rows + cosine * cols + center[1]
        ]
    )

    rotated = np.zeros(
        output_shape,
        dtype=float
    )

    ndimage.map_coordinates(
        image,
        source,
        output=rotated,
        order=1,
        mode="constant",
        cval=0
    )

    return rotated


def _corner_response(
    image: np.ndarray,
    length: int,
    width: int
) -> np.ndarray:
    """Create a response field using rotated images and blur filters.

    Create a response field of checkerboard corners using an approximated
    radon filter. This is accomplished by rotating an image, applying a
    box blur filter, and rotating the image back.

    """
    angles = [
        0,
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4
    ]

    averages = []

    for angle in angles:
        if angle == 0:
            rotated = image
        else:
            rotated = _rotate_image(
                image,
                angle
            )

        filtered = ndimage.uniform_filter(
            rotated,
            (width, length),
            mode="mirror"
        )

        if angle != 0:
            filtered = _rotate_image(
                filtered,
                -angle,
                image.shape
            )

        averages.append(filtered)

    averages = np.stack(averages)

    profile = np.rint(averages * 255)

    left, right = (
        np.roll(profile, 1, axis=0),
        np.roll(profile, -1, axis=0)
    )

    maxima = (left <= profile) & (right < profile)
    minima = (left > profile) & (right >= profile)

    junction = np.sum(maxima | minima, axis=0) == 2

    response = np.where(
        junction,
        np.ptp(averages, axis=0)**2,
        0
    )

    # As in OpenCV's FastX, suppress broad responses and smooth the corner impulses.
    high = ndimage.uniform_filter(
        response,
        width,
        mode="mirror"
    )

    low = ndimage.uniform_filter(
        response,
        length,
        mode="mirror"
    )

    return np.maximum(high - 0.8 * low, 0)


## TODO: Use OpenPIV's 3 point subpixel estimators
def _gaussian_offset(
    left: np.ndarray,
    center: np.ndarray,
    right: np.ndarray
) -> np.ndarray:
    offset = np.zeros_like(
        center,
        dtype=float
    )

    valid = (left > 0) & (center > 0) & (right > 0)

    l, c, r = (
        np.log(values[valid])
        for values in (left, center, right)
    )

    denominator = 2 * (l - 2 * c + r)

    delta = np.divide(
        l - r,
        denominator,
        out=np.zeros_like(c),
        where=denominator < -np.finfo(float).eps
    )

    offset[valid] = np.clip(delta, -0.5, 0.5)

    return offset


def _mask_candidates(
    response: np.ndarray,
    border: int,
    radius: int,
    threshold_rel: float
) -> np.ndarray:
    valid = np.zeros(
        response.shape,
        dtype=bool
    )

    valid[border:-border, border:-border] = True
    peak = response[valid].max()

    if peak <= np.finfo(float).eps:
        return np.zeros(
            response.shape,
            dtype=bool
        )

    local_maximum = ndimage.maximum_filter(
        response,
        size=2 * radius + 1
    )

    minimum_response = peak * threshold_rel

    candidates = valid.copy()
    candidates &= response >= minimum_response
    candidates &= response > 0
    candidates &= response == local_maximum

    return candidates


def _refine_candidates(
    response: np.ndarray,
    candidates: np.ndarray
) -> np.ndarray:
    labels, count = ndimage.label(
        candidates,
        structure=np.ones((3, 3))
    )

    if count == 0:
        return np.empty(
            (2, 0),
            dtype=float
        )

    positions = np.asarray(
        ndimage.maximum_position(
            response,
            labels,
            np.arange(1, count + 1)
        )
    )

    y, x = positions.T

    dx = _gaussian_offset(
        response[y, x - 1],
        response[y, x],
        response[y, x + 1]
    )

    dy = _gaussian_offset(
        response[y - 1, x],
        response[y, x],
        response[y + 1, x]
    )

    return np.array(
        [
            x + dx,
            y + dy
        ]
    )


def _filter_candidates(
    image: np.ndarray,
    points: np.ndarray,
    length: int
) -> np.ndarray:
    if points.shape[1] == 0:
        return points

    # Opposing samples agree at a checkerboard crossing, but not at an edge
    angles = (np.arange(16) + 0.5) * (2 * np.pi / 16)

    ring = points[:, :, None] + (length / 2) * np.array(
        [
            np.cos(angles),
            np.sin(angles)
        ]
    )[:, None, :]

    samples = ndimage.map_coordinates(
        image,
        ring[::-1],
        order=1,
        mode="nearest",
        prefilter=False
    )

    symmetry_error = np.mean(
        np.abs(
            samples - np.roll(samples, 8, axis=1)
        ),
        axis=1
    )

    local_contrast = np.ptp(
        samples,
        axis=1
    )

    valid = local_contrast > 0
    valid &= symmetry_error < 0.25 * local_contrast

    return points[:, valid]


def checkerboard_find_points(
    image: np.ndarray,
    robust: bool=False,
    filter_size: int=17,
    min_distance: int | None=None,
    threshold_rel: float=0.4
) -> np.ndarray:
    """Find subpixel checkerboard intersections for later point matching.

    Parameters
    ----------
    image : (height, width) array_like
        Finite, real grayscale intensities. Integer, floating and boolean
        images are accepted; intensities are normalized internally.
    robust : bool, optional
        Upsample by two before filtering and peak fitting. The filter support
        and peak separation are scaled too. Coordinates are returned in the
        original pixel frame, including the half-pixel resize correction.
        This reduces sampling error at the cost of memory and processing time.
    filter_size : int, optional
        Odd length of the directional box filter, at least 5, in original
        pixels. The odd transverse width is approximately 2/7 of this length
        (5 by 17 by default). Choose a length smaller than the tile spacing.
    min_distance : int, optional
        Peak suppression radius in original pixels (Chebyshev distance).
        Defaults to half the filter size. Must be positive and smaller than
        the distance between adjacent intersections.
    threshold_rel : float, optional
        Minimum response as a fraction of the strongest response, in (0, 1].

    Returns
    -------
    points : (2, n) ndarray
        Floating-point image coordinates: x in the first row, y in the second,
        consistent with the other calibration point functions. Pixel centers
        start at (0, 0); y increases downward.

    Notes
    -----
    Horizontal and vertical box averages are computed both before and after
    an image rotation. Their angular contrast gives a localized Radon
    response. Each local maximum is refined independently along x and y
    using a three-point Gaussian fit. A point-symmetry check rejects ordinary
    edges and tile outlines. Intersections within roughly one filter length
    of the image boundary are excluded because their filter support is
    incomplete.

    References
    ----------
    .. [1] OpenCV contributors. FastX corner detector, implemented in
       ``modules/calib3d/src/chessboard.cpp``.
       https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/chessboard.cpp

    .. [2] Alexander Duda and Udo Frese. "Accurate Detection and Localization
       of Checkerboard Corners for Calibration." Proceedings of the British
       Machine Vision Conference (BMVC), 2018.
       https://bmvc2018.org/contents/papers/0508.pdf

    Examples
    --------
    >>> from openpiv.calibration.marker_detect import checkerboard_find_points
    >>> points = checkerboard_find_points(calibration_image, robust=True)
    >>> x, y = points

    """
    image = np.asarray(image)

    if image.ndim != 2 or image.size == 0:
        raise ValueError(
            "image must be a nonempty two-dimensional grayscale array"
        )

    if image.dtype.kind not in "buif" or not np.isfinite(image).all():
        raise ValueError(
            "image must contain finite real intensities"
        )

    if (
        isinstance(filter_size, (bool, np.bool_)) or
        not isinstance(filter_size, (int, np.integer)) or
        filter_size < 5 or
        filter_size % 2 != 1
    ):
        raise ValueError(
            "filter_size must be an odd integer of at least 5"
        )

    if min_distance is None:
        min_distance = filter_size // 2

    if (
        isinstance(min_distance, (bool, np.bool_)) or
        not isinstance(min_distance, (int, np.integer)) or
        min_distance < 1
    ):
        raise ValueError(
            "min_distance must be a positive integer"
        )

    if not np.isfinite(threshold_rel) or not 0 < threshold_rel <= 1:
        raise ValueError(
            "threshold_rel must be in (0, 1]"
        )

    empty = np.empty(
        (2, 0),
        dtype=float
    )

    image = image.astype(float)

    # Scaling before subtraction also handles very large finite float values.
    magnitude = np.max(np.abs(image))

    if magnitude:
        image /= magnitude

    image -= image.min()
    contrast = image.max()

    if contrast == 0:
        return empty

    image /= contrast

    scale = 2 if robust else 1
    width = 2 * (filter_size // 7) + 1

    # Include both post-filters and the Gaussian fit's neighbors. Keep the
    # exclusion region fixed in original pixels when upsampling.
    border = scale * (
        int(
            np.ceil(
                np.hypot(filter_size, width) / 2 + filter_size / 2
            )
        ) + 2
    )

    length = (filter_size - 1) * scale + 1
    width = (width - 1) * scale + 1

    if min(image.shape) * scale <= 2 * border:
        return empty

    if robust:
        image = ndimage.zoom(
            image,
            2,
            order=1,
            mode="nearest",
            grid_mode=True,
            prefilter=False
        )

    response = _corner_response(
        image,
        length,
        width
    )

    candidates = _mask_candidates(
        response,
        border,
        min_distance * scale,
        threshold_rel
    )

    points = _refine_candidates(
        response,
        candidates
    )

    points = _filter_candidates(
        image,
        points,
        length
    )

    return (points + 0.5) / scale - 0.5
