from itertools import combinations

import numpy as np

from .._camera import BaseCamera
from ._correlation import (
    _correlate_camera_triple,
    _correlate_images,
    _fit_peak_1d,
    _fit_peak_2d,
    _get_lag_axes,
    _get_streak_profile,
    _remove_correlation_background,
)
from ._geometry import (
    _get_los_slopes,
    _get_streak_geometry,
    _minimize_image_disparities,
)
from ._images import (
    _dewarp_window,
    _get_window_mapping,
    _prepare_image_stacks,
    _prepare_world_grid,
)


__all__ = [
    'get_disparity'
]


def _check_depth_planes(z_planes: np.ndarray) -> np.ndarray:
    z = np.asarray(z_planes, dtype=np.float64)

    if z.ndim != 1 or z.size < 2:
        msg = 'z_planes must be a one-dimensional array with at least two depths'
        raise ValueError(msg)

    if not np.all(np.isfinite(z)):
        msg = 'z_planes must contain only finite depths'
        raise ValueError(msg)   
    
    steps = np.diff(z)

    if np.any(steps <= 0) or not np.allclose(steps, steps[0]):
        msg = 'z_planes must be uniformly spaced and strictly increasing'
        raise ValueError(msg)

    return z


def _get_supported_mappings(
    cams: list[BaseCamera],
    X: np.ndarray,
    Y: np.ndarray,
    z: float,
    offsets: np.ndarray,
) -> list:
    mappings = [
        _get_window_mapping(cam, X, Y, z, offset)
        for cam, offset in zip(cams, offsets, strict=True)
    ]

    if not all(np.all(mapping[2]) for mapping in mappings):
        msg = 'World grid leaves a camera sensor; reduce the window or depth range'
        raise ValueError(msg)

    return mappings


def _accumulate_pairwise_maps(
    cams: list[BaseCamera],
    stacks: list,
    X: np.ndarray,
    Y: np.ndarray,
    z: float,
    pairs: list,
    method: str,
) -> np.ndarray:
    # Equation 5; sum across correlation maps
    offsets = np.zeros((len(cams), 2))
    mappings = _get_supported_mappings(cams, X, Y, z, offsets)
    maps = None

    for frame in range(stacks[0].shape[0]):
        rectified = [
            _dewarp_window(stack[frame], mapping)
            for stack, mapping in zip(stacks, mappings, strict=True)
        ]

        current = np.asarray([
            _correlate_images(rectified[i], rectified[j], method)
            for i, j in pairs
        ])

        if maps is None:
            maps = current
        else:
            maps = maps + current

    return maps


def _measure_pairwise_disparities(
    maps: np.ndarray,
    geometries: list,
    window_shape: tuple,
    depth_interval: np.ndarray,
) -> dict:
    # Equation 11; integrate the streaks then fit the disparity
    profiles = []
    axes = []
    offsets = []
    ratios = []

    for corr, geometry in zip(maps, geometries, strict=True):
        axis, profile = _get_streak_profile(
            corr,
            geometry,
            window_shape,
            depth_interval,
        )
        peak, ratio = _fit_peak_1d(profile)

        offsets.append(axis[0] + peak)
        axes.append(axis)
        profiles.append(profile)
        ratios.append(ratio)

    return {
        'pair_disparities': np.asarray(offsets),
        'pair_peak_ratios': np.asarray(ratios),
        'streak_profiles': np.asarray(profiles),
        'streak_axes': np.asarray(axes),
    }


def _align_reference_pair(
    pair: tuple,
    pairs: list,
    measurement: dict,
    geometries: list,
    spacing: np.ndarray,
    n_cameras: int,
) -> np.ndarray:
    # Equation 17; align reference pair using the perpendicular disparity
    i, j = pair
    ordered = (min(i, j), max(i, j))
    index = pairs.index(ordered)

    foot = (
        measurement['pair_disparities'][index]
        * geometries[index]['normal']
        * spacing
    )

    offsets = np.zeros((n_cameras, 2))

    if i < j:
        offsets[j] = foot
    else:
        offsets[j] = -foot

    return offsets


def _accumulate_triple_maps(
    cams: list[BaseCamera],
    stacks: list,
    X: np.ndarray,
    Y: np.ndarray,
    z_planes: np.ndarray,
    reference_pair: tuple,
    offsets: np.ndarray,
    method: str,
) -> tuple:
    # Equation 23; Sum over Z depths and images
    i, j = reference_pair

    targets = [k for k in range(len(cams)) if k not in reference_pair]
    maps = None
    backgrounds = np.zeros(len(targets))

    for z in z_planes:
        mappings = _get_supported_mappings(cams, X, Y, z, offsets)

        for frame in range(stacks[0].shape[0]):
            images = [
                _dewarp_window(stack[frame], mapping)
                for stack, mapping in zip(stacks, mappings, strict=True)
            ]

            current = np.asarray([
                _correlate_camera_triple(images[i], images[j], images[k], method)
                for k in targets
            ])

            if maps is None:
                maps = current
            else:
                maps = maps + current

            product_mean = np.mean(images[i] * images[j])
            target_means = np.asarray([images[k].mean() for k in targets])
            backgrounds += product_mean * target_means

    return maps, targets, backgrounds


def _measure_triple_disparities(
    maps: np.ndarray,
    targets: list,
    offsets: np.ndarray,
    spacing: np.ndarray,
    window_shape: tuple,
    method: str,
    backgrounds: np.ndarray | None = None,
) -> tuple:
    offsets = offsets.copy()
    ratios = []

    if backgrounds is None:
        backgrounds = [None] * len(targets)

    for corr, camera, background in zip(maps, targets, backgrounds, strict=True):
        signal = _remove_correlation_background(
            corr,
            window_shape,
            method,
            background,
        )

        lag, ratio = _fit_peak_2d(signal, window_shape)

        offsets[camera] = lag * spacing
        ratios.append(ratio)

    return offsets, np.asarray(ratios)


def get_disparity(
    cams: list[BaseCamera],
    images: list,
    z_planes: np.ndarray,
    world_grid: tuple | None = None,
    window_size: int | tuple = 256,
    reference_cameras: tuple = (0, 1),
    method: str = 'triple',
    min_peak_ratio: float = 4.0,
    correlation_method: str = 'circular',
    single_camera: bool = False,
) -> dict:
    """Measure camera disparities using Wieneke's image-correlation VSC.

    Implements section 3.2 of Wieneke (2018), doi:10.1088/1361-6501/aacd45.
    Triple correlation is the default and supports inline camera systems.
    The pairwise mode measures streaks for sheet profiles (section 3.3).

    Parameters
    ----------
    cams : list of openpiv.calibration.BaseCamera
        List of cameras
    images : list of ndarray
        One image (height, width), or image stack (n_frames, height, width)
        per camera. Must have same length as cams.
    z_planes : 1D ndarray
        Uniformly spaced physical depth coordinates spanning the illuminated
        volume.
    world_grid : tuple of 2D ndarray, optional
        Uniform increasing X, Y meshgrid shared by all cameras and depths.
        Both arrays must have shape (height, width).
    window_size : int or tuple of int, optional
        Central interrogation window on an automatically
        generated common grid. Ignored when world_grid is supplied.
    reference_cameras : tuple of int, optional
        Two distinct camera indices defining the provisional reference
        system.
    method : {'triple', 'pairwise'}, optional
        Triple mode requires at least three cameras. Pairwise mode requires
        two and returns streak measurements without a full camera correction.
    min_peak_ratio : float, optional
        Minimum peak-to-second-peak ratio. The paper recommends about 3-4.
    correlation_method : {'circular', 'linear'}, optional
        Cyclic FFT as in the paper, or zero-padded linear FFT correlation.
    single_camera : bool, optional
        Repeat equation 20 excluding the largest-shift camera (equation 22).

    Returns
    -------
    result : dict
        A dictionary that contains the complete results of the disparity
        matching algorithm.

    """
    if method not in ('triple', 'pairwise'):
        raise ValueError("method must be 'triple' or 'pairwise'")

    if correlation_method not in ('circular', 'linear'):
        raise ValueError("correlation_method must be 'circular' or 'linear'")

    if not np.isfinite(min_peak_ratio) or min_peak_ratio < 1:
        raise ValueError('min_peak_ratio must be finite and at least one')

    stacks = _prepare_image_stacks(cams, images)

    if method == 'triple' and len(cams) < 3:
        msg = 'Triple-image disparity requires at least three cameras'
        raise ValueError(msg)

    pair = tuple(reference_cameras)

    if (
        len(pair) != 2
        or pair[0] == pair[1]
        or any(
            not isinstance(i, (int, np.integer)) or not 0 <= i < len(cams)
            for i in pair
        )
    ):
        msg = 'reference_cameras must name two distinct camera indices'
        raise ValueError(msg)

    z_planes = _check_depth_planes(z_planes)
    z0 = float((z_planes[0] + z_planes[-1]) / 2)

    X, Y, spacing = _prepare_world_grid(cams, z0, window_size, world_grid)

    reference = np.array([
        (X.min() + X.max()) / 2,
        (Y.min() + Y.max()) / 2,
        z0,
    ])

    depth_step = float(np.diff(z_planes)[0]) / 10
    slopes = _get_los_slopes(cams, reference, depth_step)

    pairs = list(combinations(range(len(cams)), 2))
    geometries = [_get_streak_geometry(slopes, p, spacing) for p in pairs]

    pair_maps = _accumulate_pairwise_maps(
        cams,
        stacks,
        X,
        Y,
        z0,
        pairs,
        correlation_method,
    )

    measurement = _measure_pairwise_disparities(
        pair_maps,
        geometries,
        X.shape,
        z_planes[[0, -1]] - z0,
    )

    pair_valid = (
        np.isfinite(measurement['pair_disparities'])
        & (measurement['pair_peak_ratios'] >= min_peak_ratio)
    )

    result = {
        'method': method,
        'valid': bool(np.all(pair_valid)),
        'world_offsets': None,
        'image_offsets': None,
        'coordinate_shift': None,
        'pairwise_maps': pair_maps,
        'pairs': pairs,
        'pair_valid': pair_valid,
        'pair_disparities': measurement['pair_disparities'],
        'geometries': geometries,
        'grid': (X, Y),
        'spacing': spacing,
        'reference': reference,
        'z_planes': z_planes,
        'window_shape': X.shape,
    }

    if method == 'pairwise':
        return result

    result['valid'] = False

    index = pairs.index(tuple(sorted(pair)))

    if not pair_valid[index]:
        return result

    offsets = _align_reference_pair(
        pair,
        pairs,
        measurement,
        geometries,
        spacing,
        len(cams),
    )

    maps, targets, backgrounds = _accumulate_triple_maps(
        cams,
        stacks,
        X,
        Y,
        z_planes,
        pair,
        offsets,
        correlation_method,
    )

    offsets, ratios = _measure_triple_disparities(
        maps,
        targets,
        offsets,
        spacing,
        X.shape,
        correlation_method,
        backgrounds,
    )

    valid = bool(
        np.all(np.isfinite(offsets))
        and np.all(ratios >= min_peak_ratio)
    )

    result['valid'] = valid

    if valid:
        correction = _minimize_image_disparities(
            cams,
            reference,
            offsets,
            slopes,
        )

        if single_camera:
            magnitudes = np.linalg.norm(correction['world_offsets'], axis=1)
            moving = int(np.argmax(magnitudes))

            correction = _minimize_image_disparities(
                cams,
                reference,
                offsets,
                slopes,
                moving,
            )

        result['world_offsets'] = correction['world_offsets']
        result['image_offsets'] = correction['image_offsets']
        result['coordinate_shift'] = correction['coordinate_shift']

    return result
