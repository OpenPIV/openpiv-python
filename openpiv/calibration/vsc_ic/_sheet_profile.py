import numpy as np
from scipy.ndimage import map_coordinates


__all__ = [
    'get_sheet_profile',
    'plot_sheet_profile',
]


def _sample_sheet_profile(
    corr: np.ndarray,
    lag: np.ndarray,
    normal: np.ndarray,
    window_shape: tuple,
    band_width: int,
) -> tuple:
    height, width = window_shape

    band = np.arange(-band_width, band_width + 1)

    center_x = lag[:, 0]
    center_y = lag[:, 1]

    offset_x = band * normal[0]
    offset_y = band * normal[1]

    x = center_x[:, np.newaxis] + offset_x[np.newaxis, :]
    y = center_y[:, np.newaxis] + offset_y[np.newaxis, :]

    supported_x = np.abs(x) < width / 4
    supported_y = np.abs(y) < height / 4

    supported = supported_x & supported_y

    overlap_x = 1 - np.abs(x) / width
    overlap_y = 1 - np.abs(y) / height

    overlap = overlap_x * overlap_y

    coords = [
        y + corr.shape[0] // 2,
        x + corr.shape[1] // 2,
    ]

    values = map_coordinates(
        corr,
        coords,
        order=3,
        mode='constant',
        cval=0,
    )

    values = values / np.maximum(overlap, np.finfo(float).eps)
    valid = np.all(supported, axis=1)

    return np.where(valid, values.mean(axis=1), np.nan), valid


def get_sheet_profile(
    disparity: dict,
    z: np.ndarray | None=None,
    normalize: bool=False,
    band_width: int=1,
    biased: bool=True,
) -> dict:
    """Calculate the light sheet profile.

    Calculate the relative intensity along each two-image correlation
    streak as a function of world depth. A separate profile is returned
    for each camera combination.

    Parameters
    ----------
    disparity : dict
        A disparity measurement returned by get_disparity.
    z : 1D np.ndarray, optional
        The Z (depth) coordinates at which to calculate the profiles, in
        world units. Coordinates must be finite and increasing. If None,
        the z_planes from the disparity measurement are used. Default is
        None.
    normalize : bool, optional
        If True, divide each profile by its positive maximum. Default is
        False.
    band_width : int, optional
        The half-width of the sampling band across the correlation streak,
        in rectified grid pixels. A value of 1 averages three samples across
        the streak. Default is 1.
    biased : bool, optional
        If True, take the square root of the overlap-corrected response
        after setting negative values to zero. If False, return the
        response without the square root. Default is True.

    Returns
    -------
    profile : dict
        A dictionary containing the depth coordinates and light sheet
        profiles.

    Notes
    -----
    The profiles follow section 3.3 of Wieneke (2018). Profiles for sheets
    thinner than 2 mm may be less reliable at viewing angles below 45
    degrees from the sheet normal. In synthetic tests with 50 recordings
    and a 0.5 mm sheet, the corrected FWHM errors were about 26%, 9%, and 3%
    at 30, 45, and 60 degrees, respectively. It should be mentioned, however,
    that this should not be considered a source of truth for light sheet
    characteristics. This utility is mostly to help analyze light sheet
    profiles which are typically only recovered after volumetric PIV.

    Examples
    --------
    Given an existing disparity measurement:

    >>> import numpy as np
    >>> from openpiv.calibration import vsc_ic

    >>> z = np.linspace(-5.0, 5.0, 201)
    >>> profile = vsc_ic.get_sheet_profile(disparity, z=z)

    """
    if not isinstance(band_width, (int, np.integer)) or band_width < 0:
        raise ValueError('band_width must be a nonnegative integer')

    if z is None:
        z = disparity['z_planes']
    else:
        z = np.asarray(z, dtype=np.float64)

        if z.ndim != 1 or z.size == 0:
            msg = 'z must be a nonempty one-dimensional array'
            raise ValueError(msg)

        if not np.all(np.isfinite(z)):
            msg = 'z must contain only finite values'
            raise ValueError(msg)

        if np.any(np.diff(z) <= 0):
            msg = 'z must be strictly increasing'
            raise ValueError(msg)

    profiles = []
    validity = []
    divisors = []
    offsets = disparity['world_offsets']

    pair_data = zip(
        disparity['pairs'],
        disparity['pairwise_maps'],
        disparity['geometries'],
        strict=True,
    )

    for index, ((i, j), corr, geometry) in enumerate(pair_data):
        if offsets is None:
            center = disparity['pair_disparities'][index] * geometry['normal']
        else:
            center = (offsets[j] - offsets[i]) / disparity['spacing']

        lag = center + (z - disparity['reference'][2])[:, None] * geometry['depth_direction']

        values, valid = _sample_sheet_profile(
            corr,
            lag,
            geometry['normal'],
            disparity['window_shape'],
            band_width,
        )

        valid &= disparity['pair_valid'][index]

        if biased:
            values = np.sqrt(np.maximum(values, np.float64(0)))

        if np.any(valid):
            peak = float(np.max(values[valid]))
        else:
            peak = 0.0

        if peak <= 0 or not np.isfinite(peak):
            valid[:] = False

        if normalize and peak > 0:
            divisor = peak
        else:
            divisor = 1.0

        profiles.append(np.where(valid, values / divisor, np.nan))
        validity.append(valid)
        divisors.append(divisor)

    return {
        'z': z.copy(),
        'profiles': np.asarray(profiles),
        'valid': np.asarray(validity),
        'pairs': list(disparity['pairs']),
        'normalization': np.asarray(divisors),
        'normalized': bool(normalize),
        'coordinate_shift': disparity['coordinate_shift'],
    }


def plot_sheet_profile(
    profile: dict,
    ax=None,
) -> tuple:
    """Plot the light sheet profiles.

    Plot the light sheet response against world depth for each
    camera combination in a calculated light sheet profile.

    Parameters
    ----------
    profile : dict
        A light sheet profile returned by get_sheet_profile.
    ax : matplotlib.axes.Axes, optional
        An instance of a Matplotlib axes object on which to plot the
        profiles. If None, a new figure and axes are created. Default is
        None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib figure containing the profiles.
    ax : matplotlib.axes.Axes
        The Matplotlib axes containing the profiles.

    Examples
    --------
    Given a calculated light sheet profile:

    >>> from openpiv.calibration import vsc_ic

    >>> fig, ax = vsc_ic.plot_sheet_profile(profile)
    >>> fig.savefig('light_sheet_profile.png')

    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    for (i, j), values in zip(profile['pairs'], profile['profiles'], strict=True):
        ax.plot(profile['z'], values, label=f'Camera {i} - {j}')

    ax.set_xlabel('Depth (calibration length units)')

    ax.set_yticks([])
    ax.set_yticks([], minor=True)
    ax.set_ylabel('Response (a.u.)')

    ax.legend()

    return ax.figure, ax
