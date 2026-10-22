import numpy as np

from scipy.ndimage import gaussian_filter, map_coordinates, label
from scipy.signal import fftconvolve
from scipy.ndimage import uniform_filter, maximum_filter
from .. import _cal_doc_utils

__all__ = [
    'detect_markers'
]


def _get_default_info_dict() -> dict[str, object]:
    return {
        'status': 'no_contrast',
        'noise_sigma': 0.,
        'min_distance': None,
        'spacing': None,
        'template_shape': None,
        'min_ncc': 0.55,
        'min_snr': 8.,
        'scale_fallback': False
    }


def _correlate_template(
    image: np.ndarray,
    template: np.ndarray,
    eps: float=1e-6
) -> np.ndarray:
    centered = template - template.mean()
    corr = fftconvolve(image, centered[::-1, ::-1], mode='same')

    mean = uniform_filter(
        image, 
        size=template.shape, 
        mode='constant'
    )
    variance = uniform_filter(
        image**2, 
        size=template.shape, 
        mode='constant'
    )
    variance -= mean**2
    np.maximum(variance, 0., out=variance)

    denominator = np.sqrt(
        variance * template.size * np.sum(centered**2)
    )
    valid = denominator > eps
    corr[valid] /= denominator[valid]
    corr[~valid] = 0.

    return corr


def _find_local_peaks(
    image: np.ndarray,
    min_distance: int=1,
    threshold_abs: float=0.,
    exclude_border: bool | tuple[int, int]=False,
    num_peaks: int | None=None
) -> np.ndarray:
    local_max = maximum_filter(
        image, size=2*min_distance + 1, mode='nearest'
    )
    mask = image == local_max

    if mask.all():
        return np.empty((0, 2), dtype=int)
    
    mask &= image > threshold_abs

    if exclude_border:
        border_y, border_x = exclude_border

        if border_y:
            mask[:border_y] = False
            mask[-border_y:] = False
        if border_x:
            mask[:, :border_x] = False
            mask[:, -border_x:] = False

    coordinates = np.argwhere(mask)
    order = np.argsort(-image[mask], kind='stable')
    coordinates = coordinates[order]
    limit = len(coordinates) if num_peaks is None else min(num_peaks, len(coordinates))
    
    if min_distance <= 1 or limit == 0:
        return coordinates[:limit]

    peaks = np.empty((limit, 2), dtype=int)
    blocked = np.zeros(image.shape, dtype=bool)
    count = 0
    radius = min_distance - 1

    for y, x in coordinates:
        if blocked[y, x]:
            continue

        peaks[count] = y, x
        count += 1

        if count == limit:
            break

        blocked[
            max(0, y-radius):y+radius+1,
            max(0, x-radius):x+radius+1
        ] = True

    return peaks[:count]


def _subpixel_approximation(
    corr: np.ndarray,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    refine_radius: int=3,
    refine_iter: int=5,
    refine_cutoff: float=0.5,
    eps: float=1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Iterative subpixel marker refinement.
    
    Subpixel iterative marker refinement using least squares 2D gaussian
    peak fitting. Least squares minimization is performed by taking the 
    pseudo inverse of a matrix and multiplying it to the logarithm of the 
    correlation matrix.
    
    Paramters
    ---------
    corr : np.ndarray
        A two dimensional array of marker response values of the shape (n, m).
    pos_x, pos_y : np.ndarray
        An one dimensional array containing local peak indexes of the shape (n,).
    refine_radius : int, optional
        Radius of the local correlation patch. Automatic detection uses 1.
    refine_iter : int, optional
        The amount of iterations to perform the gaussian subpixel estimation.
    refine_cutoff : float, optional
        The cutoff number to stop iterating. Should be between 0.25 to 0.5.
    eps : float, optional
        Small positive value used for the logarithm floor. Default is 1e-6.
        
    Returns
    -------
    new_pos_x, new_pos_y : np.ndarray
        An one dimensional array containing local peak indexes of the shape (n,).
    
    Notes
    -----
    This function may become a nxn Gaussian subpixel interpolator for large
    cross correlation peaks (e.g., usign PIV on non-particle images). Hence,
    this function is documented to remind someone about this possiblity.

    """
    new_pos_x = np.array(pos_x, dtype=np.float64, copy=True)
    new_pos_y = np.array(pos_y, dtype=np.float64, copy=True)

    _sx, _sy = np.meshgrid(
        np.arange(
            -refine_radius,
            refine_radius + 1,
            dtype=float
        ),
        np.arange(
            -refine_radius,
            refine_radius + 1,
            dtype=float
        )
    )

    nx, ny = _sx.shape

    _sx = np.ravel(_sx)
    _sy = np.ravel(_sy)

    s_arr = np.array(
        [_sx, _sy, _sx**2, _sy**2, _sx*_sy, np.ones_like(_sx)],
        dtype=float
    )

    s_arr = np.reshape(
        np.concatenate(s_arr),
        (6, nx * ny)
    ).T

    # we use a psuedo-inverse via SVD so we can solve a system of equations.
    # using iterative nonlinear methods is preferable here, though.
    s_inv = np.linalg.pinv(s_arr)

    # TODO: optimize this loop
    # TODO: See if iteration actually improves anything
    for ind in range(pos_x.shape[0]): 
        x, y = pos_x[ind], pos_y[ind]

        for _ in range(refine_iter):
            previous_x, previous_y = x, y
            x = int(np.rint(x))
            y = int(np.rint(y))
            if (x - refine_radius < 0 or y - refine_radius < 0 or
                x + refine_radius >= corr.shape[1] or
                y + refine_radius >= corr.shape[0]):
                x, y = previous_x, previous_y
                break

            slices = (
                slice(
                    y - refine_radius,
                    y + refine_radius + 1),
                slice(
                    x - refine_radius,
                    x + refine_radius + 1
                )
            )

            corr_sec = corr[slices].copy()
            corr_sec[corr_sec <= 0] = eps

            coef = np.dot(
                s_inv, 
                np.log(np.ravel(corr_sec))
            )

            determinant = 4 * coef[2] * coef[3] - coef[4]**2
            if coef[2] < 0.0 and coef[3] < 0.0 and determinant > 0.0:
                # The xy term also fits tilted elliptical correlation peaks.
                shift_x = (coef[4]*coef[1] - 2*coef[3]*coef[0]) / determinant
                shift_y = (coef[4]*coef[0] - 2*coef[2]*coef[1]) / determinant
            else:
                shift_x = 0
                shift_y = 0

            if (not np.isfinite([shift_x, shift_y]).all() or
                abs(shift_x) > refine_radius or abs(shift_y) > refine_radius):
                x, y = previous_x, previous_y
                break

            new_x = x + shift_x
            new_y = y + shift_y

            d = np.hypot(previous_x - new_x, previous_y - new_y)

            x = new_x
            y = new_y

            if d < refine_cutoff:
                break

        new_pos_x[ind] = x
        new_pos_y[ind] = y
        
    return new_pos_x, new_pos_y



def _get_auto_template(
        image: np.ndarray,
        noise_sigma: float
    ) -> tuple[np.ndarray | None, int | None, float | None]:
    # Bound autocorrelation memory for megapixel cameras. Use the most energetic
    # of five representative tiles; peak extraction still uses the entire ROI.
    if max(image.shape) > 768:
        h, w = image.shape
        th, tw = min(h, 768), min(w, 768)
        tiles = [image[y:y+th, x:x+tw] for y, x in
                 ((0, 0), (0, w-tw), (h-th, 0), (h-th, w-tw),
                  ((h-th)//2, (w-tw)//2))]
        image = max(tiles, key=lambda tile: np.var(tile))

    window = np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1]))
    centered = (image - image.mean()) * window
    ac = fftconvolve(centered, centered[::-1, ::-1], mode="full")
    cy, cx = np.array(image.shape) - 1

    # Discount the white-noise spike at zero lag before measuring dot width.
    ac[cy, cx] -= noise_sigma**2 * np.sum(window**2)
    ac[cy, cx] = max(ac[cy, cx], ac[cy-1, cx], ac[cy+1, cx],
                     ac[cy, cx-1], ac[cy, cx+1])
    height = ac[cy, cx]
    if height <= 0:
        return None, None, None

    # Use a 2D component so elliptical/oblique dots retain separate x/y scales.
    labels, _ = label(ac > 0.5 * height)
    central_label = labels[cy, cx]
    if central_label == 0:
        return None, None, None
    
    yy, xx = np.nonzero(labels == central_label)
    half_y = (yy.max() - yy.min()) / 2.
    half_x = (xx.max() - xx.min()) / 2.

    if (min(half_y, half_x) < 1 or
        max(half_y, half_x) > min(image.shape) / 8.):
        return None, None, None

    # For a Gaussian dot, autocorrelation HWHM ~= 2*sqrt(log(2))*sigma.
    sigma_y, sigma_x = np.maximum([half_y, half_x], 1.5) / 1.6651

    # Mean removal narrows the measured lobe.
    # So, flat-topped disks need more support than a Gaussian of this width.
    sigma_y *= 1.5
    sigma_x *= 1.5
    ry, rx = np.maximum(3, np.ceil(3 * np.array([sigma_y, sigma_x]))).astype(int)
    sy, sx = np.mgrid[-ry:ry+1, -rx:rx+1]

    template = np.exp(-0.5 * ((sx / sigma_x)**2 + (sy / sigma_y)**2))

    # Side peaks are optional: irregular point fields do not need a lattice.
    side_peaks = _find_local_peaks(
        ac, min_distance=max(2, int(min(half_y, half_x))),
        threshold_abs=0.2 * height, exclude_border=False, num_peaks=64
    )

    distances = np.hypot(side_peaks[:, 0] - cy, side_peaks[:, 1] - cx)
    distances = distances[distances > 2 * max(half_y, half_x) + 1]
    spacing = float(distances.min()) if distances.size else None
    min_distance = max(2, int(np.ceil(2 * min(sigma_y, sigma_x))))

    if spacing is not None:
        # For preventing accidental suppresion of markers
        min_distance = max(2, min(int(spacing / 3), min_distance))

    return template, min_distance, spacing


def _auto_marker_peaks(
    image: np.ndarray,
    template: np.ndarray,
    noise_sigma: float,
    min_distance: int,
    eps: float=1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    corr = _correlate_template(image, template, eps=eps)

    centered = template - template.mean()

    matched = fftconvolve(
        image,
        centered[::-1, ::-1],
        mode="same"
    )
    snr = matched.copy()
    snr /= max(noise_sigma * np.linalg.norm(centered), eps)

    # NCC alone can assign high scores to nearly featureless noise patches
    # So, try to filter out these random peaks
    response = corr.copy()
    response[snr < 8.] = -1.
    border = tuple(int(n // 2) for n in template.shape)
    peaks = _find_local_peaks(
        response, min_distance=min_distance, threshold_abs=0.55,
        exclude_border=border
    )

    if peaks.size:
        strengths = matched[peaks[:, 0], peaks[:, 1]]
        strong = strengths[:64]  # Highest-scoring candidates first
        floor = max(0.08 * np.median(strong), 0.005 * strengths.max())
        peaks = peaks[strengths >= floor]
        
    return peaks, corr, snr


@_cal_doc_utils.docfiller
def detect_markers(
    image: np.ndarray,
    invert: bool=False,
    roi: list[int | None] | None=None,
    template: np.ndarray | None=None,
    refine_pos: bool=True,
    return_corr: bool=False,
    return_template: bool=False,
    return_info: bool=False,
    eps: float=1e-6
) -> list[np.ndarray | dict[str, object]]:
    """Detect candidate dot centers with an automatic template estimator.

    Taking a section of the image, generate a template and approximate
    dot spacing based on autocorrelation of the calibration target. The
    template is then correlated with the image, and marker positions are
    extracted by locating local correlation peaks. Optionally, refine these
    positions using a gaussian subpixel peak fit.

    Parameters
    ----------
    image : 2D np.ndarray
        A two dimensional array of pixel intensities of shape (n, m).
        The image must contain finite values and is copied internally to
        avoid modifying the original calibration image.
    invert : bool, optional
        If True, invert the image to detect dark markers on a light
        background. Otherwise, detect bright markers on a dark background.
    roi : list, optional
        A four element list containing minimum x, y and maximum x, y in
        pixels. Maximum bounds are exclusive. If None, use the full image.
        The selected image region must be at least 16 pixels in each axis.
    template : 2D np.ndarray, optional
        A representative marker template. If None, estimate the template
        automatically. A supplied template must have odd dimensions,
        contain finite, nonconstant values, and fit inside the selected
        image region. The template should represent a bright marker,
        including when invert is True.
    refine_pos : bool, optional
        If True, refine the detected positions using a gaussian fit to
        the local correlation peaks. Otherwise, return integer peak
        locations stored as floating point coordinates.
    return_corr : bool, optional
        If True, append the normalized correlation field to the result.
        The field has the same shape as the original image, with values
        outside the selected image region set to zero.
    return_template : bool, optional
        If True, append the template used for marker detection.
    return_info : bool, optional
        If True, append a dictionary containing the detection status,
        estimated noise, template shape, peak separation, approximate
        dot spacing, and use of the alternative scale estimator.
    eps : float, optional
        Small positive value used for numerical floors. Default is 1e-6.

    Returns
    -------
    %(image_points)s

    Notes
    -----
    Image normalization and smooth background removal are performed
    internally. Autocorrelation estimates marker size and separation.
    The final template is constructed from observed marker images when
    suitable candidates are available. If the calibration grid is black on
    white (e.g., black dots), inverting the image is highly recommended.
    
    """
    image = np.array(image, dtype=np.float64, copy=True)
    if image.ndim != 2 or image.size == 0 or not np.isfinite(image).all():
        raise ValueError('image must be a nonempty, finite 2D array')
    
    full_shape = image.shape
    off_x = off_y = 0
    if roi is not None:
        if len(roi) != 4:
            raise ValueError('roi must contain xmin, ymin, xmax, ymax')
        
        x0, x1, _ = slice(roi[0], roi[2]).indices(image.shape[1])
        y0, y1, _ = slice(roi[1], roi[3]).indices(image.shape[0])
        
        image = image[y0:y1, x0:x1]
        off_x, off_y = x0, y0

    if min(image.shape) < 32:
        raise ValueError('image/roi must be at least 32 pixels on each axis')

    supplied = template is not None
    if supplied:
        template = np.array(template, dtype=np.float64, copy=True)

        # Check to make sure the template is odd, not larger than the image,
        # and is not constant
        flag = any(t > n for t, n in zip(template.shape, image.shape))
        flag |= any(n % 2 == 0 for n in template.shape)
        flag |= np.ptp(template) == 0

        if flag:
            raise ValueError('Invalid template')

    positions = np.empty((2, 0), dtype=np.float64)
    corr = np.zeros_like(image)

    # The info dict stores information about what occured during the marker
    # detection process. This is usefull because sometimes the detector fails
    # for seemingly no reason. The info dict can help figure out why it failed.
    info = _get_default_info_dict()
    
    if np.ptp(image) > 0:
        image -= image.min()
        image /= image.max()

        if invert:
            image = 1. - image

        differences = [np.diff(image, axis=axis).ravel() for axis in (0, 1)]
        noise_sigma = min(np.median(np.abs(d - np.median(d)))
                          for d in differences) / (0.67448975 * np.sqrt(2.))
        noise_sigma = max(noise_sigma, eps)

        image -= gaussian_filter(image, sigma=max(3., min(image.shape) / 16.))

        info.update(noise_sigma=noise_sigma, status="no_reliable_scale")

        if supplied:
            min_distance = max(2, min(template.shape) // 4)
            spacing = None
        else:
            template, min_distance, spacing = _get_auto_template(image, noise_sigma)
            
            if template is None:
                # Strong smooth illumination can dominate the broad AC lobe.
                # Retry scale estimation only at a narrower background scale.
                scale_image = image - gaussian_filter(image, sigma=3.)
                template, min_distance, spacing = _get_auto_template(
                    scale_image,
                    noise_sigma
                )
                info['scale_fallback'] = True

        if template is not None:
            peaks, corr, snr = _auto_marker_peaks(
                image, 
                template, 
                noise_sigma, 
                min_distance,
                eps=eps
                )
            
            if not supplied and peaks.size:
                # Highest-scoring seeds; median resists a few unusual markers
                order = np.argsort(corr[peaks[:, 0], peaks[:, 1]])[::-1][:64]
                seeds = peaks[order]
                pos_x, pos_y = _subpixel_approximation(
                    corr,
                    seeds[:, 1],
                    seeds[:, 0],
                    refine_radius=1,
                    refine_iter=1,
                    eps=eps
                )

                ry, rx = np.array(template.shape) // 2
                valid = ((pos_x >= rx) & (pos_y >= ry) &
                         (pos_x <= image.shape[1]-rx-1) &
                         (pos_y <= image.shape[0]-ry-1))
                pos_x, pos_y = pos_x[valid], pos_y[valid]
                yy, xx = np.mgrid[-ry:ry+1, -rx:rx+1]
                patches = []

                for x, y in zip(pos_x, pos_y):
                    patch = map_coordinates(image, [y + yy, x + xx], order=1)
                    edge = np.concatenate(
                        [
                            patch[0],
                            patch[-1],
                            patch[:, 0],
                            patch[:, -1]
                        ]
                    )
                    patch = np.maximum(patch - np.median(edge), 0.)

                    if patch.max() > 0:
                        patches.append(patch / patch.max())

                if patches:
                    template = np.median(patches, axis=0)
                    peaks, corr, snr = _auto_marker_peaks(
                        image, 
                        template,
                        noise_sigma,
                        min_distance,
                        eps=eps
                    )

            positions = peaks[:, ::-1].T.astype(np.float64)

            if refine_pos and positions.shape[1]:
                positions[0], positions[1] = _subpixel_approximation(
                    corr,
                    positions[0],
                    positions[1],
                    refine_radius=1,
                    refine_iter=1,
                    eps=eps
                )

            ry, rx = np.array(template.shape) // 2
            valid = ((positions[0] >= rx) & (positions[1] >= ry) &
                     (positions[0] <= image.shape[1]-rx-1) &
                     (positions[1] <= image.shape[0]-ry-1))
            positions = positions[:, valid]

            info.update(
                status="ok" if positions.shape[1] else "no_peaks",
                min_distance=min_distance, spacing=spacing,
                template_shape=tuple(template.shape)
            )

    positions += np.array([[off_x], [off_y]])
    positions = positions[:, np.lexsort((positions[1], positions[0]))]
    return_list = [positions]

    if return_corr:
        full_corr = np.zeros(full_shape, dtype=np.float64)
        full_corr[off_y:off_y+corr.shape[0], off_x:off_x+corr.shape[1]] = corr
        return_list.append(full_corr)

    if return_template:
        return_list.append(template if template is not None else np.empty((0, 0)))
    
    if return_info:
        return_list.append(info)

    return return_list