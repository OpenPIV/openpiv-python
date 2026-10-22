import numpy as np

from .. import _cal_doc_utils
from ._match_points import (
    find_nearest_points,
    find_median_point
)


__all__ = [
    'get_pairs_marching',
    'get_pairs_auto'
]


FORWARD = 1
BACKWARD = -1
HORIZONTAL = 0
VERTICAL = 1


def _initial_steps(
    image_points: np.ndarray,
    origin: np.ndarray,
    origin_x: np.ndarray | None,
    origin_y: np.ndarray | None
) -> np.ndarray:
    selected_x = None
    selected_y = None

    if origin_x is not None:
        selected_x = find_nearest_points(
            image_points,
            origin_x
        )[:, 0] - origin

    if origin_y is not None:
        selected_y = find_nearest_points(
            image_points,
            origin_y
        )[:, 0] - origin

    if selected_x is not None and selected_y is not None:
        return np.column_stack([selected_x, selected_y])

    # If the origin axes are not supplied, guestimate best candidate best
    # candidate for marching spacing

    # Find closest neighboring points to origin
    offsets = image_points - origin[:, None]
    distances = np.linalg.norm(offsets, axis=0)
    order = np.lexsort((image_points[1], image_points[0], distances))
    order = order[distances[order] > 1e-10][:8]
    
    neighbors = offsets[:, order]
    angles = np.arctan2(neighbors[1], neighbors[0])
    neighbors = neighbors[:, np.argsort(angles)]
    
    best = None
    best_score = None

    if selected_x is not None:
        candidates = [(selected_x, point) for point in neighbors.T]
    elif selected_y is not None:
        candidates = [(point, selected_y) for point in neighbors.T]
    else:
        # Make a list of all possible axes pairs for testing
        candidates = [
            (neighbors[:, i], neighbors[:, j])
            for i in range(neighbors.shape[1])
            for j in range(
                i + 1,
                neighbors.shape[1]
            )
        ]

    for step_x, step_y in candidates:
        step_x, step_y = step_x.copy(), step_y.copy()

        if selected_x is None and selected_y is None:
            alignment_x = abs(step_x[0]) / np.linalg.norm(step_x)
            alignment_y = abs(step_y[0]) / np.linalg.norm(step_y)

            if alignment_y > alignment_x:
                step_x, step_y = step_y, step_x

            if step_x[0] < 0:
                step_x = -step_x

        lengths = np.array([
            np.linalg.norm(step_x),
            np.linalg.norm(step_y)
        ])

        basis = np.column_stack([step_x, step_y])
        determinant = np.linalg.det(basis)

        # Reject directions within about 17.5 degrees of parallel
        if np.any(lengths == 0) or abs(determinant) < 0.3 * np.prod(lengths):
            continue

        # Image x usually points right; positive grid y usually points up
        if determinant > 0:
            if selected_y is not None:
                basis[:, 0] *= -1
            else:
                basis[:, 1] *= -1

        coordinates = np.linalg.solve(basis, neighbors)
        residual = np.linalg.norm(
            coordinates - np.rint(coordinates),
            axis=0
        )
        support = np.count_nonzero(residual < 0.2)
        
        cosine = abs(
            np.dot(
                basis[:, 0],
                basis[:, 1]
            )
        ) / np.prod(lengths)
        
        score = (support, -cosine, -lengths.sum())

        if best_score is None or score > best_score:
            best, best_score = basis.copy(), score

    if best is None or best_score[0] < min(
        3,
        neighbors.shape[1]
    ):
        msg = 'Cannot infer grid axes; select origin_x and origin_y'
        raise ValueError(msg)

    return best


def _find_prediction(
    image_points: np.ndarray,
    prediction: np.ndarray,
    basis: np.ndarray,
    used: np.ndarray,
    tolerance: float
) -> int:
    available = np.flatnonzero(~used)

    if available.size == 0:
        return -1

    # Measure errors in local cells so unequal spacings and skew are allowed.
    residual = np.linalg.solve(
        basis,
        image_points[:, available] - prediction[:, None]
    )
    
    nearest = find_nearest_points(
        residual,
        np.zeros(2),
        threshold=tolerance,
        flag_nans=True,
        return_indices=True
    )[0]

    return int(available[nearest]) if nearest >= 0 else -1


def _search_line(
    image_points: np.ndarray,
    origin: np.ndarray,
    step_x: np.ndarray,
    step_y: np.ndarray,
    axis: int,
    sign: int,
    tolerance: float=0.3,
    max_missing: int=2,
    used: np.ndarray | None=None,
    stop_after_first: bool=False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis = np.array(
        [step_x, step_y],
        dtype=np.float64
    ).T

    if used is None:
        used = np.zeros(image_points.shape[1], dtype=bool)
        index = find_nearest_points(
            image_points,
            origin,
            return_indices=True
        )[0]

        if np.allclose(image_points[:, index], origin):
            used[index] = True

    current = np.asarray(origin, dtype=np.float64)
    previous = current
    previous_offset = 0
    missing = 0
    indices = []
    offsets = []
    steps = []

    for offset in range(
        1,
        image_points.shape[1] + max_missing + 2
    ):
        prediction = current + sign * basis[:, axis]
        index = _find_prediction(
            image_points,
            prediction,
            basis,
            used,
            tolerance
        )

        if index < 0:
            current = prediction
            missing += 1

            if missing > max_missing:
                break

            continue

        point = image_points[:, index]
        
        # Average over intervening cells when detections are missing.
        basis[:, axis] = (point - previous) / (sign * (offset - previous_offset))
        current = point
        previous = point
        previous_offset = offset
        missing = 0
        used[index] = True
        
        indices.append(index)
        offsets.append(sign * offset)
        steps.append(basis[:, axis].copy())

        if stop_after_first:
            break

    return (
        np.asarray(indices, dtype=int),
        np.asarray(offsets, dtype=int),
        np.asarray(steps, dtype=np.float64).reshape(-1, 2).T
    )


def search_x_positive(
    image_points: np.ndarray,
    origin: np.ndarray,
    step_x: np.ndarray,
    step_y: np.ndarray,
    **kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _search_line(
        image_points,
        origin,
        step_x,
        step_y,
        HORIZONTAL,
        FORWARD,
        **kwargs
    )


def search_x_negative(
    image_points: np.ndarray,
    origin: np.ndarray,
    step_x: np.ndarray,
    step_y: np.ndarray,
    **kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _search_line(
        image_points,
        origin,
        step_x,
        step_y,
        HORIZONTAL,
        BACKWARD,
        **kwargs
    )


def search_y_positive(
    image_points: np.ndarray,
    origin: np.ndarray,
    step_x: np.ndarray,
    step_y: np.ndarray,
    **kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _search_line(
        image_points,
        origin,
        step_x,
        step_y,
        VERTICAL,
        FORWARD,
        stop_after_first=True,
        **kwargs
    )


def search_y_negative(
    image_points: np.ndarray,
    origin: np.ndarray,
    step_x: np.ndarray,
    step_y: np.ndarray,
    **kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _search_line(
        image_points,
        origin,
        step_x,
        step_y,
        VERTICAL,
        BACKWARD,
        stop_after_first=True,
        **kwargs
    )


def _fill_row(
    image_points: np.ndarray,
    origin: np.ndarray,
    step_x: np.ndarray,
    step_y: np.ndarray,
    row: int,
    indices: list[int],
    grid: list[tuple[int, int]],
    options: dict,
    column: int=0
) -> np.ndarray:
    positive, positive_columns, positive_steps = search_x_positive(
        image_points,
        origin,
        step_x,
        step_y,
        **options
    )

    negative, negative_columns, negative_steps = search_x_negative(
        image_points,
        origin,
        step_x,
        step_y,
        **options
    )

    indices.extend(positive)
    indices.extend(negative)
    grid.extend((column + offset, row) for offset in positive_columns)
    grid.extend((column + offset, row) for offset in negative_columns)

    if positive.size:
        return positive_steps[:, 0]

    if negative.size:
        return negative_steps[:, 0]

    return step_x


def _local_steps(
    image_points: np.ndarray,
    index: int,
    coordinate: tuple[int, int],
    indices: list[int],
    grid: list[tuple[int, int]],
    step_x: np.ndarray,
    step_y: np.ndarray
) -> np.ndarray:
    basis = np.array(
        [step_x, step_y],
        dtype=np.float64
    ).T

    for axis in (HORIZONTAL, VERTICAL):
        other_axis = 1 - axis
        neighbors = [
            (neighbor, cell[axis] - coordinate[axis])
            for neighbor, cell in zip(indices, grid)
            if cell[other_axis] == coordinate[other_axis]
            and cell[axis] != coordinate[axis]
        ]

        if neighbors:
            neighbor, offset = min(
                neighbors,
                key=lambda item: abs(item[1])
            )
            basis[:, axis] = (
                image_points[:, neighbor] - image_points[:, index]
            ) / offset

    return basis


def _search_next_row(
    image_points: np.ndarray,
    column: int,
    row: int,
    step_x: np.ndarray,
    step_y: np.ndarray,
    direction: int,
    indices: list[int],
    grid: list[tuple[int, int]],
    options: dict
) -> tuple[int, int, int, np.ndarray, np.ndarray] | None:
    # Sometimes, the column used for expanding fails. So, try other columns.
    seeds = [
        (index, cell) for index, cell in zip(indices, grid) if cell[1] == row
    ]
    seeds.sort(key=lambda item: abs(item[1][0] - column))

    bases = [
        _local_steps(
            image_points,
            index,
            cell,
            indices,
            grid,
            step_x,
            step_y
        )
        for index, cell in seeds
    ]

    for distance in range(1, options['max_missing'] + 2):
        offset = direction * distance

        for (seed, cell), basis in zip(seeds, bases):
            prediction = image_points[:, seed] + offset * basis[:, VERTICAL]
            index = _find_prediction(
                image_points,
                prediction,
                basis,
                options['used'],
                options['tolerance']
            )

            if index < 0:
                continue

            options['used'][index] = True
            local_y = (
                image_points[:, index] - image_points[:, seed]
            ) / offset

            return index, cell[0], row + offset, basis[:, HORIZONTAL], local_y

    return None


def _symmetric(
    image_points: np.ndarray,
    origin_index: int,
    basis: np.ndarray,
    spacing: np.ndarray,
    tolerance: float=0.3,
    max_missing: int=2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin = image_points[:, origin_index]
    used = np.zeros(
        image_points.shape[1],
        dtype=bool
    )
    used[origin_index] = True
    indices = [origin_index]
    grid = [(0, 0)]
    
    options = dict(
        tolerance=tolerance,
        max_missing=max_missing,
        used=used
    )
    
    step_x, step_y = basis[:, 0], basis[:, 1]
    
    row_step = _fill_row(
        image_points,
        origin,
        step_x,
        step_y,
        0,
        indices,
        grid,
        options
    )

    for direction in (FORWARD, BACKWARD):
        current = origin
        local_x = row_step
        local_y = step_y
        column = 0
        row = 0

        while True:
            found = _search_next_row(
                image_points,
                column,
                row,
                local_x,
                local_y,
                direction,
                indices,
                grid,
                options
            )

            if found is None:
                break

            index, column, row, local_x, local_y = found
            current = image_points[:, index]
            
            indices.append(index)
            grid.append((column, row))
            
            local_x = _fill_row(
                image_points,
                current,
                local_x,
                local_y,
                row,
                indices,
                grid,
                options,
                column=column
            )

    grid = np.asarray(
        grid,
        dtype=np.float64
    ).T
    
    order = np.lexsort((grid[0], -grid[1]))
    indices = np.asarray(
        indices,
        dtype=int
    )[order]
    
    object_points = np.zeros((3, indices.size))
    object_points[:2] = grid[:, order] * spacing[:, None]

    return image_points[:, indices].copy(), object_points, indices


def _asymmetric(
    image_points: np.ndarray,
    matched_indices: np.ndarray,
    origin: np.ndarray,
    basis: np.ndarray,
    spacing: np.ndarray,
    asymmetric_z: float,
    tolerance: float=0.3,
    max_missing: int=2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = np.ones(
        image_points.shape[1],
        dtype=bool
    )
    keep[matched_indices] = False
    
    remaining_indices = np.flatnonzero(keep)
    remaining = image_points[:, remaining_indices]
    
    prediction = origin + 0.5 * (basis[:, 1] - basis[:, 0])
    
    index = _find_prediction(
        remaining,
        prediction,
        basis,
        np.zeros(
            remaining_indices.size,
            dtype=bool
        ),
        tolerance
    )

    if index < 0:
        msg = 'No second-layer origin found near the half-cell shift'
        raise ValueError(msg)

    paired, objects, indices = _symmetric(
        remaining,
        index,
        basis,
        spacing,
        tolerance,
        max_missing
    )
    
    objects += np.array(
        [
            -spacing[0] / 2,
            spacing[1] / 2,
            asymmetric_z
        ]
    )[:, None]

    return paired, objects, remaining_indices[indices]


@_cal_doc_utils.docfiller
def get_pairs_marching(
    image_points: np.ndarray,
    origin: np.ndarray | None=None,
    origin_x: np.ndarray | None=None,
    origin_y: np.ndarray | None=None,
    spacing: float=1.0,
    asymmetric: bool=False,
    asymmetric_z: float=0.0,
    tolerance: float=0.25,
    max_missing: int=2,
    return_indices: bool=False
):
    """Match a symmetric or two-layer asymmetric target.
    
    Pair detected image points with physical grid coordinates by
    following rows and columns from a selected or automatically
    determined origin. Local image-space step vectors update after
    each accepted marker to follow changes in spacing and direction.

    Parameters
    ----------
    %(image_points)s
    origin : array_like, optional
        Approximate image position of the zero grid point, of shape (2,).
        The nearest detection is selected. If omitted for a symmetric
        target, use the marker nearest to the median.
    origin_x, origin_y : array_like, optional
        Approximate image positions of the adjacent positive-x and
        positive-y markers, each of shape (2,). Missing axes are inferred
        from up to eight nearest neighbors for symmetric targets.
    spacing : float or sequence of two floats, optional
        Positive physical spacing between adjacent markers within one
        layer. A scalar applies to both axes; (spacing_x, spacing_y)
        specifies independent spacings. Both layers use the same spacing.
        Default is 1.0.
    asymmetric : bool, optional
        Match two interleaved symmetric grids and return both layers.
        Supplied origin and axis positions identify the first layer.
        The second layer is offset by half a cell in -x and +y. Default
        is False.
    asymmetric_z : float, optional
        Second-layer z coordinate relative to the first layer, in the
        same physical units as spacing. Default is 0.0, allowing coplanar
        interleaved grids. Used only when asymmetric is True.
    tolerance : float, optional
        Maximum Euclidean prediction error after expressing the image
        displacement in local x/y grid steps. Default is 0.25.
    max_missing : int, optional
        Number of consecutive missing grid positions allowed before a
        line search ends. Default is 2.
    return_indices : bool, optional
        Also return the column indices of matched detections in the
        original image_points array. Default is False.

    Returns
    -------
    %(image_points)s
    %(object_points)s
    indices : 1D np.ndarray, optional
        Original input column indices of shape (m,), returned only when
        return_indices is True. Their ordering matches both point arrays.

    Raises
    ------
    ValueError
        If automatic origin selection receives an empty point field,
        grid axes cannot be inferred, required asymmetric seed positions
        are omitted, or no second-layer origin is found near the
        half-cell shift.
    numpy.linalg.LinAlgError
        If the projected grid basis is singular, for example when
        supplied axis positions define parallel or zero-length steps.

    Notes
    -----
    For asymmetric targets, first-layer matches are excluded before searching
    the second layer. The image-space origin is predicted by a half-cell
    shift along the selected axes, and the object coordinates receive the
    corresponding x/y/z offsets. 

    Automatic labeling can swap or reverse physical axes on rotated symmetric
    targets. Supply both axis positions to fix the physical frame. Missing
    markers on the y spine can leave entire rows unsearched. Unrelated
    detections within the acceptance tolerance can still be matched. Each
    accepted detection is used once, and input arrays are not modified.

    Origin-and-axis initialization is inspired by CalVi [1]_. Median-based
    seed selection and expansion using locally updated grid steps follow
    ideas described by Douskos et al. [2]_.

    References
    ----------
    .. [1] G. Paolillo and T. Astarita, "CalVi User Guide",
        PaIRS version 0.1.13, revised February 7, 2024,
        Sections 9 and 11, pp. 11-13 and 19-20.
        https://www.pairs.unina.it/web/CalVi-Guide.pdf

    .. [2] V. Douskos, I. Kalisperakis, G. Karras, and E. Petsa,
        "Fully automatic camera calibration using regular planar
        patterns", Int. Arch. Photogramm. Remote Sens. Spatial
        Inf. Sci., XXXVII-B5, pp. 21-26, 2008.
        See Section 2.1.2, pp. 22-23.
        https://www.isprs.org/proceedings/XXXVII/congress/5_pdf/04.pdf

    Examples
    --------
    >>> image_points, object_points, = match_pairs_marching(
    ...     image_points,
    ...     spacing=(5.0, 5.0)
    ... )

    >>> image_points, object_points, indices = match_pairs_marching(
    ...     image_points,
    ...     origin=origin,
    ...     origin_x=origin_x,
    ...     origin_y=origin_y,
    ...     spacing=(5.0, 5.0),
    ...     asymmetric=True,
    ...     asymmetric_z=2.0,
    ...     return_indices=True
    ... )

    """
    image_points = np.asarray(
        image_points,
        dtype=np.float64
    )
    spacing = np.asarray(
        spacing,
        dtype=np.float64
    )

    if spacing.ndim == 0:
        spacing = np.repeat(
            spacing,
            2
        )

    if asymmetric and (origin is None or origin_x is None or origin_y is None):
        msg = 'An asymmetric search requires origin, origin_x and origin_y'
        raise ValueError(msg)

    if spacing[0] <= 0.0 or spacing[1] <= 0.0:
        raise ValueError('Grid spacing must be greater than 0')

    if origin is None:
        origin = find_median_point(image_points)

    origin_index = find_nearest_points(
        image_points,
        origin,
        return_indices=True
    )[0]

    origin = image_points[:, origin_index]

    basis = _initial_steps(
        image_points,
        origin,
        origin_x,
        origin_y
    )

    paired, objects, indices = _symmetric(
        image_points,
        origin_index,
        basis,
        spacing,
        tolerance,
        max_missing
    )

    if asymmetric:
        paired_second, objects_second, indices_second = _asymmetric(
            image_points,
            indices,
            origin,
            basis,
            spacing,
            asymmetric_z,
            tolerance,
            max_missing
        )

        paired = np.column_stack([paired, paired_second])
        objects = np.column_stack([objects, objects_second])
        indices = np.concatenate([indices, indices_second])

    result = paired, objects
    
    return result + (indices,) if return_indices else result


@_cal_doc_utils.docfiller
def get_pairs_auto(
    image_points: np.ndarray,
    board_size: tuple[int, int],
    origin: np.ndarray | None=None,
    origin_x: np.ndarray | None=None,
    origin_y: np.ndarray | None=None,
    spacing: float | tuple[float, float]=1.0,
    set_x_long: bool=True,
    tolerance: float=0.25,
    max_missing: int=2,
    return_indices: bool=False
):
    """Match a symmetric board using its known shape and a centered origin.

    March once with unit spacing to identify the board lattice. Use its
    shape and image directions to select an origin and adjacent axis
    markers, then march again with the requested physical spacing.

    Parameters
    ----------
    %(image_points)s
    board_size : sequence of two ints
        Full marker counts along the two board directions, including
        missing detections. Either order is accepted; set_x_long controls
        which direction becomes x. For a checkerboard, these are internal
        corner counts.
    origin : array_like, optional
        Approximate image position of the desired origin, of shape (2,).
        If omitted, select board cell (nx // 2, ny // 2), counted from
        the upper-left after assigning the axes.
    origin_x, origin_y : array_like, optional
        Approximate image positions of adjacent positive-x and positive-y
        markers, each of shape (2,). Omitted positions use the origin's
        rightward and upward grid neighbors, respectively.
    spacing : float or sequence of two floats, optional
        Positive physical marker spacing. A scalar applies to both axes;
        (spacing_x, spacing_y) follows the final x/y assignment.
        Default is 1.0.
    set_x_long : bool, optional
        Assign x to the direction with more markers on rectangular
        boards. If False, assign x to the direction with fewer markers.
        Square boards use the more horizontal direction for x regardless
        of this setting. Default is True.
    tolerance : float, optional
        Maximum prediction error in local grid cells, passed to both
        marching calls. Default is 0.25.
    max_missing : int, optional
        Consecutive missing positions allowed during marching, passed to
        both calls. Default is 2.
    return_indices : bool, optional
        Also return matched column indices in the original image_points
        array. Default is False.

    Returns
    -------
    %(image_points)s
    %(object_points)s
    indices : 1D np.ndarray, optional
        Original input column indices of shape (k,), returned only when
        return_indices is True.

    Raises
    ------
    ValueError
        If either march does not recover the requested board shape, an
        automatically selected origin or axis marker is missing, or the
        marching function cannot establish the grid.

    Notes
    -----
    Input image coordinates have shape (2, n). Returned image and object
    coordinates have shapes (2, k) and (3, k). Object z is zero, and columns
    are sorted by descending object y, then ascending object x.

    Shape is measured from integer grid extents, so missing detections
    remain holes. Automatically selected axes point rightward and upward
    in the image without being forced perpendicular. Supplied seed
    positions override the corresponding automatic selections.

    The long direction is determined by marker count, not projected pixel
    length. The first-pass lattice must identify the full board extents;
    the wrapper does not reconstruct a completely missing outer row or
    column. A square board retains the first-pass choice when the two
    directions are equally horizontal.

    """
    detected_points = np.asarray(image_points, dtype=np.float64)
    expected_shape = np.sort(board_size)

    rough_image_points, rough_object_points = get_pairs_marching(
        detected_points,
        origin=origin,
        origin_x=origin_x,
        origin_y=origin_y,
        spacing=1.0,
        tolerance=tolerance,
        max_missing=max_missing
    )

    grid = np.rint(rough_object_points[:2]).astype(int)
    shape = np.ptp(grid, axis=1) + 1

    if not np.array_equal(np.sort(shape), expected_shape):
        msg = (
            f'Detected board shape {tuple(shape.tolist())} does not match '
            f'board_size={tuple(board_size)}'
        )
        raise ValueError(msg)

    # Estimate image directions from the first-pass correspondences.
    coordinates = np.column_stack([
        grid.T,
        np.ones(grid.shape[1])
    ])
    directions = np.linalg.lstsq(
        coordinates,
        rough_image_points.T,
        rcond=None
    )[0][:2].T

    if shape[0] == shape[1]:
        horizontal = abs(directions[0]) / np.linalg.norm(directions, axis=0)
        x_axis = int(np.argmax(horizontal))
    elif set_x_long:
        x_axis = int(np.argmax(shape))
    else:
        x_axis = int(np.argmin(shape))

    axes = [x_axis, 1 - x_axis]
    grid = grid[axes]
    directions = directions[:, axes]
    shape = shape[axes]

    if directions[0, 0] < 0:
        grid[0] *= -1

    if directions[1, 1] > 0:
        grid[1] *= -1

    # Board columns increase rightward; rows increase downward.
    grid[0] -= grid[0].min()
    grid[1] = grid[1].max() - grid[1]

    if origin is None:
        origin_cell = shape // 2
    else:
        index = find_nearest_points(
            rough_image_points,
            origin,
            return_indices=True
        )[0]
        origin_cell = grid[:, index]

    cells = [
        origin_cell,
        origin_cell + [1, 0],
        origin_cell + [0, -1]
    ]
    positions = [origin, origin_x, origin_y]
    names = ['origin', 'origin_x', 'origin_y']

    for i in range(3):
        if positions[i] is not None:
            continue

        index = np.flatnonzero(np.all(grid == cells[i][:, None], axis=0))

        if index.size == 0:
            msg = (
                f'The required {names[i]} marker is missing at '
                f'board cell {tuple(cells[i].tolist())}'
            )
            raise ValueError(msg)

        positions[i] = rough_image_points[:, index[0]]

    result = get_pairs_marching(
        detected_points,
        origin=positions[0],
        origin_x=positions[1],
        origin_y=positions[2],
        spacing=spacing,
        tolerance=tolerance,
        max_missing=max_missing,
        return_indices=return_indices
    )

    spacing = np.broadcast_to(np.asarray(spacing, dtype=np.float64), (2,))
    grid = np.rint(result[1][:2] / spacing[:, None]).astype(int)
    shape = np.ptp(grid, axis=1) + 1

    if not np.array_equal(np.sort(shape), expected_shape):
        msg = (
            f'Final board shape {tuple(shape.tolist())} does not match '
            f'board_size={tuple(board_size)}'
        )
        raise ValueError(msg)

    return result