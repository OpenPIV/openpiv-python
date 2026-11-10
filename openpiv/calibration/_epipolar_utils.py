from collections.abc import Callable
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from numpy.typing import DTypeLike

from . import _cal_doc_utils


__all__ = [
    'plot_epipolar_line'
]


@_cal_doc_utils.docfiller
def plot_epipolar_line(
    cam_struct: dict,
    project_to_z: Callable,
    image_points: np.ndarray,
    zlims: tuple[int, int],
    ax: Any | None = None,
    color=None
):
    """Plot 3D epipolar lines.

    Using the passed camera structure and projection function, plot the 3D
    epipolar line(s). By passing ax, multiple epipolar lines can be plotted
    as a visualization aid for camera alignment and multi-camera system
    performance.

    Parameters
    ----------
    %(cam_struct)s
    %(project_to_z_func)s
    %(image_points)s
    zlims : tuple[int, int]
        The start and end values of the epipolar line.
    ax : matplotlib.axes.Axes, optional
        The axis of which to plot the epipolar line.
    color : str, optional
        The color of the epipolar line.

    Returns
    -------
    fig, ax : matplotlib figure, optional
        If an axis is not passed, a new figure and axis will be returned.

    Notes
    -----
    This function is based on a similar utility in MyPTV, which is referenced below.
    https://github.com/ronshnapp/MyPTV

    """
    Z1, Z2 = zlims

    X1, Y1, Z1 = project_to_z(
        cam_struct,
        image_points,
        z = Z1
    )

    X2, Y2, Z2 = project_to_z(
        cam_struct,
        image_points,
        z = Z2
    )

    if ax is None:
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        if color is None:
            ax.plot3D([X1,X2], [Z1,Z2], [Y1,Y2])
        else:
            ax.plot3D([X1,X2], [Z1,Z2], [Y1,Y2], c=color)

        return fig, ax

    if color is None:
        ax.plot3D([X1,X2], [Z1,Z2], [Y1,Y2])
    else:
        ax.plot3D([X1,X2], [Z1,Z2], [Y1,Y2], c=color)

    return None


def _line_intersect(
    t1: np.ndarray,
    r1: np.ndarray,
    t2: np.ndarray,
    r2: np.ndarray
):
    """Calculate where two rays intersect.

    Using two cameras, calculate the world coordinates where two rays
    intersect. This is done through an analytical solution based on
    direction vectors (r) and camera origins/translations (O).

    Parameters
    ----------
    t1: np.ndarray
        Three element numpy arrays for camera 1 origin/translation.
    r1 : np.ndarray
        Three element numpy arrays for camera 1 direction vectors.
    t2: np.ndarray
        Three element numpy arrays for camera 2 origin/translation.
    r2 : np.ndarray
        Three element numpy arrays for camera 2 direction vectors.

    Returns
    -------
    coords : np.ndarray
        The world coordinate that is nearest to the two rays intersecting.
        Invalid point columns contain NaN.
    dist : np.ndarray
        The minimum distance between the two rays. Invalid points contain NaN.

    Raises
    ------
    ValueError
        If the ray direction arrays have different shapes.

    Notes
    -----
    Function taken from MyPTV; all rights reserved. The direct link to this
    repository is provided below.
    https://github.com/ronshnapp/MyPTV

    Examples
    --------
    >>> _line_intersect(
        t1, r1,
        t2, r2
    )

    """
    if r1.shape != r2.shape:
        msg = 'Image point counts must match between cameras'
        raise ValueError(msg)

    r1r2 = r1[0]*r2[0] + r1[1]*r2[1] + r1[2]*r2[2]
    r12 = r1[0]**2 + r1[1]**2 + r1[2]**2
    r22 = r2[0]**2 + r2[1]**2 + r2[2]**2

    dO = t2-t1
    B = [r1[0]*dO[0] + r1[1]*dO[1] + r1[2]*dO[2],
         r2[0]*dO[0] + r2[1]*dO[1] + r2[2]*dO[2]]

    denominator = r1r2**2 - r12 * r22
    tolerance = 3 * np.finfo(r1.dtype).eps * r12 * r22
    valid = np.isfinite(denominator) & (np.abs(denominator) > tolerance)

    a = np.full_like(denominator, np.nan)
    b = np.full_like(denominator, np.nan)

    np.divide(
        -r22*B[0] + r1r2*B[1],
        denominator,
        out=a,
        where=valid
    )
    np.divide(
        -r1r2*B[0] + r12*B[1],
        denominator,
        out=b,
        where=valid
    )

    # now use a and b to calculate the minimum distance
    l1 = t1 + a*r1 # line 1
    l2 = t2 + b*r2 # line 2
    dist = sum((l1 - l2)**2)**0.5 # minimum distance
    coords = (l1 + l2)*0.5 # coordinate location

    valid &= np.all(np.isfinite(coords), axis=0) & np.isfinite(dist)
    coords = np.where(valid, coords, np.nan)
    dist = np.where(valid, dist, np.nan)

    return coords, dist


def _multi_line_intersect(
    lines: list,
    dtype: DTypeLike=np.float64
):
    """Calculate where two rays intersect.

    Using two cameras, calculate the world coordinates where two rays
    intersect. This is done through an analytical solution based on
    direction vectors (r) and camera origins/translations (O).

    Parameters
    ----------
    lines : list
        A list of lines consisting of a point and direction vectors.

    Returns
    -------
    coords : np.ndarray
        The world coordinate that is nearest to the intersection of all rays.
        Invalid point columns contain NaN.

    Raises
    ------
    ValueError
        If fewer than two rays are supplied, ray directions do not have
        shape (3, N), or point counts differ between the ray arrays.

    Examples
    --------
    >>> _multi_line_intersect(
        [
            [_t1, _r1],
            [_t2, _r2],
            [_t3, _r3]
        ]
    )

    """
    n_cams = len(lines)

    if n_cams < 2:
        msg = 'Triangulation needs at least two cameras'
        raise ValueError(msg)

    if lines[0][1].ndim != 2 or lines[0][1].shape[0] != 3:
        msg = 'Ray directions must have shape (3, N)'
        raise ValueError(msg)

    n_points = lines[0][1].shape[1]

    for _, directions in lines:
        if directions.shape != (3, n_points):
            msg = 'Ray directions must have matching shape (3, N)'
            raise ValueError(msg)

    A = np.zeros([3,3], dtype=dtype)
    b = np.zeros(3, dtype=dtype)

    I = np.eye(3, dtype=dtype)

    object_points = np.full([3, n_points], np.nan, dtype=dtype)

    # TODO: Optimize this loop as python loops would be slow if we had
    # 100,000 particles to iterate
    for particle in range(n_points):
        A[:, :] = 0.0
        b[:] = 0.0
        valid = True

        for cam in range(n_cams):
            t = lines[cam][0]
            r = lines[cam][1][:, particle]

            if t.ndim == 2:
                t = t[:, particle]

            if not np.all(np.isfinite(t)) or not np.all(np.isfinite(r)):
                valid = False
                break

            norm = np.linalg.norm(r)

            if not np.isfinite(norm) or norm == 0:
                valid = False
                break

            r = r / norm

            temp = I - np.outer(r, r)

            # Update A matrix and b vector
            A += temp
            b += temp.dot(t)

        if not valid or not np.all(np.isfinite(b)):
            continue

        try:
            point, _, rank, _ = np.linalg.lstsq(
                A,
                b,
                rcond=None
            )
        except np.linalg.LinAlgError:
            continue

        expected_rank = A.shape[1]

        if rank < expected_rank or not np.all(np.isfinite(point)):
            continue

        object_points[:, particle] = point

    return object_points
