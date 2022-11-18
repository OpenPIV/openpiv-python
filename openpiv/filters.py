"""The openpiv.filters module contains some filtering/smoothing routines."""
from typing import Tuple, Optional
import numpy as np
import numpy.typing as npt
from scipy.signal import convolve
from openpiv.lib import replace_nans

__licence_ = """
Copyright (C) 2011  www.openpiv.net

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


def _gaussian_kernel(half_width: int=1)-> np.ndarray:
    """A normalized 2D Gaussian kernel array

    Parameters
    ----------
    half_width : int
        the half width of the kernel. Kernel
        has shape 2*half_width + 1 (default half_width = 1, i.e.
        a Gaussian of 3 x 3 kernel)

    Examples
    --------

    >>> from openpiv.filters import _gaussian_kernel
    >>> _gaussian_kernel(1)
    array([[ 0.04491922,  0.12210311,  0.04491922],
       [ 0.12210311,  0.33191066,  0.12210311],
       [ 0.04491922,  0.12210311,  0.04491922]])

    """
    # size = int(half_width)
    x, y = np.mgrid[-half_width:half_width + 1, -half_width:half_width + 1]
    g = np.exp(-(x ** 2 / float(half_width) + y ** 2 / float(half_width)))
    return g / g.sum()


def gaussian_kernel(sigma:float, truncate:float=4.0)->np.ndarray:
    """
    Return Gaussian that truncates at the given number of standard deviations.
    """

    radius = int(truncate * sigma + 0.5)

    x, y = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    sigma = sigma ** 2

    k = 2 * np.exp(-0.5 * (x ** 2 + y ** 2) / sigma)
    k = k / np.sum(k)

    return k


def gaussian(
    u: np.ndarray,
    v: np.ndarray,
    half_width: int=1
    )->Tuple[np.ndarray, np.ndarray]:
    """Smooths the velocity field with a Gaussian kernel.

    Parameters
    ----------
    u : 2d np.ndarray
        the u velocity component field

    v : 2d np.ndarray
        the v velocity component field

    half_width : int
        the half width of the kernel. Kernel
        has shape 2*half_width+1, default = 1

    Returns
    -------
    uf : 2d np.ndarray
        the smoothed u velocity component field

    vf : 2d np.ndarray
        the smoothed v velocity component field

    """
    g = _gaussian_kernel(half_width=half_width)
    uf = convolve(u, g, mode="same")
    vf = convolve(v, g, mode="same")
    return uf, vf


def replace_outliers(
    u: np.ndarray,
    v: np.ndarray,
    flags: np.ndarray,
    w: Optional[np.ndarray]=None,
    method: str="localmean",
    max_iter: int=5,
    tol: float=1e-3,
    kernel_size: int=1,
    )-> Tuple[np.ndarray, ...]:
    """Replace invalid vectors in an velocity field using an iterative image
        inpainting algorithm.

    The algorithm is the following:

    1) For each element in the arrays of the ``u`` and ``v`` components,
       replace it by a weighted average
       of the neighbouring elements which are not invalid themselves. The
       weights depends of the method type. If ``method=localmean`` weight
       are equal to 1/( (2*kernel_size+1)**2 -1 )

    2) Several iterations are needed if there are adjacent invalid elements.
       If this is the case, inforation is "spread" from the edges of the
       missing regions iteratively, until the variation is below a certain
       threshold.

    Parameters
    ----------

    u : 2d or 3d np.ndarray
        the u velocity component field

    v : 2d or 3d  np.ndarray
        the v velocity component field

    w : 2d or 3d  np.ndarray
        the w velocity component field

    flags : 2d array of positions with invalid vectors

    grid_mask : 2d array of positions masked by the user

    max_iter : int
        the number of iterations

    kernel_size : int
        the size of the kernel, default is 1

    method : str
        the type of kernel used for repairing missing vectors

    Returns
    -------
    uf : 2d or 3d np.ndarray
        the smoothed u velocity component field, where invalid vectors have
        been replaced

    vf : 2d or 3d np.ndarray
        the smoothed v velocity component field, where invalid vectors have
        been replaced

    wf : 2d or 3d np.ndarray
        the smoothed w velocity component field, where invalid vectors have
        been replaced

    """
    # we shall now replace NaNs only at flags positions,
    # regardless the grid_mask (which is a user-provided masked region)

    
    if not isinstance(u, np.ma.MaskedArray):
        u = np.ma.masked_array(u, mask=np.ma.nomask)
        
    # store grid_mask for reinforcement
    grid_mask = u.mask.copy()

    u[flags] = np.nan
    v[flags] = np.nan
    
    uf = replace_nans(
        u, method=method, max_iter=max_iter, tol=tol,
        kernel_size=kernel_size
    )
    vf = replace_nans(
        v, method=method, max_iter=max_iter, tol=tol,
        kernel_size=kernel_size
    )

 
    uf = np.ma.masked_array(uf, mask=grid_mask)
    vf = np.ma.masked_array(vf, mask=grid_mask)

    if isinstance(w, np.ndarray):
        w[flags] = np.nan
        wf = replace_nans(
            w, method=method, max_iter=max_iter, tol=tol,
            kernel_size=kernel_size
        )
        wf = np.ma.masked_array(wf, mask=grid_mask)
        return uf, vf, wf
    
    return uf, vf
