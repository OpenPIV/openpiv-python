"""A module for spurious vector detection."""

__licence__ = """
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


import numpy as np
from scipy.ndimage import generic_filter
import matplotlib.pyplot as plt


def global_val(u, v, u_thresholds, v_thresholds):
    """Eliminate spurious vectors with a global threshold.

    This validation method tests for the spatial consistency of the data
    and outliers vector are replaced with Nan (Not a Number) if at
    least one of the two velocity components is out of a specified global
    range.

    Parameters
    ----------
    u : 2d np.ndarray
        a two dimensional array containing the u velocity component.

    v : 2d np.ndarray
        a two dimensional array containing the v velocity component.

    u_thresholds: two elements tuple
        u_thresholds = (u_min, u_max). If ``u<u_min`` or ``u>u_max``
        the vector is treated as an outlier.

    v_thresholds: two elements tuple
        ``v_thresholds = (v_min, v_max)``. If ``v<v_min`` or ``v>v_max``
        the vector is treated as an outlier.

    Returns
    -------
    u : 2d np.ndarray
        a two dimensional array containing the u velocity component,
        where spurious vectors have been replaced by NaN.

    v : 2d np.ndarray
        a two dimensional array containing the v velocity component,
        where spurious vectors have been replaced by NaN.

    mask : boolean 2d np.ndarray
        a boolean array. True elements corresponds to outliers.

    """

    np.warnings.filterwarnings("ignore")

    ind = np.logical_or(
        np.logical_or(u < u_thresholds[0], u > u_thresholds[1]),
        np.logical_or(v < v_thresholds[0], v > v_thresholds[1]),
    )

    u[ind] = np.nan
    v[ind] = np.nan

    mask = np.zeros_like(u, dtype=bool)
    mask[ind] = True

    return u, v, mask


def global_std(u, v, std_threshold=5):
    """Eliminate spurious vectors with a global threshold defined by the
    standard deviation

    This validation method tests for the spatial consistency of the data
    and outliers vector are replaced with NaN (Not a Number) if at least
    one of the two velocity components is out of a specified global range.

    Parameters
    ----------
    u : 2d masked np.ndarray
        a two dimensional array containing the u velocity component.

    v : 2d masked np.ndarray
        a two dimensional array containing the v velocity component.

    std_threshold: float
        If the length of the vector (actually the sum of squared components) is
        larger than std_threshold times standard deviation of the flow field,
        then the vector is treated as an outlier. [default = 3]

    Returns
    -------
    u : 2d np.ndarray
        a two dimensional array containing the u velocity component,
        where spurious vectors have been replaced by NaN.

    v : 2d np.ndarray
        a two dimensional array containing the v velocity component,
        where spurious vectors have been replaced by NaN.

    mask : boolean 2d np.ndarray
        a boolean array. True elements corresponds to outliers.

    """
    # both previous nans and masked regions are not 
    # participating in the magnitude comparison

    if isinstance(u, np.ma.MaskedArray):
        vel_magnitude = u.filled(np.nan) ** 2 + v.filled(np.nan) ** 2
    else:
        vel_magnitude = u ** 2 + v ** 2

    ind = vel_magnitude > std_threshold * np.nanstd(vel_magnitude)

    if np.all(ind): # if all is True, something is really wrong
        print('Warning! everything cannot be wrong in global_std')
        ind = ~ind  

    u[ind] = np.nan
    v[ind] = np.nan

    mask = np.zeros_like(u, dtype=bool)
    mask[ind] = True

    return u, v, mask


def sig2noise_val(u, v, s2n, w=None, threshold=1.05):
    """Eliminate spurious vectors from cross-correlation signal to noise ratio.

    Replace spurious vectors with zero if signal to noise ratio
    is below a specified threshold.

    Parameters
    ----------
    u : 2d or 3d np.ndarray
        a two or three dimensional array containing the u velocity component.

    v : 2d or 3d np.ndarray
        a two or three dimensional array containing the v velocity component.

    s2n : 2d np.ndarray
        a two or three dimensional array containing the value  of the signal to
        noise ratio from cross-correlation function.
    w : 2d or 3d np.ndarray
        a two or three dimensional array containing the w (in z-direction)
        velocity component.

    threshold: float
        the signal to noise ratio threshold value.

    Returns
    -------
    u : 2d or 3d np.ndarray
        a two or three dimensional array containing the u velocity component,
        where spurious vectors have been replaced by NaN.

    v : 2d or 3d  np.ndarray
        a two or three dimensional array containing the v velocity component,
        where spurious vectors have been replaced by NaN.

    w : 2d or 3d  np.ndarray
        optional, a two or three dimensional array containing the w
        (in z-direction) velocity component, where spurious vectors
        have been replaced by NaN.

    mask : boolean 2d np.ndarray
        a boolean array. True elements corresponds to outliers.

    References
    ----------
    R. D. Keane and R. J. Adrian, Measurement Science & Technology, 1990,
        1, 1202-1215.

    """
    ind = s2n < threshold

    u[ind] = np.nan
    v[ind] = np.nan

    mask = np.zeros_like(u, dtype=bool)
    mask[ind] = True

    if isinstance(w, np.ndarray):
        w[ind] = np.nan
        return u, v, w, mask

    return u, v, mask


def local_median_val(u, v, u_threshold, v_threshold, size=1):
    """Eliminate spurious vectors with a local median threshold.

    This validation method tests for the spatial consistency of the data.
    Vectors are classified as outliers and replaced with Nan (Not a Number) if
    the absolute difference with the local median is greater than a user
    specified threshold. The median is computed for both velocity components.

    The image masked areas (obstacles, reflections) are marked as masked array:
       u = np.ma.masked(u, mask = image_mask)
    and it should not be replaced by the local median, but remain masked. 


    Parameters
    ----------
    u : 2d np.ndarray
        a two dimensional array containing the u velocity component.

    v : 2d np.ndarray
        a two dimensional array containing the v velocity component.

    u_threshold : float
        the threshold value for component u

    v_threshold : float
        the threshold value for component v

    Returns
    -------
    u : 2d np.ndarray
        a two dimensional array containing the u velocity component,
        where spurious vectors have been replaced by NaN.

    v : 2d np.ndarray
        a two dimensional array containing the v velocity component,
        where spurious vectors have been replaced by NaN.

    mask : boolean 2d np.ndarray
        a boolean array. True elements corresponds to outliers.

    """
    # make a copy of the data without the masked region, fill it also with 
    # NaN and then use generic filter with nanmean
    # 

    u = np.ma.MaskedArray(u)
    v = np.ma.MaskedArray(v)
    
    # kernel footprint
    f = np.ones((2*size+1, 2*size+1))
    f[size,size] = 0

    masked_u = np.where(~u.mask, u.data, np.nan)
    masked_v = np.where(~v.mask, v.data, np.nan)

    um = generic_filter(masked_u, np.nanmedian, mode='constant',
                        cval=np.nan, footprint=f)
    vm = generic_filter(masked_v, np.nanmedian, mode='constant',
                        cval=np.nan, footprint=f)

    ind = (np.abs((u - um)) > u_threshold) | (np.abs((v - vm)) > v_threshold)

    u[ind] = np.nan
    v[ind] = np.nan

    mask = np.zeros(u.shape, dtype=bool)
    mask[ind] = True

    return u, v, mask


def typical_validation(u, v, s2n, settings):

    """
    validation using gloabl limits and std and local median, 

    with a special option of 'no_std' for the case of completely
    uniform shift, e.g. in tests. 

    see Settings() for the parameters:

    MinMaxU : two elements tuple
        sets the limits of the u displacment component
        Used for validation.

    MinMaxV : two elements tuple
        sets the limits of the v displacment component
        Used for validation.

    std_threshold : float
        sets the  threshold for the std validation

    median_threshold : float
        sets the threshold for the median validation
    """

    if settings.show_all_plots:
        plt.figure()
        plt.quiver(u,v,color='b')

    mask = np.zeros(u.shape, dtype=bool)

    u, v, mask_g = global_val(
        u, v, settings.MinMax_U_disp, settings.MinMax_V_disp
    )
    # print(f"global filter invalidated {sum(mask_g.flatten())} vectors")
    if settings.show_all_plots:
        plt.quiver(u,v,color='m')

    u, v, mask_s = global_std(
        u, v, std_threshold=settings.std_threshold
    )
    # print(f"std filter invalidated {sum(mask_s.flatten())} vectors")
    if settings.show_all_plots:
        plt.quiver(u,v,color='k')
    

    u, v, mask_m = local_median_val(
        u,
        v,
        u_threshold=settings.median_threshold,
        v_threshold=settings.median_threshold,
        size=settings.median_size,
    )
    if settings.show_all_plots:
        plt.quiver(u,v,color='r')

    # print(f"median filter invalidated {sum(mask_m.flatten())} vectors")
    mask = mask + mask_g + mask_m + mask_s


    if settings.sig2noise_validate:
        u, v, mask_s2n = sig2noise_val(
            u, v, s2n,
            threshold=settings.sig2noise_threshold
        )
        # print(f"s2n filter invalidated {sum(mask_s2n.flatten())} vectors")
        if settings.show_all_plots:
            plt.quiver(u,v,color='g')
            plt.show()

        if settings.show_all_plots and sum(mask_s2n.flatten()): # if not all NaN
            plt.figure()
            plt.hist(s2n[s2n>0].flatten(),31)
            plt.show()

        mask += mask_s2n

    return u, v, mask