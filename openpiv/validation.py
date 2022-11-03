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


import warnings
from typing import Tuple
import numpy as np
from scipy.ndimage import generic_filter
import matplotlib.pyplot as plt



def global_val(
    u: np.ndarray,
    v: np.ndarray,
    u_thresholds: Tuple[int, int],
    v_thresholds: Tuple[int, int],
    )-> np.ndarray:
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
    flag : boolean 2d np.ndarray
        a boolean array. True elements corresponds to outliers.

    """

    warnings.filterwarnings("ignore")

    ind = np.logical_or(
        np.logical_or(u < u_thresholds[0], u > u_thresholds[1]),
        np.logical_or(v < v_thresholds[0], v > v_thresholds[1]),
    )

    return ind


def global_std(
    u: np.ndarray,
    v: np.ndarray,
    std_threshold: int=5,
    )->np.ndarray:
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
    flag : boolean 2d np.ndarray
        a boolean array. True elements corresponds to outliers.

    """
    # both previous nans and masked regions are not 
    # participating in the magnitude comparison

    # def reject_outliers(data, m=2):
    #     return data[abs(data - np.mean(data)) < m * np.std(data)]    

    # create nan filled arrays where masks
    # if u,v, are non-masked, ma.copy() adds false masks
    tmpu = np.ma.copy(u).filled(np.nan)
    tmpv = np.ma.copy(v).filled(np.nan)

    ind = np.logical_or(np.abs(tmpu - np.nanmean(tmpu)) > std_threshold * np.nanstd(tmpu), 
                        np.abs(tmpv - np.nanmean(tmpv)) > std_threshold * np.nanstd(tmpv))

    if np.all(ind): # if all is True, something is really wrong
        print('Warning! probably a uniform shift data, do not use this filter')
        ind = ~ind

    return ind


def sig2noise_val(
    s2n: np.ndarray,
    threshold: float=1.0,
    )->np.ndarray:
    """ Marks spurious vectors if signal to noise ratio is below a specified threshold.

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

    flag : boolean 2d np.ndarray
        a boolean array. True elements corresponds to outliers.

    References
    ----------
    R. D. Keane and R. J. Adrian, Measurement Science & Technology, 1990,
        1, 1202-1215.

    """
    ind = s2n < threshold

    return ind 


def local_median_val(u, v, u_threshold, v_threshold, size=1):
    """Eliminate spurious vectors with a local median threshold.

    This validation method tests for the spatial consistency of the data.
    Vectors are classified as outliers and replaced with Nan (Not a Number) if
    the absolute difference with the local median is greater than a user
    specified threshold. The median is computed for both velocity components.

    The image masked areas (obstacles, reflections) are marked as masked array:
       u = np.ma.masked(u, flag = image_mask)
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

    flag : boolean 2d np.ndarray
        a boolean array. True elements corresponds to outliers.

    """

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

    return ind


def typical_validation(
    u: np.ndarray,
    v: np.ndarray,
    s2n: np.ndarray,
    settings: "PIVSettings"
    )->np.ndarray:
    """
    validation using gloabl limits and std and local median, 

    with a special option of 'no_std' for the case of completely
    uniform shift, e.g. in tests.

    see windef.PIVSettings() for the parameters:

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
        plt.gca().invert_yaxis()
        plt.title('Before (b) and global (m) local (k)')

    # flag = np.zeros(u.shape, dtype=bool)

    # Global validation
    flag_g = global_val(u, v, settings.min_max_u_disp, settings.min_max_v_disp)

    # u[flag_g] = np.ma.masked
    # v[flag_g] = np.ma.masked

    # if settings.show_all_plots:
    #     plt.quiver(u, v, color='m')

    flag_s = global_std(
        u, v, std_threshold=settings.std_threshold
    )

    # u[flag_s] = np.ma.masked
    # v[flag_s] = np.ma.masked

    # print(f"std filter invalidated {sum(flag_s.flatten())} vectors")
    # if settings.show_all_plots:
    #     plt.quiver(u,v,color='k')
    

    flag_m = local_median_val(
        u,
        v,
        u_threshold=settings.median_threshold,
        v_threshold=settings.median_threshold,
        size=settings.median_size,
    )
    
    # u[flag_m] = np.ma.masked
    # v[flag_m] = np.ma.masked
    
    # if settings.show_all_plots:
    #     plt.quiver(u,v,color='r')

    # print(f"median filter invalidated {sum(flag_m.flatten())} vectors")
    flag = flag_g | flag_m | flag_s


    if settings.sig2noise_validate:
        flag_s2n = sig2noise_val(s2n, settings.sig2noise_threshold)
        
        # u[flag_s2n] = np.ma.masked
        # v[flag_s2n] = np.ma.masked

        # print(f"s2n filter invalidated {sum(flag_s2n.flatten())} vectors")
        # if settings.show_all_plots:
        #     plt.quiver(u,v,color='g')
        #     plt.show()

        if settings.show_all_plots and sum(flag_s2n.flatten()): # if not all NaN
            plt.figure()
            plt.hist( s2n[s2n>0], 31)
            plt.show()

        flag += flag_s2n

    return flag