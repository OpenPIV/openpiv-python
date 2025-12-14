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
from openpiv.settings import PIVSettings



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

    # Avoid unnecessary copy operations - work with masked arrays directly
    if np.ma.is_masked(u):
        tmpu = np.where(u.mask, np.nan, u.data)
        tmpv = np.where(v.mask, np.nan, v.data)
    else:
        tmpu = u
        tmpv = v

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

    This function validates velocity vectors based on the signal-to-noise ratio
    from the cross-correlation function. Vectors with a signal-to-noise ratio
    below the specified threshold are marked as outliers.

    Parameters
    ----------
    s2n : 2d or 3d np.ndarray
        A two or three dimensional array containing the value of the signal to
        noise ratio from cross-correlation function.

    threshold : float, default=1.0
        The signal to noise ratio threshold value. Vectors with s2n < threshold
        will be marked as outliers.

    Returns
    -------
    flag : boolean np.ndarray
        A boolean array with the same shape as s2n. True elements correspond to outliers
        (vectors with s2n < threshold).

    Notes
    -----
    - NaN values in s2n will result in False in the output mask, as NaN < threshold
      evaluates to False in NumPy.
    - This function works with both 2D and 3D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from openpiv.validation import sig2noise_val
    >>> s2n = np.array([[1.5, 0.7], [2.0, 1.2]])
    >>> mask = sig2noise_val(s2n, threshold=1.0)
    >>> print(mask)
    [[False  True]
     [False False]]

    References
    ----------
    R. D. Keane and R. J. Adrian, "Optimization of particle image velocimeters.
    Part I: Double pulsed systems," Measurement Science & Technology, 1990,
    1, 1202-1215.
    """
    ind = s2n < threshold

    return ind


def local_median_val(
        u: np.ndarray,
        v: np.ndarray,
        u_threshold: float,
        v_threshold: float,
        size: int=1
        )->np.ndarray:
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

    ind : boolean 2d np.ndarray
        a boolean array. True elements corresponds to outliers.

    """

    # kernel footprint
    # f = np.ones((2*size+1, 2*size+1))
    # f[size,size] = 0

    if np.ma.is_masked(u):
        masked_u = np.where(~u.mask, u.data, np.nan)
        masked_v = np.where(~v.mask, v.data, np.nan)
    else:
        masked_u = u
        masked_v = v

    um = generic_filter(masked_u, np.nanmedian, mode='constant',
                        cval=np.nan, size=(2*size+1, 2*size+1))
    vm = generic_filter(masked_v, np.nanmedian, mode='constant',
                        cval=np.nan, size=(2*size+1, 2*size+1))

    ind = (np.abs((u - um)) > u_threshold) | (np.abs((v - vm)) > v_threshold)

    return ind


def local_norm_median_val(
        u: np.ndarray,
        v: np.ndarray,
        ε: float,
        threshold: float,
        size: int=1
        )->np.ndarray:
    """This function is adapted from OpenPIV's implementation of
    validation.local_median_val(). validation.local_median_val() is,
    basically, Westerweel's original median filter (with some changes).
    The current function builts upon validation.local_median_val() and implements
    improved Westerweel's median filter (normalized filter) as described
    in 2007 edition of the German PIV book (paragraph 6.1.5) and in Westerweel's
    article J. Westerweel, F. Scarano, "Universal outlier detection for PIV data",
    Experiments in fluids, 39(6), p.1096-1100, 2005.
    For the list of parameters, see the referenced article, equation 2 on p.1097.
    The current function implements equation 2 from the referenced article in a
    manner shown in the MATLAB script at the end of the article, on p.1100.

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
        a two dimensional array containing the u velocity component

    v : 2d np.ndarray
        a two dimensional array containing the v velocity component

    ε : float
        minimum normalization level (see the referenced article, eqn.2)

    threshold : float
        the threshold to determine whether the vector is valid or not

    size: int
        the representative size of the kernel of the median filter, the
        actual size of the kernel is (2*size+1, 2*size+1) - i.e., it's the
        number of interrogation windows away from the interrogation
        window of interest

    Returns
    -------

    ind : boolean 2d np.ndarray
        a boolean array; true elements corresponds to outliers

    """
    if np.ma.is_masked(u):
        masked_u = np.where(~u.mask, u.data, np.nan)
        masked_v = np.where(~v.mask, v.data, np.nan)
    else:
        masked_u = u
        masked_v = v

    um = generic_filter(masked_u,
                        np.nanmedian,
                        mode='constant',
                        cval=np.nan,
                        size=(2*size+1, 2*size+1)
    )
    vm = generic_filter(masked_v,
                        np.nanmedian,
                        mode='constant',
                        cval=np.nan,
                        size=(2*size+1, 2*size+1)
    )

    def rfunc(x):
        """
        Implementation of r from the cited article (see the description of
        the function above). x is the array within the filtering kernel.
        I.e., every element of x is a velocity vector ui or vi.
        This function must return a scalar: https://stackoverflow.com/a/14060024/10073233
        """
        # copied from here: https://stackoverflow.com/a/60166608/10073233
        y = x.copy() # need this step, because np.put() below changes the array in place,
                     # and we can end up with a situation when the entire filtering kernel
                     # is comprised of NaNs resulting in NumPy RuntimeWarning: All-NaN slice encountered
        np.put(y, y.size//2, np.nan) # put NaN in the middle to avoid using
                                     # the middle in the calculations
        ym = np.nanmedian(y) # Um for the current filtering window
        rm = np.nanmedian(np.abs(np.subtract(y,ym))) # median of |ui-um| or |vi-vm|
        return rm

    rm_u = generic_filter(masked_u,
                          rfunc,
                          mode='constant',
                          cval=np.nan,
                          size=(2*size+1, 2*size+1)
    )
    rm_v = generic_filter(masked_v,
                          rfunc,
                          mode='constant',
                          cval=np.nan,
                          size=(2*size+1, 2*size+1)
    )

    r0ast_u = np.divide(np.abs(np.subtract(masked_u,um)), np.add(rm_u,ε)) # r0ast stands for r_0^* -
                                                                          # see formula 2 in the
                                                                          # referenced article
                                                                          # (see description of the function)
    r0ast_v = np.divide(np.abs(np.subtract(masked_v,vm)), np.add(rm_v,ε)) # r0ast stands for r_0^* -
                                                                          # see formula 2 in the
                                                                          # referenced article
                                                                          # (see description of the function)

    ind = (np.sqrt(np.add(np.square(r0ast_u),np.square(r0ast_v)))) > threshold

    return ind


def typical_validation(
    u: np.ndarray,
    v: np.ndarray,
    s2n: np.ndarray,
    settings: "PIVSettings"
    )->np.ndarray:
    """Comprehensive validation using multiple validation methods.

    This function applies a series of validation methods to identify outliers in
    PIV vector fields:

    1. Global validation: Checks if vectors are within specified min/max limits
    2. Standard deviation validation: Identifies vectors that deviate significantly
       from the mean
    3. Local median validation: Checks for spatial consistency using either standard
       or normalized median test
    4. Signal-to-noise validation: Validates vectors based on their signal-to-noise ratio

    Parameters
    ----------
    u : np.ndarray
        A two-dimensional array containing the u velocity component.

    v : np.ndarray
        A two-dimensional array containing the v velocity component.

    s2n : np.ndarray
        A two-dimensional array containing the signal-to-noise ratio values.

    settings : PIVSettings
        An object containing the validation parameters:

        - min_max_u_disp : tuple
            Two-element tuple setting the min/max limits for the u displacement component.

        - min_max_v_disp : tuple
            Two-element tuple setting the min/max limits for the v displacement component.

        - std_threshold : float
            Threshold for the standard deviation validation.

        - median_threshold : float
            Threshold for the median validation.

        - median_size : int
            Size of the kernel for median validation.

        - median_normalized : bool
            Whether to use normalized median validation.

        - sig2noise_validate : bool
            Whether to perform signal-to-noise validation.

        - sig2noise_threshold : float
            Threshold for the signal-to-noise validation.

        - show_all_plots : bool
            Whether to display validation plots.

    Returns
    -------
    flag : np.ndarray
        A boolean array with the same shape as u and v. True elements correspond
        to outliers that failed one or more validation tests.

    Notes
    -----
    - The function combines the results of multiple validation methods using
      logical OR operations.
    - If settings.show_all_plots is True, the function will display plots showing
      the vector field before and after each validation step.

    Examples
    --------
    >>> import numpy as np
    >>> from openpiv.validation import typical_validation
    >>> from openpiv.settings import PIVSettings
    >>> u = np.random.rand(10, 10)
    >>> v = np.random.rand(10, 10)
    >>> s2n = np.ones((10, 10)) * 2.0
    >>> settings = PIVSettings()
    >>> settings.min_max_u_disp = (-5, 5)
    >>> settings.min_max_v_disp = (-5, 5)
    >>> settings.std_threshold = 3
    >>> settings.median_threshold = 2
    >>> settings.median_size = 1
    >>> settings.sig2noise_validate = True
    >>> settings.sig2noise_threshold = 1.0
    >>> mask = typical_validation(u, v, s2n, settings)
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


    if settings.median_normalized:
        flag_m = local_norm_median_val(
            u,
            v,
            ε=0.2, # use the recomended value at this point, later add user's input for this
            threshold=settings.median_threshold,
            size=settings.median_size
        )
    else:
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