"""The openpiv.filters module contains some filtering/smoothing routines."""

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

from openpiv.lib import replace_nans
import numpy as np
from scipy.signal import convolve
  
    
def _gaussian_kernel( half_width=1 ):
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
    size = int(half_width)
    x, y = np.mgrid[-half_width:half_width+1, -half_width:half_width+1]
    g = np.exp(-(x**2/float(half_width)+y**2/float(half_width)))
    return g / g.sum()

def gaussian_kernel(sigma, truncate=4.0):
    """
    Return Gaussian that truncates at the given number of standard deviations. 
    """

    sigma = float(sigma)
    radius = int(truncate * sigma + 0.5)

    x, y = np.mgrid[-radius:radius+1, -radius:radius+1]
    sigma = sigma**2

    k = 2*np.exp(-0.5 * (x**2 + y**2) / sigma)
    k = k / np.sum(k)

    return k


def gaussian( u, v, half_width=1) :
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
    g = _gaussian_kernel( half_width=half_width )
    uf = convolve( u, g, mode='same')
    vf = convolve( v, g, mode='same')
    return uf, vf


def replace_outliers( u, v, w=None, method='localmean', max_iter=5, tol=1e-3, kernel_size=1):
    """Replace invalid vectors in an velocity field using an iterative image inpainting algorithm.
    
    The algorithm is the following:
    
    1) For each element in the arrays of the ``u`` and ``v`` components, replace it by a weighted average
       of the neighbouring elements which are not invalid themselves. The weights depends
       of the method type. If ``method=localmean`` weight are equal to 1/( (2*kernel_size+1)**2 -1 )
       
    2) Several iterations are needed if there are adjacent invalid elements.
       If this is the case, inforation is "spread" from the edges of the missing
       regions iteratively, until the variation is below a certain threshold. 
    
    Parameters
    ----------
    
    u : 2d or 3d np.ndarray
        the u velocity component field
        
    v : 2d or 3d  np.ndarray
        the v velocity component field

    w : 2d or 3d  np.ndarray
        the w velocity component field
        
    max_iter : int
        the number of iterations
    fil
    kernel_size : int
        the size of the kernel, default is 1
        
    method : str
        the type of kernel used for repairing missing vectors
        
    Returns
    -------
    uf : 2d or 3d np.ndarray
        the smoothed u velocity component field, where invalid vectors have been replaced
        
    vf : 2d or 3d np.ndarray
        the smoothed v velocity component field, where invalid vectors have been replaced

    wf : 2d or 3d np.ndarray
        the smoothed w velocity component field, where invalid vectors have been replaced
        
    """
    uf = replace_nans_py(u, method=method, max_iter=max_iter, tol=tol, kernel_size=kernel_size)
    vf = replace_nans_py(v, method=method, max_iter=max_iter, tol=tol, kernel_size=kernel_size)

    if isinstance(w, np.ndarray):
        wf =  replace_nans_py(w, method=method, max_iter=max_iter, tol=tol, kernel_size=kernel_size)
        return uf, vf, wf

    return uf, vf




def replace_nans_py(array, max_iter, tol, kernel_size = 2, method = 'disk'):

    """Replace NaN elements in an array using an iterative image inpainting algorithm.

      The algorithm is the following:

      1) For each element in the input array, replace it by a weighted average
         of the neighbouring elements which are not NaN themselves. The weights
         depend on the method type. See Methods below.

      2) Several iterations are needed if there are adjacent NaN elements.
         If this is the case, information is "spread" from the edges of the missing
         regions iteratively, until the variation is below a certain threshold.

      Methods:

      localmean - A square kernel where all elements have the same value,
                  weights are equal to n/( (2*kernel_size+1)**2 -1 ),
                  where n is the number of non-NaN elements.
      disk - A circular kernel where all elements have the same value,
             kernel is calculated by::
                 if ((S-i)**2 + (S-j)**2)**0.5 <= S:
                     kernel[i,j] = 1.0
                 else:
                     kernel[i,j] = 0.0
             where S is the kernel radius.
      distance - A circular inverse distance kernel where elements are
                 weighted proportional to their distance away from the
                 center of the kernel, elements farther away have less
                 weight. Elements outside the specified radius are set
                 to 0.0 as in 'disk', the remaining of the weights are
                 calculated as::
                     maxDist = ((S)**2 + (S)**2)**0.5
                     kernel[i,j] = -1*(((S-i)**2 + (S-j)**2)**0.5 - maxDist)
                 where S is the kernel radius.

      Parameters
      ----------

      array : 2d or 3d np.ndarray
          an array containing NaN elements that have to be replaced

      max_iter : int
          the number of iterations

      tol : float
          On each iteration check if the mean square difference between
          values of replaced elements is below a certain tolerance `tol`

      kernel_size : int
          the size of the kernel, default is 1

      method : str
          the method used to replace invalid values. Valid options are
          `localmean`, `disk`, and `distance`.

      Returns
      -------

      filled : 2d or 3d np.ndarray
          a copy of the input array, where NaN elements have been replaced.

      """

    DTYPEf = np.float
    DTYPEi = np.int

    filled = array.copy()
    n_dim = len(array.shape)

    # generating the kernel
    kernel = np.zeros([2 * kernel_size + 1] * len(array.shape), dtype=int)
    if method == 'localmean':
        kernel += 1
    elif method == 'disk':
        dist, dist_inv = get_dist(kernel, kernel_size)
        kernel[dist <= kernel_size] = 1
    elif method == 'distance':
        dist, dist_inv = get_dist(kernel, kernel_size)
        kernel[dist <= kernel_size] = dist_inv[dist <= kernel_size]
    else:
        raise ValueError('method not valid. Should be one of `localmean`, `disk` or `distance`.')

    # list of kernel array indices
    kernel_indices = np.indices(kernel.shape)
    kernel_indices = np.reshape(kernel_indices, (n_dim, (2 * kernel_size + 1) ** n_dim), order="C").T

    # indices where array is NaN
    nan_indices = np.array(np.nonzero(np.isnan(array))).T.astype(DTYPEi)

    # number of NaN elements
    n_nans = len(nan_indices)

    # arrays which contain replaced values to check for convergence
    replaced_new = np.zeros(n_nans, dtype=DTYPEf)
    replaced_old = np.zeros(n_nans, dtype=DTYPEf)

    # make several passes
    # until we reach convergence
    for it in range(max_iter):
        # note: identifying new nan indices and looping other the new indices would give slightly different result

        # for each NaN element
        for k in range(n_nans):
            ind = nan_indices[k] #2 or 3 indices indicating the position of a nan element
            # init to 0.0
            replaced_new[k] = 0.0
            n = 0.0

            # generating a list of indices of the convolution window in the array
            slice_indices = np.array(np.meshgrid(*[range(i-kernel_size,i+kernel_size+1) for i in ind]))
            slice_indices = np.reshape(slice_indices,( n_dim, (2 * kernel_size + 1) ** n_dim), order="C").T

            # loop over the kernel
            for s_index, k_index in zip(slice_indices, kernel_indices):
                s_index = tuple(s_index) # this is necessary for numpy array indexing
                k_index = tuple(k_index)

                # skip if we are outside of array boundaries, if the array element is nan or if the kernel element is zero
                if all([s >= 0 and s < bound for s, bound  in zip(s_index, filled.shape)]):
                    if not np.isnan(filled[s_index]) and kernel[k_index] != 0:
                    # convolve kernel with original array
                        replaced_new[k] = replaced_new[k] + filled[s_index] * kernel[k_index]
                        n = n + kernel[k_index]

                    # divide value by effective number of added elements
            if n > 0:
                replaced_new[k] = replaced_new[k] / n
            else:
                replaced_new[k] = np.nan

        # bulk replace all new values in array
        for k in range(n_nans):
            filled[tuple(nan_indices[k])] = replaced_new[k]

        # elements is below a certain tolerance
        if np.mean((replaced_new - replaced_old) ** 2) < tol:
            break
        else:
                replaced_old = replaced_new

    return filled





def get_dist(kernel,kernel_size):
    # generates a map of distances to the center of the kernel. This is later used to generate disk-shaped kernels and
    # fill in distance based weights

    if len(kernel.shape) == 2:
        # x and y coordinates for each points
        xs, ys = np.indices(kernel.shape)
        # maximal distance form center - distance to center (of each point)
        dist = np.sqrt((ys - kernel_size) ** 2 + (xs - kernel_size) ** 2)
        dist_inv = np.sqrt(2) * kernel_size - dist

    if len(kernel.shape) == 3:
        xs, ys, zs = np.indices(kernel.shape)
        dist = np.sqrt((ys - kernel_size) ** 2 + (xs - kernel_size) ** 2 + (zs - kernel_size) ** 2)
        dist_inv = np.sqrt(3) * kernel_size - dist

    return dist, dist_inv
