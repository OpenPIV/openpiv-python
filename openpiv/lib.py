import numpy as np


def replace_nans(array, max_iter, tol, kernel_size=2, method='disk'):

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

    kernel_size = int(kernel_size)
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
    # kernel_indices = np.indices(kernel.shape)
    # kernel_indices = np.reshape(kernel_indices, (n_dim, (2 * kernel_size + 1) ** n_dim), order="C").T

    # indices where array is NaN
    nan_indices = np.array(np.nonzero(np.isnan(array))).T.astype(int)

    # number of NaN elements
    n_nans = len(nan_indices)

    # arrays which contain replaced values to check for convergence
    replaced_new = np.zeros(n_nans)
    replaced_old = np.zeros(n_nans)

    # make several passes
    # until we reach convergence
    for it in range(max_iter):
        # note: identifying new nan indices and looping other the new indices would give slightly different result

        # for each NaN element
        for k in range(n_nans):
            ind = nan_indices[k] #2 or 3 indices indicating the position of a nan element
            # init to 0.0
            replaced_new[k] = 0.0

            # generating a list of indices of the convolution window in the array
            slice_indices = np.array(np.meshgrid(*[range(i-kernel_size, i+kernel_size+1) for i in ind]))

            # identifying all indices strictly inside the image edges:
            in_mask = np.array([np.logical_and(slice_indices[i] < array.shape[i], slice_indices[i] >= 0) for i in range(n_dim)])
            # logical and over x,y (and z) indices
            in_mask = np.prod(in_mask, axis=0).astype(bool)

            # extract window from array
            win = filled[tuple(slice_indices[:, in_mask])]

            # selecting the same points from the kernel
            kernel_in = kernel[in_mask]

            # sum of elements of the kernel that are not nan in the window
            non_nan = np.sum(kernel_in[~np.isnan(win)])

            if non_nan > 0:
                # convolution with the kernel
                replaced_new[k] = np.nansum(win * kernel_in) / non_nan
            else:
                # don't do anything if there is only nans around
                replaced_new[k] = np.nan

        # bulk replace all new values in array
        filled[tuple(nan_indices.T)] = replaced_new

        # check if replaced elements are below a certain tolerance
        if np.mean((replaced_new - replaced_old) ** 2) < tol:
            break
        else:
            replaced_old = replaced_new

    return filled


def get_dist(kernel, kernel_size):
    # generates a map of distances to the center of the kernel. This is later used to generate disk-shaped kernels and
    # to fill in distance based weights

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