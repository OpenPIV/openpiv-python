import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def replace_nans(array, max_iter, tol, kernel_size=2, method="disk"):
    """Replace NaN elements in an array using an iterative image inpainting
        algorithm.

      The algorithm is the following:

      1) For each element in the input array, replace it by a weighted average
         of the neighbouring elements which are not NaN themselves. The weights
         depend on the method type. See Methods below.

      2) Several iterations are needed if there are adjacent NaN elements.
         If this is the case, information is "spread" from the edges of the
         missing regions iteratively, until the variation is below a certain
         threshold.

      Methods:
        localmean - A square kernel where all elements have the same weight.
        disk - A circular kernel where all elements have the same weight.
        distance - A circular kernel where the weight of each element depends
                 on its distance from the center of the kernel. The weights
                 are given by a function of the form:
                   w_i = 1 - (d_i / d_max)^2
                 where d_i is the distance from the center, and d_max is the
                 distance of the element farthest from the center.
                 This method requires SciPy.

      Parameters
      ----------

      array : 2d or 3d np.ndarray
          an array containing NaN elements that have to be replaced
          if array is a masked array (numpy.ma.MaskedArray), then
          the mask is reapplied after the replacement

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
    # Check if there are any NaNs to replace
    if not np.any(np.isnan(array)):
        return array.copy()

    kernel_size = int(kernel_size)
    filled = array.copy()
    n_dim = len(array.shape)

    # generating the kernel
    kernel = np.zeros([2 * kernel_size + 1] * len(array.shape), dtype=int)
    if method == "localmean":
        kernel += 1
    elif method == "disk":
        dist, dist_inv = get_dist(kernel, kernel_size)
        kernel[dist <= kernel_size] = 1
    elif method == "distance":
        dist, dist_inv = get_dist(kernel, kernel_size)
        kernel[dist <= kernel_size] = dist_inv[dist <= kernel_size]
    else:
        raise ValueError(
            "Known methods are: `localmean`, `disk` or `distance`."
        )

    # list of kernel array indices
    # kernel_indices = np.indices(kernel.shape)
    # kernel_indices = np.reshape(kernel_indices,
    #   (n_dim, (2 * kernel_size + 1) ** n_dim),
    #   order="C").T

    # indices where array is NaN
    nan_indices = np.array(np.nonzero(np.isnan(array))).T.astype(int)

    # number of NaN elements
    n_nans = len(nan_indices)

    # arrays which contain replaced values to check for convergence
    replaced_old = np.zeros(n_nans)

    # broadcastable view of the kernel weights, one axis per array dimension
    # plus one matching axis per kernel dimension (for the windowed reduction)
    kernel_b = kernel.reshape((1,) * n_dim + kernel.shape)
    window_axes = tuple(range(n_dim, 2 * n_dim))

    # make several passes
    # until we reach convergence
    for _ in range(max_iter):
        # note: identifying new nan indices and looping other the new indices
        # would give slightly different result

        # NaN-pad by kernel_size so that out-of-bounds neighbours behave like
        # the boundary check in the original per-element loop (i.e. they are
        # excluded from both the weighted sum and its normalization).
        padded = np.pad(np.asarray(filled), kernel_size, mode="constant",
                         constant_values=np.nan)
        # windows.shape == array.shape + kernel.shape: one (2*kernel_size+1)^n_dim
        # neighbourhood per array element, vectorized instead of a per-NaN
        # Python loop with a fresh np.meshgrid call each time.
        windows = sliding_window_view(padded, kernel.shape)

        valid = ~np.isnan(windows)
        weights = np.where(valid, kernel_b, 0)
        non_nan = weights.sum(axis=window_axes)
        weighted_sum = (np.where(valid, windows, 0) * kernel_b).sum(axis=window_axes)

        # convolution with the kernel; stays NaN where only NaNs are around
        full_new = np.divide(weighted_sum, non_nan,
                              out=np.full(array.shape, np.nan), where=non_nan > 0)

        replaced_new = full_new[tuple(nan_indices.T)]

        # bulk replace all new values in array
        filled[tuple(nan_indices.T)] = replaced_new

        # check if replaced elements are below a certain tolerance
        if np.mean((replaced_new - replaced_old) ** 2) < tol:
            break
        else:
            replaced_old = replaced_new

    return filled


def get_dist(kernel, kernel_size):
    # generates a map of distances to the center of the kernel. This is later
    # used to generate disk-shaped kernels and
    # to fill in distance based weights

    if len(kernel.shape) == 2:
        # x and y coordinates for each points
        xs, ys = np.indices(kernel.shape)
        # maximal distance form center - distance to center (of each point)
        dist = np.sqrt((ys - kernel_size) ** 2 + (xs - kernel_size) ** 2)
        dist_inv = np.sqrt(2) * kernel_size - dist

    if len(kernel.shape) == 3:
        xs, ys, zs = np.indices(kernel.shape)
        dist = np.sqrt(
            (ys - kernel_size) ** 2 +
            (xs - kernel_size) ** 2 +
            (zs - kernel_size) ** 2
        )
        dist_inv = np.sqrt(3) * kernel_size - dist

    return dist, dist_inv
