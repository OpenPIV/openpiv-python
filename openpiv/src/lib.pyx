"""A module for various utilities and helper functions"""

import numpy as np
cimport numpy as np
cimport cython

DTYPEf = np.float
ctypedef np.float_t DTYPEf_t
DTYPEi = np.int
ctypedef np.int_t DTYPEi_t


@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False) # turn of bounds-checking for entire function
def replace_nans(np.ndarray[DTYPEf_t, ndim=2] array, int max_iter, float tol, int kernel_size=2, str method='disk'):
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
    
    array : 2d np.ndarray
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
    
    filled : 2d np.ndarray
        a copy of the input array, where NaN elements have been replaced.
        
    """
    
    cdef int i, j, I, J, it, k, l
    cdef float n

    cdef np.ndarray[DTYPEf_t, ndim=2] kernel = np.empty( (2*kernel_size+1, 2*kernel_size+1), dtype=DTYPEf ) 
    
    cdef np.ndarray[DTYPEi_t, ndim=1] inans = np.empty([array.shape[0]*array.shape[1]], dtype=DTYPEi)
    cdef np.ndarray[DTYPEi_t, ndim=1] jnans = np.empty([array.shape[0]*array.shape[1]], dtype=DTYPEi)

    cdef np.ndarray[DTYPEi_t, ndim=1] iter_seeds = np.zeros(max_iter, dtype=DTYPEi)

    # indices where array is NaN
    inans, jnans = [x.astype(DTYPEi) for x in np.nonzero(np.isnan(array))]

    # number of NaN elements
    n_nans = len(inans)
    
    # arrays which contain replaced values to check for convergence
    cdef np.ndarray[DTYPEf_t, ndim=1] replaced_new = np.zeros( n_nans, dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=1] replaced_old = np.zeros( n_nans, dtype=DTYPEf)
    
    # depending on kernel type, fill kernel array
    if method == 'localmean':
        for i in range(2*kernel_size+1):
            for j in range(2*kernel_size+1):
                kernel[i,j] = 1.0

    elif method == 'disk':
        for i in range(2*kernel_size+1):
            for j in range(2*kernel_size+1):
                if ((kernel_size-i)**2 + (kernel_size-j)**2)**0.5 <= kernel_size:
                    kernel[i,j] = 1.0
                else:
                    kernel[i,j] = 0.0

    elif method == 'distance': 
        for i in range(2*kernel_size+1):
            for j in range(2*kernel_size+1):
                if ((kernel_size-i)**2 + (kernel_size-j)**2)**0.5 <= kernel_size:
                    kernel[i,j] = kernel[i,j] = -1*(((kernel_size-i)**2 + (kernel_size-j)**2)**0.5 - ((kernel_size)**2 + (kernel_size)**2)**0.5)
                else:
                    kernel[i,j] = 0.0
    else:
        raise ValueError( 'method not valid. Should be one of `localmean`, `disk` or `distance`.')
        
    # make several passes
    # until we reach convergence 
    for it in range(max_iter):
    
        # for each NaN element
        for k in range(n_nans):
            i = inans[k]
            j = jnans[k]
           
            #init to 0.0
            replaced_new[k] = 0.0
            n = 0.0
            
            # loop over the kernel
            for I in range(2*kernel_size+1):
                for J in range(2*kernel_size+1):
                   
                    # if we are not out of the boundaries
                    if i+I-kernel_size < array.shape[0] and i+I-kernel_size >= 0:
                        if j+J-kernel_size < array.shape[1] and j+J-kernel_size >= 0:
                                                
                            # if the neighbour element is not NaN itself.
                            if not np.isnan(array[i+I-kernel_size, j+J-kernel_size]):

                                # do not bother with 0 kernel values
                                if kernel[I, J] != 0:

                                    # convolve kernel with original array
                                    replaced_new[k] = replaced_new[k] + array[i+I-kernel_size, j+J-kernel_size]*kernel[I, J]
                                    n = n + kernel[I,J]

            
            # divide value by effective number of added elements
            if n > 0:
                replaced_new[k] = replaced_new[k] / n
            else:
                replaced_new[k] = np.nan

        # bulk replace all new values in array
        for k in range(n_nans):
            array[inans[k],jnans[k]] = replaced_new[k]

        # check if mean square difference between values of replaced 
        #elements is below a certain tolerance
        if np.mean((replaced_new-replaced_old)**2) < tol:
            break
        else:
            for l in range(n_nans):
                replaced_old[l] = replaced_new[l]
    
    return array
