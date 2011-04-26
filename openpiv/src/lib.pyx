"""A cython module for common operations requiring speed."""

import numpy as np
cimport numpy as np
cimport cython

DTYPEf = np.float64
ctypedef np.float64_t DTYPEf_t

DTYPEi = np.int64
ctypedef np.int64_t DTYPEi_t

@cython.boundscheck(False) # turn of bounds-checking for entire function
@cython.wraparound(False) # turn of bounds-checking for entire function
def replace_nans( np.ndarray[DTYPEf_t, ndim=2] array, int n_iter, int kernel_size, str method='localmean'):
    
    cdef int i, j, I, J, it, n, k
    
    
    cdef np.ndarray[DTYPEf_t, ndim=2] filled = np.empty( [array.shape[0], array.shape[1]], dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] kernel = np.empty( (2*kernel_size+1, 2*kernel_size+1), dtype=DTYPEf ) 
    
    cdef np.ndarray[DTYPEi_t, ndim=1] inans 
    cdef np.ndarray[DTYPEi_t, ndim=1] jnans
    
    # indices where array is Not A Number
    inans, jnans = np.nonzero( np.isnan(array) )

    # depending on kernel type, fill kernel array
    if method == 'localmean':
        for i in range(2*kernel_size+1):
            for j in range(2*kernel_size+1):
                kernel[i,j] = 1.0
    
    # fill new array with 
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            filled[i,j] = array[i,j]

    # make several passes
    for it in range(n_iter):
        
        # for each nan element
        for k in range(len(inans)):
            i = inans[k]
            j = jnans[k]
            
            # initialize to zero
            filled[i,j] = 0.0
            n = 0
            
            # loop over the kernel
            for I in range(2*kernel_size+1):
                for J in range(2*kernel_size+1):
                   
                    # if we are not out of the boundaries
                    if i+I-kernel_size < array.shape[0] and i+I-kernel_size >= 0:
                        if j+J-kernel_size < array.shape[1] and j+J-kernel_size >= 0:
                                                
                            # if the neighbour element is not a nan
                            if not np.isnan(filled[i+I-kernel_size, j+J-kernel_size]):
                                
                                # do not sum itself
                                if I-kernel_size != 0 and J-kernel_size != 0:
                                
                                    # convolve kernel with original array
                                    filled[i,j] = filled[i,j] + filled[i+I-kernel_size, j+J-kernel_size]*kernel[I, J]
                                    n = n + 1

            # divide value by effective number of added elements
            if n != 0:
                filled[i,j] = filled[i,j] / n
            else:
                filled[i,j] = np.nan
    
    return filled
