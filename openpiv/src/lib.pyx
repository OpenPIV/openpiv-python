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
def replace_invalid( np.ndarray[DTYPEf_t, ndim=2] array, float invalid_value, int n_iter, int kernel_size=1, str method='localmean'):
    """Replace invalid in an array using an iterative image inpainting algorithm.
    
    The algorithm is the following:
    
    1) For each element in the input array replace it by a weighted average
       of the neighbouring elements which are not invalid. The weights depends
       of the method type. If ``method=localmean`` weight are equal to 1/( (2*kernel_size+1)**2 -1 )
       
    2) Several iterations are needed if there are adjacent nan elements.
       If this is the case, inforation is "spread" from the edges of the missing
       regions iteratively, until the variation is below a certain threshold. 
    
    Parameters
    ----------
    
    array : 2d np.ndarray
        an array containing invalid elements that have to be replaced
    
    invalid_value : float
        an invalid value that has to be replaced
    
    n_iter : int
        the number of iterations
    
    kernel_size : int
        the size of the kernel, default is 1
        
    method : str
        the method used to replace invalid values. Valid options are
        `localmean`.
        
    Returns
    -------
    
    filled : 2d np.ndarray
        a copy of the input array, where invalid values have been replaced.
        
    """
    
    cdef int i, j, I, J, it, n, k
    
    
    cdef np.ndarray[DTYPEf_t, ndim=2] filled = np.empty( [array.shape[0], array.shape[1]], dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] kernel = np.empty( (2*kernel_size+1, 2*kernel_size+1), dtype=DTYPEf ) 
    
    cdef np.ndarray[DTYPEi_t, ndim=1] iinvalid
    cdef np.ndarray[DTYPEi_t, ndim=1] jinvalid
    
    # indices where array is invalid
    iinvalid, jinvalid = np.nonzero( array==invalid_value )

    # depending on kernel type, fill kernel array
    if method == 'localmean':
        for i in range(2*kernel_size+1):
            for j in range(2*kernel_size+1):
                kernel[i,j] = 1.0
    
    # fill new array with input elements
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            filled[i,j] = array[i,j]

    # make several passes
    for it in range(n_iter):
        
        # for each invalid element
        for k in range(len(iinvalid)):
            i = iinvalid[k]
            j = jinvalid[k]
            
            # initialize to zero
            filled[i,j] = 0.0
            n = 0
            
            # loop over the kernel
            for I in range(2*kernel_size+1):
                for J in range(2*kernel_size+1):
                   
                    # if we are not out of the boundaries
                    if i+I-kernel_size < array.shape[0] and i+I-kernel_size >= 0:
                        if j+J-kernel_size < array.shape[1] and j+J-kernel_size >= 0:
                                                
                            # if the neighbour element is not invalid itself
                            if filled[i+I-kernel_size, j+J-kernel_size] != invalid_value:
                                
                                # do not sum itself
                                if I-kernel_size != 0 and J-kernel_size != 0:
                                
                                    # convolve kernel with original array
                                    filled[i,j] = filled[i,j] + filled[i+I-kernel_size, j+J-kernel_size]*kernel[I, J]
                                    n = n + 1

            # divide value by effective number of added elements
            if n != 0:
                filled[i,j] = filled[i,j] / n
            else:
                filled[i,j] = invalid_value
    
    return filled
