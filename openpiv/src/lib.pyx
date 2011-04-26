"""A cython module for common operations requiring speed."""

import numpy as np
cimport numpy as np
cimport cython

DTYPEf = np.float64
ctypedef np.float64_t DTYPEf_t

def replace_nans( np.ndarray[DTYPEf_t, ndim=2] array, int n_iter):
    
    cdef int i, j, k, l, I, J
    cdef np.ndarray[DTYPEi_t, ndim=2] filled = np.zeros([array.shape[0], array.shape[1]], dtype=DTYPEf)


    return filled
