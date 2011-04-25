cimport numpy as np
cimport cython


DTYPE = np.int64
ctypedef np.int64_t DTYPE_t
    

def processFFT( np.ndarray[DTYPE_t, ndim=2] frame_a, np.ndarray[DTYPE_t, ndim=2] frame_b, np.ndarray[DTYPE_t, ndim=2] u, np.ndarray[DTYPE_t, ndim=2] v ):
    """
    A cython wrapper for basic cross-correlation algorithm implemented in processFFT.cpp
    """
    return 
