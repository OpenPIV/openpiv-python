from openpiv.pyprocess import extended_search_area_piv as piv
from openpiv.tools import imread
import pathlib

import numpy as np
from .test_process import create_pair, shift_u, shift_v, threshold
from openpiv import validation

from scipy.ndimage import generic_filter, median_filter
from scipy.signal import convolve2d

import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy.typing as npt


file_a = pathlib.Path(__file__).parent / '../data/test1/exp1_001_a.bmp'
file_b = pathlib.Path(__file__).parent / '../data/test1/exp1_001_b.bmp'

frame_a = imread(file_a)
frame_b = imread(file_b)

frame_a = frame_a[:32,:32]
frame_b = frame_b[:32,:32]


def test_validation_peak2mean():
    """test of the simplest PIV run
    default window_size = 32
    """
    _, _, s2n = piv(frame_a, frame_b, 
                    window_size=32, 
                    sig2noise_method="peak2mean")

    assert np.allclose(s2n.min(),1.443882)

def test_validation_peak2peak():
    """test of the simplest PIV run
    default window_size = 32
    """
    _, _, s2n = piv(frame_a, frame_b, 
                    window_size=32, 
                    sig2noise_method="peak2peak")
    assert np.allclose(np.min(s2n), 1.24009)

def test_sig2noise_val():
    """ tests sig2noise validation """
    u = np.ones((5,5))
    v = np.ones((5,5))
    s2n_threshold = 1.05
    s2n = np.ones((5,5))*s2n_threshold
    s2n[2,2] -= 0.1
  
    mask = s2n < s2n_threshold
  
    assert not mask[0,0] # should be False
    assert mask[2,2]



def test_local_median_validation(u_threshold=3, N=3):
    """ test local median

    Args:
        u_threshold (int, optional): _description_. Defaults to 3.
        N (int, optional): _description_. Defaults to 3.
        size (int, optional): _description_. Defaults to 1.
    """
    
    u = np.random.rand(2*N+1, 2*N+1)
    u[N,N] = np.median(u)*10



    # and masked array copy
    u = np.ma.copy(u)
    u[N+1:,N+1:-1] = np.ma.masked

    # now we test our function which is just a decoration 
    # of the above steps
    flag = validation.local_median_val(u,u,u_threshold,u_threshold)

    assert flag[N,N]

def test_global_val():
    """ tests global validation
    """
    N: int=2
    U: Tuple[int,int]=(-10,10)

    u = np.random.rand(2*N+1, 2*N+1)
    u[N, N] = U[0]-.2
    u[0,0] = U[1]+.2

    u = np.ma.copy(u)
    v = np.ma.copy(u)
    v[N+1,N+1] = np.ma.masked

    mask = validation.global_val(u,u,U,U)

    assert mask[N,N]
    assert mask[0,0]

    # masked array test


    
    mask1 = validation.global_val(v,v,U,U)

    assert mask1[N,N]
    assert mask1[0,0]


def test_global_std():
    """tests global std

    Args:
        N (int, optional): array size. Defaults to 2.
        std_threshold (int, optional): threshold . Defaults to 3.
    """
    N = 2

    u = np.random.randn(2*N+1, 2*N+1)

    # print(np.nanmean(u))
    # print(np.nanstd(u))

    u[N, N] = 10.
    u[0,0] =  -10.

    v = np.ma.copy(u)
    v[N+1,N+1] = np.ma.masked


    mask = validation.global_std(u, u, 3)

    assert mask[N,N]
    assert mask[0,0] 

    mask1 = validation.global_std(v, v, std_threshold=3)

    assert mask1[N,N]
    assert mask1[0,0]

def test_uniform_shift_std(N: int=2):
    """ test for uniform shift """
    u = np.ones((2*N+1, 2*N+1))
    v = np.ma.copy(u)
    v[N+1,N+1] = np.ma.masked

    mask = validation.global_std(u, u, std_threshold=3)

    assert ~mask[N,N]


    # print(f'v.data \n {v.data}')
    # print(f'v before \n {v}')

    mask1 = validation.global_std(v, v, std_threshold=3)

    v[mask1] = np.ma.masked

    assert v.data[N,N] == 1.0
    assert v.mask[N+1,N+1]