from openpiv.pyprocess import extended_search_area_piv as piv
from openpiv.tools import imread
import pathlib

import numpy as np
from .test_process import create_pair, shift_u, shift_v, threshold
from openpiv import validation

from scipy.ndimage import generic_filter, median_filter
from scipy.signal import convolve2d

import matplotlib.pyplot as plt


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
    u = np.ones((5,5))
    v = np.ones((5,5))
    threshold = 1.05
    s2n = np.ones((5,5))*threshold
    s2n[2,2] -= 0.1

    u, v, mask = validation.sig2noise_val(u, v, s2n, w=None, threshold=1.05)

    assert np.isnan(u[2,2])
    assert np.sum(~np.isnan(u)) == 24   
    assert mask[0,0] == False
    assert mask[2,2] == True



def test_local_median_validation(u_threshold=3, N=3, size=1):
    
    u = np.random.rand(2*N+1, 2*N+1)
    u[N,N] = np.median(u)*10

    # print('mockup data')
    # print(u)


    # prepare two copies for comparison
    tmp = u.copy()

    # and masked array copy
    masked_u = np.ma.masked_array(u.copy(),np.ma.nomask)
    masked_u[N+1:,N+1:-1] = np.ma.masked
    # print('masked version, see inf')
    # print(masked_u.filled(np.inf))

 
    f = np.ones((2*size+1, 2*size+1))
    f[size,size] = 0
    # print('Kernel or footprint')
    # print(f)


    # # out = convolve2d(u, f, boundary='wrap', mode='same')/f.sum()
    # out = median_filter(u,footprint=f)
    # print('median filter does no work with nan')
    # print(out)

    um = generic_filter(u,np.nanmedian,mode='constant',cval=np.nan,footprint=f)
    # print('generic filter output with nan')
    # print(um)

    ind = np.abs((u - um)) > u_threshold
    # print('found outliers in places:')
    # print(ind)

    # mark those places
    u[ind] = np.nan
    # print('marked data and the mask')
    # print(u)

    mask = np.zeros(u.shape,dtype=bool)
    mask[ind] = True
    # print(mask)

    # now we test our function which is just a decoration 
    # of the above steps
    u1,u1,mask1 = validation.local_median_val(tmp,tmp,3,3)

    # print('data and its mask')
    # print(u1)
    # print(mask1)

    # Now we shall test a masked array (new in 0.23.3)
    # for the image masked data
    # image mask is a masked array property
    # while nan in the matrix is the previous validation step marker
    u2,u2,mask2 = validation.local_median_val(masked_u.copy(),masked_u.copy(),3,3)


    # print('data')
    # print(u2.data)
    # print('image mask')
    # print(u2.mask)
    # print('invalid vector mask')
    # print(mask2)


    # print('Assert expected results')
    assert np.isnan(u[N,N])

    assert mask[N,N]

    assert np.isnan(u1[N,N])
    assert mask1[N,N]

    assert np.isnan(u2.data[N,N])
    assert mask2[N,N]
    assert u2.mask[N+1,N+1]


def test_global_val(N=2,U=(-10,10)):
    u = np.random.rand(2*N+1, 2*N+1)
    u[N, N] = U[0]-.2
    u[0,0] = U[1]+.2

    v = np.ma.masked_array(u.copy(), np.ma.nomask)
    v[N+1,N+1] = np.ma.masked
    
    
    # print('\n\n\n')
    # print(u)    
    # print(v.data)
    # print(v.mask)

    u1, _, mask = validation.global_val(u,u,U,U)
    # print(f'u1 {u1}')
    # print(u1[N,N])
    assert np.isnan(u1[N,N])
    assert np.isnan(u1[0,0])
    assert mask[N,N]
    assert mask[0,0]

    # masked array test


    
    v1, _, mask1 = validation.global_val(v,v,U,U)
    assert isinstance(v1,np.ma.MaskedArray)
    assert np.isnan(v1.data[N,N])
    assert np.isnan(v1.data[0,0])
    # print(mask1)
    assert mask1[N,N]
    assert mask1[0,0]


def test_global_std(N=2,std_threshold=3):

    u = np.random.randn(2*N+1, 2*N+1)

    # print(np.nanmean(u))
    # print(np.nanstd(u))

    u[N, N] = 10.
    u[0,0] =  -10.

    v = np.ma.copy(u)
    v[N+1,N+1] = np.ma.masked
    
    
    # print('data')
    # print(u)    
    
    
    # print('masked')
    # print(v.data)
    # print(v.mask)

    # print('distances')
    # print(np.abs(u - np.nanmean(u))/np.nanstd(u))

    u1, _, mask = validation.global_std(u, u, std_threshold)

    # print('std of u')
    # print(3 * np.nanstd(u**2 + u**2))

    # print(f'u1 {u1}')
    # print(u1[N,N])
    # print(u1[0,0])
    assert np.isnan(u1[N,N])
    assert np.isnan(u1[0,0])
    assert mask[N,N]
    assert mask[0,0] 

    v1, _, mask1 = validation.global_std(v, v, std_threshold=3)

    # print(f'v1 {v1}')
    assert isinstance(v1,np.ma.MaskedArray)
    assert np.isnan(v1.data[N,N])
    assert np.isnan(v1.data[0,0])
    # print(mask1)
    assert mask1[N,N]
    assert mask1[0,0]

def test_uniform_shift_std(N=2, std_threshold=3):
    """ test for uniform shift """
    u = np.ones((2*N+1, 2*N+1))

    v = np.ma.copy(u)
    v[N+1,N+1] = np.ma.masked

    u1, _, mask = validation.global_std(u, u, std_threshold=3)

    # print(f'u1 {u1}')
    # print(u1[N,N])
    # print(u1[0,0])
    assert u1[N,N] == 1.0


    # print(f'v.data \n {v.data}')
    # print(f'v before \n {v}')

    v1, _, mask1 = validation.global_std(v, v, std_threshold=3)

    # print(f'v after \n {v}')
    # print(f'v1 {v1}')
    # print(v1[N,N])
    # print(v1[0,0])
    # print(mask1[N+1,N+1])

    assert isinstance(v1,np.ma.MaskedArray)
    assert v1.data[N,N] == 1.0
    assert v1.mask[N+1,N+1]