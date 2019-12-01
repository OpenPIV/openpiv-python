try:
    from openpiv.process import extended_search_area_piv as piv
except: 
    from openpiv.pyprocess import extended_search_area_piv as piv
import numpy as np


from skimage.util import random_noise
from skimage import img_as_ubyte
    
import warnings

def test_piv():
    """ test of the simplest PIV run """
    frame_a = np.zeros((32,32))
    frame_a = random_noise(frame_a)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame_a = img_as_ubyte(frame_a)
    frame_b = np.roll(np.roll(frame_a,3,axis=1),2,axis=0)
    u,v = piv(frame_a.astype(np.int32),frame_b.astype(np.int32),window_size=32)
    # print u,v
    assert(np.max(np.abs(u-3)) < 0.2)
    assert(np.max(np.abs(v+2)) < 0.2)
    
def test_piv_smaller_window():
    """ test of the search area larger than the window """
    frame_a = np.zeros((32,32))
    frame_a = random_noise(frame_a)
    frame_a = img_as_ubyte(frame_a)
    frame_b = np.roll(np.roll(frame_a,-3,axis=1),-2,axis=0)
    u,v = piv(frame_a.astype(np.int32),frame_b.astype(np.int32),window_size=16,search_area_size=32)
    # print u,v
    assert(np.max(np.abs(u+3)) < 0.2)
    assert(np.max(np.abs(v-2)) < 0.2)
    
def test_extended_search_area():
    """ test of the extended area PIV with larger image """
    frame_a = np.zeros((64,64))
    frame_a = random_noise(frame_a)
    frame_a = img_as_ubyte(frame_a)
    frame_b = np.roll(np.roll(frame_a,3,axis=1),2,axis=0)
    u,v = piv(frame_a.astype(np.int32),frame_b.astype(np.int32),window_size=16,search_area_size=32,overlap=0)
    # print u,v
    assert(np.max(np.abs(u-3)+np.abs(v+2)) <= 0.5)
    
def test_extended_search_area_overlap():
    """ test of the extended area PIV with different overlap """
    frame_a = np.zeros((64,64))
    frame_a = random_noise(frame_a)
    frame_a = img_as_ubyte(frame_a)
    frame_b = np.roll(np.roll(frame_a,3,axis=1),2,axis=0)
    u,v = piv(frame_a.astype(np.int32),frame_b.astype(np.int32),window_size=16,search_area_size=32,overlap=8)
    # print u,v
    assert(np.max(np.abs(u-3)+np.abs(v+2)) <= 0.3)
    
def test_extended_search_area_sig2noise():
    """ test of the extended area PIV with sig2peak """
    frame_a = np.zeros((64,64))
    frame_a = random_noise(frame_a)
    frame_a = img_as_ubyte(frame_a)
    frame_b = np.roll(np.roll(frame_a,3,axis=1),2,axis=0)
    u,v,s2n = piv(frame_a.astype(np.int32),frame_b.astype(np.int32),window_size=16,
                    search_area_size=32, sig2noise_method='peak2peak')
    assert(np.max(np.abs(u-3)+np.abs(v+2)) <= 0.3)
    
def test_process_extended_search_area():
    """ test of the extended area PIV from Cython """
    frame_a = np.zeros((64,64))
    frame_a = random_noise(frame_a)
    frame_a = img_as_ubyte(frame_a)
    frame_b = np.roll(np.roll(frame_a,3,axis=1),2,axis=0)
    u,v = piv(frame_a.astype(np.int32),
                                           frame_b.astype(np.int32),
window_size=16,search_area_size=32,dt=1,overlap=0)
    # print u,v
    assert(np.max(np.abs(u[:-1,:-1]-3)+np.abs(v[:-1,:-1]+2)) <= 0.3)
    
def test_piv_vs_extended_search():
    """ test of the simplest PIV run """
    import openpiv.process
    import openpiv.pyprocess
    frame_a = np.zeros((32,32))
    frame_a = random_noise(frame_a)
    frame_a = img_as_ubyte(frame_a)
    frame_b = np.roll(np.roll(frame_a,3,axis=1),2,axis=0)
    u,v = openpiv.process.extended_search_area_piv(frame_a.astype(np.int32),frame_b.astype(np.int32),window_size=32)
    u1,v1 = openpiv.pyprocess.extended_search_area_piv(frame_a.astype(np.int32),
                                           frame_b.astype(np.int32),
window_size=32,search_area_size=32,dt=1,overlap=0)
    
    print(u,v)
    print(u1,v1)
    
    assert(np.allclose(u,u1))
    assert(np.allclose(v,v1))

