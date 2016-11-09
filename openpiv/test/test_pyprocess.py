from openpiv.pyprocess import piv
import numpy as np
from skimage.util import random_noise
from skimage import img_as_ubyte

def test_piv():
    """ test of the simplest PIV run """
    frame_a = np.zeros((32,32))
    frame_a = random_noise(frame_a)
    frame_a = img_as_ubyte(frame_a)
    frame_b = np.roll(np.roll(frame_a,3,axis=1),2,axis=0)
    u,v = piv(frame_a,frame_b,window_size=32)
    print u,v
    assert(np.max(np.abs(u-3)) < 0.2)
    assert(np.max(np.abs(v+2)) < 0.2)
    
def test_piv_smaller_window():
    """ test of the simplest PIV run """
    frame_a = np.zeros((32,32))
    frame_a = random_noise(frame_a)
    frame_a = img_as_ubyte(frame_a)
    frame_b = np.roll(np.roll(frame_a,0,axis=1),2,axis=0)
    u,v = piv(frame_a,frame_b,window_size=8,search_size=25,corr_method='direct')
    print u,v
    assert(np.max(np.abs(u-3)) < 0.2)
    assert(np.max(np.abs(v+2)) < 0.2)
    
def test_extended_search_area():
    """ test of the extended area PIV """
    frame_a = np.zeros((256,256))
    frame_a = random_noise(frame_a)
    frame_a = img_as_ubyte(frame_a)
    frame_b = np.roll(np.roll(frame_a,3,axis=1),2,axis=0)
    u,v = piv(frame_a,frame_b,window_size=16,search_size=32)
    assert(np.max(np.abs(u-3)+np.abs(v+2)) <= 0.3)
#     
#     if (np.max(np.abs(u-3) + np.abs(v+2)) > 0.3):
#         scatter(u-3,v+2)
