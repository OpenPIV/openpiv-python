# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:33:21 2019

@author: Theo
"""






import numpy as np


from skimage.util import random_noise
from skimage import img_as_ubyte
    
import warnings


import openpiv.windef as windef

#this test are created only to test the displacemetn evaluation of the function.
#the validation methods are not tested here ant therefore are disabled.

#circular cross correlation
def test_first_pass_circ():
    """ test of the first pass """
    frame_a = np.zeros((1024,1024))
    frame_a = random_noise(frame_a)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame_a = img_as_ubyte(frame_a)
    frame_b = np.roll(np.roll(frame_a,3,axis=1),2,axis=0)
    x, y, u, v, sig2noise_ratio= windef.first_pass(frame_a, frame_b, window_size=64, overlap=32,iterations=1,correlation_method='circular',
                                       subpixel_method='gaussian',do_sig2noise=True, sig2noise_method='peak2peak', sig2noise_mask=2)
    # print u,v
    #print('test')
    assert(np.max(np.abs(u-3)) < 0.1)
    assert(np.max(np.abs(v+2)) < 0.1)



def test_multi_pass_circ():
    """ test fot the multipass """
    window_size=(128,64,32)
    overlap=(64,32,16)
    iterations=3
    frame_a = np.zeros((1024,1024))
    frame_a = random_noise(frame_a)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame_a = img_as_ubyte(frame_a)
    frame_b = np.roll(np.roll(frame_a,3,axis=1),2,axis=0)
    x, y, u, v, sig2noise_ratio= windef.first_pass(frame_a, frame_b, window_size[0], overlap[0],iterations,correlation_method='circular', subpixel_method='gaussian',do_sig2noise=True, sig2noise_method='peak2peak', sig2noise_mask=2)
    u_old=u
    v_old=v
    i=1
    for i in range(2,iterations+1):
        x, y, u, v, sig2noise_ratio, mask=windef.multipass_img_deform(frame_a, frame_b, window_size[i-1], overlap[i-1],iterations,i, x, y, u, v, correlation_method='circular',
                             subpixel_method='gaussian', do_sig2noise=False, sig2noise_method='peak2peak', sig2noise_mask=2,
                             MinMaxU=(-100, 50), MinMaxV=(-50, 50), std_threshold=1000000, median_threshold=200000,median_size=1, filter_method='localmean',
                             max_filter_iteration=10, filter_kernel_size=2, interpolation_order=3)
    assert(np.max(np.abs(u-3)) < 0.1 and np.any(u!=u_old))
    assert(np.max(np.abs(v+2)) < 0.1 and np.any(v!=v_old))
    #the second condition is to check if the multpass is done. It need's a little numerical inaccuracy.
    
##################################################################################
####################################################################################
#linear cross correlation

def test_first_pass_lin():
    """ test of the first pass """
    frame_a = np.zeros((1024,1024))
    frame_a = random_noise(frame_a)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame_a = img_as_ubyte(frame_a)
    frame_b = np.roll(np.roll(frame_a,3,axis=1),2,axis=0)
    x, y, u, v, sig2noise_ratio= windef.first_pass(frame_a, frame_b, window_size=64, overlap=32,iterations=1,correlation_method='linear',
                                       subpixel_method='gaussian',do_sig2noise=True, sig2noise_method='peak2peak', sig2noise_mask=2)
    # print u,v
    #print('test')
    assert(np.max(np.abs(u-3)) < 0.1)
    assert(np.max(np.abs(v+2)) < 0.1)



def test_multi_pass_lin():
    """ test fot the multipass """
    window_size=(128,64,32)
    overlap=(64,32,16)
    iterations=3
    frame_a = np.zeros((1024,1024))
    frame_a = random_noise(frame_a)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame_a = img_as_ubyte(frame_a)
    frame_b = np.roll(np.roll(frame_a,3,axis=1),2,axis=0)
    x, y, u, v, sig2noise_ratio= windef.first_pass(frame_a, frame_b, window_size[0], overlap[0],iterations,correlation_method='linear', subpixel_method='gaussian',do_sig2noise=True, sig2noise_method='peak2peak', sig2noise_mask=2)
    u_old=u
    v_old=v
    i=1
    for i in range(2,iterations+1):
        x, y, u, v, sig2noise_ratio, mask=windef.multipass_img_deform(frame_a, frame_b, window_size[i-1], overlap[i-1],iterations,i, x, y, u, v, correlation_method='linear',
                             subpixel_method='gaussian', do_sig2noise=False, sig2noise_method='peak2peak', sig2noise_mask=2,
                             MinMaxU=(-100, 50), MinMaxV=(-50, 50), std_threshold=1000000, median_threshold=200000,median_size=1, filter_method='localmean',
                             max_filter_iteration=10, filter_kernel_size=2, interpolation_order=3)
    assert(np.max(np.abs(u-3)) < 0.1 and np.any(u!=u_old))
    assert(np.max(np.abs(v+2)) < 0.1 and np.any(v!=v_old))
    #the second condition is to check if the multpass is done. It need's a little numerical inaccuracy.
    