# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:33:21 2019

@author: Theo
"""


import numpy as np
import openpiv.windef as windef
from test_process import create_pair

frame_a, frame_b = create_pair(image_size=1024)

# this test are created only to test the displacement evaluation of the
# function the validation methods are not tested here ant therefore
# are disabled.


# circular cross correlation
def test_first_pass_circ():
    """ test of the first pass """
    x, y, u, v, s2n = windef.first_pass(frame_a, frame_b,
                                        window_size=64,
                                        overlap=32,
                                        iterations=1,
                                        correlation_method='circular',
                                        subpixel_method='gaussian',
                                        do_sig2noise=True,
                                        sig2noise_method='peak2peak',
                                        sig2noise_mask=2)
    # print u,v
    assert(np.max(np.abs(u-3)) < 0.1)
    assert(np.max(np.abs(v+2)) < 0.1)


def test_multi_pass_circ():
    """ test fot the multipass """
    window_size = (128, 64, 32)
    overlap = (64, 32, 16)
    iterations = 3

    x, y, u, v, s2n = windef.first_pass(frame_a, frame_b, window_size[0],
                                        overlap[0], iterations,
                                        correlation_method='circular',
                                        subpixel_method='gaussian',
                                        do_sig2noise=True,
                                        sig2noise_method='peak2peak',
                                        sig2noise_mask=2)
    u_old = u.copy()
    v_old = v.copy()
    i = 1
    for i in range(2, iterations+1):
        x, y,\
        u, v,\
        s2n, mask = windef.multipass_img_deform(frame_a, frame_b,
                                                window_size[i-1], overlap[i-1],
                                                iterations, i, x, y, u, v,
                                                correlation_method='circular',
                                                subpixel_method='gaussian',
                                                do_sig2noise=False,
                                                sig2noise_method='peak2peak',
                                                sig2noise_mask=2,
                                                MinMaxU=(-100, 50),
                                                MinMaxV=(-50, 50),
                                                std_threshold=1000000,
                                                median_threshold=200000,
                                                median_size=1,
                                                filter_method='localmean',
                                                max_filter_iteration=10,
                                                filter_kernel_size=2,
                                                interpolation_order=3)
    assert(np.max(np.abs(u-3)) < 0.1 and np.any(u != u_old))
    assert(np.max(np.abs(v+2)) < 0.1 and np.any(v != v_old))
    # the second condition is to check if the multipass is done.
    # It need's a little numerical inaccuracy.


# linear cross correlation
def test_first_pass_lin():
    """ test of the first pass """
    x, y, u, v, s2n = windef.first_pass(frame_a, frame_b, window_size=64,
                                        overlap=32, iterations=1,
                                        correlation_method='linear',
                                        subpixel_method='gaussian',
                                        do_sig2noise=True,
                                        sig2noise_method='peak2peak',
                                        sig2noise_mask=2)
    # print u,v
    assert(np.max(np.abs(u-3)) < 0.1)
    assert(np.max(np.abs(v+2)) < 0.1)


def test_multi_pass_lin():
    """ test fot the multipass """
    window_size = (128, 64, 32)
    overlap = (64, 32, 16)
    iterations = 3

    x, y, u, v, s2n = windef.first_pass(frame_a, frame_b, window_size[0],
                                        overlap[0], iterations,
                                        correlation_method='linear',
                                        subpixel_method='gaussian',
                                        do_sig2noise=True,
                                        sig2noise_method='peak2peak',
                                        sig2noise_mask=2)
    u_old = u.copy()
    v_old = v.copy()
    i = 1
    for i in range(2, iterations+1):
        x, y,\
        u, v,\
        sn, m = windef.multipass_img_deform(frame_a, frame_b,
                                            window_size[i-1],
                                            overlap[i-1],
                                            iterations, i, x, y,
                                            u, v,
                                            correlation_method='linear',
                                            subpixel_method='gaussian',
                                            do_sig2noise=False,
                                            sig2noise_method='peak2peak',
                                            sig2noise_mask=2,
                                            MinMaxU=(-100, 50),
                                            MinMaxV=(-50, 50),
                                            std_threshold=1000000,
                                            median_threshold=200000,
                                            median_size=1,
                                            filter_method='localmean',
                                            max_filter_iteration=10,
                                            filter_kernel_size=2,
                                            interpolation_order=3)
    assert(np.max(np.abs(u-3)) < 0.1 and np.any(u != u_old))
    assert(np.max(np.abs(v+2)) < 0.1 and np.any(v != v_old))
    # the second condition is to check if the multipass is done.
    # It need's a little numerical inaccuracy.
