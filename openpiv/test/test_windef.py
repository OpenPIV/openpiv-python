# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:33:21 2019

@author: Theo
"""


import numpy as np
import openpiv.windef as windef
from test_process import create_pair, shift_u, shift_v, threshold

frame_a, frame_b = create_pair(image_size=256)

# this test are created only to test the displacement evaluation of the
# function the validation methods are not tested here ant therefore
# are disabled.


# circular cross correlation
def test_first_pass_circ():
    """ test of the first pass """
    x, y, u, v, s2n = windef.first_pass(
        frame_a,
        frame_b,
        window_size=64,
        overlap=32,
        iterations=1,
        correlation_method="circular",
        subpixel_method="gaussian",
        do_sig2noise=True,
        sig2noise_method="peak2peak",
        sig2noise_mask=2,
    )
    print("\n", x, y, u, v, s2n)
    assert np.mean(np.abs(u - shift_u)) < threshold
    assert np.mean(np.abs(v - shift_v)) < threshold


def test_multi_pass_circ():
    """ test fot the multipass """
    window_size = (128, 64, 32)
    overlap = (64, 32, 16)
    iterations = 3

    x, y, u, v, s2n = windef.first_pass(
        frame_a,
        frame_b,
        window_size[0],
        overlap[0],
        iterations,
        correlation_method="circular",
        subpixel_method="gaussian",
        do_sig2noise=True,
        sig2noise_method="peak2peak",
        sig2noise_mask=2,
    )
    u_old = u.copy()
    v_old = v.copy()
    for i in range(1, iterations ):
        x, y, u, v, s2n = windef.multipass_img_deform(
            frame_a,
            frame_b,
            window_size[i],
            overlap[i],
            iterations,
            i,
            x,
            y,
            u,
            v,
            correlation_method="circular",
            subpixel_method="gaussian",
            deformation_method="symmetric",
            do_sig2noise=False,
            sig2noise_method="peak2peak",
            sig2noise_mask=2,
            interpolation_order=3,
        )

    print("\n", x, y, u, v, s2n)
    assert np.mean(np.abs(u - shift_u)) < threshold
    assert np.mean(np.abs(v - shift_v)) < threshold
    # the second condition is to check if the multipass is done.
    # It need's a little numerical inaccuracy.


# linear cross correlation
def test_first_pass_lin():
    """ test of the first pass """
    x, y, u, v, s2n = windef.first_pass(
        frame_a,
        frame_b,
        window_size=64,
        overlap=32,
        iterations=1,
        correlation_method="linear",
        subpixel_method="gaussian",
        do_sig2noise=True,
        sig2noise_method="peak2peak",
        sig2noise_mask=2,
    )
    print("\n", x, y, u, v, s2n)
    assert np.mean(np.abs(u - shift_u)) < threshold
    assert np.mean(np.abs(v - shift_v)) < threshold


def test_multi_pass_lin():
    """ test fot the multipass """
    window_size = (128, 64, 32)
    overlap = (64, 32, 16)
    iterations = 3

    x, y, u, v, s2n = windef.first_pass(
        frame_a,
        frame_b,
        window_size[0],
        overlap[0],
        iterations,
        correlation_method="linear",
        subpixel_method="gaussian",
        do_sig2noise=True,
        sig2noise_method="peak2peak",
        sig2noise_mask=2,
    )
    u_old = u.copy()
    v_old = v.copy()
    for i in range(1, iterations):
        x, y, u, v, sn = windef.multipass_img_deform(
            frame_a,
            frame_b,
            window_size[i],
            overlap[i],
            iterations,
            i,
            x,
            y,
            u,
            v,
            correlation_method="linear",
            subpixel_method="gaussian",
            deformation_method="symmetric",
            do_sig2noise=False,
            sig2noise_method="peak2peak",
            sig2noise_mask=2,
            interpolation_order=3,
        )

    print("\n", x, y, u, v, s2n)
    assert np.mean(np.abs(u - shift_u)) < threshold
    assert np.mean(np.abs(v - shift_v)) < threshold

    # the second condition is to check if the multipass is done.
    # It need's a little numerical inaccuracy.
