# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:33:21 2019

@author: Theo
"""


import numpy as np
import openpiv.windef as windef
from test_process import create_pair, shift_u, shift_v, threshold
import pathlib
import os

frame_a, frame_b = create_pair(image_size=256)

# this test are created only to test the displacement evaluation of the
# function the validation methods are not tested here ant therefore
# are disabled.

settings = windef.Settings()
settings.windowsizes = (64,)
settings.overlap = (32,)
settings.num_iterations = 1
settings.correlation_method = 'circular'
settings.sig2noise_method = 'peak2peak'
settings.subpixel_method = 'gaussian'
settings.sig2noise_mask = 2


# circular cross correlation
def test_first_pass_circ():
    """ test of the first pass """
    x, y, u, v, s2n = windef.first_pass(
        frame_a,
        frame_b,
        settings
    )
    print("\n", x, y, u, v, s2n)
    assert np.mean(np.abs(u - shift_u)) < threshold
    assert np.mean(np.abs(v - shift_v)) < threshold


def test_multi_pass_circ():
    """ test fot the multipass """
    settings.windowsizes = (64, 32, 16)
    settings.overlap = (32, 16, 8)
    settings.num_iterations = 3
    settings.interpolation_order = 3
    x, y, u, v, s2n = windef.first_pass(
        frame_a,
        frame_b,
        settings,
    )
    u_old = u.copy()
    v_old = v.copy()
    print("\n", x, y, u_old, v_old, s2n)
    assert np.mean(np.abs(u_old - shift_u)) < threshold
    assert np.mean(np.abs(v_old - shift_v)) < threshold
    for i in range(1,settings.num_iterations):
        x, y, u, v, s2n, mask = windef.multipass_img_deform(
            frame_a,
            frame_b,
            i,
            x,
            y,
            np.ma.array(u, mask=np.ma.nomask),
            np.ma.array(v, mask=np.ma.nomask),
            settings
        )

    print("\n", x, y, u, v, s2n)
    assert np.mean(np.abs(u - shift_u)) < threshold
    assert np.mean(np.abs(v - shift_v)) < threshold
    # the second condition is to check if the multipass is done.
    # It need's a little numerical inaccuracy.


# linear cross correlation
def test_first_pass_lin():
    """ test of the first pass """
    settings.correlation_method = 'linear'

    x, y, u, v, s2n = windef.first_pass(
        frame_a,
        frame_b,
        settings,
    )
    print("\n", x, y, u, v, s2n)
    assert np.mean(np.abs(u - shift_u)) < threshold
    assert np.mean(np.abs(v - shift_v)) < threshold


def test_invert():
    """ Test windef.piv with invert option """

    settings = windef.Settings()
    'Data related settings'
    # Folder with the images to process
    settings.filepath_images = pathlib.Path(__file__).parent / '../examples/test1'
    settings.save_path = '.'
    # Root name of the output Folder for Result Files
    settings.save_folder_suffix = 'test'
    # Format and Image Sequence
    settings.frame_pattern_a = 'exp1_001_a.bmp'
    settings.frame_pattern_b = 'exp1_001_b.bmp'

    settings.num_iterations = 1
    settings.show_plot = True
    settings.scale_plot = 100
    settings.show_all_plots = True
    settings.invert = True

    windef.piv(settings)


def test_multi_pass_lin():
    """ test fot the multipass """
    settings.windowsizes = (64, 32, 16)
    settings.overlap = (32, 16, 8)
    settings.num_iterations = 3
    settings.sig2noise_validate = True
    settings.correlation_method = 'linear'

    x, y, u, v, s2n = windef.first_pass(
        frame_a,
        frame_b,
        settings,
    )
    u_old = u.copy()
    v_old = v.copy()

    print("\n", x, y, u_old, v_old, s2n)
    assert np.mean(np.abs(u_old - shift_u)) < threshold
    assert np.mean(np.abs(v_old - shift_v)) < threshold

    for i in range(1, settings.num_iterations):
        x, y, u, v, s2n, mask = windef.multipass_img_deform(
            frame_a,
            frame_b,
            i,
            x,
            y,
            np.ma.array(u, mask=np.ma.nomask),
            np.ma.array(v, mask=np.ma.nomask),
            settings,
        )

    print("\n", x, y, u, v, s2n)
    assert np.mean(np.abs(u - shift_u)) < threshold
    assert np.mean(np.abs(v - shift_v)) < threshold

    # the second condition is to check if the multipass is done.
    # It need's a little numerical inaccuracy.
