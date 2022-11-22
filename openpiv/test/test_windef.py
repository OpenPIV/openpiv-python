"""
Created on Fri Oct  4 14:33:21 2019

@author: Theo
"""

import pathlib
import numpy as np
from openpiv import windef
from openpiv.test import test_process

frame_a, frame_b = test_process.create_pair(image_size=256)
shift_u, shift_v, threshold = test_process.shift_u, test_process.shift_v, \
                              test_process.threshold

# this test are created only to test the displacement evaluation of the
# function the validation methods are not tested here ant therefore
# are disabled.

# settings = windef.PIVSettings()
# settings.windowsizes = (64,)
# settings.overlap = (32,)
# settings.num_iterations = 1
# settings.correlation_method = 'circular'
# settings.sig2noise_method = 'peak2peak'
# settings.subpixel_method = 'gaussian'
# settings.sig2noise_mask = 2


# circular cross correlation
def test_first_pass_circ():
    """ test of the first pass """
    settings = windef.PIVSettings()
    settings.windowsizes = (64,)
    settings.overlap = (32,)
    settings.num_iterations = 1
    settings.correlation_method = 'circular'
    settings.sig2noise_method = 'peak2peak'
    settings.subpixel_method = 'gaussian'
    settings.sig2noise_mask = 2

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
    settings = windef.PIVSettings()
    settings.windowsizes = (64, 64, 16)
    settings.overlap = (32, 32, 8)
    settings.num_iterations = 2
    settings.interpolation_order = 3
    settings.validation_first_pass = True
    settings.sig2noise_validate = False
    # settings.show_all_plots = True

    x, y, u, v, s2n = windef.first_pass(
        frame_a,
        frame_b,
        settings,
    )
    print("first pass\n")
    print("\n", x, y, u, v, s2n)
    assert np.allclose(u, shift_u, atol = threshold)
    assert np.allclose(v, shift_v, atol = threshold)

    u = np.ma.masked_array(u, mask=np.ma.nomask)
    v = np.ma.masked_array(v, mask=np.ma.nomask)

    for i in range(1,settings.num_iterations):
        x, y, u, v, s2n, _ = windef.multipass_img_deform(
            frame_a,
            frame_b,
            i,
            x,
            y,
            u,
            v,
            settings
        )

    print(f"Pass {i}\n")
    print(x)
    print(y)
    print(u) 
    print(v)
    print(s2n)
    assert np.mean(np.abs(u - shift_u)) < threshold
    assert np.mean(np.abs(v - shift_v)) < threshold
    # the second condition is to check if the multipass is done.
    # It need's a little numerical inaccuracy.


# linear cross correlation
def test_first_pass_lin():
    """ test of the first pass """
    settings = windef.PIVSettings()
    settings.windowsizes = (64,)
    settings.overlap = (32,)
    settings.num_iterations = 1
    settings.correlation_method = 'circular'
    settings.sig2noise_method = 'peak2peak'
    settings.subpixel_method = 'gaussian'
    settings.sig2noise_mask = 2
    settings.correlation_method = 'linear'

    x, y, u, v, s2n = windef.first_pass(
        frame_a,
        frame_b,
        settings,
    )
    print("\n", x, y, u, v, s2n)
    assert np.mean(np.abs(u - shift_u)) < threshold
    assert np.mean(np.abs(v - shift_v)) < threshold


def test_save_plot():
    """ Test save plot """

    settings = windef.PIVSettings()
    settings.save_plot = True
    settings.save_path = settings.filepath_images.parent.parent / "test"
    windef.piv(settings)

    save_path_string = \
        f"OpenPIV_results_{settings.windowsizes[settings.num_iterations-1]}_{settings.save_folder_suffix}"
    save_path = \
        settings.save_path / save_path_string

    png_file = save_path / f'field_A{0:04d}.png'
    assert png_file.exists()


def test_invert_and_piv():
    """ Test windef.piv with invert option """

    settings = windef.PIVSettings()
    # Folder with the images to process
    settings.filepath_images = pathlib.Path(__file__).parent / '../data/test1'
    settings.save_path = pathlib.Path('.')
    # Root name of the output Folder for Result Files
    settings.save_folder_suffix = 'test'
    # Format and Image Sequence
    settings.frame_pattern_a = 'exp1_001_a.bmp'
    settings.frame_pattern_b = 'exp1_001_b.bmp'

    settings.num_iterations = 1
    settings.show_plot = False
    settings.scale_plot = 100
    settings.show_all_plots = False
    settings.invert = True

    windef.piv(settings)


def test_multi_pass_lin():
    """ test fot the multipass """
    settings = windef.PIVSettings()
    # settings.windowsizes = (64,)
    # settings.overlap = (32,)
    # settings.num_iterations = 1
    # settings.correlation_method = 'circular'
    # settings.sig2noise_method = 'peak2peak'
    # settings.subpixel_method = 'gaussian'
    # settings.sig2noise_mask = 2

    settings.windowsizes = (64, 32)
    settings.overlap = (32, 16)
    settings.num_iterations = 1
    settings.sig2noise_validate = True
    settings.correlation_method = 'linear'
    settings.normalized_correlation = True
    settings.sig2noise_threshold = 1.0 # note the value for linear/normalized

    x, y, u, v, s2n = windef.first_pass(
        frame_a,
        frame_b,
        settings,
    )

    print("\n", x, y, u, v, s2n)
    assert np.mean(np.abs(u - shift_u)) < threshold
    assert np.mean(np.abs(v - shift_v)) < threshold


    mask_coords = []
    u = np.ma.masked_array(u, mask=np.ma.nomask)
    v = np.ma.masked_array(v, mask=np.ma.nomask)

    for i in range(1, settings.num_iterations):
        x, y, u, v, s2n, _ = windef.multipass_img_deform(
            frame_a,
            frame_b,
            i,
            x,
            y,
            u,
            v,
            settings,
        )
        # print(f"Iteration {i}")
        # print(u[:2,:2],v[:2,:2])
        assert np.allclose(u, shift_u, atol=threshold)
        assert np.allclose(v, shift_v, atol=threshold)

    # the second condition is to check if the multipass is done.
    # It need's a little numerical inaccuracy.

def test_simple_multipass():
    """ Test simple multipass """
    settings = windef.PIVSettings()
    settings.windowsizes = (64,)
    settings.overlap = (32,)
    settings.num_iterations = 1
    settings.correlation_method = 'circular'
    settings.sig2noise_method = 'peak2peak'
    settings.subpixel_method = 'gaussian'
    settings.sig2noise_mask = 2

    x, y, u, v, mask = windef.simple_multipass(
        frame_a,
        frame_b,
        settings,
    )
    # print("simple multipass\n")
    # print(x,y,u,v,mask)
    # print(u[:4,:4])
    # print(v[:4,:4])
    # print(shift_u)
    # print(shift_v)

#     # note the -shift_v 
#     # the simple_multipass also transforms units, so 
#     # the plot is in the image-like space

#     assert np.allclose(u, shift_u, atol=threshold)
#     assert np.allclose(v, -shift_v, atol=threshold)


#    # windowsizes = (64, 32, 16)
#     x, y, u, v, s2n = windef.simple_multipass(
#         frame_a,
#         frame_b,
#         settings,
#         windows=(32,16,8),
#     )
#     print("simple multipass\n")
#     print(u[:4,:4])
#     print(v[:4,:4])
#     assert np.allclose(u, shift_u, atol=threshold)
#     assert np.allclose(v, -shift_v, atol=threshold)

#     # the second condition is to check if the multipass is done.
#     # It need's a little numerical inaccuracy.


def test_simple_rectangular_window():
    """ Test simple multipass """
    print('test simple pass with rectangular windows')

    settings = windef.PIVSettings()


    x, y, _,_,_ = windef.simple_multipass(
        frame_a,
        frame_b,
        settings,
    )
    
        
    settings.windowsizes = ((64, 32),)
    settings.overlap = ((32, 16),)
    settings.num_iterations = 1
    settings.correlation_method = 'circular'
    settings.sig2noise_method = 'peak2peak'
    settings.subpixel_method = 'gaussian'
    settings.sig2noise_mask = 2

    x, y, _,_,_ = windef.simple_multipass(
        frame_a,
        frame_b,
        settings,
    )
    # print("rectangular windows\n")
    # print(x,y,u,v,mask)
    # print(np.diff(x[0,:2]))
    # print( np.diff(y[:2,0]))
    assert np.diff(x[0,:2]) == 16
    assert np.diff(y[:2,0]) == -32


    settings.windowsizes = ((32, 64),(16, 32))
    settings.overlap = ((16, 32), (8, 16))
    settings.num_iterations = 2

    x, y, _, _, _ = windef.simple_multipass(
        frame_a,
        frame_b,
        settings,
    )
    assert np.diff(x[0,:2]) == 16
    assert np.diff(y[:2,0]) == -8

    settings.show_all_plots = False
    settings.show_plot = True

    windef.piv(settings)
