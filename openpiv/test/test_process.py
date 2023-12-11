import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
from skimage import img_as_ubyte
from scipy.ndimage import shift as shift_img
# import pkg_resources as pkg
from importlib_resources import files
from openpiv.pyprocess import extended_search_area_piv as piv
from openpiv.pyprocess import fft_correlate_images, \
                              correlation_to_displacement


threshold = 0.25

# define "PIV" shift, i.e. creating u,v values that we want to get
# -5.5 pixels to the left and 3.2 pixels upwards
# if we translate it to the scipy.ndimage.shift values
# the first value is 2 pixels positive downwards, positive rows, 
# the second value is 1 pixel positive to the right 
# shifted_digit_image=shift(some_digit_image,[2,1]) 
# so we expect to get later
# shift(image, [-1*shift_v, shift_u])


# <------
shift_u = -3.5  # shift to the left, should be placed in columns, axis=1
# ^
# |
# |
shift_v = 2.5   # shift upwards, should be placed in rows, axis=0 


def create_pair(image_size=32, u=shift_u, v=shift_v):
    """ creates a pair of images with a roll/shift """
    frame_a = np.zeros((image_size, image_size))
    frame_a = random_noise(frame_a)
    frame_a = img_as_ubyte(frame_a)
    # note rolling positive vertical +2 means we create
    # negative vertical velocity as our origin is at 0,0
    # bottom left corner, and the image is rolled from the
    # top left corner

    # frame_b = np.roll(np.roll(frame_a, u, axis=1), v, axis=0)
    # scipy shift allows to shift by floating values
    frame_b = shift_img(frame_a, (v, u), mode='wrap')

    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].imshow(frame_a, cmap=plt.cm.gray)
    # ax[1].imshow(frame_b, cmap=plt.cm.gray)
    # plt.show()

    return frame_a.astype(np.int32), frame_b.astype(np.int32)


def test_piv():
    """test of the simplest PIV run
    default window_size = 32
    """
    frame_a, frame_b = create_pair(image_size=32)
    # extended_search_area_piv returns image based coordinate system
    u, v, _ = piv(frame_a, frame_b, window_size=32)
    print(u, v)
    assert np.allclose(u, shift_u, atol=threshold)
    assert np.allclose(v, shift_v, atol=threshold)


def test_piv_smaller_window():
    """ test of the search area larger than the window """
    frame_a, frame_b = create_pair(image_size=32, u=-3.5, v=-2.1)
    u, v, _ = piv(frame_a, frame_b, window_size=16, search_area_size=32)
    assert np.allclose(u, -3.5, atol=threshold)
    assert np.allclose(v, -2.1, atol=threshold)


def test_extended_search_area():
    """ test of the extended area PIV with larger image """
    frame_a, frame_b = create_pair(image_size=64, u=-3.5, v=-2.1)
    u, v, _ = piv(frame_a, frame_b,
                  window_size=16, 
                  search_area_size=32, 
                  overlap=0)

    assert np.allclose(u, -3.5, atol=threshold)
    assert np.allclose(v, -2.1, atol=threshold)
    # assert dist(u, shift_u) < threshold
    # assert dist(v, shift_v) < threshold


def test_extended_search_area_overlap():
    """ test of the extended area PIV with different overlap """
    frame_a, frame_b = create_pair(image_size=72)
    u, v, _ = piv(frame_a, frame_b,
                  window_size=16, 
                  search_area_size=32,
                  overlap=8)
    print(f"\n u={u}\n v={v}\n")
    assert np.allclose(u, shift_u, atol=threshold)
    assert np.allclose(v, shift_v, atol=threshold)


def test_extended_search_area_sig2noise():
    """ test of the extended area PIV with sig2peak """
    frame_a, frame_b = create_pair(image_size=64, u=-3.5, v=2.1)
    u, v, _ = piv(
        frame_a,
        frame_b,
        window_size=16,
        search_area_size=32,
        sig2noise_method="peak2peak",
        subpixel_method="gaussian"
    )

    assert np.allclose(u, -3.5, atol=threshold)
    assert np.allclose(v, 2.1, atol=threshold)


def test_process_extended_search_area():
    """ test of the extended area PIV from Cython """
    frame_a, frame_b = create_pair(image_size=64)
    u, v, _ = piv(frame_a, frame_b, window_size=16,
                  search_area_size=32, dt=2., overlap=0)

    assert np.allclose(u, shift_u/2., atol=threshold)
    assert np.allclose(v, shift_v/2., atol=threshold)


def test_sig2noise_ratio():
    """ s2n ratio test """
    from openpiv import tools 
    im1 = files('openpiv.data').joinpath('test1/exp1_001_a.bmp')
    im2 = files('openpiv.data').joinpath('test1/exp1_001_b.bmp')
    

    frame_a = tools.imread(im1)
    frame_b = tools.imread(im2)
    
    u, v, s2n = piv(
        frame_a.astype(np.int32),
        frame_b.astype(np.int32),
        window_size=32,
        search_area_size=64,
        sig2noise_method="peak2peak",
        subpixel_method="gaussian"
    )   
    # print(s2n.flatten().min(),s2n.mean(),s2n.max())
    assert np.allclose(s2n.mean(), 1.422, rtol=1e-3)
    assert np.allclose(s2n.max(), 2.264, rtol=1e-3)


def test_fft_correlate():
    frame_a, frame_b = create_pair(image_size=32)
    corr = fft_correlate_images(frame_a, frame_b)
    u, v = correlation_to_displacement(corr[np.newaxis, ...], 1, 1)
    assert np.allclose(u, shift_u, atol=threshold)
    assert np.allclose(v, shift_v, atol=threshold)

