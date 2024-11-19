""" Testing basic PIV processes """
import numpy as np
from skimage.util import random_noise
from skimage import img_as_ubyte
from scipy.ndimage import shift as shift_img
# import pkg_resources as pkg
from importlib_resources import files
from openpiv.pyprocess import extended_search_area_piv as piv
from openpiv.pyprocess import fft_correlate_images, \
                              correlation_to_displacement
from openpiv import tools 



THRESHOLD = 0.25

# define "PIV" shift, i.e. creating u,v values that we want to get
# -5.5 pixels to the left and 3.2 pixels upwards
# if we translate it to the scipy.ndimage.shift values
# the first value is 2 pixels positive downwards, positive rows, 
# the second value is 1 pixel positive to the right 
# shifted_digit_image=shift(some_digit_image,[2,1]) 
# so we expect to get later
# shift(image, [-1*SHIFT_V, SHIFT_U])


# <------
SHIFT_U = -3.5  # shift to the left, should be placed in columns, axis=1
# ^
# |
# |
SHIFT_V = 2.5   # shift upwards, should be placed in rows, axis=0 


def create_pair(image_size=32, u=SHIFT_U, v=SHIFT_V):
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
    assert np.allclose(u, SHIFT_U, atol=THRESHOLD)
    assert np.allclose(v, SHIFT_V, atol=THRESHOLD)


def test_piv_smaller_window():
    """ test of the search area larger than the window """
    frame_a, frame_b = create_pair(image_size=32, u=-3.5, v=-2.1)
    u, v, _ = piv(frame_a, frame_b, window_size=16, search_area_size=32)
    assert np.allclose(u, -3.5, atol=THRESHOLD)
    assert np.allclose(v, -2.1, atol=THRESHOLD)


def test_extended_search_area():
    """ test of the extended area PIV with larger image """
    frame_a, frame_b = create_pair(image_size=64, u=-3.5, v=-2.1)
    u, v, _ = piv(frame_a, frame_b,
                  window_size=16, 
                  search_area_size=32, 
                  overlap=0)

    assert np.allclose(u, -3.5, atol=THRESHOLD)
    assert np.allclose(v, -2.1, atol=THRESHOLD)
    # assert dist(u, SHIFT_U) < THRESHOLD
    # assert dist(v, SHIFT_V) < THRESHOLD


def test_extended_search_area_overlap():
    """ test of the extended area PIV with different overlap """
    frame_a, frame_b = create_pair(image_size=72)
    u, v, _ = piv(frame_a, frame_b,
                  window_size=16, 
                  search_area_size=32,
                  overlap=8)
    print(f"\n u={u}\n v={v}\n")
    assert np.allclose(u, SHIFT_U, atol=THRESHOLD)
    assert np.allclose(v, SHIFT_V, atol=THRESHOLD)


def test_extended_search_area_sig2noise():
    """ test of the extended area PIV with sig2peak """
    success_count = 0
    num_trials = 10
    for _ in range(num_trials):
        frame_a, frame_b = create_pair(image_size=64, u=SHIFT_U, v=SHIFT_V)
        u, v, _ = piv(
            frame_a,
            frame_b,
            window_size=16,
            search_area_size=32,
            sig2noise_method="peak2peak",
            subpixel_method="gaussian"
        )
        if np.allclose(u, SHIFT_U, atol=THRESHOLD) and np.allclose(v, SHIFT_V, atol=THRESHOLD):
            success_count += 1

    assert success_count >= 7, f"Test failed: {success_count} out of {num_trials} trials were successful"


def test_process_extended_search_area():
    """ test of the extended area PIV from Cython """
    frame_a, frame_b = create_pair(image_size=64)
    u, v, _ = piv(frame_a, frame_b, window_size=16,
                  search_area_size=32, dt=2., overlap=0)

    assert np.allclose(u, SHIFT_U/2., atol=THRESHOLD)
    assert np.allclose(v, SHIFT_V/2., atol=THRESHOLD)


def test_sig2noise_ratio():
    """ s2n ratio test """
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
    """ test of the fft correlation """
    frame_a, frame_b = create_pair(image_size=32)
    corr = fft_correlate_images(frame_a, frame_b)
    u, v = correlation_to_displacement(corr[np.newaxis, ...], 1, 1)
    assert np.allclose(u, SHIFT_U, atol=THRESHOLD)
    assert np.allclose(v, SHIFT_V, atol=THRESHOLD)


def test_new_overlap_setting():
    """ test of the new overlap setting changed on 19/11/2024"""
    frame_a, frame_b = create_pair(image_size=72)
    u, v, _ = piv(frame_a, frame_b,
                  window_size=16,
                  search_area_size=32,
                  overlap=22)

    assert u.shape == (5, 5) and v.shape == (5, 5)

    u, v, _ = piv(frame_a, frame_b,
                  window_size=16,
                  search_area_size=32,
                  overlap=21)
    assert u.shape == (4, 4) and v.shape == (4, 4)

    u, v, _ = piv(frame_a, frame_b,
                  window_size=16,
                  search_area_size=32,
                  overlap=19)
    assert u.shape == (4, 4) and v.shape == (4, 4)

