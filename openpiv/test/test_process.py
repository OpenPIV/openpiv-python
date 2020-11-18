from openpiv.pyprocess import extended_search_area_piv as piv

import numpy as np


from skimage.util import random_noise
from skimage import img_as_ubyte
from scipy.ndimage import shift

threshold = 0.2
shift_u = -5.5
shift_v = 3.2


def dist(u, shift):
    return np.mean(np.abs(u - shift))


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
    frame_b = shift(frame_a, (-v, u), mode='wrap')
    return frame_a.astype(np.int32), frame_b.astype(np.int32)


def test_piv():
    """test of the simplest PIV run
    default window_size = 32
    """
    frame_a, frame_b = create_pair(image_size=64)
    u, v, _ = piv(frame_a, frame_b, window_size=64)
    print(u, v)
    assert dist(u, shift_u) < threshold
    assert dist(v, shift_v) < threshold


def test_piv_smaller_window():
    """ test of the search area larger than the window """
    frame_a, frame_b = create_pair(image_size=32, u=-3, v=-2)
    u, v, _ = piv(frame_a, frame_b, window_size=16, search_area_size=32)
    assert dist(u, -3) < threshold
    assert dist(v, -2) < threshold


def test_extended_search_area():
    """ test of the extended area PIV with larger image """
    frame_a, frame_b = create_pair(image_size=64)
    u, v, _ = piv(frame_a, frame_b,
                  window_size=16, 
                  search_area_size=32, 
                  overlap=0)

    assert (dist(u, shift_u) + dist(v, shift_v)) < 2 * threshold


def test_extended_search_area_overlap():
    """ test of the extended area PIV with different overlap """
    frame_a, frame_b = create_pair(image_size=64)
    u, v, _ = piv(frame_a, frame_b,
                  window_size=16, 
                  search_area_size=32,
                  overlap=8)
    assert (dist(u, shift_u) + dist(v, shift_v)) < 2 * threshold


def test_extended_search_area_sig2noise():
    """ test of the extended area PIV with sig2peak """
    frame_a, frame_b = create_pair(image_size=64)
    u, v, s2n = piv(
        frame_a,
        frame_b,
        window_size=16,
        search_area_size=32,
        sig2noise_method="peak2peak",
    )
    assert (dist(u, shift_u) + dist(v, shift_v)) < 2 * threshold


def test_process_extended_search_area():
    """ test of the extended area PIV from Cython """
    frame_a, frame_b = create_pair(image_size=64)
    u, v, _ = piv(frame_a, frame_b, window_size=16, 
                  search_area_size=32, dt=1, overlap=0)
    # assert(np.max(np.abs(u[:-1,:-1]-3)+np.abs(v[:-1,:-1]+2)) <= 0.3)
    assert (dist(u, shift_u) + dist(v, shift_v)) < 2 * threshold


def test_sig2noise_ratio():
    return False
