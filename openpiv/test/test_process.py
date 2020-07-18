try:
    from openpiv.process import extended_search_area_piv as piv
except ImportError:
    from openpiv.pyprocess import extended_search_area_piv as piv
import numpy as np


from skimage.util import random_noise
from skimage import img_as_ubyte

threshold = 0.1


def dist(u, shift):
    return np.mean(np.abs(u-shift))


def create_pair(image_size=32, u=3, v=2):
    """ creates a pair of images with a roll/shift """
    frame_a = np.zeros((image_size, image_size))
    frame_a = random_noise(frame_a)
    frame_a = img_as_ubyte(frame_a)
    # note rolling positive vertical +2 means we create
    # negative vertical velocity as our origin is at 0,0
    # bottom left corner, and the image is rolled from the
    # top left corner

    frame_b = np.roll(np.roll(frame_a, u, axis=1), v, axis=0)
    return frame_a.astype(np.int32), frame_b.astype(np.int32)


def test_piv():
    """ test of the simplest PIV run
        default window_size = 32
    """
    frame_a, frame_b = create_pair(image_size=32)
    u, v = piv(frame_a, frame_b, window_size=32)
    assert(dist(u, 3) < threshold)
    assert(dist(v, -2) < threshold)


def test_piv_smaller_window():
    """ test of the search area larger than the window """
    frame_a, frame_b = create_pair(image_size=32, u=-3, v=-2)
    u, v = piv(frame_a, frame_b, window_size=16, search_area_size=32)
    assert(dist(u, -3) < threshold)
    assert(dist(v, 2) < threshold)


def test_extended_search_area():
    """ test of the extended area PIV with larger image """
    frame_a, frame_b = create_pair(image_size=64)
    u, v = piv(frame_a, frame_b, window_size=16,
               search_area_size=32, overlap=0)

    assert((dist(u, 3)+dist(v, -2)) < 2*threshold)


def test_extended_search_area_overlap():
    """ test of the extended area PIV with different overlap """
    frame_a, frame_b = create_pair(image_size=64)
    u, v = piv(frame_a, frame_b, window_size=16,
               search_area_size=32, overlap=8)
    assert((dist(u, 3)+dist(v, -2)) < 2*threshold)


def test_extended_search_area_sig2noise():
    """ test of the extended area PIV with sig2peak """
    frame_a, frame_b = create_pair(image_size=64)
    u, v, s2n = piv(frame_a, frame_b, window_size=16,
                    search_area_size=32, sig2noise_method='peak2peak')
    assert((dist(u, 3)+dist(v, -2)) < 2*threshold)


def test_process_extended_search_area():
    """ test of the extended area PIV from Cython """
    frame_a, frame_b = create_pair(image_size=64)
    u, v = piv(frame_a, frame_b,
               window_size=16, search_area_size=32, dt=1, overlap=0)
    # assert(np.max(np.abs(u[:-1,:-1]-3)+np.abs(v[:-1,:-1]+2)) <= 0.3)
    assert((dist(u, 3)+dist(v, -2)) < 2*threshold)


def test_piv_vs_extended_search():
    """ test of the simplest PIV run """
    import openpiv.process
    import openpiv.pyprocess
    frame_a, frame_b = create_pair()
    u, v = openpiv.process.extended_search_area_piv(frame_a, frame_b,
                                                    window_size=32)
    u1, v1 = openpiv.pyprocess.extended_search_area_piv(frame_a, frame_b,
                                                        window_size=32,
                                                        search_area_size=32,
                                                        dt=1, overlap=0)

    assert(np.allclose(u, u1))
    assert(np.allclose(v, v1))
