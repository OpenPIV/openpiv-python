from openpiv.pyprocess3D import extended_search_area_piv3D
import numpy as np

from skimage.util import random_noise
from skimage import img_as_ubyte

import warnings

threshold = 0.1


def dist(u, shift):
    return np.mean(np.abs(u - shift))


def create_pair(image_size=32, u=3, v=2, w=1):
    """ creates a pair of images with a roll/shift """
    frame_a = np.zeros((image_size, image_size, image_size))
    frame_a = random_noise(frame_a)
    frame_a = img_as_ubyte(frame_a)
    # note rolling positive vertical +2 means we create
    # negative vertical velocity as our origin is at 0,0
    # bottom left corner, and the image is rolled from the
    # top left corner

    frame_b = np.roll(np.roll(np.roll(frame_a, w, axis=2), u, axis=1), v, axis=0)
    return frame_a.astype(np.int32), frame_b.astype(np.int32)


def test_piv():
    """
    test for 3D PIV with window_size==search_area_size
    """
    frame_a, frame_b = create_pair(image_size=32)
    u, v, w = extended_search_area_piv3D(frame_a, frame_b, window_size=(10, 10, 10), search_area_size=(10, 10, 10))
    assert (dist(u, 3) < threshold)
    assert (dist(v, -2) < threshold)
    assert (dist(w, -1) < threshold)


def test_piv_extended_search_area():

    """
    test for 3D PIV with larger search_area_size
    """
    frame_a, frame_b = create_pair(image_size=32)
    u, v, w = extended_search_area_piv3D(frame_a, frame_b, window_size=(10, 10, 10), search_area_size=(15, 15, 15))
    assert (dist(u, 3) < threshold)
    assert (dist(v, -2) < threshold)
    assert (dist(w, -1) < threshold)
