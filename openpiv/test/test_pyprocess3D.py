import numpy as np
import pytest
from scipy.ndimage import shift as shift_img
from skimage import img_as_ubyte
from skimage.util import random_noise

from openpiv.pyprocess3D import (
    extended_search_area_piv3D,
    find_second_peak_3D,
    sig2noise_ratio,
)

# import warnings

threshold = 0.1


@pytest.mark.parametrize("width", [0, 2])
@pytest.mark.parametrize("explicit_peak", [False, True])
@pytest.mark.parametrize("shape,first,second", [
    ((7, 8, 20), (3, 4, 15), (3, 4, 5)),
    ((7, 8, 20), (3, 4, 18), (3, 4, 5)),
    ((7, 20, 8), (3, 10, 4), (3, 5, 4)),
    ((9, 12, 12), (4, 6, 6), (4, 3, 6)),
])
def test_second_peak_uses_depth(shape, first, second, explicit_peak, width):
    corr = np.zeros(shape)
    corr[first] = 10.0
    corr[second] = 4.0
    original = corr.copy()
    args = first if explicit_peak else ()

    position, value = find_second_peak_3D(corr, *args, width=width)

    assert position == second
    assert value == 4.0
    np.testing.assert_array_equal(corr, original)


@pytest.mark.parametrize("depth", [16, 20])
def test_signal_to_noise_excludes_first_peak_in_deep_volume(depth):
    corr = np.zeros((7, 8, depth))
    corr[3, 4, depth - 3] = 10.0
    corr[3, 4, 5] = 4.0

    assert sig2noise_ratio(corr, sig2noise_method="peak2peak") == 2.5


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

    frame_b = shift_img(frame_a, (v, u, w))
    return frame_a.astype(np.int32), frame_b.astype(np.int32)


def test_piv():
    """
    test for 3D PIV with window_size==search_area_size
    """
    frame_a, frame_b = create_pair(image_size=32)
    u, v, w = extended_search_area_piv3D(
        frame_a, frame_b, window_size=(10, 10, 10), search_area_size=(10, 10, 10)
    )
    assert dist(u, -3) < threshold
    assert dist(v, 2) < threshold
    assert dist(w, 1) < threshold


def test_piv_extended_search_area():

    """
    test for 3D PIV with larger search_area_size
    """
    frame_a, frame_b = create_pair(image_size=32)
    u, v, w = extended_search_area_piv3D(
        frame_a, frame_b, window_size=(10, 10, 10), search_area_size=(15, 15, 15)
    )
    assert dist(u, -3) < threshold
    assert dist(v, 2) < threshold
    assert dist(w, 1) < threshold
