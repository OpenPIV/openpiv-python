from openpiv.pyprocess import extended_search_area_piv as piv

import numpy as np
from test_process import create_pair, shift_u, shift_v, threshold, dist

frame_a, frame_b = create_pair(image_size=128)

def test_validation_peak2mean():
    """test of the simplest PIV run
    default window_size = 32
    """
    u, v, s2n = piv(frame_a, frame_b, 
                    window_size=32, 
                    sig2noise_method="peak2mean")
    print(s2n)
    assert np.min(s2n) > 100

def test_validation_peak2peak():
    """test of the simplest PIV run
    default window_size = 32
    """
    u, v, s2n = piv(frame_a, frame_b, 
                    window_size=32, 
                    sig2noise_method="peak2peak")
    print(s2n)
    assert np.min(s2n) > 2.