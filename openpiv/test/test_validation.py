from openpiv.pyprocess import extended_search_area_piv as piv
from openpiv.tools import imread
import pathlib

import numpy as np
from test_process import create_pair, shift_u, shift_v, threshold



file_a = pathlib.Path(__file__).parent / '../examples/test1/exp1_001_a.bmp'
file_b = pathlib.Path(__file__).parent / '../examples/test1/exp1_001_b.bmp'

frame_a = imread(file_a)
frame_b = imread(file_b)

frame_a = frame_a[:32,:32]
frame_b = frame_b[:32,:32]


def test_validation_peak2mean():
    """test of the simplest PIV run
    default window_size = 32
    """
    _, _, s2n = piv(frame_a, frame_b, 
                    window_size=32, 
                    sig2noise_method="peak2mean")

    assert np.allclose(s2n.min(),1.443882)

def test_validation_peak2peak():
    """test of the simplest PIV run
    default window_size = 32
    """
    _, _, s2n = piv(frame_a, frame_b, 
                    window_size=32, 
                    sig2noise_method="peak2peak")
    assert np.allclose(np.min(s2n), 1.24009)
