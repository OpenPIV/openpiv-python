""" tests windef functionality """
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.testing import compare, decorators
from openpiv.tools import imread, save, display_vector_field, transform_coordinates
from openpiv.pyprocess import extended_search_area_piv, get_coordinates


_file_a = pathlib.Path(__file__).parent / '../data/test1/exp1_001_a.bmp'
_file_b = pathlib.Path(__file__).parent / '../data/test1/exp1_001_b.bmp'

_test_file = pathlib.Path(__file__).parent / 'test_tools.png'


def test_imread(image_file=_file_a):
    """test imread

    Args:
        image_file (_type_, optional): image path and filename. Defaults to _file_a.
    """
    frame_a = imread(image_file)
    assert frame_a.shape == (369, 511)
    assert frame_a[0, 0] == 8
    assert frame_a[-1, -1] == 15


def test_display_vector_field(
    file_a=_file_a,
    file_b=_file_b,
    test_file=_test_file
    ):
    """ tests display vector field """
    a = imread(file_a)
    b = imread(file_b)

    window_size = 32
    overlap = 16
    search_area_size = 40

    u, v, _ = extended_search_area_piv(a, b, window_size,
                                         search_area_size=search_area_size,
                                         overlap=overlap,
                                         correlation_method='circular',
                                         normalized_correlation=False)

    x, y = get_coordinates(a.shape, search_area_size=search_area_size, overlap=overlap)

    x, y, u, v = transform_coordinates(x, y, u, v)

    mask = np.zeros_like(x, dtype=int)
    flags = np.zeros_like(x, dtype=int)
    flags[-1,1] = 1 # test of invalid vector plot
    save('tmp.txt', x, y, u, v, flags, mask)
    fig, ax = plt.subplots(figsize=(6, 6))
    display_vector_field('tmp.txt', on_img=True, image_name=file_a, ax=ax)
    decorators.remove_ticks_and_titles(fig)
    fig.savefig('./tmp.png')
    res = compare.compare_images('./tmp.png', test_file, 0.05)
    assert res is None

def test_file_patterns():
    """ 
    tools.Multiprocesser() class has a couple of options to process
    pairs of images or create pairs from sequential list of files

    # Format and Image Sequence 
        settings.frame_pattern_a = 'exp1_001_a.bmp'
        settings.frame_pattern_b = 'exp1_001_b.bmp'

        # or if you have a sequence:
        # settings.frame_pattern_a = '000*.tif'
        # settings.frame_pattern_b = '(1+2),(2+3)'
        # settings.frame_pattern_b = '(1+3),(2+4)'
        # settings.frame_pattern_b = '(1+2),(3+4)'
    """

    
