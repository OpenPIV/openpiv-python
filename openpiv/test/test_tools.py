from openpiv.tools import imread, save, display_vector_field, transform_coordinates
from openpiv.pyprocess import extended_search_area_piv, get_coordinates
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.testing import compare, decorators

_file_a = pathlib.Path(__file__).parent / '../data/test1/exp1_001_a.bmp'
_file_b = pathlib.Path(__file__).parent / '../data/test1/exp1_001_b.bmp'

_test_file = pathlib.Path(__file__).parent / 'test_tools.png'


def test_imread(image_file=_file_a):
    a = imread(image_file)
    assert a.shape == (369, 511)
    assert a[0, 0] == 8
    assert a[-1, -1] == 15


def test_display_vector_field(file_a=_file_a, file_b=_file_b, test_file=_test_file):
    a = imread(file_a)
    b = imread(file_b)

    window_size = 32
    overlap = 16
    search_area_size = 40

    u, v, s2n = extended_search_area_piv(a, b, window_size,
                                         search_area_size=search_area_size,
                                         overlap=overlap,
                                         correlation_method='circular',
                                         normalized_correlation=False)

    x, y = get_coordinates(a.shape, search_area_size=search_area_size, overlap=overlap)

    x, y, u, v = transform_coordinates(x, y, u, v)

    save(x, y, u, v, np.zeros_like(x), 'tmp.txt')
    fig, ax = plt.subplots(figsize=(6, 6))
    display_vector_field('tmp.txt', on_img=True, image_name=file_a, ax=ax)
    decorators.remove_ticks_and_titles(fig)
    fig.savefig('./tmp.png')
    res = compare.compare_images('./tmp.png', test_file, 0.001)
    assert res is None
