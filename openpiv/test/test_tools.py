""" tests windef functionality """
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import pytest
from matplotlib.testing import compare, decorators
from openpiv.tools import (
    imread, save, display_vector_field, transform_coordinates,
    display_vector_field_from_arrays, negative, Multiprocesser
)
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


def test_imread_edge_cases():
    """Test imread with different file types and edge cases"""
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        imread('non_existent_file.tif')
    
    # Test with different file formats if applicable
    # Create temporary test images if needed


def test_display_vector_field_with_warnings_suppressed():
    """Test the display_vector_field function with warnings suppressed"""
    import warnings
    
    # Create a temporary vector file with more data points
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[1, 2], [3, 4]])
    u = np.array([[0.1, 0.2], [0.3, 0.4]])
    v = np.array([[0.1, 0.2], [0.3, 0.4]])
    flags = np.zeros_like(u)
    mask = np.zeros_like(u)
    
    save('temp_test.vec', x, y, u, v, mask)
    
    # Test with different parameters, suppressing warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        display_vector_field('temp_test.vec', scale=10)
        display_vector_field('temp_test.vec', width=0.005)
    
    # Clean up
    import os
    os.remove('temp_test.vec')


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

def test_transform_coordinates():
    """Test the transform_coordinates function"""
    # Create test data
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[1, 2], [3, 4]])
    u = np.array([[0.1, 0.2], [0.3, 0.4]])
    v = np.array([[0.1, 0.2], [0.3, 0.4]])
    
    # Store original v for comparison
    v_original = v.copy()
    
    # Apply the transformation
    x_new, y_new, u_new, v_new = transform_coordinates(x, y, u, v)
    
    # Check that the transformation was applied correctly
    # The function reverses the order of rows in y and negates v
    assert np.allclose(x_new, x)
    assert np.allclose(y_new, y[::-1, :])  # Reversed rows
    assert np.allclose(u_new, u)
    assert np.allclose(v_new, -v_original)  # Negated v


def test_save_and_load():
    """Test saving and loading vector data"""
    # Create test data
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[1, 2], [3, 4]])
    u = np.array([[0.1, 0.2], [0.3, 0.4]])
    v = np.array([[0.1, 0.2], [0.3, 0.4]])
    mask = np.zeros_like(u)
    
    # Save data
    filename = 'temp_save_test.vec'
    save(filename, x, y, u, v, mask)
    
    # Load data
    data = np.loadtxt(filename)
    
    # Verify data
    assert data.shape[0] == x.size
    assert data.shape[1] >= 4  # At least x, y, u, v columns
    
    # Clean up
    import os
    os.remove(filename)


def test_negative():
    """Test the negative function"""
    # Create a test image
    img = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    
    # Apply negative function
    neg_img = negative(img)
    
    # Check results
    assert np.all(neg_img == 255 - img)
    
    # Test with float image
    img_float = np.array([[0.1, 0.2], [0.3, 0.4]])
    neg_img_float = negative(img_float)
    assert np.allclose(neg_img_float, 255 - img_float)  # Subtracts from 255, not 1.0


def test_display_vector_field_from_arrays_with_warnings_suppressed():
    """Test display_vector_field_from_arrays function with warnings suppressed"""
    import warnings
    
    # Create test data
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[1, 2], [3, 4]])
    u = np.array([[0.1, 0.2], [0.3, 0.4]])
    v = np.array([[0.1, 0.2], [0.3, 0.4]])
    flags = np.zeros_like(u)
    mask = np.zeros_like(u)
    
    # Test with warnings suppressed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        # Test basic functionality
        display_vector_field_from_arrays(x, y, u, v, flags, mask)
        
        # Test with width parameter
        display_vector_field_from_arrays(x, y, u, v, flags, mask, width=0.01)
        
        # Test with custom axes
        fig, ax = plt.subplots()
        display_vector_field_from_arrays(x, y, u, v, flags, mask, ax=ax)
        plt.close(fig)


def test_multiprocesser():
    """Test the Multiprocesser class"""
    # Create a temporary directory with test files
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create a few empty test files
        for i in range(3):
            open(os.path.join(tmpdirname, f'img_a_{i}.tif'), 'w').close()
            open(os.path.join(tmpdirname, f'img_b_{i}.tif'), 'w').close()
        
        # Create a Multiprocesser instance
        mp = Multiprocesser(
            data_dir=pathlib.Path(tmpdirname),
            pattern_a='img_a_*.tif',
            pattern_b='img_b_*.tif'
        )
        
        # Check if files were found
        assert len(mp.files_a) == 3
        assert len(mp.files_b) == 3
        
        # Define a simple processing function
        def process_func(args):
            file_a, file_b, counter = args
            # Just return the filenames to verify they're passed correctly
            return (file_a.name, file_b.name, counter)
        
        # We won't actually run the process since it would try to read the empty files
        # But we can check that the class was initialized correctly


def test_imread():
    """Test the imread function"""
    import tempfile
    from PIL import Image
    
    # Create a temporary image file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        # Create a simple test image
        img = np.zeros((10, 10), dtype=np.uint8)
        img[2:8, 2:8] = 255  # white square on black background
        
        # Save the image
        Image.fromarray(img).save(tmp.name)
    
    try:
        # Read the image using the imread function
        read_img = imread(tmp.name)
        
        # Check that the image was read correctly
        assert read_img.shape == (10, 10)
        assert np.all(read_img[2:8, 2:8] == 255)
        assert np.all(read_img[0:2, 0:2] == 0)
    finally:
        # Clean up
        import os
        os.unlink(tmp.name)
