"""Tests for the piv module"""
import numpy as np
import pytest
from importlib_resources import files
from openpiv import piv, tools
from openpiv.pyprocess import extended_search_area_piv
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing plots
import matplotlib.pyplot as plt
from unittest.mock import patch

# Create synthetic image pairs for testing
def create_test_pair(image_size=32, shift=(2, 2)):
    """Create a pair of synthetic images with known displacement"""
    # Create a random image with stronger patterns for better correlation
    np.random.seed(42)  # For reproducibility
    frame_a = np.zeros((image_size, image_size))
    # Add some particle-like features
    for _ in range(image_size * 2):
        x = np.random.randint(0, image_size)
        y = np.random.randint(0, image_size)
        r = np.random.randint(2, 5)
        frame_a[max(0, y-r):min(image_size, y+r), max(0, x-r):min(image_size, x+r)] = 1.0
    
    # Apply Gaussian blur to make particles more realistic
    from scipy.ndimage import gaussian_filter
    frame_a = gaussian_filter(frame_a, sigma=1.5)
    
    # Shift the image to create the second frame
    dx, dy = shift
    frame_b = np.zeros_like(frame_a)
    for y in range(image_size):
        for x in range(image_size):
            new_y = (y + dy) % image_size
            new_x = (x + dx) % image_size
            frame_b[new_y, new_x] = frame_a[y, x]
            
    return frame_a, frame_b


def test_simple_piv_with_arrays():
    """Test simple_piv with numpy arrays as input"""
    # Create test images
    frame_a, frame_b = create_test_pair(image_size=64, shift=(2, 2))
    
    # Run simple_piv with plot=False to avoid display during tests
    x, y, u, v, s2n = piv.simple_piv(frame_a, frame_b, plot=False)
    
    # Check that results are not None
    assert x is not None
    assert y is not None
    assert u is not None
    assert v is not None
    assert s2n is not None
    
    # Check shapes
    assert x.shape == y.shape == u.shape == v.shape == s2n.shape
    
    # Check that at least some values are valid (not NaN)
    assert not np.all(np.isnan(u))
    assert not np.all(np.isnan(v))
    
    # Check that the mean displacement of valid values has the expected sign
    valid_mask = ~np.isnan(u)
    if np.any(valid_mask):
        # Just check that u is positive and v is negative
        assert np.mean(u[valid_mask]) > 0
        assert np.mean(v[valid_mask]) < 0


def test_simple_piv_with_file_paths():
    """Test simple_piv with file paths as input"""
    # Get example image paths
    im1 = files('openpiv.data').joinpath('test1/exp1_001_a.bmp')
    im2 = files('openpiv.data').joinpath('test1/exp1_001_b.bmp')
    
    # Run simple_piv with plot=False
    x, y, u, v, s2n = piv.simple_piv(str(im1), str(im2), plot=False)
    
    # Check that results are not None
    assert x is not None
    assert y is not None
    assert u is not None
    assert v is not None
    assert s2n is not None
    
    # Check shapes
    assert x.shape == y.shape == u.shape == v.shape == s2n.shape


@pytest.mark.parametrize("validation_method", [None, "sig2noise", "global_std"])
def test_simple_piv_validation_methods(validation_method):
    """Test simple_piv with different validation methods"""
    # Create test images
    frame_a, frame_b = create_test_pair(image_size=64, shift=(2, 2))
    
    # Run simple_piv with the specified validation method
    x, y, u, v, s2n = piv.simple_piv(
        frame_a, frame_b, 
        validation_method=validation_method,
        plot=False
    )
    
    # Check that results are not None
    assert x is not None
    assert y is not None
    assert u is not None
    assert v is not None
    assert s2n is not None


@pytest.mark.parametrize("window_size,overlap,search_area_size", [
    (16, 8, 32),
    (32, 16, 64),
    (64, 32, 64)
])
def test_simple_piv_parameters(window_size, overlap, search_area_size):
    """Test simple_piv with different parameter combinations"""
    # Create test images
    frame_a, frame_b = create_test_pair(image_size=128, shift=(2, 2))
    
    # Run simple_piv with the specified parameters
    x, y, u, v, s2n = piv.simple_piv(
        frame_a, frame_b,
        window_size=window_size,
        overlap=overlap,
        search_area_size=search_area_size,
        plot=False
    )
    
    # Check that results are not None
    assert x is not None
    assert y is not None
    assert u is not None
    assert v is not None
    assert s2n is not None
    
    # Just check that shapes are consistent with each other
    assert x.shape == y.shape == u.shape == v.shape == s2n.shape


def test_piv_example_no_plots():
    """Test piv_example with plotting disabled"""
    # Run piv_example with plotting disabled
    x, y, u, v = piv.piv_example(plot_animation=False, plot_quiver=False)
    
    # Check that results are not None
    assert x is not None
    assert y is not None
    assert u is not None
    assert v is not None
    
    # Check shapes
    assert x.shape == y.shape == u.shape == v.shape


def test_process_pair():
    """Test process_pair function"""
    # Create test images
    frame_a, frame_b = create_test_pair(image_size=64, shift=(2, 2))
    
    # Run process_pair
    x, y, u, v, mask = piv.process_pair(
        frame_a, frame_b,
        window_size=32,
        overlap=16,
        search_area_size=32,
        plot=False
    )
    
    # Check that results are not None
    assert x is not None
    assert y is not None
    assert u is not None
    assert v is not None
    assert mask is not None
    
    # Check shapes
    assert x.shape == y.shape == u.shape == v.shape == mask.shape
    
    # Check that at least some values are valid (not NaN)
    assert not np.all(np.isnan(u))
    assert not np.all(np.isnan(v))
    
    # Check that the mean displacement of valid values has the expected sign
    valid_mask = ~np.isnan(u)
    if np.any(valid_mask):
        # Just check that u is positive and v is negative
        assert np.mean(u[valid_mask]) > 0
        assert np.mean(v[valid_mask]) < 0


@pytest.mark.parametrize("validation_method", [None, "sig2noise", "global_std"])
def test_process_pair_validation_methods(validation_method):
    """Test process_pair with different validation methods"""
    # Create test images
    frame_a, frame_b = create_test_pair(image_size=64, shift=(2, 2))
    
    # Run process_pair with the specified validation method
    x, y, u, v, mask = piv.process_pair(
        frame_a, frame_b,
        validation_method=validation_method,
        plot=False
    )
    
    # Check that results are not None
    assert x is not None
    assert y is not None
    assert u is not None
    assert v is not None
    assert mask is not None


@pytest.mark.parametrize("filter_method", ["localmean", "disk", "distance"])
def test_process_pair_filter_methods(filter_method):
    """Test process_pair with different filter methods"""
    # Create test images
    frame_a, frame_b = create_test_pair(image_size=64, shift=(2, 2))
    
    # Add some outliers to test filtering
    frame_b[10:15, 10:15] = 0  # Create a region with bad correlation
    
    # Run process_pair with the specified filter method
    x, y, u, v, mask = piv.process_pair(
        frame_a, frame_b,
        validation_method="sig2noise",
        s2n_threshold=1.5,  # Higher threshold to create more outliers
        filter_method=filter_method,
        filter_kernel_size=2,
        plot=False
    )
    
    # Check that results are not None
    assert x is not None
    assert y is not None
    assert u is not None
    assert v is not None
    assert mask is not None


def test_process_pair_with_real_images():
    """Test process_pair with real images from the package data"""
    # Get example image paths
    im1 = files('openpiv.data').joinpath('test1/exp1_001_a.bmp')
    im2 = files('openpiv.data').joinpath('test1/exp1_001_b.bmp')
    
    # Load images
    frame_a = tools.imread(im1)
    frame_b = tools.imread(im2)
    
    # Run process_pair
    x, y, u, v, mask = piv.process_pair(
        frame_a, frame_b,
        window_size=32,
        overlap=16,
        search_area_size=64,
        plot=False
    )
    
    # Check that results are not None
    assert x is not None
    assert y is not None
    assert u is not None
    assert v is not None
    assert mask is not None
    
    # Check shapes
    assert x.shape == y.shape == u.shape == v.shape == mask.shape


def test_piv_example_with_quiver_only():
    """Test piv_example with only quiver plotting enabled"""
    # Save the current backend
    original_backend = plt.get_backend()
    plt.switch_backend('Agg')
    
    try:
        # Mock plt.show to prevent actual display
        with patch('matplotlib.pyplot.show') as mock_show:
            # Run piv_example with only quiver plot enabled, no animation
            x, y, u, v = piv.piv_example(plot_animation=False, plot_quiver=True)
            
            # Check that plt.show was called at least once
            assert mock_show.called
        
        # Check that results are not None
        assert x is not None
        assert y is not None
        assert u is not None
        assert v is not None
        
    finally:
        # Restore the original backend
        plt.switch_backend(original_backend)
        plt.close('all')


def test_simple_piv_with_plotting():
    """Test simple_piv with plotting enabled"""
    # Save the current backend
    original_backend = plt.get_backend()
    plt.switch_backend('Agg')
    
    try:
        # Create test images
        frame_a, frame_b = create_test_pair(image_size=64, shift=(2, 2))
        
        # Mock plt.show to prevent actual display
        with patch('matplotlib.pyplot.show') as mock_show:
            # Run simple_piv with plot=True
            x, y, u, v, s2n = piv.simple_piv(frame_a, frame_b, plot=True)
            
            # Check that plt.show was called
            assert mock_show.called
        
        # Check that results are not None
        assert x is not None
        assert y is not None
        assert u is not None
        assert v is not None
        assert s2n is not None
        
    finally:
        # Restore the original backend
        plt.switch_backend(original_backend)
        plt.close('all')


def test_process_pair_with_plotting():
    """Test process_pair with plotting enabled"""
    # Save the current backend
    original_backend = plt.get_backend()
    plt.switch_backend('Agg')
    
    try:
        # Create test images
        frame_a, frame_b = create_test_pair(image_size=64, shift=(2, 2))
        
        # Mock plt.show to prevent actual display
        with patch('matplotlib.pyplot.show') as mock_show:
            # Run process_pair with plot=True
            x, y, u, v, mask = piv.process_pair(
                frame_a, frame_b,
                window_size=32,
                overlap=16,
                search_area_size=32,
                plot=True
            )
            
            # Check that plt.show was called
            assert mock_show.called
        
        # Check that results are not None
        assert x is not None
        assert y is not None
        assert u is not None
        assert v is not None
        assert mask is not None
        
    finally:
        # Restore the original backend
        plt.switch_backend(original_backend)
        plt.close('all')


@pytest.mark.parametrize("dt", [0.5, 1.0, 2.0])
def test_simple_piv_with_different_dt(dt):
    """Test simple_piv with different dt values"""
    # Create test images
    frame_a, frame_b = create_test_pair(image_size=64, shift=(2, 2))
    
    # Run simple_piv with the specified dt
    x, y, u, v, s2n = piv.simple_piv(
        frame_a, frame_b,
        dt=dt,
        plot=False
    )
    
    # Check that results are not None
    assert x is not None
    assert y is not None
    assert u is not None
    assert v is not None
    assert s2n is not None
    
    # Check that the velocity values are consistent regardless of dt
    # (The implementation doesn't scale velocities with dt)
    valid_mask = ~np.isnan(u)
    if np.any(valid_mask):
        # Just verify that we have positive u values and negative v values
        assert np.mean(u[valid_mask]) > 0
        assert np.mean(v[valid_mask]) < 0


def test_simple_piv_with_invalid_inputs():
    """Test simple_piv with invalid inputs"""
    # Test with empty arrays
    with pytest.raises((ValueError, IndexError, ZeroDivisionError)):
        piv.simple_piv(np.array([]), np.array([]), plot=False)
    
    # Test with arrays of different sizes
    frame_a = np.random.rand(32, 32)
    frame_b = np.random.rand(64, 64)
    
    with pytest.raises((ValueError, IndexError, AssertionError), match=""):
        piv.simple_piv(frame_a, frame_b, plot=False)


def test_process_pair_with_different_parameters():
    """Test process_pair with different parameter combinations"""
    # Create test images
    frame_a, frame_b = create_test_pair(image_size=64, shift=(2, 2))
    
    # Test with different s2n_threshold values
    for s2n_threshold in [1.0, 1.5, 2.0]:
        x, y, u, v, mask = piv.process_pair(
            frame_a, frame_b,
            validation_method="sig2noise",
            s2n_threshold=s2n_threshold,
            plot=False
        )
        assert x is not None
        assert y is not None
        assert u is not None
        assert v is not None
        assert mask is not None
    
    # Test with different filter_kernel_size values
    for kernel_size in [1, 2, 3]:
        x, y, u, v, mask = piv.process_pair(
            frame_a, frame_b,
            filter_method="localmean",
            filter_kernel_size=kernel_size,
            plot=False
        )
        assert x is not None
        assert y is not None
        assert u is not None
        assert v is not None
        assert mask is not None
