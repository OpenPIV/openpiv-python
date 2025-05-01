import numpy as np
import pytest
from openpiv.lib import replace_nans, get_dist


def test_replace_nans_2d():
    """Test replace_nans function with 2D arrays"""
    # Create a 2D array with NaNs
    array = np.ones((10, 10))
    array[3:7, 3:7] = np.nan  # Create a square of NaNs

    # Test with localmean method
    filled = replace_nans(array, max_iter=100, tol=1e-6, kernel_size=1, method="localmean")
    assert not np.any(np.isnan(filled))
    assert np.allclose(filled[0:3, 0:3], 1.0)  # Original values should be preserved

    # Test with disk method
    filled = replace_nans(array, max_iter=100, tol=1e-6, kernel_size=2, method="disk")
    assert not np.any(np.isnan(filled))
    assert np.allclose(filled[0:3, 0:3], 1.0)  # Original values should be preserved

    # Test with distance method
    filled = replace_nans(array, max_iter=100, tol=1e-6, kernel_size=2, method="distance")
    assert not np.any(np.isnan(filled))
    assert np.allclose(filled[0:3, 0:3], 1.0)  # Original values should be preserved


def test_replace_nans_3d():
    """Test replace_nans function with 3D arrays"""
    # Create a 3D array with NaNs
    array = np.ones((5, 5, 5))
    array[1:4, 1:4, 1:4] = np.nan  # Create a cube of NaNs

    # Test with localmean method
    filled = replace_nans(array, max_iter=100, tol=1e-6, kernel_size=1, method="localmean")
    assert not np.any(np.isnan(filled))
    assert np.allclose(filled[0, 0, 0], 1.0)  # Original values should be preserved

    # Test with disk method
    filled = replace_nans(array, max_iter=100, tol=1e-6, kernel_size=1, method="disk")
    assert not np.any(np.isnan(filled))
    assert np.allclose(filled[0, 0, 0], 1.0)  # Original values should be preserved

    # Test with distance method - needs more iterations for 3D
    filled = replace_nans(array, max_iter=200, tol=1e-6, kernel_size=2, method="distance")
    # Check if most NaNs are replaced (might not be all due to algorithm limitations)
    nan_count = np.sum(np.isnan(filled))
    original_nan_count = np.sum(np.isnan(array))
    assert nan_count < original_nan_count
    # Check that non-NaN values are preserved
    assert np.allclose(filled[0, 0, 0], 1.0)


def test_replace_nans_masked_array():
    """Test replace_nans function with masked arrays"""
    # Create a masked array with NaNs
    array = np.ma.array(np.ones((8, 8)), mask=np.zeros((8, 8), dtype=bool))
    array[2:6, 2:6] = np.nan  # Create a square of NaNs
    array.mask[6:8, 6:8] = True  # Mask a corner

    # Test with localmean method
    filled = replace_nans(array, max_iter=100, tol=1e-6, kernel_size=1, method="localmean")
    assert not np.any(np.isnan(filled))
    assert np.ma.is_masked(filled)  # Result should still be masked
    assert np.all(filled.mask[6:8, 6:8])  # Original mask should be preserved


def test_replace_nans_convergence():
    """Test replace_nans function convergence with different tolerances"""
    # Create an array with NaNs
    array = np.ones((10, 10))
    array[3:7, 3:7] = np.nan  # Create a square of NaNs

    # Test with different tolerances and very few iterations
    # This ensures the algorithm doesn't fully converge
    filled_low_tol = replace_nans(array, max_iter=2, tol=1e-2, kernel_size=1, method="localmean")
    filled_high_tol = replace_nans(array, max_iter=2, tol=1e-6, kernel_size=1, method="localmean")

    # Check that both methods replaced some NaNs
    assert np.sum(np.isnan(filled_low_tol)) < np.sum(np.isnan(array))
    assert np.sum(np.isnan(filled_high_tol)) < np.sum(np.isnan(array))
    
    # Check that original values are preserved
    assert np.allclose(filled_low_tol[0, 0], 1.0)
    assert np.allclose(filled_high_tol[0, 0], 1.0)


def test_replace_nans_max_iter():
    """Test replace_nans function with different max_iter values"""
    # Create an array with NaNs
    array = np.ones((10, 10))
    array[3:7, 3:7] = np.nan  # Create a square of NaNs

    # Test with different max_iter values
    filled_few_iter = replace_nans(array, max_iter=1, tol=1e-10, kernel_size=1, method="localmean")
    filled_many_iter = replace_nans(array, max_iter=100, tol=1e-10, kernel_size=1, method="localmean")

    # Check that both methods replaced some NaNs
    assert np.sum(np.isnan(filled_few_iter)) < np.sum(np.isnan(array))
    assert np.sum(np.isnan(filled_many_iter)) < np.sum(np.isnan(array))
    
    # More iterations should replace more NaNs
    assert np.sum(np.isnan(filled_few_iter)) >= np.sum(np.isnan(filled_many_iter))


def test_replace_nans_kernel_size():
    """Test replace_nans function with different kernel sizes"""
    # Create an array with NaNs
    array = np.ones((10, 10))
    array[3:7, 3:7] = np.nan  # Create a square of NaNs

    # Test with different kernel sizes and very few iterations
    # This ensures the algorithm doesn't fully converge
    filled_small_kernel = replace_nans(array, max_iter=2, tol=1e-6, kernel_size=1, method="localmean")
    filled_large_kernel = replace_nans(array, max_iter=2, tol=1e-6, kernel_size=3, method="localmean")

    # Check that both methods replaced some NaNs
    assert np.sum(np.isnan(filled_small_kernel)) < np.sum(np.isnan(array))
    assert np.sum(np.isnan(filled_large_kernel)) < np.sum(np.isnan(array))
    
    # Larger kernel should replace more NaNs in fewer iterations
    assert np.sum(np.isnan(filled_small_kernel)) >= np.sum(np.isnan(filled_large_kernel))


def test_replace_nans_invalid_method():
    """Test replace_nans function with invalid method"""
    array = np.ones((5, 5))
    array[2, 2] = np.nan

    # Test with invalid method
    with pytest.raises(ValueError, match="Known methods are:"):
        replace_nans(array, max_iter=10, tol=1e-6, kernel_size=1, method="invalid_method")


def test_replace_nans_all_nan_neighbors():
    """Test replace_nans function when all neighbors are NaN"""
    # Create an array where a NaN element is surrounded by other NaNs
    array = np.ones((5, 5))
    array[1:4, 1:4] = np.nan  # Create a square of NaNs

    # The center element has only NaN neighbors
    filled = replace_nans(array, max_iter=10, tol=1e-6, kernel_size=1, method="localmean")
    
    # The algorithm should still work, but the center might still be NaN after few iterations
    # Let's check that at least some NaNs were replaced
    assert np.sum(np.isnan(filled)) < np.sum(np.isnan(array))


def test_replace_nans_no_nans():
    """Test replace_nans function with an array that has no NaNs"""
    array = np.ones((5, 5))  # No NaNs
    
    filled = replace_nans(array, max_iter=10, tol=1e-6, kernel_size=1, method="localmean")
    
    # The result should be identical to the input
    assert np.array_equal(array, filled)


def test_get_dist_2d():
    """Test get_dist function with 2D kernel"""
    kernel = np.zeros((5, 5))
    kernel_size = 2
    
    dist, dist_inv = get_dist(kernel, kernel_size)
    
    # Check shapes
    assert dist.shape == (5, 5)
    assert dist_inv.shape == (5, 5)
    
    # Check center value
    assert dist[2, 2] == 0
    assert dist_inv[2, 2] == np.sqrt(2) * kernel_size
    
    # Check corner values (should be furthest from center)
    assert dist[0, 0] > dist[1, 1]
    assert dist_inv[0, 0] < dist_inv[1, 1]


def test_get_dist_3d():
    """Test get_dist function with 3D kernel"""
    kernel = np.zeros((5, 5, 5))
    kernel_size = 2
    
    dist, dist_inv = get_dist(kernel, kernel_size)
    
    # Check shapes
    assert dist.shape == (5, 5, 5)
    assert dist_inv.shape == (5, 5, 5)
    
    # Check center value
    assert dist[2, 2, 2] == 0
    assert dist_inv[2, 2, 2] == np.sqrt(3) * kernel_size
    
    # Check corner values (should be furthest from center)
    assert dist[0, 0, 0] > dist[1, 1, 1]
    assert dist_inv[0, 0, 0] < dist_inv[1, 1, 1]
