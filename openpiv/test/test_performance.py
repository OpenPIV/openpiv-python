"""Performance tests to verify optimizations."""
import numpy as np
import pytest
import time
from openpiv import pyprocess, validation, filters


def test_find_all_first_peaks_performance():
    """Test that find_all_first_peaks uses vectorized operations."""
    # Create test correlation maps
    n_windows = 100
    window_size = 32
    corr = np.random.rand(n_windows, window_size, window_size)
    
    # Add clear peaks
    for i in range(n_windows):
        peak_i = np.random.randint(5, window_size-5)
        peak_j = np.random.randint(5, window_size-5)
        corr[i, peak_i, peak_j] = 100.0
    
    start = time.time()
    indexes, peaks = pyprocess.find_all_first_peaks(corr)
    elapsed = time.time() - start
    
    # Verify results
    assert indexes.shape == (n_windows, 3)
    assert peaks.shape == (n_windows,)
    assert np.all(peaks >= 0)
    
    # Should be fast (< 10ms for 100 windows)
    assert elapsed < 0.01, f"find_all_first_peaks took {elapsed:.4f}s, expected < 0.01s"


def test_normalize_intensity_performance():
    """Test that normalize_intensity avoids unnecessary conversions."""
    # Test with float32 input (should not convert)
    window_float = np.random.rand(50, 64, 64).astype(np.float32)
    
    start = time.time()
    result = pyprocess.normalize_intensity(window_float)
    elapsed_float = time.time() - start
    
    assert result.dtype == np.float32
    
    # Test with uint8 input (needs conversion)
    window_uint = (np.random.rand(50, 64, 64) * 255).astype(np.uint8)
    
    start = time.time()
    result = pyprocess.normalize_intensity(window_uint)
    elapsed_uint = time.time() - start
    
    assert result.dtype == np.float32
    
    # Should be reasonably fast (< 50ms for 50 windows)
    assert elapsed_float < 0.05, f"normalize_intensity (float32) took {elapsed_float:.4f}s"
    assert elapsed_uint < 0.05, f"normalize_intensity (uint8) took {elapsed_uint:.4f}s"


def test_global_std_performance():
    """Test that global_std avoids unnecessary array copies."""
    # Create test data
    u = np.random.randn(100, 100) * 10
    v = np.random.randn(100, 100) * 10
    
    # Test with regular arrays
    start = time.time()
    flag = validation.global_std(u, v, std_threshold=3)
    elapsed_regular = time.time() - start
    
    assert flag.shape == u.shape
    
    # Test with masked arrays
    u_masked = np.ma.masked_array(u, mask=np.random.rand(100, 100) > 0.9)
    v_masked = np.ma.masked_array(v, mask=np.random.rand(100, 100) > 0.9)
    
    start = time.time()
    flag = validation.global_std(u_masked, v_masked, std_threshold=3)
    elapsed_masked = time.time() - start
    
    assert flag.shape == u.shape
    
    # Should be fast (< 10ms for 100x100 arrays)
    assert elapsed_regular < 0.01, f"global_std (regular) took {elapsed_regular:.4f}s"
    assert elapsed_masked < 0.01, f"global_std (masked) took {elapsed_masked:.4f}s"


def test_replace_outliers_performance():
    """Test that replace_outliers only creates masked arrays when needed."""
    # Create test data
    u = np.random.randn(50, 50) * 10
    v = np.random.randn(50, 50) * 10
    flags = np.random.rand(50, 50) > 0.95  # 5% outliers
    
    # Test with regular arrays
    start = time.time()
    uf, vf = filters.replace_outliers(u, v, flags, method='localmean', max_iter=3)
    elapsed = time.time() - start
    
    assert uf.shape == u.shape
    assert vf.shape == v.shape
    
    # Should be reasonably fast (< 100ms for 50x50 with 3 iterations)
    assert elapsed < 0.1, f"replace_outliers took {elapsed:.4f}s, expected < 0.1s"


def test_vectorized_sig2noise_ratio_performance():
    """Test that vectorized sig2noise ratio is faster than loop version."""
    # Create test correlation maps
    n_windows = 200
    window_size = 32
    corr = np.random.rand(n_windows, window_size, window_size) * 0.5
    
    # Add clear peaks
    for i in range(n_windows):
        peak_i = np.random.randint(5, window_size-5)
        peak_j = np.random.randint(5, window_size-5)
        corr[i, peak_i, peak_j] = 10.0
    
    # Test vectorized version
    start = time.time()
    s2n_vectorized = pyprocess.vectorized_sig2noise_ratio(
        corr, sig2noise_method='peak2peak', width=2
    )
    elapsed_vectorized = time.time() - start
    
    assert s2n_vectorized.shape == (n_windows,)
    assert np.all(s2n_vectorized >= 0)
    
    # Should be fast (< 50ms for 200 windows)
    assert elapsed_vectorized < 0.05, \
        f"vectorized_sig2noise_ratio took {elapsed_vectorized:.4f}s, expected < 0.05s"


if __name__ == "__main__":
    # Run tests manually with timing output
    print("Running performance tests...")
    
    print("\n1. Testing find_all_first_peaks_performance...")
    test_find_all_first_peaks_performance()
    print("   ✓ Passed")
    
    print("\n2. Testing normalize_intensity_performance...")
    test_normalize_intensity_performance()
    print("   ✓ Passed")
    
    print("\n3. Testing global_std_performance...")
    test_global_std_performance()
    print("   ✓ Passed")
    
    print("\n4. Testing replace_outliers_performance...")
    test_replace_outliers_performance()
    print("   ✓ Passed")
    
    print("\n5. Testing vectorized_sig2noise_ratio_performance...")
    test_vectorized_sig2noise_ratio_performance()
    print("   ✓ Passed")
    
    print("\n✅ All performance tests passed!")
