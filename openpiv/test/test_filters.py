from openpiv import filters
from openpiv.lib import replace_nans
import numpy as np
import pytest


def test_gaussian_kernel():
    """ test of a _gaussian_kernel """

    assert np.allclose(
        filters._gaussian_kernel(1),
        np.array(
            [
                [0.04491922, 0.12210311, 0.04491922],
                [0.12210311, 0.33191066, 0.12210311],
                [0.04491922, 0.12210311, 0.04491922],
            ]
        ),
    )

    # Test the case when half_width is 0
    assert filters._gaussian_kernel(0) == 1


def test_gaussian_kernel_function():
    """Test the gaussian_kernel function"""
    # Test with sigma=1.0 and default truncate=4.0
    kernel = filters.gaussian_kernel(1.0)
    assert kernel.shape == (9, 9)  # Should be 2*radius+1 where radius = int(truncate*sigma+0.5)
    assert np.isclose(np.sum(kernel), 1.0)  # Kernel should be normalized
    
    # Test with different sigma and truncate values
    kernel = filters.gaussian_kernel(0.5, truncate=2.0)
    assert kernel.shape == (3, 3)  # Should be 2*radius+1 where radius = int(truncate*sigma+0.5)
    assert np.isclose(np.sum(kernel), 1.0)  # Kernel should be normalized


def test_gaussian():
    """ test of a Gaussian filter """
    u = np.ones((3, 3))
    v = np.eye(3)
    uf, vf = filters.gaussian(u, v, 1)
    assert np.allclose(
        uf,
        np.array(
            [
                [0.62103611, 0.78805844, 0.62103611],
                [0.78805844, 1.0, 0.78805844],
                [0.62103611, 0.78805844, 0.62103611],
            ]
        ),
    )
    assert np.allclose(
        vf,
        np.array(
            [
                [0.37682989, 0.24420622, 0.04491922],
                [0.24420622, 0.42174911, 0.24420622],
                [0.04491922, 0.24420622, 0.37682989],
            ]
        ),
    )


def test_replace_nans():
    """ test of NaNs inpainting """

    u = np.nan * np.ones((5, 5))
    u[2, 2] = 1
    u = replace_nans(u, 2, 1e-3)
    assert ~np.all(np.isnan(u))

    u = np.ones((9, 9))
    u[1:-1, 1:-1] = np.nan
    u = replace_nans(u, 1, 1e-3, method="disk")
    assert np.sum(np.isnan(u)) == 9  # central core is nan

    u = np.ones((9, 9))
    u[1:-1, 1:-1] = np.nan
    u = replace_nans(u, 2, 1e-3, method="disk")
    assert np.allclose(np.ones((9, 9)), u)


def test_replace_outliers():
    """ test of replacing outliers """
    v = np.ma.array(np.ones((5, 5)), mask=np.ma.nomask)
    v[3:,3:] = np.ma.masked

    v_copy = np.ma.copy(v) # without NaNs

    v[1, 1] = np.nan
    invalid_mask = np.isnan(v)
    u = v.copy()
    uf, vf = filters.replace_outliers(u, v, invalid_mask)

    assert np.ma.allclose(v_copy, uf)
    assert isinstance(uf, np.ma.MaskedArray)


def test_replace_outliers_with_w():
    """Test replace_outliers with w parameter"""
    # Create test data
    u = np.ma.array(np.ones((5, 5)), mask=np.ma.nomask)
    v = np.ma.array(np.ones((5, 5)), mask=np.ma.nomask)
    w = np.ma.array(np.ones((5, 5)), mask=np.ma.nomask)
    
    # Add some masked values
    u[3:, 3:] = np.ma.masked
    v[3:, 3:] = np.ma.masked
    w[3:, 3:] = np.ma.masked
    
    # Create copies for comparison
    u_copy = np.ma.copy(u)
    v_copy = np.ma.copy(v)
    w_copy = np.ma.copy(w)
    
    # Add some NaN values
    u[1, 1] = np.nan
    v[1, 1] = np.nan
    w[1, 1] = np.nan
    
    # Create invalid mask
    invalid_mask = np.isnan(u.data)
    
    # Call replace_outliers with w parameter
    uf, vf, wf = filters.replace_outliers(u, v, invalid_mask, w=w)
    
    # Check results
    assert np.ma.allclose(u_copy, uf)
    assert np.ma.allclose(v_copy, vf)
    assert np.ma.allclose(w_copy, wf)
    assert isinstance(uf, np.ma.MaskedArray)
    assert isinstance(vf, np.ma.MaskedArray)
    assert isinstance(wf, np.ma.MaskedArray)


def test_replace_outliers_different_methods():
    """Test replace_outliers with different methods"""
    # Create test data
    u = np.ma.array(np.ones((7, 7)), mask=np.ma.nomask)
    v = np.ma.array(np.ones((7, 7)), mask=np.ma.nomask)
    
    # Add some masked values
    u[5:, 5:] = np.ma.masked
    v[5:, 5:] = np.ma.masked
    
    # Add some NaN values in a pattern
    u[1:4, 1:4] = np.nan
    v[1:4, 1:4] = np.nan
    
    # Create invalid mask
    invalid_mask = np.isnan(u.data)
    
    # Test different methods
    for method in ['localmean', 'disk', 'distance']:
        uf, vf = filters.replace_outliers(
            u.copy(), v.copy(), invalid_mask, 
            method=method, max_iter=10, kernel_size=2
        )
        
        # Check that NaNs were replaced
        assert not np.any(np.isnan(uf))
        assert not np.any(np.isnan(vf))
        
        # Check that masks are preserved
        assert np.all(uf.mask[5:, 5:])
        assert np.all(vf.mask[5:, 5:])


def test_replace_outliers_non_masked_input():
    """Test replace_outliers with non-masked input arrays"""
    # Create regular numpy arrays (not masked)
    u = np.ones((5, 5))
    v = np.ones((5, 5))
    
    # Add some NaN values
    u[1, 1] = np.nan
    v[1, 1] = np.nan
    
    # Create invalid mask
    invalid_mask = np.isnan(u)
    
    # Call replace_outliers
    uf, vf = filters.replace_outliers(u, v, invalid_mask)
    
    # Check results
    assert isinstance(uf, np.ma.MaskedArray)
    assert isinstance(vf, np.ma.MaskedArray)
    assert not np.any(np.isnan(uf))
    assert not np.any(np.isnan(vf))
