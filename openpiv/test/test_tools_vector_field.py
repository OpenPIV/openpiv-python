"""Tests for vector field operations in tools.py"""
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import pytest
from openpiv.tools import (
    save, display_vector_field, display_vector_field_from_arrays,
    transform_coordinates, display_windows_sampling
)


def test_save_basic():
    """Test basic functionality of save function"""
    # Create test data
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])
    u = np.array([[0.1, 0.2], [0.3, 0.4]])
    v = np.array([[0.5, 0.6], [0.7, 0.8]])

    # Save data to temporary file
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
        save(tmp.name, x, y, u, v)

    try:
        # Load data back
        data = np.loadtxt(tmp.name)

        # Check shape
        assert data.shape == (4, 6)  # 4 points, 6 columns (x, y, u, v, flags, mask)

        # Check data
        assert np.allclose(data[:, 0], [1, 2, 3, 4])  # x
        assert np.allclose(data[:, 1], [5, 6, 7, 8])  # y
        assert np.allclose(data[:, 2], [0.1, 0.2, 0.3, 0.4])  # u
        assert np.allclose(data[:, 3], [0.5, 0.6, 0.7, 0.8])  # v
        assert np.allclose(data[:, 4], [0, 0, 0, 0])  # flags (default 0)
        assert np.allclose(data[:, 5], [0, 0, 0, 0])  # mask (default 0)
    finally:
        # Clean up
        os.unlink(tmp.name)


def test_save_with_flags_and_mask():
    """Test save function with flags and mask"""
    # Create test data
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])
    u = np.array([[0.1, 0.2], [0.3, 0.4]])
    v = np.array([[0.5, 0.6], [0.7, 0.8]])
    flags = np.array([[1, 0], [0, 1]])
    mask = np.array([[0, 1], [1, 0]])

    # Save data to temporary file
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
        save(tmp.name, x, y, u, v, flags, mask)

    try:
        # Load data back
        data = np.loadtxt(tmp.name)

        # Check shape
        assert data.shape == (4, 6)  # 4 points, 6 columns (x, y, u, v, flags, mask)

        # Check data
        assert np.allclose(data[:, 0], [1, 2, 3, 4])  # x
        assert np.allclose(data[:, 1], [5, 6, 7, 8])  # y
        assert np.allclose(data[:, 2], [0.1, 0.2, 0.3, 0.4])  # u
        assert np.allclose(data[:, 3], [0.5, 0.6, 0.7, 0.8])  # v
        assert np.allclose(data[:, 4], [1, 0, 0, 1])  # flags
        assert np.allclose(data[:, 5], [0, 1, 1, 0])  # mask
    finally:
        # Clean up
        os.unlink(tmp.name)


def test_save_with_masked_array():
    """Test save function with masked arrays"""
    # Create test data
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])
    u = np.ma.array([[0.1, 0.2], [0.3, 0.4]], mask=[[True, False], [False, True]])
    v = np.ma.array([[0.5, 0.6], [0.7, 0.8]], mask=[[True, False], [False, True]])

    # Save data to temporary file
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
        save(tmp.name, x, y, u, v)

    try:
        # Load data back
        data = np.loadtxt(tmp.name)

        # Check shape
        assert data.shape == (4, 6)  # 4 points, 6 columns (x, y, u, v, flags, mask)

        # Check data
        assert np.allclose(data[:, 0], [1, 2, 3, 4])  # x
        assert np.allclose(data[:, 1], [5, 6, 7, 8])  # y
        # Masked values should be filled with 0
        assert np.allclose(data[:, 2], [0.0, 0.2, 0.3, 0.0])  # u
        assert np.allclose(data[:, 3], [0.0, 0.6, 0.7, 0.0])  # v
        assert np.allclose(data[:, 4], [0, 0, 0, 0])  # flags (default 0)
        assert np.allclose(data[:, 5], [0, 0, 0, 0])  # mask (default 0)
    finally:
        # Clean up
        os.unlink(tmp.name)


def test_save_with_custom_format():
    """Test save function with custom format"""
    # Create test data
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])
    u = np.array([[0.1, 0.2], [0.3, 0.4]])
    v = np.array([[0.5, 0.6], [0.7, 0.8]])

    # Save data to temporary file with custom format
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
        save(tmp.name, x, y, u, v, fmt='%.2f', delimiter=',')

    try:
        # Load data back
        data = np.loadtxt(tmp.name, delimiter=',')

        # Check shape
        assert data.shape == (4, 6)  # 4 points, 6 columns (x, y, u, v, flags, mask)

        # Check data (with reduced precision due to format)
        assert np.allclose(data[:, 0], [1.00, 2.00, 3.00, 4.00])  # x
        assert np.allclose(data[:, 1], [5.00, 6.00, 7.00, 8.00])  # y
        assert np.allclose(data[:, 2], [0.10, 0.20, 0.30, 0.40])  # u
        assert np.allclose(data[:, 3], [0.50, 0.60, 0.70, 0.80])  # v
    finally:
        # Clean up
        os.unlink(tmp.name)


def test_transform_coordinates_2d():
    """Test transform_coordinates with 2D arrays"""
    # Create test data
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])
    u = np.array([[0.1, 0.2], [0.3, 0.4]])
    v = np.array([[0.5, 0.6], [0.7, 0.8]])

    # Store original values for comparison
    v_orig = v.copy()

    # Transform coordinates
    x_new, y_new, u_new, v_new = transform_coordinates(x, y, u, v)

    # Check results
    assert np.array_equal(x_new, x)  # x should be unchanged
    assert np.array_equal(y_new, y[::-1])  # y should be flipped vertically
    assert np.array_equal(u_new, u)  # u should be unchanged
    assert np.array_equal(v_new, -v_orig)  # v should be negated


def test_transform_coordinates_1d():
    """Test transform_coordinates with 1D arrays"""
    # Create test data
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    u = np.array([0.1, 0.2, 0.3])
    v = np.array([0.4, 0.5, 0.6])

    # Store original values for comparison
    v_orig = v.copy()

    # Transform coordinates
    x_new, y_new, u_new, v_new = transform_coordinates(x, y, u, v)

    # Check results
    assert np.array_equal(x_new, x)  # x should be unchanged
    assert np.array_equal(y_new, y[::-1])  # y should be flipped
    assert np.array_equal(u_new, u)  # u should be unchanged
    assert np.array_equal(v_new, -v_orig)  # v should be negated


@pytest.mark.parametrize("show_invalid", [True, False])
def test_display_vector_field_from_arrays(show_invalid):
    """Test display_vector_field_from_arrays function"""
    # Create test data
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])
    u = np.array([[0.1, 0.2], [0.3, 0.4]])
    v = np.array([[0.5, 0.6], [0.7, 0.8]])
    flags = np.array([[1, 0], [0, 0]])  # One invalid vector
    mask = np.zeros_like(flags)

    # Create a figure and axes for testing
    fig, ax = plt.subplots()

    # Call function with show_invalid parameter
    fig_out, ax_out = display_vector_field_from_arrays(
        x, y, u, v, flags, mask, ax=ax, show_invalid=show_invalid
    )

    # Check that the function returns the same figure and axes
    assert fig_out is fig
    assert ax_out is ax

    # Clean up
    plt.close(fig)


@pytest.mark.parametrize("method", ["standard", "random"])
@pytest.mark.skip(reason="Requires interactive matplotlib backend")
def test_display_windows_sampling(method):
    """Test display_windows_sampling function"""
    # Create test data
    x = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    y = np.array([[4, 4, 4], [5, 5, 5], [6, 6, 6]])
    window_size = 16

    # Create a figure for testing
    fig = plt.figure()

    # Temporarily replace plt.show to avoid displaying the figure
    original_show = plt.show
    plt.show = lambda: None

    try:
        # Call function
        display_windows_sampling(x, y, window_size, skip=1, method=method)
    finally:
        # Restore plt.show
        plt.show = original_show

        # Clean up
        plt.close(fig)
