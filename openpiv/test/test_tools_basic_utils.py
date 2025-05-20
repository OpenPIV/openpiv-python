"""Tests for basic utility functions in tools.py"""
import pathlib
import numpy as np
import pytest
from openpiv.tools import natural_sort, sorted_unique, display, negative


def test_natural_sort():
    """Test the natural_sort function with different inputs"""
    # Test with numeric filenames
    files = [
        pathlib.Path('file10.txt'),
        pathlib.Path('file2.txt'),
        pathlib.Path('file1.txt')
    ]
    sorted_files = natural_sort(files)

    # Check that files are sorted correctly (1, 2, 10 instead of 1, 10, 2)
    assert sorted_files[0].name == 'file1.txt'
    assert sorted_files[1].name == 'file2.txt'
    assert sorted_files[2].name == 'file10.txt'

    # Test with mixed alphanumeric filenames
    files = [
        pathlib.Path('file_b10.txt'),
        pathlib.Path('file_a2.txt'),
        pathlib.Path('file_a10.txt'),
        pathlib.Path('file_b2.txt')
    ]
    sorted_files = natural_sort(files)

    # Check that files are sorted correctly
    assert sorted_files[0].name == 'file_a2.txt'
    assert sorted_files[1].name == 'file_a10.txt'
    assert sorted_files[2].name == 'file_b2.txt'
    assert sorted_files[3].name == 'file_b10.txt'

    # Test with empty list
    assert natural_sort([]) == []


def test_sorted_unique():
    """Test the sorted_unique function with different inputs"""
    # Test with simple array
    arr = np.array([3, 1, 2, 1, 3])
    result = sorted_unique(arr)

    # Check that result contains the unique values
    assert set(result) == set([1, 2, 3])
    # Check that result is sorted
    assert np.all(np.diff(result) > 0)

    # Test with more complex array
    arr = np.array([10, 5, 10, 2, 5, 1])
    result = sorted_unique(arr)

    # Check that result contains the unique values
    assert set(result) == set([1, 2, 5, 10])
    # Check that result is sorted
    assert np.all(np.diff(result) > 0)

    # Test with already sorted array
    arr = np.array([1, 2, 3, 4])
    result = sorted_unique(arr)

    # Check that result contains the same values
    assert set(result) == set(arr)
    # Check that result is sorted
    assert np.all(np.diff(result) > 0)

    # Test with empty array
    arr = np.array([])
    result = sorted_unique(arr)

    # Check that result is empty
    assert result.size == 0


def test_display(capsys):
    """Test the display function"""
    # Test with simple message
    display("Test message")
    captured = capsys.readouterr()

    # Check that message was printed
    assert captured.out == "Test message\n"

    # Test with empty message
    display("")
    captured = capsys.readouterr()

    # Check that empty line was printed
    assert captured.out == "\n"

    # Test with multi-line message
    display("Line 1\nLine 2")
    captured = capsys.readouterr()

    # Check that message was printed correctly
    assert captured.out == "Line 1\nLine 2\n"


def test_negative():
    """Test the negative function with different inputs"""
    # Test with uint8 array
    img = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    result = negative(img)

    # Check that result is correct
    assert np.array_equal(result, 255 - img)

    # Test with float array
    img = np.array([[0.1, 0.2], [0.3, 0.4]])
    result = negative(img)

    # Check that result is correct
    assert np.allclose(result, 255 - img)

    # Test with all zeros
    img = np.zeros((3, 3))
    result = negative(img)

    # Check that result is all 255s
    assert np.array_equal(result, np.full((3, 3), 255))

    # Test with all 255s
    img = np.full((3, 3), 255)
    result = negative(img)

    # Check that result is all zeros
    assert np.array_equal(result, np.zeros((3, 3)))
