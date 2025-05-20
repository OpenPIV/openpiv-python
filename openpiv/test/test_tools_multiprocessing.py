"""Tests for multiprocessing functions in tools.py"""
import os
import tempfile
import pathlib
import numpy as np
import pytest
from openpiv.tools import Multiprocesser


def test_multiprocesser_basic():
    """Test basic functionality of Multiprocesser class"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create a few test files
        for i in range(3):
            open(os.path.join(tmpdirname, f'img_a_{i}.tif'), 'w').close()
            open(os.path.join(tmpdirname, f'img_b_{i}.tif'), 'w').close()
        
        # Create a Multiprocesser instance
        mp = Multiprocesser(
            data_dir=pathlib.Path(tmpdirname),
            pattern_a='img_a_*.tif',
            pattern_b='img_b_*.tif'
        )
        
        # Check that files were found
        assert len(mp.files_a) == 3
        assert len(mp.files_b) == 3
        
        # Check that files are in the correct order
        assert mp.files_a[0].name == 'img_a_0.tif'
        assert mp.files_a[1].name == 'img_a_1.tif'
        assert mp.files_a[2].name == 'img_a_2.tif'
        
        assert mp.files_b[0].name == 'img_b_0.tif'
        assert mp.files_b[1].name == 'img_b_1.tif'
        assert mp.files_b[2].name == 'img_b_2.tif'


def test_multiprocesser_pattern_1_plus_2():
    """Test Multiprocesser with pattern_b='(1+2),(2+3)'"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create a sequence of test files
        for i in range(5):
            open(os.path.join(tmpdirname, f'{i:04d}.tif'), 'w').close()
        
        # Create a Multiprocesser instance with pattern_b='(1+2),(2+3)'
        mp = Multiprocesser(
            data_dir=pathlib.Path(tmpdirname),
            pattern_a='*.tif',
            pattern_b='(1+2),(2+3)'
        )
        
        # Check that files were paired correctly
        assert len(mp.files_a) == 4  # 0001, 0002, 0003, 0004
        assert len(mp.files_b) == 4  # 0002, 0003, 0004, 0005
        
        # Check specific pairs
        assert mp.files_a[0].name == '0000.tif'
        assert mp.files_b[0].name == '0001.tif'
        
        assert mp.files_a[1].name == '0001.tif'
        assert mp.files_b[1].name == '0002.tif'
        
        assert mp.files_a[2].name == '0002.tif'
        assert mp.files_b[2].name == '0003.tif'
        
        assert mp.files_a[3].name == '0003.tif'
        assert mp.files_b[3].name == '0004.tif'


def test_multiprocesser_pattern_1_plus_3():
    """Test Multiprocesser with pattern_b='(1+3),(2+4)'"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create a sequence of test files
        for i in range(7):
            open(os.path.join(tmpdirname, f'{i:04d}.tif'), 'w').close()
        
        # Create a Multiprocesser instance with pattern_b='(1+3),(2+4)'
        mp = Multiprocesser(
            data_dir=pathlib.Path(tmpdirname),
            pattern_a='*.tif',
            pattern_b='(1+3),(2+4)'
        )
        
        # Check that files were paired correctly
        assert len(mp.files_a) == 5  # 0000, 0001, 0002, 0003, 0004
        assert len(mp.files_b) == 5  # 0002, 0003, 0004, 0005, 0006
        
        # Check specific pairs
        assert mp.files_a[0].name == '0000.tif'
        assert mp.files_b[0].name == '0002.tif'
        
        assert mp.files_a[1].name == '0001.tif'
        assert mp.files_b[1].name == '0003.tif'
        
        assert mp.files_a[2].name == '0002.tif'
        assert mp.files_b[2].name == '0004.tif'
        
        assert mp.files_a[3].name == '0003.tif'
        assert mp.files_b[3].name == '0005.tif'
        
        assert mp.files_a[4].name == '0004.tif'
        assert mp.files_b[4].name == '0006.tif'


def test_multiprocesser_pattern_1_plus_2_3_plus_4():
    """Test Multiprocesser with pattern_b='(1+2),(3+4)'"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create a sequence of test files
        for i in range(6):
            open(os.path.join(tmpdirname, f'{i:04d}.tif'), 'w').close()
        
        # Create a Multiprocesser instance with pattern_b='(1+2),(3+4)'
        mp = Multiprocesser(
            data_dir=pathlib.Path(tmpdirname),
            pattern_a='*.tif',
            pattern_b='(1+2),(3+4)'
        )
        
        # Check that files were paired correctly
        assert len(mp.files_a) == 3  # 0000, 0002, 0004
        assert len(mp.files_b) == 3  # 0001, 0003, 0005
        
        # Check specific pairs
        assert mp.files_a[0].name == '0000.tif'
        assert mp.files_b[0].name == '0001.tif'
        
        assert mp.files_a[1].name == '0002.tif'
        assert mp.files_b[1].name == '0003.tif'
        
        assert mp.files_a[2].name == '0004.tif'
        assert mp.files_b[2].name == '0005.tif'


def test_multiprocesser_run():
    """Test the run method of Multiprocesser"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create a few test files
        for i in range(3):
            open(os.path.join(tmpdirname, f'img_a_{i}.tif'), 'w').close()
            open(os.path.join(tmpdirname, f'img_b_{i}.tif'), 'w').close()
        
        # Create a Multiprocesser instance
        mp = Multiprocesser(
            data_dir=pathlib.Path(tmpdirname),
            pattern_a='img_a_*.tif',
            pattern_b='img_b_*.tif'
        )
        
        # Define a simple processing function
        results = []
        def process_func(args):
            file_a, file_b, counter = args
            results.append((file_a.name, file_b.name, counter))
        
        # Run the processing function
        mp.run(process_func, n_cpus=1)
        
        # Check that all files were processed
        assert len(results) == 3
        
        # Check that files were processed in the correct order
        assert results[0][0] == 'img_a_0.tif'
        assert results[0][1] == 'img_b_0.tif'
        assert results[0][2] == 0
        
        assert results[1][0] == 'img_a_1.tif'
        assert results[1][1] == 'img_b_1.tif'
        assert results[1][2] == 1
        
        assert results[2][0] == 'img_a_2.tif'
        assert results[2][1] == 'img_b_2.tif'
        assert results[2][2] == 2


def test_multiprocesser_error_handling():
    """Test error handling in Multiprocesser"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Test with no matching files
        with pytest.raises(ValueError, match="No images were found"):
            Multiprocesser(
                data_dir=pathlib.Path(tmpdirname),
                pattern_a='nonexistent_*.tif',
                pattern_b='nonexistent_*.tif'
            )
        
        # Test with unequal number of files
        open(os.path.join(tmpdirname, 'img_a_0.tif'), 'w').close()
        open(os.path.join(tmpdirname, 'img_a_1.tif'), 'w').close()
        open(os.path.join(tmpdirname, 'img_b_0.tif'), 'w').close()
        
        with pytest.raises(ValueError, match="equal number"):
            Multiprocesser(
                data_dir=pathlib.Path(tmpdirname),
                pattern_a='img_a_*.tif',
                pattern_b='img_b_*.tif'
            )
