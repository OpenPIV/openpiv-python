"""Tests for synimage module integration in openpiv.tools."""

import pytest
import numpy as np
from openpiv.tools import synimage


class TestSynimageIntegration:
    """Test synimage module integration with openpiv.tools."""

    def test_synimage_import(self):
        """Test that synimage can be imported from openpiv.tools."""
        assert hasattr(synimage, 'synimagegen')
        assert hasattr(synimage, 'generate_particle_image')
        assert hasattr(synimage, 'create_synimage_parameters')
        assert hasattr(synimage, 'continuous_flow_field')

    def test_synimagegen_basic(self):
        """Test synimagegen with default parameters."""
        image_a, image_b = synimage.synimagegen(128)
        
        assert isinstance(image_a, np.ndarray)
        assert isinstance(image_b, np.ndarray)
        assert image_a.shape == (128, 128)
        assert image_b.shape == (128, 128)

    def test_synimagegen_different_sizes(self):
        """Test synimagegen with different image sizes."""
        sizes = [64, 128, 256]
        
        for size in sizes:
            image_a, image_b = synimage.synimagegen(size)
            assert image_a.shape == (size, size)
            assert image_b.shape == (size, size)

    def test_synimagegen_output_range(self):
        """Test that generated images have valid pixel values."""
        image_a, image_b = synimage.synimagegen(128, bit_depth=8)
        
        # Check that values are non-negative
        assert np.all(image_a >= 0)
        assert np.all(image_b >= 0)
        
        # Check that values are within reasonable range for 8-bit
        # (may exceed 255 slightly due to noise and processing)
        assert np.max(image_a) < 2**10  # reasonable upper bound
        assert np.max(image_b) < 2**10

    def test_synimagegen_custom_parameters(self):
        """Test synimagegen with custom parameters."""
        image_a, image_b = synimage.synimagegen(
            image_size=100,
            dt=0.2,
            x_bound=(0, 1),
            y_bound=(0, 1),
            den=0.01,
            bit_depth=8
        )
        
        assert image_a.shape == (100, 100)
        assert image_b.shape == (100, 100)

    def test_generate_particle_image_basic(self):
        """Test generate_particle_image function directly."""
        height, width = 64, 64
        x = np.array([32, 16, 48])
        y = np.array([32, 16, 48])
        particle_diameters = np.array([5, 8, 6])
        particle_max_intensities = np.array([1.0, 0.8, 0.9])
        bit_depth = 8
        
        image = synimage.generate_particle_image(
            height, width, x, y, 
            particle_diameters, particle_max_intensities, bit_depth
        )
        
        assert image.shape == (height, width)
        assert isinstance(image, np.ndarray)

    def test_continuous_flow_field(self):
        """Test continuous_flow_field class."""
        cff = synimage.continuous_flow_field(None, inter=False)
        
        # Test that we can get velocity values
        u, v = cff.get_U_V(0.5, 0.5)
        
        assert isinstance(u, (int, float, np.number))
        assert isinstance(v, (int, float, np.number))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
