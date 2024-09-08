import unittest
import numpy as np
from synimagegen import generate_particle_image

class TestGenerateParticleImage(unittest.TestCase):

    def setUp(self):
        # Common setup for the tests
        self.height = 100
        self.width = 100
        self.bit_depth = 8
        self.x = np.array([50, 20, 80])
        self.y = np.array([50, 20, 80])
        self.particle_diameters = np.array([5, 10, 15])
        self.particle_max_intensities = np.array([255, 128, 64])

    def test_image_dimensions(self):
        """Test if the generated image has the correct dimensions."""
        image = generate_particle_image(
            self.height, self.width, self.x, self.y, 
            self.particle_diameters, self.particle_max_intensities, self.bit_depth
        )
        self.assertEqual(image.shape, (self.height, self.width))

    def test_image_type(self):
        """Test if the generated image is of type numpy array."""
        image = generate_particle_image(
            self.height, self.width, self.x, self.y, 
            self.particle_diameters, self.particle_max_intensities, self.bit_depth
        )
        self.assertIsInstance(image, np.ndarray)

    def test_image_non_negative(self):
        """Test if the generated image has non-negative pixel values."""
        image = generate_particle_image(
            self.height, self.width, self.x, self.y, 
            self.particle_diameters, self.particle_max_intensities, self.bit_depth
        )
        self.assertTrue(np.all(image >= 0))

    def test_image_max_value(self):
        """Test if the generated image has pixel values within the bit depth range."""
        image = generate_particle_image(
            self.height, self.width, self.x, self.y, 
            self.particle_diameters, self.particle_max_intensities, self.bit_depth
        )
        max_value = 2**self.bit_depth - 1
        self.assertTrue(np.all(image <= max_value))

    def test_empty_particles(self):
        """Test if the function handles empty particle arrays correctly."""
        x_empty = np.array([])
        y_empty = np.array([])
        particle_diameters_empty = np.array([])
        particle_max_intensities_empty = np.array([])
        image = generate_particle_image(
            self.height, self.width, x_empty, y_empty, 
            particle_diameters_empty, particle_max_intensities_empty, self.bit_depth
        )
        self.assertEqual(image.shape, (self.height, self.width))
        self.assertTrue(np.all(image >= 0))
        self.assertTrue(np.all(image <= 2**self.bit_depth - 1))

    def test_single_particle(self):
        """Test if the function handles a single particle correctly."""
        x_single = np.array([50])
        y_single = np.array([50])
        particle_diameters_single = np.array([10])
        particle_max_intensities_single = np.array([255])
        image = generate_particle_image(
            self.height, self.width, x_single, y_single, 
            particle_diameters_single, particle_max_intensities_single, self.bit_depth
        )
        self.assertEqual(image.shape, (self.height, self.width))
        self.assertTrue(np.all(image >= 0))
        self.assertTrue(np.all(image <= 2**self.bit_depth - 1))

if __name__ == '__main__':
    unittest.main()