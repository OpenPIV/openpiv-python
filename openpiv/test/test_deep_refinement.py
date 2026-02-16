import unittest
import numpy as np
from scipy import ndimage
from openpiv import deep_refinement

class TestDeepRefinement(unittest.TestCase):
    def setUp(self):
        """
        Generate synthetic particle images for testing.
        """
        # 1. Create a random speckle pattern (Image A)
        # We use random noise smoothed by a Gaussian to simulate particles
        np.random.seed(42)
        noise = np.random.rand(64, 64)
        self.frame_a = ndimage.gaussian_filter(noise, sigma=1.5)
        
        # 2. Define a known Ground Truth displacement
        # Let's say particles moved u=3.5px, v=-2.0px
        self.u_gt = 3.5
        self.v_gt = -2.0
        
        # 3. Create Image B by shifting Image A by the ground truth
        # (Simulates the particles moving)
        self.frame_b = ndimage.shift(self.frame_a, shift=(self.v_gt, self.u_gt), order=3)

        # 4. Create a "Coarse" PIV grid (Simulating a standard Cross-Correlation result)
        # We simulate a coarse grid of 3x3 vectors
        # Let's assume standard PIV found an integer approximation (u=3, v=-2)
        grid_h, grid_w = 3, 3
        self.u_piv = np.full((grid_h, grid_w), 3.0) 
        self.v_piv = np.full((grid_h, grid_w), -2.0)
        
        # Create coordinates for these vectors (center of windows)
        # Just simple linspace for testing upscaling
        y = np.linspace(16, 48, grid_h)
        x = np.linspace(16, 48, grid_w)
        self.x_piv, self.y_piv = np.meshgrid(x, y)

    def test_upscale_flow(self):
        """
        Test if the coarse grid is correctly interpolated to pixel resolution.
        """
        target_shape = (64, 64)
        u_dense, v_dense = deep_refinement.upscale_flow(
            self.u_piv, self.v_piv, self.x_piv, self.y_piv, target_shape
        )
        
        # The result should be dense (64x64)
        self.assertEqual(u_dense.shape, target_shape)
        
        # Since our coarse input was uniform (all 3.0), the output should be uniform
        np.testing.assert_allclose(u_dense, 3.0, rtol=1e-5)
        np.testing.assert_allclose(v_dense, -2.0, rtol=1e-5)

    def test_image_warping_integrity(self):
        """
        Crucial Test: Validate the warping logic defined by Choi et al.
        If we warp Frame B back by the Ground Truth flow, it should match Frame A.
        """
        # Create a dense flow field representing the Ground Truth
        h, w = self.frame_a.shape
        u_field = np.full((h, w), self.u_gt) # 3.5
        v_field = np.full((h, w), self.v_gt) # -2.0
        
        # Warp Frame B "backwards" using the flow
        # Expected: warped_b should look like frame_a
        warped_b = deep_refinement.warp_image(self.frame_b, u_field, v_field)
        
        # Crop borders to avoid shifting artifacts when comparing
        border = 5
        diff = np.abs(self.frame_a[border:-border, border:-border] - 
                      warped_b[border:-border, border:-border])
        
        # The difference should be very small (close to 0)
        # We allow small tolerance due to interpolation errors
        mae = np.mean(diff)
        self.assertLess(mae, 0.05, "Warping failed to align Frame B with Frame A")

    def test_pipeline_with_mock_model(self):
        """
        Test the full 'refine' pipeline.
        Since we don't have a trained CNN model file in the repo,
        we Mock the 'predict_residual' method.
        """
        
        # 1. Initialize the Refiner
        refiner = deep_refinement.DeepRefiner(model_path=None, device='cpu')
        
        # 2. Mock the CNN output
        # The Standard PIV found u=3.0. The Ground Truth is u=3.5.
        # The CNN *should* find the residual: 0.5.
        # We force our mock to return exactly that.
        def mock_predict(img1, img2_warped):
            h, w = img1.shape
            # Return uniform residual of 0.5 for u, 0.0 for v
            return np.full((h, w), 0.5), np.full((h, w), 0.0)
        
        # Inject the mock
        refiner.predict_residual = mock_predict
        
        # 3. Run the pipeline
        u_final, v_final = refiner.refine(
            self.frame_a, self.frame_b, 
            self.u_piv, self.v_piv, 
            self.x_piv, self.y_piv
        )
        
        # 4. Assertions
        # Coarse (3.0) + Residual (0.5) = 3.5 (Ground Truth)
        expected_u = 3.5
        
        # Check center pixels (avoid boundary interpolation issues)
        center_u = u_final[32, 32]
        
        self.assertAlmostEqual(center_u, expected_u, places=3)
        print(f"Pipeline Test: Input=3.0, Residual=0.5, Result={center_u} (Expected 3.5)")

if __name__ == '__main__':
    unittest.main()