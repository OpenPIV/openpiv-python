""" Testing validation functions """
from typing import Tuple
import numpy as np
from importlib_resources import files
import matplotlib.pyplot as plt

from openpiv.pyprocess import extended_search_area_piv as piv
from openpiv.tools import imread
from openpiv import validation
from openpiv.settings import PIVSettings


file_a = files('openpiv.data').joinpath('test1/exp1_001_a.bmp')
file_b = files('openpiv.data').joinpath('test1/exp1_001_b.bmp')

frame_a = imread(file_a)
frame_b = imread(file_b)

frame_a = frame_a[:32, :32]
frame_b = frame_b[:32, :32]


def test_validation_peak2mean():
    """test of the simplest PIV run
    default window_size = 32
    """
    _, _, s2n = piv(frame_a, frame_b,
                    window_size=32,
                    sig2noise_method="peak2mean")

    assert np.allclose(s2n.min(), 1.443882)


def test_validation_peak2peak():
    """test of the simplest PIV run
    default window_size = 32
    """
    _, _, s2n = piv(frame_a, frame_b,
                    window_size=32,
                    sig2noise_method="peak2peak")
    assert np.allclose(np.min(s2n), 1.24009)


def test_sig2noise_val():
    """ tests sig2noise validation """
    s2n_threshold = 1.05
    s2n = np.ones((5, 5))*s2n_threshold
    s2n[2, 2] -= 0.1

    mask = validation.sig2noise_val(s2n, threshold=s2n_threshold)

    assert not mask[0, 0]  # should be False
    assert mask[2, 2]


def test_sig2noise_val_3d():
    """ tests sig2noise validation with 3D data """
    s2n_threshold = 1.05
    s2n = np.ones((3, 5, 5))*s2n_threshold
    s2n[1, 2, 2] -= 0.1
    s2n[2, 3, 3] -= 0.2

    mask = validation.sig2noise_val(s2n, threshold=s2n_threshold)

    # Check shape
    assert mask.shape == (3, 5, 5)

    # Check values
    assert not mask[0, 0, 0]  # should be False
    assert mask[1, 2, 2]      # should be True
    assert mask[2, 3, 3]      # should be True


def test_sig2noise_val_edge_cases():
    """ tests sig2noise validation with edge cases """
    # Test with all values below threshold
    s2n_threshold = 2.0
    s2n = np.ones((3, 3))*1.0  # All below threshold

    mask = validation.sig2noise_val(s2n, threshold=s2n_threshold)

    assert np.all(mask)  # All should be flagged

    # Test with all values above threshold
    s2n = np.ones((3, 3))*3.0  # All above threshold

    mask = validation.sig2noise_val(s2n, threshold=s2n_threshold)

    assert not np.any(mask)  # None should be flagged

    # Test with NaN values
    s2n = np.ones((3, 3))*3.0
    s2n[1, 1] = np.nan

    mask = validation.sig2noise_val(s2n, threshold=s2n_threshold)

    assert not mask[0, 0]  # Regular value should not be flagged
    # The current implementation treats NaN as False in the comparison
    # This is the default behavior of NumPy's comparison operators with NaN
    assert not mask[1, 1]  # NaN < threshold is False in NumPy


def test_local_median_validation(u_threshold=3, N=3):
    """ test local median

    Args:
        u_threshold (int, optional): _description_. Defaults to 3.
        N (int, optional): _description_. Defaults to 3.
        size (int, optional): _description_. Defaults to 1.
    """

    u = np.random.rand(2*N+1, 2*N+1)
    u[N, N] = np.median(u)*10

    # and masked array copy
    u = np.ma.copy(u)
    u[N+1:, N+1:-1] = np.ma.masked

    # now we test our function which is just a decoration
    # of the above steps
    flag = validation.local_median_val(u, u, u_threshold, u_threshold)

    assert flag[N, N]


def test_local_norm_median_validation():
    """ test normalized local median

    Args:
        threshold (int, optional): _description_.
        N (int, optional): _description_.
    """

    # The idea is the following. Let's create an array of the size of one filtering kernel.
    # Calculate normalized median filter by hand and using local_norm_median_val(). See if
    # the results are comparable. As outlined in the description of local_norm_median_val(),
    # follow the paper J. Westerweel, F. Scarano, "Universal outlier detection for PIV data",
    # Experiments in fluids, 39(6), p.1096-1100, 2005, equation 2, to perform calculations
    # by hand.

    # Case study:
    u = np.asarray([[1,2,3],[4,5,6],[7,8,9]])
    v = np.asarray([[2,3,4],[5,6,7],[8,9,10]])
    ε = 0.1
    thresh = 2

    # Calculations by hand (according to equation 2 in the referenced article).
    u0 = 5
    v0 = 6
    # Median formula for even number of observations (the center is excluded)
    # that are placed in the ascending order (just like the given u and v arrays):
    # Median = [(n/2)th term + ((n/2) + 1)th term]/2 (https://www.cuemath.com/data/median/).
    um = 5 # = (4 + 6) / 2
    vm = 6 # = (5 + 7) / 2
    # Arrays of residuals given by the formula rui = |ui-um| and rvi = |vi-vm|
    rui = np.asarray([[4,3,2],[1,0,1],[2,3,4]])
    rvi = np.asarray([[4,3,2],[1,0,1],[2,3,4]])
    # Median residuals values (center excluded) must be calculated for ascending arrays:
    ruiascend = [1, 1, 2, 2, 3, 3, 4, 4]
    rviascend = [1, 1, 2, 2, 3, 3, 4, 4]
    rum = 2.5 # = (2 + 3) / 2
    rvm = 2.5 # = (2 + 3) / 2
    # Now implement equation 2 from the referenced article:
    r0ast_u = np.abs(u0 - um) / (rum + ε)
    r0ast_v = np.abs(v0 - vm) / (rvm + ε)
    # The method of comparison to the threshold is given at the very end of the referenced
    # article in the MATLAB code:
    byHand = (r0ast_u**2 + r0ast_v**2)**0.5 > thresh
    # Here, we calculated byHand for the velocity vector (u0,v0) coordinates of which are
    # located at u[1,1] and v[1,1].

    # Calculations using local_norm_median_val() function.
    byFunc = validation.local_norm_median_val(u, v, ε, thresh)
    # Here by func returns a boolean matrix containing either True or False for each velocity
    # vector (ui,vi).

    # Compare the two results:
    assert byHand == byFunc[1,1]



def test_global_val():
    """ tests global validation
    """
    N: int = 2
    U: Tuple[int, int] = (-10, 10)

    u = np.random.rand(2*N+1, 2*N+1)
    u[N, N] = U[0]-.2
    u[0, 0] = U[1]+.2

    u = np.ma.copy(u)
    v = np.ma.copy(u)
    v[N+1, N+1] = np.ma.masked

    mask = validation.global_val(u, u, U, U)

    assert mask[N, N]
    assert mask[0, 0]

    # masked array test

    mask1 = validation.global_val(v, v, U, U)

    assert mask1[N, N]
    assert mask1[0, 0]


def test_global_std():
    """tests global std

    Args:
        N (int, optional): array size. Defaults to 2.
        std_threshold (int, optional): threshold . Defaults to 3.
    """
    N = 2

    u = np.random.randn(2*N+1, 2*N+1)

    # print(np.nanmean(u))
    # print(np.nanstd(u))

    u[N, N] = 10.
    u[0, 0] = -10.

    v = np.ma.copy(u)
    v[N+1, N+1] = np.ma.masked

    mask = validation.global_std(u, u, 3)

    assert mask[N, N]
    assert mask[0, 0]

    mask1 = validation.global_std(v, v, std_threshold=3)

    assert mask1[N, N]
    assert mask1[0, 0]


def test_uniform_shift_std(N: int = 2):
    """ test for uniform shift """
    u = np.ones((2*N+1, 2*N+1))
    v = np.ma.copy(u)
    v[N+1, N+1] = np.ma.masked

    mask = validation.global_std(u, u, std_threshold=3)

    assert ~mask[N, N]

    # print(f'v.data \n {v.data}')
    # print(f'v before \n {v}')

    mask1 = validation.global_std(v, v, std_threshold=3)

    v[mask1] = np.ma.masked

    assert v.data[N, N] == 1.0
    assert v.mask[N+1, N+1]


def test_typical_validation_basic():
    """Test the typical_validation function with basic settings."""
    # Create test data
    u = np.random.rand(10, 10)
    v = np.random.rand(10, 10)
    s2n = np.ones((10, 10)) * 2.0

    # Create outliers
    u[5, 5] = 100.0  # Global outlier
    v[7, 7] = -100.0  # Global outlier
    u[2, 2] = 10.0   # Local outlier
    s2n[3, 3] = 0.5  # Signal-to-noise outlier

    # Create settings
    settings = PIVSettings()
    settings.min_max_u_disp = (-5, 5)
    settings.min_max_v_disp = (-5, 5)
    settings.std_threshold = 3
    settings.median_threshold = 2
    settings.median_size = 1
    settings.sig2noise_validate = True
    settings.sig2noise_threshold = 1.0
    settings.show_all_plots = False

    # Run validation
    mask = validation.typical_validation(u, v, s2n, settings)

    # Check that outliers are detected
    assert mask[5, 5]  # Global outlier
    assert mask[7, 7]  # Global outlier
    assert mask[2, 2]  # Local outlier
    assert mask[3, 3]  # Signal-to-noise outlier

    # Check that normal values are not flagged
    assert not mask[0, 0]


def test_typical_validation_normalized_median():
    """Test the typical_validation function with normalized median."""
    # Create test data
    u = np.random.rand(10, 10)
    v = np.random.rand(10, 10)
    s2n = np.ones((10, 10)) * 2.0

    # Create outliers
    u[5, 5] = 100.0  # Global outlier

    # Create settings
    settings = PIVSettings()
    settings.min_max_u_disp = (-5, 5)
    settings.min_max_v_disp = (-5, 5)
    settings.std_threshold = 3
    settings.median_threshold = 2
    settings.median_size = 1
    settings.median_normalized = True
    settings.sig2noise_validate = False
    settings.show_all_plots = False

    # Run validation
    mask = validation.typical_validation(u, v, s2n, settings)

    # Check that outliers are detected
    assert mask[5, 5]  # Global outlier

    # Check that normal values are not flagged
    assert not mask[0, 0]


def test_typical_validation_no_s2n():
    """Test the typical_validation function without s2n validation."""
    # Create test data
    u = np.random.rand(10, 10)
    v = np.random.rand(10, 10)
    s2n = np.ones((10, 10)) * 2.0

    # Create outliers
    u[5, 5] = 100.0  # Global outlier
    s2n[3, 3] = 0.5  # Signal-to-noise outlier that should be ignored

    # Create settings
    settings = PIVSettings()
    settings.min_max_u_disp = (-5, 5)
    settings.min_max_v_disp = (-5, 5)
    settings.std_threshold = 3
    settings.median_threshold = 2
    settings.median_size = 1
    settings.sig2noise_validate = False
    settings.show_all_plots = False

    # Run validation
    mask = validation.typical_validation(u, v, s2n, settings)

    # Check that outliers are detected
    assert mask[5, 5]  # Global outlier

    # Check that s2n outlier is not flagged
    assert not mask[3, 3]  # Signal-to-noise outlier should be ignored


def test_typical_validation_with_plots():
    """Test the typical_validation function with plots enabled."""
    # Create test data
    u = np.random.rand(10, 10)
    v = np.random.rand(10, 10)
    s2n = np.ones((10, 10)) * 2.0

    # Create outliers
    u[5, 5] = 100.0  # Global outlier
    s2n[3, 3] = 0.5  # Signal-to-noise outlier

    # Create settings
    settings = PIVSettings()
    settings.min_max_u_disp = (-5, 5)
    settings.min_max_v_disp = (-5, 5)
    settings.std_threshold = 3
    settings.median_threshold = 2
    settings.median_size = 1
    settings.sig2noise_validate = True
    settings.sig2noise_threshold = 1.0
    settings.show_all_plots = True

    # Temporarily disable plt.show to avoid displaying plots during tests
    original_show = plt.show
    plt.show = lambda: None

    try:
        # Run validation
        mask = validation.typical_validation(u, v, s2n, settings)

        # Check that outliers are detected
        assert mask[5, 5]  # Global outlier
        assert mask[3, 3]  # Signal-to-noise outlier
    finally:
        # Restore plt.show
        plt.show = original_show
