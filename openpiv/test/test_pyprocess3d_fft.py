import numpy as np
import pytest
from scipy.signal import correlate

from openpiv.pyprocess3D import correlate_windows, extended_search_area_piv3D


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16])
@pytest.mark.parametrize("extra_depth", [0, 1, 2])
@pytest.mark.parametrize("first_shape,second_shape", [
    ((3, 4, 5), (3, 4, 5)),
    ((4, 3, 4), (5, 4, 6)),
])
def test_correlation_preserves_fft_depth(dtype, extra_depth, first_shape,
                                         second_shape):
    rng = np.random.default_rng(0)
    first = rng.integers(0, 8, size=first_shape).astype(dtype)
    second = rng.integers(0, 8, size=second_shape).astype(dtype)
    original_first, original_second = first.copy(), second.copy()
    full_shape = np.array(first_shape) + second_shape - 1
    fft_depth = int(full_shape[2] + extra_depth)

    actual = correlate_windows(first, second, nffty=int(full_shape[1] + 2),
                               nfftz=fft_depth)

    reference = correlate(first - first.mean(), second - second.mean(),
                          mode="full", method="direct")
    expected_shape = (full_shape[0] + 1, full_shape[1] + 1,
                      min(full_shape[2] + 1, fft_depth))
    expected = np.zeros(expected_shape)
    expected[tuple(slice(0, n) for n in full_shape)] = reference
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(first, original_first)
    np.testing.assert_array_equal(second, original_second)


@pytest.mark.parametrize("fft_depth", [15, 17, 19])
@pytest.mark.parametrize("dt", [(1., 1., 1.), (.5, .25, .75)])
def test_stationary_volume_with_odd_fft_depth(fft_depth, dt):
    frame = np.random.default_rng(0).integers(0, 256, size=(16, 16, 16))
    original = frame.copy()

    velocity = extended_search_area_piv3D(
        frame, frame, window_size=(8, 8, 8), search_area_size=(8, 8, 8),
        nfftz=fft_depth, dt=dt,
    )

    np.testing.assert_allclose(velocity, 0., atol=1e-10)
    np.testing.assert_array_equal(frame, original)
