from openpiv import filters
from openpiv.lib import replace_nans
import numpy as np


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

    # assert(np.isnan(filters._gaussian_kernel(0))) # issues a Warning


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
    grid_mask = np.zeros_like(v, dtype=bool)
    u = v.copy()
    uf, _ = filters.replace_outliers(u,v, invalid_mask)

    assert np.ma.allclose(v_copy,uf)
    assert isinstance(uf, np.ma.MaskedArray)
