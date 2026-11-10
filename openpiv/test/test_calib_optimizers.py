import numpy as np
import pytest
from numpy.testing import (
    assert_,
    assert_allclose,
    assert_array_equal,
    assert_equal,
)
from scipy.optimize import least_squares

from openpiv.calibration import pinhole_model
from openpiv.calibration._optimizers import optimize_board, optimize_general
from openpiv.calibration.pinhole_model._optimize import (
    _camera_columns,
    _camera_vector,
    _set_camera_vector,
)


@pytest.mark.parametrize("options", [
    {"max_iter": 0},
    {"max_iter": 1.5},
    {"tolerance": 0},
    {"tolerance": np.nan},
    {"tolerance": np.inf},
])
def test_invalid_iteration_controls(
    options: dict
):
    with pytest.raises(ValueError, match='max_iter|tolerance'):
        optimize_general(
            lambda parameters: parameters - 1,
            [0.0],
            lambda parameters: np.ones((1, 1)),
            **options
        )


@pytest.mark.parametrize("parameters", ([], [[0.0]], [np.nan]))
def test_invalid_initial_parameters(
    parameters: list
):
    with pytest.raises(ValueError, match='Initial parameters'):
        optimize_general(
            lambda parameters: parameters,
            parameters,
            lambda parameters: np.eye(parameters.size)
        )


@pytest.mark.parametrize("residual", ([], [[1.0]], [np.nan], [np.inf]))
def test_invalid_initial_residual(
    residual: list
):
    with pytest.raises(ValueError, match='Residuals|Initial calibration'):
        optimize_general(
            lambda parameters: residual,
            [0.0],
            lambda parameters: np.ones((1, 1))
        )


@pytest.mark.parametrize("jacobian", (np.ones((2, 1)), [[np.nan]]))
def test_invalid_initial_jacobian(
    jacobian: np.ndarray | list
):
    with pytest.raises(ValueError, match='Jacobian|Initial calibration'):
        optimize_general(
            lambda parameters: parameters - 1,
            [0.0],
            lambda parameters: jacobian
        )


def test_too_few_observations():
    with pytest.raises(ValueError, match='at least as many residuals'):
        optimize_general(
            lambda parameters: np.array([parameters.sum()]),
            [0.0, 0.0],
            lambda parameters: [[1, 1]]
        )


def test_general_fit_matches_scipy():
    rng = np.random.default_rng(17)
    design = rng.normal(size=(30, 3))
    coefficients = np.array([0.5, -2.0, 1.0])
    observed = np.dot(design, coefficients) + rng.normal(0, 0.01, 30)

    def residual(
        parameters: np.ndarray
    ):
        return np.dot(design, parameters) - observed

    fitted, result = optimize_general(
        residual,
        np.zeros(3),
        lambda parameters: design
    )
    reference = least_squares(
        residual,
        np.zeros(3),
        jac=lambda parameters: design,
        method='lm'
    )

    assert_(result['success'])
    assert_(not result['rank_deficient'])
    assert_equal(result['rank'], 3)
    assert_equal(result['expected_rank'], 3)

    assert_allclose(
        fitted,
        reference.x,
        atol=1e-8
    )


def test_dependent_parameters_report_rank():
    design = np.array(
        [[1, 1], [2, 2], [3, 3]],
        dtype=np.float64
    )
    observed = np.array(
        [2, 4, 6],
        dtype=np.float64
    )

    msg = 'rank deficient: rank 1, expected 2'

    with pytest.warns(RuntimeWarning, match=msg):
        fitted, result = optimize_general(
            lambda parameters: np.dot(design, parameters) - observed,
            np.zeros(2),
            lambda parameters: design
        )

    assert_(result['success'])
    assert_(result['rank_deficient'])

    assert_allclose(
        np.dot(design, fitted),
        observed,
        atol=1e-7
    )


def test_invalid_trial_jacobian_does_not_replace_accepted_state():
    initial = np.array([0.0])

    def jacobian(
        parameters: np.ndarray
    ):
        if parameters[0] == 0:
            return np.array([[1.0]])

        return np.array([[np.nan]])

    fitted, result = optimize_general(
        lambda parameters: parameters - 1,
        initial,
        jacobian,
        max_iter=1
    )

    assert_(not result['success'])
    assert_(not result['history'][0]['accepted'])
    assert_equal(result['cost'], 0.5)

    assert_array_equal(
        fitted,
        initial
    )


def test_mismatched_board_blocks():
    def residual(
        parameters: np.ndarray,
        poses: np.ndarray
    ):
        return [np.zeros(10), np.zeros(10)]

    def jacobian(
        parameters: np.ndarray,
        poses: np.ndarray
    ):
        return [np.ones((10, 1))], [np.ones((10, 6))]

    with pytest.raises(ValueError, match='Jacobian block counts'):
        optimize_board(
            residual,
            np.zeros(1),
            np.zeros((2, 6)),
            jacobian
        )


@pytest.mark.parametrize("deficiency", (None, 'camera', 'board'))
def test_board_rank_matches_complete_jacobian(
    deficiency: str | None
):
    rng = np.random.default_rng(23)
    cameras = [rng.normal(size=(12, 2)) for _ in range(2)]
    boards = [rng.normal(size=(12, 6)) for _ in range(2)]

    if deficiency == 'camera':
        for camera, board in zip(cameras, boards):
            camera[:, 0] = board[:, 0]
    elif deficiency == 'board':
        boards[0][:, 1] = boards[0][:, 0]

    complete = np.zeros((24, 14))

    for i in range(2):
        complete[12*i:12*(i+1), :2] = cameras[i]
        complete[12*i:12*(i+1), 2+6*i:2+6*(i+1)] = boards[i]

    def residual(
        parameters: np.ndarray,
        poses: np.ndarray
    ):
        residuals = []

        for camera, board, pose in zip(cameras, boards, poses):
            residuals.append(
                np.dot(camera, parameters) + np.dot(board, pose)
            )

        return residuals

    def fit():
        return optimize_board(
            residual,
            np.zeros(2),
            np.zeros((2, 6)),
            lambda parameters, poses: (cameras, boards)
        )

    if deficiency is None:
        _, _, result = fit()
    else:
        with pytest.warns(RuntimeWarning, match='rank deficient'):
            _, _, result = fit()

    assert_equal(
        result['rank'],
        np.linalg.matrix_rank(complete)
    )
    assert_equal(
        result['expected_rank'],
        complete.shape[1]
    )
    assert_equal(
        result['rank_deficient'],
        deficiency is not None
    )


def test_full_brown_initial_rank_deficiency_is_not_rejected():
    cam = pinhole_model.PinCam(
        'full',
        (1280, 960),
        translation=[0, 0, 0],
        focal=[800, 820],
        distortion_model='full'
    )
    rng = np.random.default_rng(29)
    points = rng.uniform(
        [-0.2, -0.2, 0.8],
        [0.2, 0.2, 1.2],
        size=(100, 3)
    ).T
    observed = cam.project_points(points)
    values = _camera_vector(cam)
    columns = _camera_columns(cam)

    def set_parameters(
        parameters: np.ndarray
    ):
        values[columns] = parameters
        _set_camera_vector(cam, values)

    def residual(
        parameters: np.ndarray
    ):
        set_parameters(parameters)

        return (cam.project_points(points) - observed).T.ravel()

    def jacobian(
        parameters: np.ndarray
    ):
        set_parameters(parameters)
        _, camera, _ = cam.project_jacobian(points)

        return camera[:, :, columns].reshape(-1, len(columns))

    msg = 'rank deficient: rank 15, expected 16'

    with pytest.warns(RuntimeWarning, match=msg):
        _, result = optimize_general(
            residual,
            values[columns],
            jacobian
        )

    assert_(result['success'])
    assert_(result['rank_deficient'])
