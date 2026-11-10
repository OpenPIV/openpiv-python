from collections.abc import Callable
from warnings import warn

import numpy as np


__all__ = [
    'optimize_general',
    'optimize_board'
]


def _check_residual(
    residual: np.ndarray,
    expected_size: int | None=None
):
    residual = np.asarray(
        residual,
        dtype=np.float64
    )

    if residual.ndim != 1 or residual.size == 0:
        msg = 'Residuals must be a nonempty 1D array'
        raise ValueError(msg)

    if expected_size is not None and residual.size != expected_size:
        msg = 'The number of residuals must remain constant'
        raise ValueError(msg)

    if not np.all(np.isfinite(residual)):
        msg = 'Calibration residuals must be finite'
        raise FloatingPointError(msg)

    return residual


def _check_jacobian(
    jacobian: np.ndarray,
    expected_shape: tuple[int, int]
):
    jacobian = np.asarray(
        jacobian,
        dtype=np.float64
    )

    if jacobian.shape != expected_shape:
        msg = (
            f'Jacobian has shape {jacobian.shape}; '
            f'expected {expected_shape}'
        )
        raise ValueError(msg)

    if not np.all(np.isfinite(jacobian)):
        msg = 'Calibration Jacobian must be finite'
        raise FloatingPointError(msg)

    return jacobian


def _linearization_rank(
    blocks: tuple,
    mode: str
):
    if mode == 'points':
        jacobian = blocks[2]
        jacobian = jacobian * _column_scales(jacobian)
        rank = np.linalg.matrix_rank(jacobian)
        expected_rank = jacobian.shape[1]

        return int(rank), expected_rank

    _, _, camera_jacobians, board_jacobians = blocks

    camera_scale, board_scales = _target_scales(
        camera_jacobians,
        board_jacobians
    )

    expected_rank = len(camera_scale)
    n_residuals = 0
    squared_norm = 0.0
    scaled_blocks = []

    for camera, board, scale in zip(
        camera_jacobians, board_jacobians, board_scales
    ):
        camera = camera * camera_scale
        board = board * scale
        scaled_blocks.append((camera, board))
        expected_rank += board.shape[1]
        n_residuals += board.shape[0]
        squared_norm += np.sum(camera**2) + np.sum(board**2)

    # Use the same tolerance for the camera and board blocks.
    tolerance = (
        np.finfo(np.float64).eps
        * max(n_residuals, expected_rank)
        * np.sqrt(squared_norm)
    )
    board_rank = 0
    reduced_camera = []

    # Remove the board columns before checking the shared camera columns
    for camera, board in scaled_blocks:
        left, singular_values, _ = np.linalg.svd(
            board,
            full_matrices=False
        )

        rank = int(np.count_nonzero(singular_values > tolerance))
        board_rank += rank
        basis = left[:, :rank]
        projection = np.dot(basis, np.dot(basis.T, camera))
        reduced_camera.append(camera - projection)

    camera_rank = 0

    if len(camera_scale):
        camera_rank = np.linalg.matrix_rank(
            np.vstack(reduced_camera),
            tol=tolerance
        )

    return int(board_rank + camera_rank), expected_rank


def _column_scales(
    jacobian: np.ndarray
):
    column_norm = np.linalg.norm(jacobian, axis=0)

    return 1 / np.maximum(column_norm, 1e-6)


def _target_scales(
    camera_jacobians: list,
    board_jacobians: list
):
    camera_norm = np.zeros(camera_jacobians[0].shape[1])

    for jacobian in camera_jacobians:
        camera_norm = np.hypot(
            camera_norm,
            np.linalg.norm(jacobian, axis=0)
        )

    camera_scale = 1 / np.maximum(camera_norm, 1e-6)
    board_scale = [_column_scales(jacobian) for jacobian in board_jacobians]

    return camera_scale, board_scale


def _qr_solve(
    jacobian: np.ndarray,
    rhs: np.ndarray
):
    n_parameters = jacobian.shape[1]
    augmented = np.column_stack([jacobian, rhs])
    factor = np.linalg.qr(augmented, mode='r')

    return np.linalg.solve(
        factor[:n_parameters, :n_parameters],
        factor[:n_parameters, n_parameters]
    )


def _point_step(
    residual: np.ndarray,
    jacobian: np.ndarray,
    damping: float
):
    scale = _column_scales(jacobian)
    n_parameters = jacobian.shape[1]

    augmented = np.vstack([
        jacobian * scale,
        np.sqrt(damping) * np.eye(n_parameters)
    ])
    rhs = np.concatenate([-residual, np.zeros(n_parameters)])

    step = scale * _qr_solve(augmented, rhs)
    change = np.dot(jacobian, step)
    predicted = -np.dot(residual, change) - 0.5 * np.dot(change, change)

    return step, predicted


def _target_step(
    residuals: list,
    camera_jacobians: list,
    board_jacobians: list,
    damping: float
):
    sc, sb = _target_scales(camera_jacobians, board_jacobians)
    n_camera = len(sc)
    reduced_camera = []
    reduced_rhs = []
    recovery = []
    root_damping = np.sqrt(damping)

    for residual, camera, board, scale in zip(
        residuals, camera_jacobians, board_jacobians, sb
    ):
        n_rows = len(residual)
        n_board = board.shape[1]
        block = np.zeros((n_rows+n_board, n_board+n_camera+1))

        block[:n_rows, :n_board] = board * scale
        block[:n_rows, n_board:n_board+n_camera] = camera * sc
        block[:n_rows, -1] = residual
        block[n_rows:, :n_board] = root_damping * np.eye(n_board)

        factor = np.linalg.qr(block, mode='r')
        recovery.append(factor[:n_board])
        reduced_camera.append(factor[n_board:, n_board:n_board+n_camera])
        reduced_rhs.append(-factor[n_board:, -1])

    reduced_camera.append(root_damping * np.eye(n_camera))
    reduced_rhs.append(np.zeros(n_camera))
    camera_step = _qr_solve(
        np.vstack(reduced_camera),
        np.concatenate(reduced_rhs)
    )

    board_steps = []

    for factor, scale in zip(recovery, sb):
        n_board = len(scale)
        rhs = -factor[:, -1] - np.dot(
            factor[:, n_board:n_board+n_camera],
            camera_step
        )
        board_step = np.linalg.solve(factor[:, :n_board], rhs)
        board_steps.append(scale * board_step)

    dc = sc * camera_step
    db = np.array(board_steps)
    predicted = 0.0

    for residual, camera, board, step in zip(
        residuals, camera_jacobians, board_jacobians, db
    ):
        change = np.dot(camera, dc) + np.dot(board, step)
        predicted -= np.dot(residual, change) + 0.5 * np.dot(change, change)

    return dc, db, predicted


def _levenberg_marquardt(
    state,
    linearize: Callable,
    update: Callable,
    evaluate: Callable,
    max_iter: int=100,
    tolerance: float=1e-10,
    mode: str='target'
):
    if mode not in ('target', 'points'):
        msg = 'LM mode must be "target" or "points"'
        raise ValueError(msg)

    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        msg = 'max_iter must be a positive integer'
        raise ValueError(msg)

    if not np.isfinite(tolerance) or tolerance <= 0:
        msg = 'tolerance must be finite and positive'
        raise ValueError(msg)

    damping = 1e-3
    multiplier = 2.0

    try:
        blocks = linearize(state)
    except FloatingPointError as error:
        msg = 'Initial calibration residuals and Jacobian must be finite'
        raise ValueError(msg) from error

    if not np.isfinite(blocks[0]):
        msg = 'Initial calibration cost must be finite'
        raise ValueError(msg)

    initial_cost = blocks[0]
    history = []
    success = False
    message = 'Maximum iterations reached'

    for _ in range(max_iter):
        if mode == 'target':
            cost, residuals, camera_jacobians, board_jacobians = blocks
            sc, sb = _target_scales(camera_jacobians, board_jacobians)
            camera_gradient = np.zeros(len(sc))
            board_gradient = []

            for residual, camera, board, scale in zip(
                residuals, camera_jacobians, board_jacobians, sb
            ):
                camera_gradient += sc * np.dot(camera.T, residual)
                board_gradient.append(scale * np.dot(board.T, residual))

            gradient = np.concatenate([camera_gradient, *board_gradient])
        else:
            cost, residual, jacobian = blocks
            scale = _column_scales(jacobian)
            gradient = scale * np.dot(jacobian.T, residual)

        if np.max(np.abs(gradient)) < tolerance:
            success = True
            message = 'Gradient tolerance reached'
            break

        try:
            if mode == 'target':
                dc, db, predicted = _target_step(
                    residuals, camera_jacobians, board_jacobians, damping
                )
                if (
                    not np.all(np.isfinite(dc))
                    or not np.all(np.isfinite(db))
                ):
                    msg = 'Non-finite calibration step'
                    raise FloatingPointError(msg)

                trial = update(state, dc, db)
            else:
                step, predicted = _point_step(residual, jacobian, damping)

                if not np.all(np.isfinite(step)):
                    msg = 'Non-finite calibration step'
                    raise FloatingPointError(msg)

                trial = update(state, step)

            trial_cost = evaluate(trial)

        except (np.linalg.LinAlgError, FloatingPointError):
            predicted = 0.0
            trial_cost = np.inf

        actual = cost - trial_cost

        if np.isfinite(predicted) and predicted > 0 and np.isfinite(trial_cost):
            ratio = actual / predicted
        else:
            ratio = -np.inf

        accepted = np.isfinite(ratio) and ratio > 0

        if accepted:
            try:
                trial_blocks = linearize(trial)

                if not np.isfinite(trial_blocks[0]):
                    msg = 'Non-finite calibration cost'
                    raise FloatingPointError(msg)

            except (np.linalg.LinAlgError, FloatingPointError):
                accepted = False

        history.append({
            'cost': cost,
            'trial_cost': trial_cost,
            'damping': damping,
            'ratio': ratio,
            'accepted': accepted
        })

        if accepted:
            state = trial
            blocks = trial_blocks
            damping *= max(1 / 3, 1 - (2 * ratio - 1)**3)
            damping = max(damping, 1e-15)
            multiplier = 2.0

            if actual < tolerance * max(1.0, cost):
                success = True
                message = 'Cost tolerance reached'
                break
        else:
            damping *= multiplier
            multiplier *= 2

            if damping > 1e16:
                message = 'Unable to find a decreasing step'
                break

    rank, expected_rank = _linearization_rank(blocks, mode)

    if rank < expected_rank:
        msg = (
            f'Calibration Jacobian is rank deficient: rank {rank}, '
            f'expected {expected_rank}'
        )
        warn(msg, RuntimeWarning, stacklevel=2)

    return state, {
        'success': success,
        'message': message,
        'iterations': len(history),
        'initial_cost': initial_cost,
        'cost': blocks[0],
        'history': history,
        'rank': rank,
        'expected_rank': expected_rank,
        'rank_deficient': rank < expected_rank
    }


def optimize_general(
    fun: Callable,
    x0: np.ndarray,
    jac: Callable,
    max_iter: int=100,
    tolerance: float=1e-10
):
    """Fit a parameter vector using QR Levenberg-Marquardt.

    Parameters
    ----------
    fun : callable
        Return the residual vector for the supplied parameters.
    x0 : np.ndarray
        Initial parameter vector with shape (M,).
    jac : callable
        Return the residual Jacobian with shape (N, M) for the
        supplied parameters.
    max_iter : int, optional
        Maximum attempted LM steps. Default is 100.
    tolerance : float, optional
        Scaled gradient and cost reduction tolerance. Default is 1e-10.

    Returns
    -------
    parameters : np.ndarray
        Final accepted parameter vector.
    result : dict
        Convergence status, costs, trial history, and final undamped
        Jacobian rank. Rank deficiency emits a RuntimeWarning and is
        reported separately from convergence.

    Raises
    ------
    ValueError
        If iteration controls, initial parameters, residuals or Jacobian
        shapes are invalid, or there are fewer residuals than parameters.

    """
    initial = np.array(
        x0,
        dtype=np.float64
    )

    if initial.ndim != 1 or initial.size == 0:
        msg = 'Initial parameters must be a nonempty 1D array'
        raise ValueError(msg)

    if not np.all(np.isfinite(initial)):
        msg = 'Initial parameters must be finite'
        raise ValueError(msg)

    n_residuals = None

    def linearize(parameters):
        nonlocal n_residuals
        residual = _check_residual(fun(parameters), n_residuals)

        if residual.size < initial.size:
            msg = 'There must be at least as many residuals as parameters'
            raise ValueError(msg)

        n_residuals = residual.size
        jacobian = _check_jacobian(
            jac(parameters),
            (n_residuals, initial.size)
        )
        cost = 0.5 * np.dot(residual, residual)

        return cost, residual, jacobian

    def update(parameters, step):
        return parameters + step

    def evaluate(parameters):
        residual = _check_residual(fun(parameters), n_residuals)

        return 0.5 * np.dot(residual, residual)

    return _levenberg_marquardt(
        initial,
        linearize,
        update,
        evaluate,
        max_iter=max_iter,
        tolerance=tolerance,
        mode='points'
    )


def optimize_board(
    fun: Callable,
    initial_camera_parameters: np.ndarray,
    initial_board_poses: np.ndarray,
    jac: Callable,
    max_iter: int=100,
    tolerance: float=1e-10
):
    """Fit shared camera parameters and one board pose per capture.

    Parameters
    ----------
    fun : callable
        Given the camera vector and board poses, return a list of
        residual vectors in capture order.
    initial_camera_parameters : np.ndarray
        Initial shared camera parameter vector with shape (M,).
    initial_board_poses : np.ndarray
        Initial board poses with shape (captures, 6).
    jac : callable
        Given the camera vector and board poses, return two lists of
        Jacobians in capture order. For capture j, the camera Jacobian
        has shape (N_j, M) and the board Jacobian has shape (N_j, 6).
    max_iter : int, optional
        Maximum attempted LM steps. Default is 100.
    tolerance : float, optional
        Scaled gradient and cost reduction tolerance. Default is 1e-10.

    Returns
    -------
    camera_parameters : np.ndarray
        Final accepted shared camera parameters.
    board_poses : np.ndarray
        Final accepted board poses.
    result : dict
        Convergence status, costs, trial history, and final undamped
        Jacobian rank for the coupled camera and board parameters.
        Rank deficiency emits a RuntimeWarning and is reported separately
        from convergence.

    Raises
    ------
    ValueError
        If iteration controls, initial parameters, residual or Jacobian
        blocks are invalid, or there are fewer residuals than parameters.

    Notes
    -----
    The QR step adapts the square-root elimination in Demmel et al. [1]_
    to six-parameter board poses. Each capture's pose is eliminated
    before solving for the shared camera parameters, similar to how it is
    done in Schur optimization commonly used in bundle adjustment algorithms.
    QR decomposition allows for better handeling of the residuals and
    jacobians since it avoids normal matrices which can amplify noise.

    References
    ----------
    .. [1] Demmel, N., Sommer, C., Cremers, D., and Usenko, V. (2021).
        Square Root Bundle Adjustment for Large-Scale Reconstruction.
        Proceedings of the IEEE/CVF Conference on Computer Vision and
        Pattern Recognition, 11723-11732.
        https://doi.org/10.1109/CVPR46437.2021.01155

    """
    camera_parameters = np.array(
        initial_camera_parameters,
        dtype=np.float64
    )
    board_poses = np.array(
        initial_board_poses,
        dtype=np.float64
    )

    if camera_parameters.ndim != 1:
        msg = 'Initial camera parameters must be a 1D array'
        raise ValueError(msg)

    if board_poses.ndim != 2 or board_poses.shape[1] != 6:
        msg = 'Initial board poses must have shape (captures, 6)'
        raise ValueError(msg)

    if board_poses.shape[0] == 0:
        msg = 'At least one board pose is required'
        raise ValueError(msg)

    if not np.all(np.isfinite(camera_parameters)):
        msg = 'Initial camera parameters must be finite'
        raise ValueError(msg)

    if not np.all(np.isfinite(board_poses)):
        msg = 'Initial board poses must be finite'
        raise ValueError(msg)

    n_boards = board_poses.shape[0]
    n_camera = camera_parameters.size
    residual_sizes = [None] * n_boards

    def residual_blocks(
        state: tuple
    ):
        residuals = fun(*state)

        if len(residuals) != n_boards:
            msg = 'Residual block count must match the board count'
            raise ValueError(msg)

        checked_residuals = []

        for i, residual in enumerate(residuals):
            residual = _check_residual(residual, residual_sizes[i])
            residual_sizes[i] = residual.size
            checked_residuals.append(residual)

        return checked_residuals

    def linearize(state):
        residuals = residual_blocks(state)

        if sum(residual_sizes) < n_camera + board_poses.size:
            msg = 'There must be at least as many residuals as parameters'
            raise ValueError(msg)

        cost = sum(0.5 * np.dot(residual, residual) for residual in residuals)
        camera_jacobians, board_jacobians = jac(*state)

        if (
            len(camera_jacobians) != n_boards
            or len(board_jacobians) != n_boards
        ):
            msg = 'Jacobian block counts must match the board count'
            raise ValueError(msg)

        checked_camera = []
        checked_board = []

        for i in range(n_boards):
            camera = _check_jacobian(
                camera_jacobians[i],
                (residual_sizes[i], n_camera)
            )
            board = _check_jacobian(
                board_jacobians[i],
                (residual_sizes[i], 6)
            )
            checked_camera.append(camera)
            checked_board.append(board)

        return cost, residuals, checked_camera, checked_board

    def update(state, camera_step, board_step):
        camera_parameters, board_poses = state

        return camera_parameters + camera_step, board_poses + board_step

    def evaluate(state):
        residuals = residual_blocks(state)

        return sum(0.5 * np.dot(residual, residual) for residual in residuals)

    initial = (
        camera_parameters,
        board_poses
    )
    state, result = _levenberg_marquardt(
        initial,
        linearize,
        update,
        evaluate,
        max_iter=max_iter,
        tolerance=tolerance,
        mode='target'
    )
    camera_parameters, board_poses = state

    return camera_parameters, board_poses, result