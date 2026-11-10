from copy import deepcopy

import numpy as np

from .._optimizers import optimize_board
from ._optimize import (
    _camera_columns,
    _camera_vector,
    _set_camera_vector
)
from ._global_pose import _estimate_global_pose
from ._utils import _rotation, _rotation_angles


__all__ = [
    'align_reference_plate',
    'bundle_adjust'
]


def _board_points(
    object_points: np.ndarray
):
    points = np.asarray(object_points, dtype=np.float64)

    if points.ndim != 2 or points.shape[0] not in (2, 3):
        raise ValueError('Board points must have shape (2, N) or (3, N)')

    if points.shape[0] == 2:
        points = np.vstack([
            points,
            np.zeros(points.shape[1])
        ])

    if not np.all(np.isfinite(points)) or not np.allclose(points[2], 0):
        msg = 'Board points must be finite and have Z=0 in the board frame'
        raise ValueError(msg)

    return points


def _board_world(
    points: np.ndarray,
    pose: np.ndarray
):
    rotation, derivatives = _rotation(pose[3:])
    world = np.dot(rotation, points) + pose[:3, None]

    jac = np.zeros((points.shape[1], 3, 6))
    jac[:, :, :3] = np.eye(3)

    for i in range(3):
        jac[:, :, 3+i] = np.dot(derivatives[i], points).T

    return world, jac


def _prepare_observations(
    cameras: list,
    points: np.ndarray,
    image_points: list
):
    if len(cameras) == 0 or len(image_points) != len(cameras):
        raise ValueError('Camera - image size mismatch')

    n_boards = len(image_points[0])

    if (
        n_boards == 0
        or any(len(views) != n_boards for views in image_points)
    ):
        msg = 'Each camera must have one image entry per capture'
        raise ValueError(msg)

    observations = []
    board_counts = np.zeros(n_boards, dtype=int)
    links = [[] for _ in range(len(cameras) + n_boards)]

    for i, (cam, views) in enumerate(zip(cameras, image_points)):
        cam._check_parameters()

        for j, image in enumerate(views):
            if image is None:
                continue

            image = np.asarray(image, dtype=np.float64)

            if image.shape != (2, points.shape[1]):
                msg = 'Image points must have shape (2, number of board corners)'
                raise ValueError(msg)

            valid = np.all(np.isfinite(image), axis=0)

            if not np.any(valid):
                continue

            observations.append((
                i,
                j,
                valid,
                image[:, valid]
            ))

            board_counts[j] += np.count_nonzero(valid)
            links[i].append(len(cameras) + j)
            links[len(cameras) + j].append(i)

    if np.any(board_counts < 4):
        msg = 'Each capture needs at least four observed corners'
        raise ValueError(msg)

    # Camera 0 anchors the common world frame for all connected captures.
    connected = {0}
    pending = [0]

    while pending:
        node = pending.pop()

        for neighbor in links[node]:
            if neighbor not in connected:
                connected.add(neighbor)
                pending.append(neighbor)

    if len(connected) != len(links):
        msg = 'All cameras and captures must be connected to camera 0'
        raise ValueError(msg)

    return observations, n_boards


def _bundle_blocks(
    state: tuple,
    points: np.ndarray,
    observations: list,
    columns: list,
    correct_distortion: bool,
    jacobian: bool=True
):
    cameras, poses = state

    offsets = np.cumsum(
        [0] + [len(active) for active in columns]
    )
    size = offsets[-1]

    residuals = [[] for _ in poses]
    camera_jacobians = [[] for _ in poses]
    board_jacobians = [[] for _ in poses]

    for i, j, valid, image in observations:
        world, board_jac = _board_world(
            points[:, valid],
            poses[j]
        )

        cam = cameras[i]

        if not jacobian:
            camera_points = np.dot(
                cam.rotation.T,
                world - cam.translation[:, None]
            )
            depth = camera_points[2]

            if np.any(depth <= 0) or np.any(cam.focal <= 0):
                residual = np.full(image.size, np.inf)
            else:
                projected = cam.project_points(
                    world,
                    correct_distortion
                )
                residual = (projected - image).T.ravel()

            residuals[j].append(residual)
            continue

        _, camera_jac, point_jac = cam.project_jacobian(
            world,
            correct_distortion
        )

        camera_rows = np.zeros((image.size, size))
        section = slice(offsets[i], offsets[i+1])
        camera_rows[:, section] = camera_jac[:, :, columns[i]].reshape(
            -1,
            len(columns[i])
        )

        board_rows = np.einsum('nij,njk->nik', point_jac, board_jac)

        camera_jacobians[j].append(camera_rows)
        board_jacobians[j].append(board_rows.reshape(-1, 6))

    if jacobian:
        return (
            [np.vstack(rows) for rows in camera_jacobians],
            [np.vstack(rows) for rows in board_jacobians]
        )

    return [np.concatenate(rows) for rows in residuals]


def _change_frame(
    cameras: list,
    poses: np.ndarray,
    origin: np.ndarray,
    rotation: np.ndarray
):
    origin = np.array(origin, copy=True)
    rotation = np.array(rotation, copy=True)

    new_poses = poses.copy()

    for cam in cameras:
        cam._get_rotation_matrix()

        cam.translation = np.dot(
            rotation.T,
            cam.translation - origin
        )
        cam.orientation = _rotation_angles(
            np.dot(rotation.T, cam.rotation)
        )

        cam._get_rotation_matrix()

    for j, pose in enumerate(poses):
        board_rotation, _ = _rotation(pose[3:])

        new_poses[j, :3] = np.dot(
            rotation.T,
            pose[:3] - origin
        )
        new_poses[j, 3:] = _rotation_angles(
            np.dot(rotation.T, board_rotation)
        )

    return new_poses


def bundle_adjust(
    cameras: list,
    object_points: np.ndarray,
    image_points: list,
    board_poses: np.ndarray | None=None,
    correct_focal: bool=True,
    correct_distortion: bool=True,
    max_iter: int=100,
    tolerance: float=1e-10
):
    """Calibrate cameras and shared target poses using square-root QR LM.

    Refine the camera parameters and one six-parameter board pose per
    capture by minimizing pixel reprojection errors. Hold camera 0's pose
    fixed to define the common world frame.

    Parameters
    ----------
    cameras : list
        Pinhole cameras, updated in place after optimization. When board
        poses are omitted, initialize the cameras from the target views.
    object_points : np.ndarray
        Fixed board coordinates with shape (2, N) or (3, N), with Z=0.
    image_points : list
        Image coordinates indexed by camera and capture. Each view has
        shape (2, N). Missing views are None and missing corners are NaN.
    board_poses : np.ndarray, optional
        Initial board-to-world poses with shape (captures, 6), containing
        translation followed by Rx Ry Rz Euler angles in radians. If
        omitted, initialize intrinsics with Zhang's method, align the grid
        correspondences geometrically, and estimate camera and board
        poses in camera 0's frame. Each camera needs at least three
        non-degenerate views. If supplied, use the provided camera and
        board parameters and the input correspondences as given.
    correct_focal : bool, optional
        Refine both focal lengths. Principal points are always refined.
        Default is True.
    correct_distortion : bool, optional
        Include and refine the enabled Brown and sensor tilt coefficients.
        If False, use an ideal pinhole projection and hold the stored
        distortion coefficients fixed. Default is True.
    max_iter : int, optional
        Maximum number of attempted LM steps. Default is 100.
    tolerance : float, optional
        Scaled gradient tolerance. The optimizer also stops when an
        accepted cost reduction is less than tolerance times
        max(1, current cost). Default is 1e-10.

    Returns
    -------
    board_poses : np.ndarray
        Fitted shared poses with shape (captures, 6), in camera 0's frame.
        For supplied initial poses, camera 0's initial center and
        orientation define the change from the supplied world frame.
    result : dict
        Convergence success and message, number of attempted steps,
        initial and final costs, RMS per scalar pixel residual, and
        attempted-step history. If optimization does not converge, the
        last accepted state is returned. Automatic initialization also
        returns result['permutations'][camera][capture] column indices
        mapping original observations to the common board indexing.
        Missing views have a permutation of None.

    Notes
    -----
    When performing camera calibration of a PIV imaging system using
    bundle adjustment on planes of unkown origins, it is highly, highly,
    recommended that either the first frame or the last frame contains the
    plane that sets the origin of the system. Without this alignment plane,
    regular Tomo-PIV and associated reconstruction algorithms will fail as
    there is no set origin in the system.

    The implementation of the bundle adjustment algorithm is based on a
    square-root Levenberg-Marquardt formulation with initialization from
    Zhang's method [Zhang2000Bundle]_. Zhang's method estimates camera
    intrinsics and each board capture's translation and rotation.
    By holding one camera's pose fixed as a reference, the camera
    parameters and global board poses can be refined using reprojection
    errors and a bundle adjustment algorithm as described by Triggs et al.
    [Triggs2000Bundle]_.

    Typically, bundle adjustment algorithms use some form of Schur
    elimination, such as that described by Lourakis and Argyros
    [Lourakis2009Bundle]_. For this implementation, each capture's
    six-parameter board-pose block can be eliminated from the damped normal
    equations during Levenberg-Marquardt optimization. However, since normal
    equations square the Jacobian's condition number, which increases
    sensitivity to floating-point errors, a QR-based formulation that
    eliminates the board-pose increments directly from the augmented
    Jacobian was implemented based on the work of Demmel et al.
    [Demmel2021Bundle]_. This avoids forming normal equations and the
    associated loss of numerical accuracy. As in Schur-based implementations,
    the reduced camera system is solved first, and back-substitution is used
    to recover the board-pose updates.

    References
    ----------
    .. [Zhang2000Bundle] Zhang, Z. (2000). A flexible new technique for
    camera calibration. IEEE Transactions on Pattern Analysis and
    Machine Intelligence, 22(11), 1330-1334.
    https://doi.org/10.1109/34.888718

    .. [Triggs2000Bundle] Triggs, B., McLauchlan, P. F., Hartley, R. I.,
    and Fitzgibbon, A. W. (2000). Bundle adjustment - A modern synthesis.
    Vision Algorithms: Theory and Practice, 298-372.
    https://doi.org/10.1007/3-540-44480-7_21

    .. [Lourakis2009Bundle] Lourakis, M. I. A., and Argyros, A. A. (2009).
    SBA: A software package for generic sparse bundle adjustment.
    ACM Transactions on Mathematical Software, 36(1), Article 2.
    https://doi.org/10.1145/1486525.1486527

    .. [Demmel2021Bundle] Demmel, N., Sommer, C., Cremers, D., and
    Usenko, V. (2021). Square Root Bundle Adjustment for Large-Scale
    Reconstruction. Proceedings of the IEEE/CVF Conference on Computer
    Vision and Pattern Recognition, 11723-11732.
    https://doi.org/10.1109/CVPR46437.2021.01155

    """
    points = _board_points(object_points)

    _, n_boards = _prepare_observations(
        cameras,
        points,
        image_points
    )

    if max_iter < 1 or tolerance <= 0:
        raise ValueError('max_iter and tolerance must be positive')

    state_cameras = deepcopy(cameras)

    for cam in state_cameras:
        cam.dtype = np.dtype(np.float64)
        values = _camera_vector(cam).astype(np.float64)

        _set_camera_vector(
            cam,
            values
        )

    permutations = None

    if board_poses is None:
        board_poses, image_points, permutations = _estimate_global_pose(
            state_cameras,
            points,
            image_points,
            correct_focal=correct_focal,
            correct_distortion=correct_distortion
        )

    poses = np.array(board_poses, dtype=np.float64)

    if poses.shape != (n_boards, 6) or not np.all(np.isfinite(poses)):
        msg = 'Board poses must have shape (captures, 6) and be finite'
        raise ValueError(msg)

    observations, _ = _prepare_observations(
        state_cameras,
        points,
        image_points
    )

    # Use camera 0's initial center and orientation as the world frame.
    poses = _change_frame(
        state_cameras,
        poses,
        state_cameras[0].translation,
        state_cameras[0].rotation
    )

    state_cameras[0].translation = np.zeros(3)
    state_cameras[0].orientation = np.zeros(3)
    state_cameras[0]._get_rotation_matrix()

    columns = []

    for i, cam in enumerate(state_cameras):
        active = _camera_columns(
            cam,
            correct_focal=correct_focal,
            correct_distortion=correct_distortion,
            pose=i != 0
        )

        columns.append(active)

    n_residuals = sum(
        2 * image.shape[1]
        for _, _, _, image in observations
    )
    n_parameters = sum(len(active) for active in columns) + 6 * n_boards

    if n_residuals < n_parameters:
        msg = 'Too few observations for the number of fitted parameters'
        raise ValueError(msg)

    camera_values = [_camera_vector(cam) for cam in state_cameras]
    offsets = np.cumsum([0] + [len(active) for active in columns])

    initial = np.concatenate([
        values[active]
        for values, active in zip(camera_values, columns)
    ])

    def set_parameters(parameters):
        for i, (cam, values, active) in enumerate(
            zip(state_cameras, camera_values, columns)
        ):
            values[active] = parameters[offsets[i]:offsets[i+1]]
            _set_camera_vector(cam, values)

    def residuals(parameters, board_poses):
        set_parameters(parameters)

        return _bundle_blocks(
            (state_cameras, board_poses),
            points,
            observations,
            columns,
            correct_distortion,
            jacobian=False
        )

    def jacobian(parameters, board_poses):
        set_parameters(parameters)

        return _bundle_blocks(
            (state_cameras, board_poses),
            points,
            observations,
            columns,
            correct_distortion
        )

    parameters, fitted_poses, result = optimize_board(
        residuals,
        initial,
        poses,
        jac=jacobian,
        max_iter=max_iter,
        tolerance=tolerance
    )

    set_parameters(parameters)

    for cam, fitted in zip(cameras, state_cameras):
        values = _camera_vector(fitted).astype(cam.dtype)
        _set_camera_vector(cam, values)

    result['rms'] = np.sqrt(
        2 * result['cost'] / n_residuals
    )

    if permutations is not None:
        result['permutations'] = permutations

    return fitted_poses, result


def align_reference_plate(
    cameras: list,
    board_poses: np.ndarray,
    reference: int=0
):
    """Align a fitted system with a reference plate.

    Transform the cameras and board poses together so the designated
    capture defines the world origin and axes.

    Parameters
    ----------
    cameras : list
        Fitted cameras, transformed in place.
    board_poses : np.ndarray
        Fitted board poses with shape (captures, 6), containing
        translation followed by Rx Ry Rz Euler angles.
    reference : int, optional
        Reference capture index. Default is 0.

    Returns
    -------
    board_poses : np.ndarray
        Poses in the reference-plate frame, with shape (captures, 6).
        The reference board has identity pose in this frame.

    """
    poses = np.asarray(board_poses, dtype=np.float64)
    pose = poses[reference]
    rotation, _ = _rotation(pose[3:])

    return _change_frame(
        cameras,
        poses,
        pose[:3],
        rotation
    )