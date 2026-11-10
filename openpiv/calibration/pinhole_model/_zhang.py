from copy import deepcopy

import numpy as np

from ..dlt_model import calibrate_dlt
from ._utils import _rotation, _rotation_angles


__all__ = [
    'calibrate_intrinsics',
    'calibrate_zhang'
]


def _get_all_homography(
    all_object_points: np.ndarray,
    all_image_points: np.ndarray
):
    n_views = len(all_image_points)

    if n_views < 3:
        msg = 'Zhang calibration needs at least three target orientations'
        raise ValueError(msg)

    if len(all_object_points) == 0:
        raise ValueError('Object points must not be empty')

    first_points = np.asarray(all_object_points[0])

    if first_points.ndim == 1:
        boards = [all_object_points] * n_views
    else:
        boards = all_object_points

    if len(boards) != n_views:
        raise ValueError('Object point image point view mismatch')

    homographies = []

    for board, image in zip(boards, all_image_points):
        homography, _ = calibrate_dlt(
            board,
            image,
            enforce_coplanar=True
        )
        homographies.append(homography)

    return np.array(homographies)


def _get_Vij(
    all_h: np.ndarray,
    i: int,
    j: int
):
    column_i = all_h[:, :, i]
    column_j = all_h[:, :, j]
    v_ij = np.zeros((all_h.shape[0], 6), dtype=np.float64)

    v_ij[:, 0] = column_i[:, 0] * column_j[:, 0]
    v_ij[:, 1] = (
        column_i[:, 0] * column_j[:, 1]
        + column_i[:, 1] * column_j[:, 0]
    )
    v_ij[:, 2] = column_i[:, 1] * column_j[:, 1]
    v_ij[:, 3] = (
        column_i[:, 2] * column_j[:, 0]
        + column_i[:, 0] * column_j[:, 2]
    )
    v_ij[:, 4] = (
        column_i[:, 2] * column_j[:, 1]
        + column_i[:, 1] * column_j[:, 2]
    )
    v_ij[:, 5] = column_i[:, 2] * column_j[:, 2]

    return v_ij


def _get_B(
    all_h: np.ndarray
):
    v_00 = _get_Vij(all_h, 0, 0)
    v_01 = _get_Vij(all_h, 0, 1)
    v_11 = _get_Vij(all_h, 1, 1)

    constraints = np.zeros((all_h.shape[0] * 2, 6), dtype=np.float64)
    constraints[0::2, :] = v_01
    constraints[1::2, :] = v_00 - v_11

    row_norms = np.linalg.norm(constraints, axis=1)
    constraints /= row_norms[:, np.newaxis]

    _, singular_values, right = np.linalg.svd(constraints)

    if singular_values[-2] < singular_values[0] * 1e-10:
        msg = 'Degenerate target orientations for Zhang calibration'
        raise ValueError(msg)

    values = right[-1]

    if values[0] < 0:
        values = -values

    b0, b1, b2, b3, b4, b5 = values

    return np.array([
        [b0, b1, b3],
        [b1, b2, b4],
        [b3, b4, b5]
    ])


def _get_intrinsics(
    B: np.ndarray
):
    denominator = B[0, 0] * B[1, 1] - B[0, 1]**2
    principal_term = B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2]
    cy = principal_term / denominator

    scale = B[2, 2] - (B[0, 2]**2 + cy * principal_term) / B[0, 0]
    fx = np.sqrt(scale / B[0, 0])
    fy = np.sqrt((scale * B[0, 0]) / denominator)

    skew = -(B[0, 1] * fx**2 * fy) / scale
    cx = (skew * cy / fy) - (B[0, 2] * fx**2) / scale

    return fx, fy, cx, cy, skew


def calibrate_intrinsics(
    all_object_points: np.ndarray,
    all_image_points: np.ndarray
):
    """Calculate intrinsic parameters using linear Zhang calibration.

    Parameters
    ----------
    all_object_points : np.ndarray or list
        Fixed board-frame coordinates with shape (2, N) or (3, N),
        or one such array per view. Three-dimensional inputs have Z=0.
    all_image_points : list
        Image coordinates with shape (2, N) per view. Supply at least
        three distinct, non-degenerate target orientations.

    Returns
    -------
    intrinsics : tuple
        Focal lengths fx, fy, principal point cx, cy, and pixel skew.

    Raises
    ------
    ValueError
        If the point arrays or view counts are invalid, target
        orientations are degenerate, or positive finite intrinsics
        cannot be recovered.
    np.linalg.LinAlgError
        If homography or intrinsic decomposition fails.

    """
    homographies = _get_all_homography(
        all_object_points,
        all_image_points
    )
    constraints = _get_B(homographies)
    intrinsics = _get_intrinsics(constraints)

    if not np.all(np.isfinite(intrinsics)) or min(intrinsics[:2]) <= 0:
        msg = 'Unable to recover positive focal lengths from homographies'
        raise ValueError(msg)

    return intrinsics


def _pose_from_homography(
    homography: np.ndarray,
    intrinsics: np.ndarray
):
    h = np.linalg.solve(
        intrinsics,
        homography
    )

    scale = 2 / (
        np.linalg.norm(h[:, 0]) + np.linalg.norm(h[:, 1])
    )

    if h[2, 2] < 0:
        scale = -scale

    r1 = scale * h[:, 0]
    r2 = scale * h[:, 1]

    rotation = np.column_stack([
        r1,
        r2,
        np.cross(r1, r2)
    ])

    u, _, vh = np.linalg.svd(rotation)

    rotation = u @ np.diag([
        1,
        1,
        np.linalg.det(u @ vh)
    ]) @ vh

    translation = scale * h[:, 2]

    return rotation, translation


def _estimate_board_poses(
    cameras: list,
    points: np.ndarray,
    image_points: list,
    correct_distortion: bool=True
):
    poses = []

    for j in range(len(image_points[0])):
        for cam, images in zip(cameras, image_points):
            if images[j] is None:
                continue

            image = np.asarray(images[j], dtype=np.float64)
            valid = np.all(np.isfinite(image), axis=0)

            if np.count_nonzero(valid) < 4:
                continue

            normalized = cam._normalize_image_points(
                image[:, valid]
            )

            if correct_distortion:
                normalized = cam._undistort_points_brown(
                    *normalized
                )

            h, _ = calibrate_dlt(
                points[:, valid],
                normalized,
                enforce_coplanar=True
            )

            rotation, translation = _pose_from_homography(
                h,
                np.eye(3)
            )

            cam._get_rotation_matrix()

            poses.append(np.hstack([
                cam.rotation @ translation + cam.translation,
                _rotation_angles(cam.rotation @ rotation)
            ]))

            break

        else:
            msg = 'Pose initialization needs four corners in one view per capture'
            raise ValueError(msg)

    return np.array(poses)


def _estimate_radial_distortion(
    cam,
    points: np.ndarray,
    images: list,
    poses: np.ndarray
):
    design = []
    residual = []

    for image, pose in zip(images, poses):
        rotation, _ = _rotation(pose[3:])
        translation = pose[:3].reshape(3, 1)
        world = rotation @ points + translation

        normalized = cam._normalize_world_points(world)
        r2 = np.sum(normalized**2, axis=0)

        ideal = cam.project_points(
            world,
            correct_distortion=False
        )

        principal = np.asarray(cam.principal).reshape(2, 1)
        offset = ideal - principal

        view_design = np.stack([
            offset * r2,
            offset * r2**2
        ], axis=-1)
        view_design = view_design.reshape(-1, 2)
        design.append(view_design)

        view_residual = image - ideal
        residual.append(view_residual.reshape(-1))

    design_matrix = np.vstack(design)
    residual_vector = np.concatenate(residual)

    radial, _, _, _ = np.linalg.lstsq(
        design_matrix,
        residual_vector,
        rcond=None
    )

    return radial


def _zhang_residuals(
    cam,
    object_points: np.ndarray,
    image_points: list,
    board_poses: np.ndarray,
    correct_distortion: bool
):
    residuals = []

    for image, pose in zip(image_points, board_poses):
        rotation, _ = _rotation(pose[3:])
        translation = pose[:3].reshape(3, 1)
        world_points = rotation @ object_points + translation

        projected = cam.project_points(
            world_points,
            correct_distortion=correct_distortion
        )
        residual = (projected - image).T
        residuals.append(residual.reshape(-1))

    return np.concatenate(residuals)


def calibrate_zhang(
    cam,
    object_points: np.ndarray,
    image_points: list,
    correct_distortion: bool=True,
    refine: bool=True,
    max_iter: int=200,
    tolerance: float=1e-10
):
    """Calibrate a pinhole camera using Zhang's method.

    Estimate intrinsics and board poses from planar homographies, then
    estimate k1 and k2 linearly. Optionally refine the intrinsics, selected
    distortion coefficients, and board poses together using Schur LM.

    Parameters
    ----------
    cam : pinhole_model.PinCam
        Camera to update. Its reduced/full distortion setting and tilt
        flag select the coefficients fitted during nonlinear refinement.
    object_points : np.ndarray
        Fixed board coordinates with shape (2, N) or (3, N), with Z=0.
        Corner spacing defines the physical scale.
    image_points : list
        Image coordinates with shape (2, N) for each view. Supply at
        least three different board orientations and all corners.
    correct_distortion : bool, optional
        Estimate k1 and k2 linearly and fit the selected distortion
        coefficients during refinement. If False, use an ideal pinhole
        projection. Default is True.
    refine : bool, optional
        Refine using analytic derivatives and Schur LM. Default is True.
        If False, return only the linear initialization.
    max_iter : int, optional
        Maximum attempted Schur LM steps. Default is 200.
    tolerance : float, optional
        Convergence tolerance for Schur LM. Default is 1e-10.

    Returns
    -------
    board_poses : np.ndarray
        Board-to-camera poses with shape (views, 6), containing translation
        followed by Rx Ry Rz Euler angles in radians. The fitted camera
        is at the origin with identity orientation.
    result : dict
        Success, message, initial and final costs, RMS per scalar pixel
        residual, linear k1/k2 estimate, and selected solver. Also includes
        the Schur step count, history, and residual-evaluation count.

    Raises
    ------
    ValueError
        If camera parameters or point-array shapes are invalid, the board
        is not on Z=0, target views are insufficient or degenerate, or
        positive finite focal lengths cannot be estimated.
    np.linalg.LinAlgError
        If a homography, intrinsic, or pose decomposition fails.

    Notes
    -----
    k1/k2 initialization follows Equation (13) in Zhang [Zhang2000]_. The joint
    pixel refinement follows Equation (14), extended to the selected Brown
    and sensor-tilt coefficients. When performing non-linear optimization,
    solver failures will return the last fitted state and success=False.

    References
    ----------
    .. [Zhang2000] Zhang, Z. (2000). A flexible new technique for camera
        calibration. IEEE Transactions on Pattern Analysis and Machine
        Intelligence, 22(11), 1330-1334. See Sections 3.1-3.4 of
        Microsoft Research report MSR-TR-98-71, Equations (13)-(14).
        https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf

    """
    cam._check_parameters()

    points = np.array(object_points, dtype=np.float64)
    images = np.array(image_points, dtype=np.float64)

    if points.ndim != 2 or points.shape[0] not in (2, 3):
        raise ValueError('Board points must have shape (2, N) or (3, N)')

    if points.shape[0] == 2:
        points = np.vstack([
            points,
            np.zeros(points.shape[1])
        ])

    if not np.allclose(points[2], 0):
        raise ValueError('Board points must have Z=0 in the board frame')

    if images.shape != (len(images), 2, points.shape[1]):
        raise ValueError('Image points must have shape (views, 2, N)')

    # Linear intrinsic and board-pose initialization.
    homographies = _get_all_homography(
        points,
        images
    )
    constraints = _get_B(homographies)
    fx, fy, cx, cy, skew = _get_intrinsics(constraints)

    if not np.all(np.isfinite([fx, fy, cx, cy, skew])) or min(fx, fy) <= 0:
        msg = 'Unable to recover positive focal lengths from homographies'
        raise ValueError(msg)

    intrinsic = np.array([
        [fx, skew, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    poses = []

    for homography in homographies:
        rotation, translation = _pose_from_homography(
            homography,
            intrinsic
        )
        angles = _rotation_angles(rotation)
        poses.append(np.hstack([translation, angles]))

    poses = np.array(poses)

    initial = deepcopy(cam)
    initial.dtype = np.dtype(np.float64)
    initial.translation = np.zeros(3)
    initial.orientation = np.zeros(3)
    initial.focal = np.array([fx, fy])
    initial.principal = np.array([cx, cy])
    initial.distortion1 = np.zeros(8)
    initial._get_rotation_matrix()

    # Linear k1/k2 initialization. Other coefficients start at zero.
    radial = np.zeros(2)

    if correct_distortion:
        radial = _estimate_radial_distortion(
            initial,
            points,
            images,
            poses
        )
        initial.distortion1[:2] = radial

    residuals = _zhang_residuals(
        initial,
        points,
        images,
        poses,
        correct_distortion
    )
    initial_cost = 0.5 * residuals @ residuals

    result = {
        'success': True,
        'message': 'Linear initialization only',
        'solver': 'linear',
        'iterations': 0,
        'nfev': 0,
        'history': []
    }

    if refine:
        # Import bundle adjust here to avoid circular imports between _zhang and _bundle_adjust
        from ._bundle_adjustment import bundle_adjust

        poses, result = bundle_adjust(
            [initial],
            points,
            [images],
            board_poses=poses,
            correct_focal=True,
            correct_distortion=correct_distortion,
            max_iter=max_iter,
            tolerance=tolerance
        )
        result['solver'] = 'schur'
        accepted_steps = sum(step['accepted'] for step in result['history'])
        result['nfev'] = 1 + result['iterations'] + int(accepted_steps)

    residuals = _zhang_residuals(
        initial,
        points,
        images,
        poses,
        correct_distortion
    )

    cam.translation = initial.translation.astype(cam.dtype)
    cam.orientation = initial.orientation.astype(cam.dtype)
    cam.focal = initial.focal.astype(cam.dtype)
    cam.principal = initial.principal.astype(cam.dtype)
    cam.distortion1 = initial.distortion1.astype(cam.dtype)
    cam._get_rotation_matrix()

    result['initial_cost'] = initial_cost
    result['cost'] = 0.5 * residuals @ residuals
    result['rms'] = np.sqrt(np.mean(residuals**2))
    result['linear_radial'] = radial

    return poses, result
