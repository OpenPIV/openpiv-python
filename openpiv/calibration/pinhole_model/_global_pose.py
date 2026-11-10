import numpy as np

from ..dlt_model import calibrate_dlt
from ._utils import _rotation_angles
from ._zhang import _get_B, _get_intrinsics, _pose_from_homography


def _board_permutations(
    points: np.ndarray
):
    board_min = points.min(axis=1)
    board_max = points.max(axis=1)
    center = (board_min + board_max) / 2

    board_extent = np.ptp(points, axis=1)
    tolerance = 1e-8 * max(1.0, board_extent.max())

    n_points = points.shape[1]
    point_indices = np.arange(n_points)
    original_points = points.T
    original_points = original_points[np.newaxis, :, :]

    permutations = []
    transforms = []

    for swap in [False, True]:
        for sx, sy in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
            axes = np.diag([sx, sy])

            if swap:
                axes = axes[:, ::-1]

            rotation = np.eye(3)
            rotation[:2, :2] = axes
            rotation[2, 2] = np.linalg.det(axes)

            translation = center - rotation @ center
            transformed = rotation @ points + translation.reshape(3, 1)

            # Compare every transformed point with every original point.
            transformed_points = transformed.T
            transformed_points = transformed_points[:, np.newaxis, :]
            difference = transformed_points - original_points
            squared_distance = np.sum(difference**2, axis=2)

            indices = np.argmin(squared_distance, axis=1)
            matched_distance = squared_distance[point_indices, indices]

            matches_geometry = np.max(matched_distance) <= tolerance**2
            one_to_one = len(np.unique(indices)) == n_points

            if not matches_geometry or not one_to_one:
                continue

            already_added = any(
                np.array_equal(indices, previous)
                for previous in permutations
            )

            if already_added:
                continue

            permutations.append(indices)
            transforms.append((rotation, translation))

    return permutations, transforms


def _mean_rotation(
    rotations: np.ndarray
):
    mean_matrix = np.mean(rotations, axis=0)
    left, _, right = np.linalg.svd(mean_matrix)

    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(left @ right)

    rotation = left @ correction @ right

    return rotation


def _consistent_rotation(
    rotations: np.ndarray
):
    best_cost = np.inf
    capture_indices = np.arange(rotations.shape[0])
    hypotheses = rotations.reshape(-1, 3, 3)

    for candidate in hypotheses:
        difference = rotations - candidate
        squared_distance = np.sum(difference**2, axis=(2, 3))
        nearest_distance = np.min(squared_distance, axis=1)
        cost = np.sum(nearest_distance)

        if cost < best_cost:
            best_cost = cost
            choices = np.argmin(squared_distance, axis=1)

    for _ in range(10):
        selected = rotations[capture_indices, choices]
        rotation = _mean_rotation(selected)

        difference = rotations - rotation
        squared_distance = np.sum(difference**2, axis=(2, 3))
        updated_choices = np.argmin(squared_distance, axis=1)

        if np.array_equal(updated_choices, choices):
            break

        choices = updated_choices

    selected = rotations[capture_indices, choices]
    rotation = _mean_rotation(selected)

    return rotation, choices


def _estimate_global_pose(
    cameras: list,
    points: np.ndarray,
    image_points: list,
    correct_focal: bool=True,
    correct_distortion: bool=True
):
    permutations, transforms = _board_permutations(points)
    n_cameras = len(cameras)
    n_boards = len(image_points[0])
    candidates = [[None] * n_boards for _ in cameras]

    # Estimate intrinsics and local board poses via Zhang's algo.
    for i, camera in enumerate(cameras):
        homographies = {}

        for j, image in enumerate(image_points[i]):
            if image is None:
                continue

            image = np.asarray(image, dtype=np.float64)
            valid = np.all(np.isfinite(image), axis=0)

            if np.count_nonzero(valid) < 4:
                continue

            try:
                homography, _ = calibrate_dlt(
                    points[:, valid],
                    image[:, valid],
                    enforce_coplanar=True
                )
            except ValueError:
                continue

            homographies[j] = homography

        if len(homographies) < 3:
            msg = f'{camera.name} needs three non-degenerate board orientations'
            raise ValueError(msg)

        homography_matrices = np.array(list(homographies.values()))
        constraints = _get_B(homography_matrices)
        fx, fy, cx, cy, _ = _get_intrinsics(constraints)

        if not np.all(np.isfinite([fx, fy, cx, cy])) or min(fx, fy) <= 0:
            msg = f'Unable to initialize intrinsics for {camera.name}'
            raise ValueError(msg)

        if correct_focal:
            camera.focal = np.array([fx, fy], dtype=camera.dtype)

        camera.principal = np.array([cx, cy], dtype=camera.dtype)

        if correct_distortion:
            camera.distortion1[:] = 0

        intrinsic = np.array([
            [camera.focal[0], 0, cx],
            [0, camera.focal[1], cy],
            [0, 0, 1]
        ])

        for j, homography in homographies.items():
            board_rotation, board_translation = _pose_from_homography(
                homography,
                intrinsic
            )
            view_candidates = []

            for local_rotation, local_translation in transforms:
                candidate_rotation = board_rotation @ local_rotation
                candidate_translation = board_rotation @ local_translation
                candidate_translation += board_translation

                view_candidates.append((
                    candidate_rotation,
                    candidate_translation
                ))

            candidates[i][j] = view_candidates

    # Anchor the world frame on camera 0.
    cameras[0].translation = np.zeros(3)
    cameras[0].orientation = np.zeros(3)
    cameras[0]._get_rotation_matrix()

    world_poses = [None] * n_boards

    for j, view_candidates in enumerate(candidates[0]):
        if view_candidates is not None:
            world_poses[j] = view_candidates[0]

    placed_cameras = {0}

    while len(placed_cameras) < n_cameras:
        progress = False

        for i, camera in enumerate(cameras):
            if i in placed_cameras:
                continue

            shared_captures = []

            for j in range(n_boards):
                if world_poses[j] is not None and candidates[i][j] is not None:
                    shared_captures.append(j)

            if len(shared_captures) < 3:
                continue

            relative_rotations = []

            for j in shared_captures:
                world_rotation, _ = world_poses[j]
                capture_rotations = []

                for local_rotation, _ in candidates[i][j]:
                    relative_rotation = world_rotation @ local_rotation.T
                    capture_rotations.append(relative_rotation)

                relative_rotations.append(capture_rotations)

            relative_rotations = np.array(relative_rotations)
            camera_rotation, choices = _consistent_rotation(relative_rotations)

            centers = []

            for k, (j, choice) in enumerate(zip(shared_captures, choices)):
                _, world_translation = world_poses[j]
                _, local_translation = candidates[i][j][choice]
                relative_rotation = relative_rotations[k, choice]

                center = world_translation - relative_rotation @ local_translation
                centers.append(center)

            camera.translation = np.mean(centers, axis=0)
            camera.orientation = _rotation_angles(camera_rotation)
            camera._get_rotation_matrix()

            for j, view_candidates in enumerate(candidates[i]):
                if world_poses[j] is not None or view_candidates is None:
                    continue

                local_rotation, local_translation = view_candidates[0]
                world_rotation = camera_rotation @ local_rotation
                world_translation = camera_rotation @ local_translation
                world_translation += camera.translation

                world_poses[j] = (world_rotation, world_translation)

            placed_cameras.add(i)
            progress = True

        if not progress:
            msg = 'Global pose initialization needs three shared captures per camera link'
            raise ValueError(msg)

    if any(pose is None for pose in world_poses):
        msg = 'Each capture needs four non-collinear corners in one camera'
        raise ValueError(msg)

    poses = []

    for rotation, translation in world_poses:
        angles = _rotation_angles(rotation)
        poses.append(np.hstack([translation, angles]))

    poses = np.array(poses)
    images = []
    order = []

    for i, camera in enumerate(cameras):
        views = []
        indices = []

        for j, image in enumerate(image_points[i]):
            if image is None:
                views.append(None)
                indices.append(None)
                continue

            image = np.asarray(image, dtype=np.float64)
            world_rotation, world_translation = world_poses[j]
            distance = []

            if candidates[i][j] is not None:
                for local_rotation, _ in candidates[i][j]:
                    estimated_rotation = camera.rotation @ local_rotation
                    difference = estimated_rotation - world_rotation
                    distance.append(np.sum(difference**2))
            else:
                world_points = world_rotation @ points
                world_points += world_translation.reshape(3, 1)

                projected = camera.project_points(
                    world_points,
                    correct_distortion=False
                )

                for permutation in permutations:
                    view = image[:, permutation]
                    valid = np.all(np.isfinite(view), axis=0)
                    residual = projected[:, valid] - view[:, valid]
                    distance.append(np.sum(residual**2))

            choice = np.argmin(distance)
            permutation = permutations[choice]
            aligned = image[:, permutation].copy()

            views.append(aligned)
            indices.append(permutation.tolist())

        images.append(views)
        order.append(indices)

    return poses, images, order