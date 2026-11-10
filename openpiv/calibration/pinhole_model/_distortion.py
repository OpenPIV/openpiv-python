import numpy as np


def _active_distortion(
    self
):
    if self.distortion_model == 'reduced':
        indices = [0, 1, 2, 3]
    else:
        indices = [0, 1, 2, 3, 4, 5]

    if self.tilt:
        indices += [6, 7]

    return np.array(indices, dtype=int)


def _brown(
    x: np.ndarray,
    y: np.ndarray,
    coefficients: np.ndarray,
    jacobian: bool=False
):
    k1, k2, p1, p2, k3, k4 = coefficients[:6]

    x2 = x * x
    y2 = y * y
    xy = x * y
    r2 = x2 + y2
    r4 = r2 * r2
    r6 = r4 * r2

    numerator = 1 + k1 * r2 + k2 * r4 + k3 * r6
    denominator = 1 + k4 * r2
    radial = numerator / denominator

    tangential_x = 2 * p1 * xy + p2 * (r2 + 2 * x2)
    tangential_y = p1 * (r2 + 2 * y2) + 2 * p2 * xy

    points = np.array([
        x * radial + tangential_x,
        y * radial + tangential_y
    ])

    if not jacobian:
        return points

    # Derivative of the radial factor with respect to r2.
    numerator_derivative = k1 + 2 * k2 * r2 + 3 * k3 * r4
    radial_derivative = numerator_derivative * denominator - numerator * k4
    radial_derivative /= denominator**2

    position_jacobian = np.empty((x.size, 2, 2))

    position_jacobian[:, 0, 0] = (
        radial + 2 * x2 * radial_derivative + 2 * p1 * y + 6 * p2 * x
    )
    position_jacobian[:, 0, 1] = (
        2 * xy * radial_derivative + 2 * p1 * x + 2 * p2 * y
    )
    position_jacobian[:, 1, 0] = position_jacobian[:, 0, 1]
    position_jacobian[:, 1, 1] = (
        radial + 2 * y2 * radial_derivative + 6 * p1 * y + 2 * p2 * x
    )

    coefficient_jacobian = np.zeros((x.size, 2, 8))

    # k1 and k2.
    radial_k1 = r2 / denominator
    radial_k2 = r4 / denominator

    coefficient_jacobian[:, 0, 0] = x * radial_k1
    coefficient_jacobian[:, 1, 0] = y * radial_k1
    coefficient_jacobian[:, 0, 1] = x * radial_k2
    coefficient_jacobian[:, 1, 1] = y * radial_k2

    # p1 and p2.
    coefficient_jacobian[:, 0, 2] = 2 * xy
    coefficient_jacobian[:, 1, 2] = r2 + 2 * y2
    coefficient_jacobian[:, 0, 3] = r2 + 2 * x2
    coefficient_jacobian[:, 1, 3] = 2 * xy

    # k3 and the denominator coefficient k4.
    radial_k3 = r6 / denominator
    radial_k4 = -radial * r2 / denominator

    coefficient_jacobian[:, 0, 4] = x * radial_k3
    coefficient_jacobian[:, 1, 4] = y * radial_k3
    coefficient_jacobian[:, 0, 5] = x * radial_k4
    coefficient_jacobian[:, 1, 5] = y * radial_k4

    return points, position_jacobian, coefficient_jacobian


def _tilt_matrix(
    t1: float,
    t2: float
):
    cos_x = np.cos(t1)
    sin_x = np.sin(t1)
    cos_y = np.cos(t2)
    sin_y = np.sin(t2)

    matrix = np.array([
        [cos_x, 0, 0],
        [-sin_x * sin_y, cos_y, 0],
        [sin_y, -cos_y * sin_x, cos_y * cos_x]
    ])

    derivative_x = np.array([
        [-sin_x, 0, 0],
        [-cos_x * sin_y, 0, 0],
        [0, -cos_y * cos_x, -cos_y * sin_x]
    ])

    derivative_y = np.array([
        [0, 0, 0],
        [-sin_x * cos_y, -sin_y, 0],
        [cos_y, sin_y * sin_x, -sin_y * cos_x]
    ])

    derivatives = np.array([
        derivative_x,
        derivative_y
    ])

    return matrix, derivatives


def _distort_normalized(
    self,
    x: np.ndarray,
    y: np.ndarray,
    jacobian: bool=False
):
    if jacobian:
        points, position_jacobian, coefficient_jacobian = _brown(
            x,
            y,
            self.distortion1,
            jacobian=True
        )
    else:
        points = _brown(
            x,
            y,
            self.distortion1
        )

    if self.tilt:
        t1, t2 = self.distortion1[6:8]
        matrix, derivatives = _tilt_matrix(t1, t2)

        homogeneous = np.vstack([
            points,
            np.ones(x.size)
        ])
        tilted = matrix @ homogeneous
        depth = tilted[2]
        points = tilted[:2] / depth

        if jacobian:
            # Quotient-rule derivatives of tilted x and y.
            tilt_jacobian = np.empty((x.size, 2, 2))

            tilt_jacobian[:, 0, 0] = (matrix[0, 0] - points[0] * matrix[2, 0]) / depth
            tilt_jacobian[:, 0, 1] = (matrix[0, 1] - points[0] * matrix[2, 1]) / depth
            tilt_jacobian[:, 1, 0] = (matrix[1, 0] - points[1] * matrix[2, 0]) / depth
            tilt_jacobian[:, 1, 1] = (matrix[1, 1] - points[1] * matrix[2, 1]) / depth

            position_jacobian = tilt_jacobian @ position_jacobian
            coefficient_jacobian = tilt_jacobian @ coefficient_jacobian

            for i in range(2):
                derivative = derivatives[i] @ homogeneous
                derivative_x = (derivative[0] - points[0] * derivative[2]) / depth
                derivative_y = (derivative[1] - points[1] * derivative[2]) / depth

                coefficient_jacobian[:, 0, 6+i] = derivative_x
                coefficient_jacobian[:, 1, 6+i] = derivative_y

    if jacobian:
        disabled = np.ones(8, dtype=bool)
        disabled[_active_distortion(self)] = False
        coefficient_jacobian[:, :, disabled] = 0

        return points, position_jacobian, coefficient_jacobian

    return points.astype(self.dtype)


def _distort_points_brown(
    self,
    x: np.ndarray,
    y: np.ndarray
):
    x = np.asarray(x)
    y = np.asarray(y)

    return _distort_normalized(
        self,
        x,
        y
    )


def _undistort_points_brown(
    self,
    xd: np.ndarray,
    yd: np.ndarray,
    max_iter: int=30,
    tolerance: float=1e-12
):
    target = np.array([xd, yd], dtype=np.float64)

    if self.tilt:
        t1, t2 = self.distortion1[6:8]
        matrix, _ = _tilt_matrix(t1, t2)
        homogeneous = np.vstack([
            target,
            np.ones(target.shape[1])
        ])

        untilted = np.linalg.solve(matrix, homogeneous)
        target = untilted[:2] / untilted[2]

    points = target.copy()

    for _ in range(max_iter):
        projected, position_jacobian, _ = _brown(
            *points,
            self.distortion1,
            jacobian=True
        )

        residual = (projected - target).T
        rhs = residual[:, :, np.newaxis]

        step = np.linalg.solve(position_jacobian, rhs)
        step = step[:, :, 0].T

        points -= step

        if np.max(np.abs(step)) < tolerance:
            return points.astype(self.dtype)

    raise RuntimeError('Brown distortion inversion failed to converge')