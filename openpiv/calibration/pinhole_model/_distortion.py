import numpy as np


def _undistort_points_brown(
    self,
    xd: np.ndarray, 
    yd: np.ndarray
):
    """Undistort normalized points.
    
    Undistort normalized camera points using a radial and tangential
    distortion model. 
    
    Parameters
    ----------
    xd : 1D np.ndarray
        Distorted x-coordinates.
    yd : 1D np.ndarray
        Distorted y-coordinates.
        
    Returns
    -------
    x : 1D np.ndarray
        Undistorted x-coordinates.
    y : 1D np.ndarray
        Undistorted y-coordinates.
    
    Notes
    -----
    Distortion model is based off of OpenCV. The direct link where the
    distortion model was accessed is provided below.
    https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html
    
    """    
    k = self.distortion1
    dtype = self.dtype
    
    r2 = xd*xd + yd*yd
    r4 = r2 * r2
    r6 = r4 * r2
    
    num = 1 + k[0]*r2 + k[1]*r4 + k[4]*r6
    den = 1 + k[5]*r2 + k[6]*r4 + k[7]*r6
    
    delta_x = k[2]*2*xd*yd + k[3]*(r2 + 2*xd*xd)
    delta_y = k[3]*2*xd*yd + k[2]*(r2 + 2*yd*yd)
    
    x = xd*(num / den) + delta_x
    y = yd*(num / den) + delta_y
    
    return np.array([x, y], dtype=dtype)


def _distort_points_brown(
    self,
    x: np.ndarray, 
    y: np.ndarray
):
    """Distort normalized points.
    
    Distort normalized camera points using a radial and tangential
    distortion model. 
    
    Parameters
    ----------
    x : 1D np.ndarray
        Undistorted x-coordinates.
    y : 1D np.ndarray
        Undistorted y-coordinates.
        
    Returns
    -------
    xd : 1D np.ndarray
        Distorted x-coordinates.
    yd : 1D np.ndarray
        Distorted y-coordinates.
    
    Notes
    -----
    Distortion model is based off of OpenCV. The direct link where the
    distortion model was accessed is provided below.
    https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html
    
    """    
    k = self.distortion1
    dtype = self.dtype
    
    r2 = x*x + y*y
    r4 = r2 * r2
    r6 = r4 * r2
    
    den = 1 + k[0]*r2 + k[1]*r4 + k[4]*r6
    num = 1 + k[5]*r2 + k[6]*r4 + k[7]*r6
    
    delta_x = k[2]*2*x*y + k[3]*(r2 + 2*x*x)
    delta_y = k[3]*2*x*y + k[2]*(r2 + 2*y*y)

    xd = (x - delta_x) * (num / den)
    yd = (y - delta_y) * (num / den)
    
    return np.array([xd, yd], dtype=dtype)


def _undistort_points_poly(
    self,
    xd: np.ndarray, 
    yd: np.ndarray
):
    """Undistort normalized points.
    
    Undistort normalized camera points using a polynomial distortion model.
    
    Parameters
    ----------
    xd : 1D np.ndarray
        Distorted x-coordinates.
    yd : 1D np.ndarray
        Distorted y-coordinates.
        
    Returns
    -------
    x : 1D np.ndarray
        Undistorted x-coordinates.
    y : 1D np.ndarray
        Undistorted y-coordinates.
    
    Notes
    -----
    Distortion model based wholly on by MyPTV. The link is provided below.
    https://github.com/ronshnapp/MyPTV/tree/master/myptv
    
    The polynomial is of the 2nd order type, with the coefficients arranged
    like such: coeff = [1, x, y, x**2, y**2, x*y]. This effectively allows
    any distortion in the x and y axes to be compensated. However, the 
    polynomial model is not stable when extrapolating, so beware of
    artifcacts.
    
    References
    ----------
    .. [1] Shnapp, R. (2022). MyPTV: A Python Package for 3D Particle
        Tracking. J. Open Source Softw., 7, 4398.
        
    """    
    dtype = self.dtype
    coeffs = self.distortion2
    
    poly = np.array([xd, yd, xd**2, yd**2, xd * yd])

    e_ = np.dot(coeffs, poly)

    # Calculate derivatives of the polynomials
    e_0 = e_[0]
    a, b, c, d, ee = coeffs[0,:]
    e_xd_0 = a + 2*c*xd + ee*yd
    e_yd_0 = b + 2*d*yd + ee*xd

    e_1 = e_[1]
    a, b, c, d, ee = coeffs[1,:]
    e_xd_1 = a + 2*c*xd + ee*yd
    e_yd_1 = b + 2*d*yd + ee*xd
    
    # Calculate the inverse of the polynomials using derivatives
    A11 = 1.0 + e_xd_0
    A12 = e_yd_0
    A21 = e_xd_1
    A22 = 1.0 + e_yd_1

    rhs1 = xd*(1.0 + e_xd_0) + yd*e_yd_0 - e_0
    rhs2 = yd*(1.0 + e_yd_1) + xd*e_xd_1 - e_1

    Ainv = np.array([[A22, -A12],[-A21, A11]]) / (A11*A22 - A12*A21)
    
    xn = Ainv[0, 0] * rhs1 + Ainv[0, 1] * rhs2
    yn = Ainv[1, 0] * rhs1 + Ainv[1, 1] * rhs2

    return np.array([xn, yn], dtype=dtype)


def _distort_points_poly(
    self,
    xn: np.ndarray, 
    yn: np.ndarray
):
    """Distort normalized points.
    
    Distort normalized camera points using a polynomial distortion model.
    
    Parameters
    ----------
    xn : 1D np.ndarray
        Undistorted x-coordinates.
    yn : 1D np.ndarray
        Undistorted y-coordinates.
        
    Returns
    -------
    xd : 1D np.ndarray
        Distorted x-coordinates.
    yd : 1D np.ndarray
        Distorted y-coordinates.
    
    Notes
    -----
    Distortion model is inspired by MyPTV. The link is provided below.
    https://github.com/ronshnapp/MyPTV/tree/master/myptv
    
    The polynomial is of the 2nd order type, with the coefficients arranged
    like such: coeff = [x, y, x**2, y**2, x*y]. This effectively allows
    any distortion in the x and y axes to be compensated. However, the 
    polynomial model is not stable when extrapolating, so beware of
    artifcacts.
    
    To compute the inverse of the distortion model, we use compute the error
    term and add it to the normalized camera coordinated. This method does
    not require further iterations for refiunement.
    
    References
    ----------
    .. [1] Shnapp, R. (2022). MyPTV: A Python Package for 3D Particle
        Tracking. J. Open Source Softw., 7, 4398.
    
    """    
    dtype = self.dtype
    coeffs = self.distortion2
    
    poly = np.array([xn, yn, xn**2, yn**2, xn * yn])
    
    e_x, e_y = np.dot(coeffs, poly)
    
    xd = xn + e_x
    yd = yn + e_y
    
    return np.array([xd, yd], dtype=dtype)