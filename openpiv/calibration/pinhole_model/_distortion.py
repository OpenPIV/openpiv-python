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
    Distortion model is inspired by MyPTV. The link is provided below.
    https://github.com/ronshnapp/MyPTV/tree/master/myptv
    
    The polynomial is of the 2nd order type, with the coefficients arranged
    like such: coeff = [1, x, y, x**2, y**2, x*y]. This effectively allows
    any distortion in the x and y axes to be compensated. However, the 
    polynomial model is not stable when extrapolating, so beware of
    artifcacts.
    
    """    
    k1, k2, _, _ = self.distortion2
    dtype = self.dtype
    
    poly = np.array([np.ones_like(xd), xd, yd, xd**2, yd**2, xd * yd], dtype=dtype)
    
    x = np.dot(k1, poly)
    y = np.dot(k2, poly)
    
    return np.array([x, y], dtype=dtype)


def _distort_points_poly(
    self,
    x: np.ndarray, 
    y: np.ndarray
):
    """Distort normalized points.
    
    Distort normalized camera points using a polynomial distortion model.
    
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
    Distortion model is inspired by MyPTV. The link is provided below.
    https://github.com/ronshnapp/MyPTV/tree/master/myptv
    
    The polynomial is of the 2nd order type, with the coefficients arranged
    like such: coeff = [1, x, y, x**2, y**2, x*y]. This effectively allows
    any distortion in the x and y axes to be compensated. However, the 
    polynomial model is not stable when extrapolating, so beware of
    artifcacts.
    
    To compute the inverse of the distortion model, we simply minimize a
    new inverse matrix by projecting world points into image points and
    correcting for distortion. This approach is different from MyPTV as it
    does not use a Taylor Series expansion on the error terms for inversion.
    
    """    
    _, _, k1, k2 = self.distortion2
    dtype = self.dtype
    
    poly = np.array([np.ones_like(x), x, y, x**2, y**2, x * y], dtype=dtype)
    
    xd = np.dot(k1, poly)
    yd = np.dot(k2, poly)
    
    return np.array([xd, yd], dtype=dtype)