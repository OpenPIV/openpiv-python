import numpy as np
from typing import Tuple

from ..dlt_model import (camera as dlt_camera, 
                         line_intersect as dlt_line_intersect)


__all__ = [
    "multi_line_intersect"
]


def _dFdx(
    cam: "poly_model.camera",
    object_points: np.ndarray
) -> np.ndarray:
    dtype = cam.dtype
    a = cam.poly_wi
    
    object_points = np.array(object_points, dtype=dtype)
    
    X = object_points[0]
    Y = object_points[1]
    Z = object_points[2]
    
    polynomial_dx = np.array(
        [
            np.ones_like(X),
            Y, Z,
            2*X, 
            3*X*X, 2*X*Y, 2*X*Z,
            Y*Y,
            Z*Z, Y*Z
        ],
        dtype=dtype
    ).T
    
    polynomial_dx_a = np.array(
        [
            a[1, :],
            a[4, :], a[5, :],
            a[7, :],
            a[10, :], a[11, :], a[12, :],
            a[14, :],
            a[16, :], a[18, :]
        ],
        dtype=dtype
    )
    
    return np.dot(
        polynomial_dx,
        polynomial_dx_a
    ).T


def _dFdy(
    cam: "poly_model.camera",
    object_points: np.ndarray
) -> np.ndarray:
    dtype = cam.dtype
    a = cam.poly_wi
    
    object_points = np.array(object_points, dtype=dtype)
    
    X = object_points[0]
    Y = object_points[1]
    Z = object_points[2]
    
    polynomial_dy = np.array(
        [
            np.ones_like(Y),
            X, Z,
            2*Y, 
            X*X,
            3*Y*Y, 2*Y*X, 2*Y*Z,
            Z*Z, X*Z
        ],
        dtype=dtype
    ).T
    
    polynomial_dy_a = np.array(
        [
            a[2, :],
            a[4, :], a[6, :],
            a[8, :],
            a[11, :],
            a[13, :], a[14, :], a[15, :],
            a[17, :], a[18, :]
        ],
        dtype=dtype
    )
    
    return np.dot(
        polynomial_dy,
        polynomial_dy_a
    ).T


def _dFdz(
    cam: "poly_model.camera",
    object_points: np.ndarray
) -> np.ndarray:
    dtype = cam.dtype
    a = cam.poly_wi
    
    object_points = np.array(object_points, dtype=dtype)
    
    X = object_points[0]
    Y = object_points[1]
    Z = object_points[2]
    
    polynomial_dz = np.array(
        [
            np.ones_like(Y),
            X, Y,
            2*Z, 
            X*X,
            Y*Y,
            2*X*Z, 2*Y*Z, X*Y
        ],
        dtype=dtype
    ).T
    
    polynomial_dz_a = np.array(
        [
            a[3, :],
            a[5, :], a[6, :],
            a[9, :],
            a[12, :],
            a[15, :],
            a[16, :], a[17, :], a[18, :]
        ],
        dtype=dtype
    )
    
    return np.dot(
        polynomial_dz,
        polynomial_dz_a
    ).T


def _refine_jac3D(
    cams: list,
    object_points: np.ndarray
) -> np.ndarray:
    dtype = cams[0].dtype
    
    object_points = np.array(object_points, dtype=dtype)
    
    # if a 1D array is given, extend it
    if len(object_points.shape) == 1:
        object_points = object_points[:, np.newaxis]
            
    n_points = object_points.shape[1]
    n_cams = len(cams)
    
    jac = np.zeros([n_points, n_cams*2, 3], dtype=dtype)
        
    for i in range(n_cams):
        xdx, ydx = _dFdx(cams[i], object_points)
        xdy, ydy = _dFdy(cams[i], object_points)
        xdz, ydz = _dFdz(cams[i], object_points)
        
        jac[:, i*2, :] = np.array([xdx, xdy, xdz], dtype=dtype).T
        jac[:, (i*2)+1, :] = np.array([ydx, ydy, ydz], dtype=dtype).T

    return jac


def _refine_func3D(
    cams: list,
    object_points: np.ndarray,
    image_points: list
) -> np.ndarray:
    dtype = cams[0].dtype
    
    n_cams = len(cams)
    
    # make sure each camera has a pair of image points
    assert(n_cams == len(image_points))
        
    # if a 1D array is given, extend it
    if len(object_points.shape) == 1:
        object_points = object_points[:, np.newaxis]
        
    if len(image_points[0].shape) == 1:
        for i in range(len(image_points)):
            image_points[i] = image_points[i][:, np.newaxis]
    
    n_points = object_points.shape[1]
    
    residuals = np.zeros([n_points, n_cams, 2], dtype=dtype)
    
    for i in range(n_cams):
        res = cams[i].project_points(
            object_points
        ) - image_points[i]
        
        residuals[:, i, :] = res.T
        
    return residuals.reshape((n_points, n_cams*2))


# TODO: move this function somewhere else?
def _minimize_gradient(
    jac: np.ndarray,
    residual: np.ndarray,
    object_point: np.ndarray
):
    object_point += np.linalg.lstsq(jac, -residual, rcond=None)[0]
    
    return object_point


def _refine_pos3D(
    cams: list,
    object_points: np.ndarray,
    image_points: list,
    iterations: int=3
):
    dtype = cams[0].dtype
    n_cams = len(cams)
    
    # if a 1D array is given, extend it
    if len(object_points.shape) == 1:
        object_points = object_points[:, np.newaxis]
        
    if len(image_points[0].shape) == 1:
        for i in range(len(image_points)):
            image_points[i] = image_points[i][:, np.newaxis]
    
    n_points = object_points.shape[1]
    
    new_object_points = np.zeros([3, n_points], dtype=dtype)
    new_object_points[:, :] = object_points
        
    # TODO: I bet this loop is hellish slow for a large number of particles
    for i in range(iterations):
        jac = _refine_jac3D(
            cams,
            new_object_points
        )
        
        residuals = _refine_func3D(
            cams,
            new_object_points,
            image_points,
        )
        
        for particle in range(n_points):
            img_points = []
            for cam in range(n_cams):
                img_points.append(image_points[cam][:, particle])

            obj_point = new_object_points[:, particle]

            new_object_points[:, particle] = _minimize_gradient(
                jac[particle],
                residuals[particle],
                obj_point
            ).T

    return new_object_points
    

def _estimate_pos(
    cams: list,
    image_points: list
):
    # only need two cameras for analytical solution
    cam1 = dlt_camera(
        "dummy",
        resolution = cams[0].resolution,
        coeffs      = cams[0].dlt,
        dtype      = cams[0].dtype
    )
    
    cam2 = dlt_camera(
        "dummy",
        resolution = cams[1].resolution,
        coeffs     = cams[1].dlt,
        dtype      = cams[1].dtype
    )
    
    # TODO: Should we use DLT's multi_line_intersect?
    return dlt_line_intersect(
        cam1,
        cam2,
        image_points[0],
        image_points[1]
    )[0]


# Now the good part starts :)
def multi_line_intersect(
    cameras: list,
    image_points: list,
    init_pos: Tuple[float, float, float]=None,
    iterations: int=3
):
    """Estimate 3D positions using a gradient descent algorithm.
    
    Using an approximated initial position, optimize the particle locations
    such that the residuals between the image points and the projected
    object points are minimized. This is performed by calculating an 
    analytical solution for the derivatives of the projection function and
    finding a least squares solution by iteratively updating each point until
    a specified amount of iterations have been completed. The least squares
    solution is performed via SVD, making it robust to noise and artifacts.
    This is necessary since the intitial particle positions are approximated
    using the driect linear transforms (DLT) algorithm which ignores
    all distortion artifacts.
    
    Parameters
    ----------
    cameras : list
        A list of instances of polynomial cameras.
    img_points : list
        A list of image coordinates for each canera structure.
    init_pos : list, optional
        An initial position for all particles given by a tuple of three
        floats. The ordering of the tuple is (X, Y, Z) in world coordinates
        and should be the center of the volume (e.g., (0, 0, 0)). If not
        given, the particle positions are approximated using a DLT-based 
        algorithm.
    iterations : int, optional
        The number of iterations each object point recieves.
        
    Returns
    -------
    coords : np.ndarray
        The world coordinate that mininmize all projection residuals.
        
    References
    ----------
    .. [Herzog] Herzog, S., Schiepel, D., Guido, I. et al. A Probabilistic
            Particle Tracking Framework for Guided and Brownian Motion
            Systems with High Particle Densities. SN COMPUT. SCI. 2, 485
            (2021). https://doi.org/10.1007/s42979-021-00879-z
    
    """
    n_cams = len(cameras)
    n_imgs = len(image_points)
    
    # make sure each camera has a set of images
    if n_cams != n_imgs:
        raise ValueError(
            f"Camera - image size mismatch. Got {n_cams} cameras and " +
            f"{n_imgs} images"
        )
    
    # check each camera structure
    for cam in range(n_cams):
        cameras[cam]._check_parameters()
        
    # all cameras should have the same dtype
    dtype1 = cameras[0].dtype
    for cam in range(1, n_cams):
        dtype2 = cameras[cam].dtype
        
        if dtype1 != dtype2:
            raise ValueError(
                "Dtypes between camera structures must match"
            )
    
    # TODO: we should also check for individual image mismatches
    # if a 1D array is given, extend it
    if len(image_points[0].shape) == 1:
        for i in range(len(image_points)):
            image_points[i] = image_points[i][:, np.newaxis]
    
    if init_pos is not None:
        n_particles = image_points[0].shape[1]
        
        object_points = np.zeros(
            [3, n_particles],
            dtype=dtype1
        )
        
        object_points[0, :] = init_pos[0]
        object_points[1, :] = init_pos[1]
        object_points[2, :] = init_pos[2]
        
    else:
        object_points = _estimate_pos(
            cameras,
            image_points
        )
    
    object_points = _refine_pos3D(
        cameras,
        object_points,
        image_points,
        iterations=iterations
    )
    
    return object_points