import numpy as np

from ..dlt_model import calibrate_dlt


__all__ = [
    "calibrate_intrinsics"
]


def _get_all_homography(
    all_object_points: np.ndarray,
    all_image_points: np.ndarray
):
    if len(all_object_points) < 2:
        raise ValueError(
            "Too little planes to calibrate"
        )
        
    all_H = []
    
    for i in range(len(all_image_points)):
        H, err = calibrate_dlt(
            all_object_points,
            all_image_points[i],
            enforce_coplanar=True
        )
        all_H.append(H)
        
    return np.array(all_H, dtype="float64").reshape((len(all_H), 3, 3))

    
def _get_Vij(
    all_h: np.ndarray,
    i: int,
    j: int
):
    v_ij = np.zeros((all_h.shape[0], 6), dtype="float64")
    
    v_ij[:, 0] = all_h[:, 0, i] * all_h[:, 0, j]
    v_ij[:, 1] = all_h[:, 0, i] * all_h[:, 1, j] + all_h[:, 1, i] * all_h[:, 0, j]
    v_ij[:, 2] = all_h[:, 1, i] * all_h[:, 1, j]
    v_ij[:, 3] = all_h[:, 2, i] * all_h[:, 0, j] + all_h[:, 0, i] * all_h[:, 2, j]
    v_ij[:, 4] = all_h[:, 2, i] * all_h[:, 1, j] + all_h[:, 1, i] * all_h[:, 2, j]
    v_ij[:, 5] = all_h[:, 2, i] * all_h[:, 2, j]
    
    return v_ij
    

def _get_B(
    all_h: np.ndarray
):    
    v_00 = _get_Vij(all_h, 0, 0)
    v_01 = _get_Vij(all_h, 0, 1)
    v_11 = _get_Vij(all_h, 1, 1)

    v = np.zeros((all_h.shape[0] * 2, 6), dtype = "float64")
    
    v[0::2, :] = v_01
    v[1::2, :] = v_00 - v_11
                 
    U, E, V = np.linalg.svd(v)
    
    b = V[-1, :]
    
    B0, B1, B2, B3, B4, B5 = b

    # Rearrage B to form B = K^-T K^-1
    B = np.array([[B0, B1, B3],
                  [B1, B2, B4],
                  [B3, B4, B5]])
    
    return B


def _get_intrinsics(
    B: np.ndarray
):
    v0 = (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2]) / (B[0, 0] * B[1, 1] - B[0, 1]**2)
    lambda_ = B[2, 2] - (B[0, 2]**2 + v0 * (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2])) / B[0, 0]
    alpha = np.sqrt(lambda_ / B[0, 0])
    beta = np.sqrt((lambda_ * B[0, 0]) / (B[0, 0] * B[1, 1] - B[0, 1]**2))
    gamma = -(B[0, 1] * alpha**2 * beta) / lambda_
    u0 = (gamma * v0 / beta) - (B[0, 2] * alpha**2) / lambda_
            
    return alpha, beta, u0, v0, gamma # <-- gamma is skew


def calibrate_intrinsics(
    all_object_points: np.ndarray,
    all_image_points: np.ndarray
):
    """Intrinsic calibration using Zhang's method.
    
    Using multiple views of planar targets, calculate the intrinsic
    parameters that best fits all views using a closed-form solution.
    
    Parameters
    ----------
    all_object_points : np.ndarray
        Lab coordinates with a structured like [[X, Y, Z]'] * number
        of planes.
        
    all_image_points : np.ndarray
        Image coordinates with a structured like [[x, y]'] * number
        of planes.
        
    Returns
    -------
    intrinsics : np.ndarray
        5 elements that contain camera intrinsic information and are in the
        order as such: fx, fy, cx, cy, and gamma (skew).
    
    """
    all_h = _get_all_homography(
        all_object_points,
        all_image_points
    )
    
    B = _get_B(all_h)
    
    return _get_intrinsics(B)