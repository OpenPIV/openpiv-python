import numpy as np
from typing import Tuple
from scipy.spatial import ConvexHull

from .dlt_model import calibrate_dlt
from ._calib_utils import homogenize
from ._target_grids import get_simple_grid


__all__ = [
    "show_calibration_image",
    "find_nearest_points",
    "get_pairs_dlt",
    "get_pairs_proj"    
]
    

# @author: Theo
# Created on Thu Mar 25 21:03:47 2021

# @ErichZimmer - Changes (June 2, 2023):
# Revised function

# @ErichZimmer - Changes (Decemmber 2, 2023):
# Revised function
def show_calibration_image(
    image: np.ndarray, 
    markers: np.ndarray,
    figsize=(8,11),
    radius: int=30,
    fontsize=20
):
    """Plot markers on image.
    
    Plot markers on image and their associated index. This allows one to find the
    origin, x-axis, and y-axis point indexes for object-image point matching.
    
    Parameters
    ----------
    image : 2D np.ndarray
        A 2D array containing grayscale pixel intensities.
    markers : 2D np.ndarray
        A 2D array containing image marker coordinates in [x, y]` image coordinates.
    radius : int, optional
        The radius of the circle drawn around the marker point.
    
    Returns
    -------
    None
    
    Examples
    --------
    Examples
    --------
    >>> import numpy as np
    >>> from openpiv import calib_utils
    >>> from openpiv.data.test5 import cal_image
    
    >>> cal_img = cal_image(z=0)
    
    >>> marks_pos = calib_utils.detect_markers_template(
            cal_img,
            window_size = 64,
            template_radius=5,
            min_peak_height = 0.2,
            merge_radius = 10,
            merge_iter=5,
            min_count=8,
        )
    
    >>> calib_utils.show_calibration_image(
        cal_img,
        marks_pos
    )
    
    """
    from PIL import Image, ImageFont, ImageDraw
    from matplotlib import pyplot as plt
    
    markers = markers.T
    
    # funtction to show th clalibration iamge with numbers and circles
    plt.close('all')
    
    marker_numbers=np.arange(0,np.size(markers[:,0]))
    
    image_p = Image.fromarray(np.uint8((image/np.max(image[::]))*255))
    
    draw = ImageDraw.Draw(image_p)
    font = ImageFont.truetype("arial.ttf", fontsize)
    
    for i in range(0, np.size(markers, 0)):
        x, y=markers[i,:]
        draw.text((x, y), str(marker_numbers[i]), fill=(255),
                  anchor='mb',font=font)
        
    plt.figure(1)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image_p)
    
    for marker in markers:
        x, y = marker
        c = plt.Circle((x, y), radius, color='red', linewidth=2, fill=False)
        ax.add_patch(c)
        
    plt.show()
    plt.pause(1)
    
    
def _reorder_corners(
    corners: np.ndarray,
    is_square: bool=False
):
    """Reorder corners clock-wise and axis-aligned.
    
    Reorder corner points clock-wise with the x-axis being the largest axis
    of the rectangular grid.
    
    Parameters
    ----------
    corners : 2D np.ndarray
        A 2D np.ndarray of containing corner points of a rectangular.
    is_square : bool, optional
        If the target grid is a square, then omit axis alignment.
        
    Returns
    -------
    corners : 2D np.ndarray
        A 2D array of containing corner points of a rectangle ordered in a 
        clock-wise fashion.
    
    """
    x0 = np.mean(corners[0])
    y0 = np.mean(corners[1])
    
    theta = np.arctan2(corners[1] - y0, corners[0] - x0)
    
    index = np.argsort(theta)
    
    corners = corners[:, index]
    
    dist1 = np.linalg.norm(corners[:, 0] - corners[:, 1])
    dist1 = np.linalg.norm(corners[:, 0] - corners[:, -1]) 

    if dist2 > dist1 and not is_square:
        num_corners = corners.shape[1]
        new_index = [(i - 1) % num_corners for i in range(num_corners)]
        corners = corners[:, new_index]
        
    return corners


def _get_angle(
    point1: np.ndarray,
    point2: np.ndarray,
    point3: np.ndarray
):
    """Calculate an angle between 3 points.
    
    Calculate an angle between 0 and 360 degrees. This is used for locating
    the correct corners of a possibly distorted rectangle.
    
    Parameters
    ----------
    point1 : 2D np.ndarray
        A 2D np.ndarray of containing points for the origin of the triangle.
    
    point1, point2 : 2D np.ndarray
        A 2D np.ndarray of containing points for the sides of the triangle.
        
    Returns
    -------
    angle : float
        The angle between points 2 and 3 where point 1 is the vertex.
    
    """
    
    angle1 = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])
    angle2 = np.arctan2(point3[1] - point1[1], point3[0] - point1[0])
    
    angle = angle1 - angle2
    
    pi2 = np.pi * 2
    
    if angle < 0:
        angle += pi2
    
    if angle > np.pi:
        angle = pi2 - angle
    
    return (360 * angle) / pi2


def _find_corners(
    image_points: np.ndarray,
    asymmetric: bool=False,
    is_square=False
):
    """Locate corners of a rectangle using a convex hull.
    
    Locate the corners of a rectangle that may be distorted by obtaining
    the perimeter points using a convex hull of the image points and
    sorting the perimeter points by its angle between two neighboring 
    points. For a symmetric grid, the first four points are assumed to be
    the correct corner candidates, otherwise the first six points are
    selected.
    
    Parameter
    ---------
    image_points : 2D np.ndarray
        A 2D array of image points in [x, y]' image coordinates.
    asymmetric: bool, optional
        If true, validate that 6 corners have been found instead of 4.
    is_square : bool, optional
        If the target grid is a square, then omit axis alignment.
        
    Returns
    -------
    corners : 2D np.ndarray
        A 2D array of corner points in [x, y]' image coordinates.
        
    Raises
    ------
        ValueError
            if an incorrect amount of corners are detected.
    
    """
    indexes = ConvexHull(image_points.T).vertices
    
    candidates = image_points.T[indexes]
    
    min_ind = 0
    max_ind = candidates.shape[0]-1
    
    angles = []
    for i, corner in enumerate(candidates):
        
        ind1 = i - 1
        ind2 = i + 1
        
        if ind1 < 0:
            ind1 = max_ind
        
        if ind2 > max_ind:
            ind2 = min_ind
        
        point1 = candidates[ind1, :]
        point2 = candidates[ind2, :]
        
        angle = _get_angle(corner, point1, point2)
                
        angles.append(angle)
        
    indexes = np.argsort(angles)
    
    num_corners = 4
    
    if asymmetric == True:
 #       num_corners += 2
        raise ValueError(
            "Asymmetric grid detection is not currently implemented"
        )
    
    corners = candidates[indexes[:num_corners], :]
    
    corners = np.array(corners, dtype="float64").T
    
    return _reorder_corners(corners, is_square)


def find_nearest_points(
    image_points: np.ndarray,
    point: np.ndarray,
    threshold: float=None,
    flag_nans: bool=False
):
    """Locate the nearest image point.
    
    Locate the closest image point to the user selected image point.
    This function is implemented by find the minimum distance to the
    point of interest, so it will always return a point no matter how
    (un)realistic that point is.
    
    Parameters
    ----------
    image_points : 2D np.ndarray
        A 2D array of image points in [x, y]' image coordinates.
    point : 2D np.ndarray
        A 2D array of points of interest in [x, y]' image coordinates.
    threshold : float, optional
        If set, distances that are greater than the threshold are ignored.
    flag_nans : bool, optional
        If enabled, set flagged points that exceed the threshold to nan.
        Otherwise, the flagged points are not returned.
        
    Returns
    -------
    points : 2D np.ndarray
        A 2D array of image points in [x, y]' image coordinates.
    
    """
    dist = np.linalg.norm(
        [   
            image_points[0][:, np.newaxis] - point[0],
            image_points[1][:, np.newaxis] - point[1]
        ],
        axis=0
    )
        
    index = np.argmin(dist, axis = 0)
    
    if threshold is not None:
        min_dist = np.min(dist, axis=0)
        
        if flag_nans == True:
            set_to_nan = np.zeros_like(index).astype("bool")
            set_to_nan[min_dist > threshold] = True
            
        else:
            index = index[min_dist < threshold]
            set_to_nan = np.zeros_like(index).astype("bool")
        
    candidate_points = image_points[:, index]
    
    candidate_points[:, set_to_nan] = np.nan
    
    return candidate_points


def get_pairs_dlt(
    image_points: np.ndarray,
    grid_shape: list,
    grid: np.ndarray=None,
    corners: np.ndarray=None,
    asymmetric: bool=False
    
):  
    """Match object points to image points via homography.
    
    Match image points to lab points using the direct linear transformation
    correspondence of four corner points. Using the DLT, the correspondences
    of the remaining image points can be found and paired with lab points
    under the assumption that there is little distortion and the grid is
    planar.
    
    Parameters
    ----------
    image_points : 2D np.ndarray
        Image coordinates. The ndarray is structured like [x, y]'.
    grid_shape : tuple
        The shape of the grid for the x and y axis respectively.
    grid : 2D np.ndarray, optional
        Lab coordinates with an array structed like [x, y]'. If no grid
        is supplied, a simple one is automatically created.
    corners : 2D np.ndarray, optional
        Corners used for point correspondences. If not supplied, corners
        are automatically detected using a convex-hull algorithm.
    asymmetric : bool
        If true, use asymmetric point matching.
    
    Returns
    -------
    img_points : 2D np.ndarray
        2D matched image points of [x, y]' in image coordinates.
    obj_points : 2D np.ndarray
        2D matched object points of [x, y, z]' in world coordinates.
        
    Notes
    -----
    Since the direct linear transformation is the similar to the pinhole
    camera model without distortion modeling, this algorithm will fail with
    the presence of distortion and high point densities. Additionally,
    non-planar calibration targets are not compatible with this function.
    
    """
    if asymmetric:
        raise ValueError(
            "Asymmetric grid detection is not currently implemented"
        )
        
    if not isinstance(grid, np.ndarray):
        grid = get_simple_grid(
            grid_shape[0], grid_shape[1],
            0, 0, 0,
            flip_y=False
        )
        
    grid = grid[:2, :].reshape((2, grid_shape[1], grid_shape[0]))
    
    real_corners = np.array([
        grid[:, 0, 0],
        grid[:, 0, grid_shape[0] - 1],
        grid[:, grid_shape[1] - 1, grid_shape[0] - 1],
        grid[:, grid_shape[1] - 1, 0]
    ]).T
    
    grid = grid.reshape([2, -1])
    
    if not isinstance(corners, np.ndarray):
        corners = _find_corners(
            image_points,
            asymmetric,
            is_square=grid[0] == grid[1]
        )
    
    H, _ = calibrate_dlt(
        real_corners, 
        corners, 
        enforce_coplanar=True
    )
    
    rectified = np.dot(
        H, 
        homogenize(grid)
    )
    
    rectified = rectified / rectified[2, :]
    rectified = rectified[:2, :]
    
    tolerance1 = np.linalg.norm(corners[:, 0] - corners[:, 1]) / grid_shape[0]
    tolerance2 = np.linalg.norm(corners[:, 0] - corners[:, -1]) / grid_shape[1]
    
    tolerance = min(tolerance1, tolerance2)
    
    reordered_points = find_nearest_points(
        image_points, 
        rectified,
        tolerance, 
        flag_nans=True
    )
    
    # check for duplicates (only happens with distortion)
    mask = ~np.isnan(reordered_points[0, :])
    
    good_points = np.unique(
        reordered_points[:, mask], 
        axis=1
    )
    
    if good_points.shape[1] != reordered_points[:, mask].shape[1]:
        raise Exception(
            "Failed to sort image points due to multiple points sharing " +
            "the same location (this is most likely caused by distortion)"
        )
            
    return reordered_points[:, mask], grid[:, mask]


def get_pairs_proj(
    cam_struct: dict,
    proj_func: "function",
    object_points: np.ndarray,
    image_points: np.ndarray,
    tolerance: float=10
): 
    """Match object points to image points via projection.
    
    Match object points to image points by projection with a rough
    calibration of at least 6 points. This method is more reliable than
    matching image points to object points based on homographies since
    image points and object points are being projected and compared to 
    each other to find a best match. This allows non-planar calibration
    plates to be used.
    
    Parameters
    ----------
    cam_struct : dict
        A dictionary structure of camera parameters.
    proj_func : function
        Projection function with the following signiture:
        res = func(cam_struct, object_points).
    image_points : 2D np.ndarray
        2D np.ndarray of [x, y]` image coordinates.
    object_points : 2D np.ndarray
        2D np.ndarray of [X, Y, Z]` world coordinates.
    tolerance : float, optional
        The maximum RMS error between the image point and an object point.
    
    Returns
    -------
    img_points : 2D np.ndarray
        2D matched image points of [x, y] in image coordinates.
    obj_points : 2D np.ndarray
        2D matched object points of [x, y, z] in world coordinates.
    
    Notes
    -----
    This function is used when a rough calibration is performed over some points of
    the calibration plate. These points are usually manually selected and given
    world point coordinates. At least 9 points for a pinhole model or 19 points for
    a polynomial model are needed since this gives a good enough calibration to pair
    the correct object points to image points.
    
    """
    object_points = np.array(object_points, dtype="float64")
    image_points = np.array(image_points, dtype="float64")
    
    image_points_proj = proj_func(
        cam_struct,
        object_points
    )
    
    obj_pairs = []
    img_pairs = []
    
    for i in range(image_points.shape[1]):
        min_j = -1
        min_rmse = 1000
        
        for j in range(image_points_proj.shape[1]):
            rmse = np.mean(
                np.sqrt(
                    np.square(
                        image_points[:, i] - image_points_proj[:, j]
                    ),
                )
            )
            
            if rmse < min_rmse:
                min_rmse = rmse
                min_j = j
        
        if min_rmse < tolerance:
            if min_j == -1:
                continue
                
            obj_pairs.append(object_points[:, min_j])
            img_pairs.append(image_points[:, i])
    
    return (
        np.array(img_pairs, dtype="float64").T, 
        np.array(obj_pairs, dtype="float64").T
    )