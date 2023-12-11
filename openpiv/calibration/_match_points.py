import numpy as np
from typing import Tuple
from scipy.spatial import ConvexHull

__all__ = [
    "_reorder_corners",
    "find_corners",
    "find_nearest_points",
    "reorder_image_points",
    "show_calibration_image",
    "get_pairs_proj"    
]
    

def _reorder_corners(
    corners: np.ndarray
):
    """Reorder corners clock-wise and axis-aligned.
    
    Reorder corner points clock-wise with the x-axis being the largest axis
    of the rectangular grid.
    
    Parameters
    ----------
    corners : 2D np.ndarray
        A 2D np.ndarray of containing corner points of a rectangular.
        
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
    
    corners = corners.T[index].T
    
    dist1 = np.sqrt(
        (corners[0][0] - corners[0][1])**2 + 
        (corners[1][0] - corners[1][1])**2
    )
    
    dist2 = np.sqrt(
        (corners[0][0] - corners[0][3])**2 + 
        (corners[1][0] - corners[1][3])**2
    )

    if dist2 > dist1:
        new_index = [3, 0, 1, 2]
        corners = corners.T[new_index].T
        
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


def find_corners(
    image_points: np.ndarray,
    asymmetric: bool=False
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

    corners = []
    
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
    
    corners = candidates[indexes[:num_corners]]
    
    corners = np.array(corners, dtype="float64").T
    
    return corners


def find_nearest_points(
    image_points: np.ndarray,
    point: np.ndarray,
    threshold: float=None
):
    """Locate the nearest image point.
    
    Locate the the closest image point to the user selected image point.
    This function is implemented by find the mininmum distance to the
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
        
    Returns
    -------
    points : 2D np.ndarray
        A 2D array of image points in [x, y]' image coordinates.
    
    """
    dist = np.sqrt(
        np.sum(
            np.square(
                [   
                    image_points[0][:, np.newaxis] - point[0],
                    image_points[1][:, np.newaxis] - point[1]
                ]
            ),
            axis=0
        )
    )
        
    index = np.argmin(dist, axis = 0)
    
    if threshold is not None:
        min_dist = np.min(dist, axis=0)
        index = index[min_dist < threshold]
        
    return np.array([image_points[0][index], image_points[1][index]], dtype="float64")


def _find_line_points(
    image_points: np.ndarray,
    point1: tuple,
    point2: tuple,
    num_points: int,
    tolerance: float
):
    corner_pos_x = np.linspace(point1[0], point2[0], num_points, endpoint=True)
    corner_pos_y = np.linspace(point1[1], point2[1], num_points, endpoint=True)
    
    corner_pos = np.array([corner_pos_x, corner_pos_y], dtype="float64")
    
    points = find_nearest_points(image_points, corner_pos, tolerance)
    
    return points


def _find_line_points_march(
    image_points: np.ndarray,
    point1: tuple,
    point2: tuple,
    num_points: int,
    tolerance: float
):  
    points_x = []
    points_y = []
    
    points_x.append(point1[0])
    points_y.append(point1[1])
    
    for i in range(num_points - 2):
        numel_elem = int((num_points - i)*1.5)
        
        corner_pos_x = np.linspace(point1[0], point2[0], numel_elem, endpoint=True)
        corner_pos_y = np.linspace(point1[1], point2[1], numel_elem, endpoint=True)
        
        band_x = corner_pos_x[0:3]
        band_y = corner_pos_y[0:3]
        
        corner_pos = np.array([band_x, band_y], dtype="float64")

        found = find_nearest_points(image_points, corner_pos, tolerance)
        
        if found.shape[1] != 3:
            return np.array([[], []], dtype="float64")
        
        elif found[0][1] != point1[0] and found[1][1] != point1[1]:        
            point1 = [found[0][1], found[1][1]]
        
        elif found[0][2] != point1[0] and found[1][2] != point1[1]:        
            point1 = [found[0][2], found[1][2]]
            
        else:
            return np.array([[], []], dtype="float64")
        
        points_x.append(point1[0])
        points_y.append(point1[1])
    
    points_x.append(point2[0])
    points_y.append(point2[1])
    
    points_x = np.array(points_x, dtype="float64").ravel()
    points_y = np.array(points_y, dtype="float64").ravel()
    
    points = np.array(
        [points_x, 
         points_y], 
        dtype="float64"
    )
    
    return points
    

def reorder_image_points(
    image_points: np.ndarray,
    corners: np.ndarray,
    grid_size: tuple,
    tolerance: float=None,
    march: bool=False
):
    """Reorder marker points in ascending order.
    
    Reorder marker points in ascending order so that they could be assigned
    world/lab coordinates. This algorithm worker by first locating all
    points in the y-axis in-between two corner points. Then, estimated 
    locations of the marker points in the x-axis are created and the nearest
    marker point is assumed to be the correct point. 
    
    image_points : 2D np.ndarray
        A 2D array of image points in [x, y]' image coordinates.
    corners : 2D np.ndarray
        A 2D array of containing corner points of a rectangle ordered in a 
        clock-wise fashion.
    grid_size : tuple
        The grid size in number of rows to number of columns.
    tolerance : float, optional
        The tolerance between the estimated marker point and the actual
        marker point. Points that exceed this tolerance are ignored. This
        parameter should not be set when utilizing the marching scheme.
    march : bool, optional
        If true, march from point to point in a line to better detect image
        points when distortion is severe.
        
    Returns
    -------
    image_points : 2D np.ndarray
        A 2D array of reordered image points in [x, y]' image coordinates.
        
    Raises
    ------
        ValueError
            if all marker points are not found and reordered.
        
    Notes
    -----
    This algorithm may fail with severe distortion due to the distorted
    points in the middle being located farther out than the distorted
    points in the corner. In the presence of strong distortions, enabling
    the marching algorithm may help lower the chances of an error being
    raised due to marker points not being properly located. If this
    algorithm fails, then manual calibration is the next option.
    
    The marching scheme is implemented as such. First, the starting and
    ending points are selected. Next, a segmented line is created where the
    number of segments is the width or height of the grid multiplied by
    1.5. Using the first three elements of this segmented line, the nearest
    marker points are then located. The line segment whos location is not
    equal to the starting point is assumed to be the correct point. Finally,
    the loop is repeated where the assumed correct point is the new starting
    point and the number of line segments are decreased by one. This scheme
    allows for barrel, pincushion, and other distortions that do not have
    sudden changes in distortion.
    
    """
    if march == True:
        line_to_point = _find_line_points_march
    else:
        line_to_point = _find_line_points
    
    # find points between corners 0 and 3
    points_03 = line_to_point(
        image_points,
        corners[:, 0], 
        corners[:, 3], 
        grid_size[1], 
        tolerance
    )
    
    # find points between corners 1 and 2
    points_12 = line_to_point(
        image_points,
        corners[:, 1], 
        corners[:, 2], 
        grid_size[1], 
        tolerance
    )
    
    num_points_recovered_1 = points_03.shape[1]
    num_points_recovered_2 = points_12.shape[1]
    
    if (num_points_recovered_1 != grid_size[1]) or \
        (num_points_recovered_2 != grid_size[1]):
        raise ValueError(
            "Not enough points recovered in y-axis. This is most likely " +
            "due to severe distortion and a low tolerance"
        )
    
    # now that we got the y-axis points, now get the x-axis in order from corner 0 going down to corner 3
    reordered_img_pairs_x = []
    reordered_img_pairs_y = []
    for i in range(grid_size[1]):
        points = line_to_point(
            image_points,
            points_03[:, i], 
            points_12[:, i], 
            grid_size[0], 
            tolerance
        )
        
        num_points_recovered = points.shape[1]
        
        if (num_points_recovered != grid_size[0]):
            raise ValueError(
                "Not enough points recovered in x-axis. This is most likely " +
                "due to severe distortion and a low tolerance"
            )
        
        reordered_img_pairs_x.append(points[0])
        reordered_img_pairs_y.append(points[1])
    
    reordered_img_pairs_x = np.array(reordered_img_pairs_x, dtype="float64").ravel()
    reordered_img_pairs_y = np.array(reordered_img_pairs_y, dtype="float64").ravel()
    
    reordered_img_pairs = np.array(
        [reordered_img_pairs_x, 
         reordered_img_pairs_y], 
        dtype="float64"
    )
    
    # remove duplicates (only happens with severe distortion)
    good_ind = np.sort(
        np.unique(
            reordered_img_pairs, 
            axis=1, 
            return_index=True
        )[1]
    )
    
    reordered_img_pairs = reordered_img_pairs.T[good_ind].T
    num_points_recovered = reordered_img_pairs.shape[1]
    
    if num_points_recovered != np.prod(grid_size):
        raise ValueError(
            f"{num_points_recovered} points recovered, however there should " +
            f"been {np.prod(grid_size)} points"
        )
        
    return reordered_img_pairs


# @author: Theo
# Created on Thu Mar 25 21:03:47 2021

# @ErichZimemr - Changes (June 2, 2023):
# Revised function

# @ErichZimemr - Changes (Decemmber 2, 2023):
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


def get_pairs_proj(
    cam_struct: dict,
    proj_func: "function",
    object_points: np.ndarray,
    image_points: np.ndarray,
    tolerance: float=5
): 
    """Match object points to image points via projection.
    
    Match object points to imahge points by projection with a rough calibration
    of at least 6 points. This method is more reliable than matching image points
    to object points based on analyitics since image points and object points are
    being projected and compared to each other to find a best match. This allows
    non-planar calibration plates to be used.
    
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