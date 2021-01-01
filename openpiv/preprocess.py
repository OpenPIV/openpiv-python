from scipy.ndimage import median_filter, gaussian_filter, binary_fill_holes
from skimage import io, img_as_float, exposure, data, img_as_uint, img_as_ubyte
from skimage.filters import sobel, rank, threshold_otsu
from skimage.measure import find_contours, approximate_polygon, points_in_poly
import numpy as np
import matplotlib.pyplot as plt

"""This module contains image processing routines that improve
images prior to PIV processing."""

__licence_ = """
Copyright (C) 2011  www.openpiv.net

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


def dynamic_masking(image, method="edges", filter_size=7, threshold=0.005):
    """Dynamically masks out the objects in the PIV images

    Parameters
    ----------
    image: image
        a two dimensional array of uint16, uint8 or similar type

    method: string
        'edges' or 'intensity':
        'edges' method is used for relatively dark and sharp objects, with visible edges, on
        dark backgrounds, i.e. low contrast
        'intensity' method is useful for smooth bright objects or dark objects or vice versa,
        i.e. images with high contrast between the object and the background

    filter_size: integer
        a scalar that defines the size of the Gaussian filter

    threshold: float
        a value of the threshold to segment the background from the object
        default value: None, replaced by sckimage.filter.threshold_otsu value

    Returns
    -------
    image : array of the same datatype as the incoming image with the object masked out
        as a completely black region(s) of zeros (integers or floats).


    Example
    --------
    frame_a  = openpiv.tools.imread( 'Camera1-001.tif' )
    imshow(frame_a) # original

    frame_a = dynamic_masking(frame_a,method='edges',filter_size=7,threshold=0.005)
    imshow(frame_a) # masked

    """
    imcopy = np.copy(image)
    # stretch the histogram
    image = exposure.rescale_intensity(img_as_float(image), in_range=(0, 1))
    # blur the image, low-pass
    blurback = img_as_ubyte(gaussian_filter(image, filter_size))
    if method == "edges":
        # identify edges
        edges = sobel(blurback)
        blur_edges = gaussian_filter(edges, 21)
        # create the boolean mask
        mask = blur_edges > threshold
        mask = img_as_ubyte(binary_fill_holes(mask))
        imcopy -= blurback
        imcopy[mask] = 0
    elif method == "intensity":
        background = gaussian_filter(median_filter(image, filter_size), filter_size)
        mask = background > threshold_otsu(background)
        imcopy[mask] = 0
    else:
        raise ValueError(f"method {method} is not implemented")

    return imcopy, mask




def mask_coordinates(image_mask, tolerance=1.5, min_length=10, plot=False):
    """ Creates set of coordinates of polygons from the image mask
    
    Inputs:
        mask : binary image of a mask.

        [tolerance] : float - tolerance for approximate_polygons, default = 1.5

        [min_length] : int - minimum length of the polygon, filters out 
        the small polygons like noisy regions, default = 10
    
    Outputs:
        mask_coord : list of mask coordinates in pixels
    
    Example:
        # if masks of image A and B are slightly different:
        image_mask = np.logical_and(image_mask_a, image_mask_b)
        mask_coords = mask_coordinates(image_mask)
        
    """
    
    mask_coords = []
    if plot:
        plt.figure()
        plt.imshow(image_mask)
    for contour in find_contours(image_mask, 0):
        coords = approximate_polygon(contour, tolerance=tolerance)
        if len(coords) > min_length:
            if plot:
                plt.plot(coords[:, 1], coords[:, 0], '-r', linewidth=2)
            mask_coords = coords.copy()
            
    return mask_coords

def prepare_mask_on_grid(x,y,mask_coords):
    """ Converts mask coordinates of the image mask 
    to the grid of 1/0 on the x,y grid
    Inputs:
        x,y : grid of x,y points
        mask_coords : array of coordinates in pixels of the image_mask

    Outputs:
        grid of points of the mask, of the shape of x  
    """
    xymask = points_in_poly(np.c_[y.flatten(), x.flatten()], mask_coords)
    return xymask.reshape(x.shape).astype(np.int)


