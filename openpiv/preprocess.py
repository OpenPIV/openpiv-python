"""This module contains image processing routines that improve
images prior to PIV processing."""

import numpy as np
from scipy.ndimage import median_filter, gaussian_filter, binary_fill_holes,\
     map_coordinates
from skimage import img_as_float, exposure, img_as_ubyte
from skimage import filters
from skimage.measure import find_contours, approximate_polygon, points_in_poly
from skimage.transform import rescale
import matplotlib.pyplot as plt
from openpiv.tools import imread

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
        'edges' method is used for relatively dark and sharp objects,
        with visible edges, on
        dark backgrounds, i.e. low contrast
        'intensity' method is useful for smooth bright objects or dark objects
        or vice versa,
        i.e. images with high contrast between the object and the background

    filter_size: integer
        a scalar that defines the size of the Gaussian filter

    threshold: float
        a value of the threshold to segment the background from the object
        default value: None, replaced by sckimage.filter.threshold_otsu value

    Returns
    -------
    image : array of the same datatype as the incoming image with the
    object masked out
    as a completely black region(s) of zeros (integers or floats).


    Example
    --------
    frame_a  = openpiv.tools.imread( 'Camera1-001.tif' )
    imshow(frame_a) # original

    frame_a = dynamic_masking(frame_a,method='edges',filter_size=7,
    threshold=0.005)
    imshow(frame_a) # masked

    """
    imcopy = np.copy(image)
    # stretch the histogram
    image = exposure.rescale_intensity(img_as_float(image), in_range=(0, 1))
    # blur the image, low-pass
    blurback = img_as_ubyte(gaussian_filter(image, filter_size))
    if method == "edges":
        # identify edges
        edges = filters.sobel(blurback)
        blur_edges = gaussian_filter(edges, 21)
        # create the boolean mask
        mask = blur_edges > threshold
        mask = img_as_ubyte(binary_fill_holes(mask))
        imcopy -= blurback
        imcopy[mask] = 0
    elif method == "intensity":
        background = gaussian_filter(median_filter(image, filter_size),
                                     filter_size)
        mask = background > filters.threshold_otsu(background)
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
                plt.plot(coords[:, 1], coords[:, 0], '-r', linewidth=3)
            mask_coords = coords.copy()

    return mask_coords


def prepare_mask_from_polygon(x, y, mask_coords):
    """ Converts mask coordinates of the image mask
    to the grid of 1/0 on the x,y grid
    Inputs:
        x,y : grid of x,y points
        mask_coords : array of coordinates in pixels of the image_mask

    Outputs:
        grid of points of the mask, of the shape of x
    
    """
    xymask = points_in_poly(np.c_[y.flatten(), x.flatten()], mask_coords)
    return xymask.reshape(x.shape)

def prepare_mask_on_grid(
    x: np.ndarray,
    y: np.ndarray,
    image_mask: np.ndarray,
)->np.array:
    """_summary_

    Args:
        x (np.ndarray): x coordinates of vectors in pixels
        y (np.ndarray): y coordinates of vectors in pixels
        image_mask (np.ndarray): image of the mask, 1 or True is to be masked

    Returns:
        np.ndarray: boolean array of the size of x,y with 1 where the values are masked
    """
    return map_coordinates(image_mask, [y,x]).astype(bool)


def normalize_array(array, axis = None):
    """
    Min/max normalization to [0,1].
    
    Parameters
    ----------
    array: np.ndarray
        array to normalize
        
    axis: int, tuple
        axis to find values for normalization
        
    Returns
    -------
    array: np.ndarray
        normalized array
    
    """
    array = array.astype(np.float32)
    if axis is None:
        return((array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array)))
    else:
        return((array - np.nanmin(array, axis = axis)) / 
               (np.nanmax(array, axis = axis) - np.nanmin(array, axis = axis)))

    
def standardize_array(array, axis = None):
    """
    Standardize an array.
    
    Parameters
    ----------
    array: np.ndarray
        array to normalize
        
    axis: int, tuple
        axis to find values for standardization
        
    Returns
    -------
    array: np.ndarray
        normalized array
    
    """
    array = array.astype(np.float32)
    if axis is None:
        return((array - np.nanmean(array) / np.nanstd(array)))  
    else:
        return((array - np.nanmean(array, axis = axis) / np.nanstd(array, axis = axis)))
    
    
def instensity_cap(img, std_mult = 2):
    """
    Simple intensity capping.
    
    Parameters
    ----------
    img: image
        a two dimensional array of float32 or float64,
        but can be uint16, uint8 or similar type
        
    std_mult: int
        how strong the intensity capping is. Lower values
        yields a lower threshold
        
    Returns
    -------
    img: image
        a filtered two dimensional array of the input image
    
    """
    upper_limit = np.mean(img) + std_mult * img.std()
    img[img > upper_limit] = upper_limit
    return img


def intensity_clip(img, min_val = 0, max_val = None, flag = 'clip'):
    """
    Simple intensity clipping
    
    Parameters
    ----------
    img: image
        a two dimensional array of float32 or float64,
        but can be uint16, uint8 or similar type
        
    min_val: int or float
        min allowed pixel intensity
        
    max_val: int or float
        min allowed pixel intensity
        
    flag: str
        one of two methods to set invalid pixels intensities
        
    Returns
    -------
    img: image
        a filtered two dimensional array of the input image
    
    """
    if flag not in ['clip', 'cap']:
        raise ValueError(f'Flag not supported {flag}')
    if flag == 'clip':
        flag_min, flag_max = 0 , 0
    elif flag == 'cap':
        flag_min, flag_max = min_val, max_val
    img[img < min_val] = flag_min
    if max_val is not None:
        img[img > max_val] = flag_max
    return img


def high_pass(img, sigma = 5, clip = False):
    """
    Simple high pass filter
    
    Parameters
    ----------
    img: image
        a two dimensional array of float32 or float64, 
        but can be uint16, uint8 or similar type
        
    sigma: float
        sigma value of the gaussian filter
        
    Returns
    -------
    img: image
        a filtered two dimensional array of the input image
    
    """
    low_pass = gaussian_filter(img, sigma = sigma)
    img -= low_pass
    if clip:
        img[img < 0] = 0
    return img


def local_variance_normalization(img, sigma_1 = 2, sigma_2 = 1, clip = True):
    """
    Local variance normalization by two gaussian filters.
    This method is used by common commercial softwares
    
    Parameters
    ----------
    img: image
        a two dimensional array of float32 or float64, 
        but can be uint16, uint8 or similar type
        
    sigma_1: float
        sigma value of the first gaussian low pass filter
        
    sigma_2: float
        sigma value of the second gaussian low pass filter
        
    clip: bool
        set negative pixels to zero
        
    Returns
    -------
    img: image
        a filtered two dimensional array of the input image
    
    """
    _high_pass = img - gaussian_filter(img, sigma_1)
    img_blur = gaussian_filter(_high_pass * _high_pass, sigma = sigma_2)
    den = np.sqrt(img_blur)
    img = np.divide( # stops image from being all black
        _high_pass, den,
        out = np.zeros_like(img),
        where = (den != 0.0)
    )    
    if clip:
        img[img < 0] = 0 
    img = (img - img.min()) / (img.max() - img.min())
    return img



def contrast_stretch(img, lower_limit = 2, upper_limit = 98):
    """
    Simple percentile-based contrast stretching
    
    Parameters
    ----------
    img: image
        a two dimensional array of float32 or float64, 
        but can be uint16, uint8 or similar type
        
    lower_limit: int
        lower percentile limit
        
    upper_limit: int
        upper percentile limit
        
    Returns
    -------
    img: image
        a filtered two dimensional array of the input image  
    
    """
    if lower_limit < 0:
        lower_limit = 0
    if upper_limit > 100:
        upper_limit = 100
                
    lower = np.percentile(img, lower_limit)
    upper = np.percentile(img, upper_limit)
    img = exposure.rescale_intensity(img, in_range = (lower, upper))
    return img

def threshold_binarize(img, threshold, max_val = 255):
    """
    Simple binarizing threshold
    
    Parameters
    ----------
    img: image
        a two dimensional array of float32 or float64, 
        but can be uint16, uint8 or similar type
        
    threshold: int or float
        boundary where pixels set lower than the threshold are set to zero
        and values higher than the threshold are set to the maximum user selected value
        
    max_val: int or float
        maximum pixel value of the image
        
    Returns
    -------
    img: image
        a filtered two dimensional array of the input image
    
    """
    img[img < threshold] = 0
    img[img > threshold] = max_val
    return img


def gen_min_background(img_list, resize = 255):
    """
    Generate a background by averaging the minimum intensity 
    of all images in an image list.
    Apply by subtracting generated background image.
    
    Parameters
    ----------
    img_list: list
        list of image directories
        
    resize: int or float
        disabled by default, normalize array and set value to user
        selected max pixel intensity
        
    Returns
    -------
    img: image
        a mean of all images
    
    """
    background = imread(img_list[0])
    if resize is not None:
        background = normalize_array(background) * resize
    for img in img_list: 
        if img == img_list: # the original image is already included, so skip it in the for loop
            pass
        else:
            img = imread(img)
            if resize is not None:
                img = normalize_array(img) * resize
            background = np.min(np.array([background, img]), axis = 0)
    return(background)


def gen_lowpass_background(img_list, sigma = 3, resize = None):
    """
    Generate a background by averaging a low pass of all images in an image list.
    Apply by subtracting generated background image.
    
    Parameters
    ----------
    img_list: list
        list of image directories
        
    sigma: float
        sigma of the gaussian filter
        
    resize: int or float
        disabled by default, normalize array and set value to user
        selected max pixel intensity
        
    Returns
    -------
    img: image
        a mean of all low-passed images
    
    """
    for img_file in img_list:
        if resize is not None:
            img = normalize_array(imread(img_file)) * resize
        else:
            img = imread(img_file)
        img = gaussian_filter(img, sigma = sigma)
        if img_file == img_list[0]:
            background = img
        else:
            background += img
    return (background / len(img_list))


def offset_image(img, offset_x, offset_y, pad = 'zero'):
    """
    Offset an image by padding.
    
    Parameters
    ----------
    img: image
        a two dimensional array of float32 or float64, 
        but can be uint16, uint8 or similar type
        
    offset_x: int
        offset an image by integer values. Positive values shifts 
        the image to the right and negative values shift to the left
        
    offset_y: int
        offset an image by integer values. Positive values shifts 
        the image to the top and negative values shift to the bottom
        
    pad: str
        pad the shift with zeros or a reflection of the shift
        
    Returns
    -------
    img: image
        a transformed two dimensional array of the input image
    
    """
    if pad not in [
        'zero', 'reflect'
    ]:
        raise ValueError(f'pad method not supported: {pad}')
    end_y, end_x = img.shape
    start_x = 0; start_y = 0
    if offset_x > 0:
        offset_x1 = offset_x
        offset_x2 = 0
    else:
        offset_x1 = 0
        offset_x2 = offset_x * -1
        start_x = offset_x2
        end_x += offset_x2
    if offset_y > 0:
        offset_y1 = offset_y
        offset_y2 = 0
    else:
        offset_y1 = 0
        offset_y2 = offset_y * -1
        start_y = offset_y2
        end_y += offset_y2   
    if pad == 'zero':
        pad = 'constant'
    img = np.pad(
        img,
        ((offset_y1, offset_y2),
        (offset_x1, offset_x2)),
        mode = pad
    )
    return img[start_y:end_y, start_x:end_x]


def stretch_image(img,
                  x_axis = 0,
                  y_axis = 0,
                 ):
    """
    Stretch an image by interplation.
    
    Parameters
    ----------
    img: image
        a two dimensional array of float32 or float64, 
        but can be uint16, uint8 or similar type
        
    x_axis: float
        stretch the x-axis of an image where 0 == no stretching
        
    y_axis: float
        stretch the y-axis of an image where 0 == no stretching
        
    Returns
    -------
    img: image
        a transformed two dimensional array of the input image  
    
    """
    y_axis += 1 # set so zero = no stretch
    x_axis += 1

    x_axis = max(x_axis, 1)
    y_axis = max(y_axis, 1)

    return rescale(img, (y_axis, x_axis))
