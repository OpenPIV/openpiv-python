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


from scipy.ndimage import median_filter, gaussian_filter, binary_fill_holes
from skimage import io, img_as_float, exposure, data, img_as_uint, img_as_ubyte
from skimage.filters import sobel, rank, threshold_otsu
import numpy as np


def dynamic_masking(image,method='edges',filter_size=7,threshold=0.005):
    """ Dynamically masks out the objects in the PIV images
    
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
    blurback = img_as_ubyte(gaussian_filter(image,filter_size))
    if method == 'edges':
        # identify edges
        edges = sobel(blurback)
        blur_edges = gaussian_filter(edges,21)
        # create the boolean mask 
        bw = (blur_edges > threshold)
        bw = img_as_ubyte(binary_fill_holes(bw))
        imcopy -= blurback
        imcopy[bw] = 0.0
    elif method == 'intensity':
        background = gaussian_filter(median_filter(image,filter_size),filter_size)
        imcopy[background > threshold_otsu(background)] = 0

        
    return imcopy #image
