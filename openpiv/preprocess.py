
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

from skimage import io, img_as_float, img_as_int, exposure, data
from skimage.util.dtype import dtype_range
from scipy.ndimage import median_filter, gaussian_filter, binary_fill_holes



def dynamic_masking(image):
    """ Dynamically masks out the objects in the PIV images
    
    Parameters
    ----------
    image: image
        a two dimensional array of uint16, uint8 or similar type
            
    Returns
    -------
    image : 2d np.ndarray of floats
        
    """
    image = exposure.rescale_intensity(image, in_range=(0, 1))
    blurback = gaussian_filter(median_filter(image,size=3),sigma=3)
    # create the boolean mask 
    bw = (blurback > .3).astype('bool')
    bw = binary_fill_holes(bw)
    image[bw] = 0.0    # mask out the white regions
    image -= blurback  # subtrack the blurred background
    # subtraction causes negative values, we need to rescale it back to 0-1 interval
    image = img_as_int(exposure.rescale_intensity(image,in_range=(0,1)))
    
    return image

