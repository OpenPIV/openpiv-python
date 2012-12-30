
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

from skimage import img_as_float
from scipy.ndimage import median_filter, gaussian_filter
from skimage.filter import threshold_otsu



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
	image = img_as_float(image)
	background = gaussian_filter(median_filter(image,3),1)
	image[background > threshold_otsu(background)/5.0] = 0.0
    
    return image

