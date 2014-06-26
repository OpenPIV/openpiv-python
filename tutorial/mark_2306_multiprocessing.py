# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# Follow the example in tutorial 2: http://openpiv.readthedocs.org/en/latest/src/tutorial.html

# <codecell>

import sys, os
sys.path.append(os.path.abspath('/Users/alex/Documents/OpenPIV/openpiv-python/'))
import openpiv.tools
import openpiv.process
import openpiv.scaling
import openpiv.preprocess

import numpy as np
# import matplotlib.pyplot as plt

from skimage import io, img_as_float, exposure, data, img_as_uint
from skimage.filter import sobel
from skimage.util.dtype import dtype_range
from scipy.ndimage import median_filter, gaussian_filter, binary_fill_holes
from skimage.filter import rank, threshold_otsu
from skimage.morphology import disk

# <codecell>

def dynamic_masking(image,filter_size=7,threshold=0.005):
    """ Dynamically masks out the objects in the PIV images
    """
    imcopy = np.copy(image)
    image = exposure.rescale_intensity(img_as_float(image), in_range=(0, 1))
    blurback = gaussian_filter(image,filter_size)
    edges = sobel(blurback)
    blur_edges = gaussian_filter(edges,21)
    # create the boolean mask 
    bw = (blur_edges > threshold)
    bw = binary_fill_holes(bw)
    imcopy -= blurback
    imcopy[bw] = 0.0
    return imcopy #image

# <codecell>

def func( args ):
    """A function to process each image pair."""

    # this line is REQUIRED for multiprocessing to work
    # always use it in your custom function

    file_a, file_b, counter = args
    
    print file_a, file_b, counter

    # typical parameters:
    window_size = 32 #pixels
    overlap = 16 # pixels
    search_area_size = 64 # pixels 
    frame_rate = 40 # fps

    # read images into numpy arrays
    frame_a  = openpiv.tools.imread( file_a )
    frame_b  = openpiv.tools.imread( file_b )
    
    # %%timeit
    im_a  = io.imread( file_a )
    im_b  = io.imread( file_b )
    
    # let's crop the region of interest
    frame_a =  im_a[600:1600,800:1400]
    frame_b =  im_b[600:1600,800:1400]
    # mask 
    frame_a = dynamic_masking(frame_a)
    frame_b = dynamic_masking(frame_b)

    # process again with the masked images, for comparison# process once with the original images
    u, v, sig2noise = openpiv.process.extended_search_area_piv(
                                                           frame_a.astype(np.int32) , frame_b.astype(np.int32), 
                                                           window_size = window_size,
                                                           overlap = overlap, 
                                                           dt=1./frame_rate, 
                                                           search_area_size = search_area_size, 
                                                           sig2noise_method = 'peak2peak')
    x, y = openpiv.process.get_coordinates( image_size = frame_a.shape, window_size = window_size, overlap = overlap )
    u, v, mask = openpiv.validation.global_val( u, v, (-100.,100.),(-100.,100.))
    u, v, mask = openpiv.validation.sig2noise_val( u, v, sig2noise, threshold = 1.6 )
    # u, v = openpiv.filters.replace_outliers( u, v, method='localmean', max_iter=1, kernel_size=2)
    # x, y, u, v = openpiv.scaling.uniform(x, y, u, v, scaling_factor = 96.52 )
    # save to a file
    openpiv.tools.save(x, y, u, v, mask, '/Users/alex/Desktop/mark_23_06/5ppm_6hz/5ppm_6hz_%03d.txt' % counter, fmt='%8.7f', delimiter='\t')

# <codecell>

# '/Users/alex/Desktop/mark_23_06/5ppm_6hz/Camera1-0100.tif'
task = openpiv.tools.Multiprocesser( data_dir = '/Users/alex/Desktop/mark_23_06/5ppm_6hz/', pattern_a='Camera1-0*.tif')
task.run( func = func, n_cpus = 4 )


