# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# Follow the example in tutorial 2: http://openpiv.readthedocs.org/en/latest/src/tutorial.html

# <codecell>

import sys, os, glob
sys.path.append(os.path.abspath(r'D:\OpenPIV\OpenPIV-0.11'))

import openpiv.tools
import openpiv.process
import openpiv.scaling
import openpiv.preprocess

import numpy as np
# import matplotlib.pyplot as plt

def func( args ):
    """A function to process each image pair."""

    # this line is REQUIRED for multiprocessing to work
    # always use it in your custom function

    file_a, file_b, counter = args
    
    print file_a, file_b, counter

    filepath, filename = os.path.split(file_a)
    filedrive, filepath = os.path.splitdrive(filepath)
    filename = os.path.splitext(filename)
    filepath = os.path.join('d:',filepath)
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    filename = os.path.join(filepath,filename[0])
    
    # print filepath

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
    u, v, mask = openpiv.validation.global_val( u, v, (-300.,300.),(-300.,300.))
    # u, v, mask = openpiv.validation.sig2noise_val( u, v, sig2noise, threshold = 1.6 )
    # u, v = openpiv.filters.replace_outliers( u, v, method='localmean', max_iter=1, kernel_size=2)
    # x, y, u, v = openpiv.scaling.uniform(x, y, u, v, scaling_factor = 96.52 )
    # save to a file
    openpiv.tools.save(x, y, u, v, mask, filename+'_%05d.txt' % counter, fmt='%8.7f', delimiter='\t')

# <codecell>

# '/Users/alex/Desktop/mark_23_06/5ppm_6hz/Camera1-0100.tif'

if __name__ == '__main__':
	# listdir = glob.glob(r'Z:\Water_10PPM\6_86Hz\*')
	# for data_dir in listdir[3:]:
		# task = openpiv.tools.Multiprocesser( data_dir = data_dir, pattern_a=r'Camera1-*.tif', pattern_b = None)
		# task.run( func = func, n_cpus = 4 )

	# listdir = glob.glob(r'Z:\Water_10PPM\8_4Hz\*')
	# for data_dir in listdir[2:]:
		# task = openpiv.tools.Multiprocesser( data_dir = data_dir, pattern_a=r'Camera1-*.tif', pattern_b = None)
		# task.run( func = func, n_cpus = 4 )


	listdir = glob.glob(r'Z:\Water_10PPM\10_53Hz\*')
	for data_dir in listdir:
		task = openpiv.tools.Multiprocesser( data_dir = data_dir, pattern_a=r'Camera1-*.tif', pattern_b = None)
		task.run( func = func, n_cpus = 4 )


