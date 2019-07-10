from openpiv import tools, scaling, process, validation, filters
import os
import numpy as np

def func( args ):
    """A function to process each image pair."""

    # this line is REQUIRED for multiprocessing to work
    # always use it in your custom function

    file_a, file_b, counter = args


    #####################
    # Here goes you code
    #####################

    # read images into numpy arrays
    frame_a  = tools.imread( os.path.join(path,file_a) )
    frame_b  = tools.imread( os.path.join(path,file_b) )

    frame_a = (frame_a*1024).astype(np.int32)
    frame_b = (frame_b*1024).astype(np.int32)


    # process image pair with extended search area piv algorithm.
    u, v, sig2noise = process.extended_search_area_piv( frame_a, frame_b, \
        window_size=64, overlap=32, dt=0.02, search_area_size=128, sig2noise_method='peak2peak')
    u, v, mask = validation.sig2noise_val( u, v, sig2noise, threshold = 1.5 )
    u, v = filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)
    # get window centers coordinates
    x, y = process.get_coordinates( image_size=frame_a.shape, window_size=64, overlap=32 )
    # save to a file
    tools.save(x, y, u, v, mask, 'test2_%03d.txt' % counter)
    tools.display_vector_field('test2_%03d.txt' % counter)

path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(path,'../test2/')
task = tools.Multiprocesser( data_dir = path, pattern_a='2image_*0.tif', pattern_b='2image_*1.tif' )
task.run( func = func, n_cpus=1 )


