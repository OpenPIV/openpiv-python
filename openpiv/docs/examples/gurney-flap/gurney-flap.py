#!/usr/bin/python

"""Another example of openpiv module usage, showing 
how to specify how many core to use and how to load custom 
processing parameters from a file."""

import os

# import utilities
import openpiv.tools



# define  a function which will be executed for each image pair. 
# Define here processing parameters

# this function should accept a tuple of three elements: the two image
# filenames and an index, identifying the image number.
def func( args ):
    """'exp1_%03d.txt' % counter
    """
    file_a, file_b, counter = args
    
    # read images into numpy arrays
    frame_a  = openpiv.tools.imread( file_a )
    frame_b  = openpiv.tools.imread( file_b )
        
    # process image pair with the purepython implementation
    u, v = openpiv.pyprocess.piv( frame_a, frame_b, window_size=32, overlap=16, dt=0.02, sig2noise_method = 'peak2peak', sig2noise_lim = 1.0  )
    
    # get window centers coordinates
    x, y = openpiv.pyprocess.get_coordinates( image_size=frame_a.shape, window_size=32, overlap=16 )
    
    # get flow field in dimensional units
    x, y, u, v = openpiv.scaling.uniform(x, y, u, v, scaling_factor = 1236.6 )
        
    # save to a file
    openpiv.tools.save(x, y, u, v, 'exp1_%03d.txt' % counter, fmt='%8.7f', delimiter='\t' )
    
    openpiv.tools.display( 'Done image pair number %d' % counter )
    


# process image in parallel
m = openpiv.tools.Multiprocesser( data_dir = '.', pattern_a='2image_*0.tif', pattern_b='2image_*1.tif' )
m.run( func = func, n_cpus= 6 )

# display one for the sake of test
for f in os.listdir('.'):
    if f.endswith('.txt'):
            openpiv.tools.display_vector_field(f)




