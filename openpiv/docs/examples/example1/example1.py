#!/usr/bin/python

"""Example of openpiv module usage"""

# import utilities
import openpiv


# read images into numpy arrays
basename = 'exp1_001'
frame_a  = openpiv.tools.imread( basename+'_a.bmp' )
frame_b  = openpiv.tools.imread( basename+'_b.bmp' )

# process image pair with the purepython implementation
u, v = openpiv.pyprocess.piv( frame_a, frame_b, window_size=32, overlap=16, dt=0.02, corr_method = 'fft', sig2noise_method = 'peak2peak', sig2noise_lim = 1.0  )
    
# get window centers coordinates
x, y = openpiv.pyprocess.get_coordinates( image_size=frame_a.shape, window_size=32, overlap=16 )

# get flow field in dimensional units
x, y, u, v = openpiv.scaling.uniform(x, y, u, v, scaling_factor = 1.0 )


# save to a file
openpiv.tools.save(x, y, u, v, basename+'.txt')

# display velocity field
openpiv.tools.display_vector_field(basename+'.txt')
