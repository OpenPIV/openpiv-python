import sys
sys.path.append('/Users/alex/Documents/OpenPIV/alexlib/openpiv-python')

import openpiv.tools
#import openpiv.process
import openpiv.scaling
import numpy as np

frame_a  = openpiv.tools.imread( 'test001.png' )
frame_b  = openpiv.tools.imread( 'test002.png')


print frame_a.shape 
print frame_a.dtype


u, v, sig2noise = openpiv.process.extended_search_area_piv( 
                                                           frame_a.astype(np.int32), 
frame_b.astype(np.int32), window_size=24, overlap=12, dt=0.02, 
search_area_size=48, sig2noise_method='peak2peak' )

x, y = openpiv.process.get_coordinates( image_size=frame_a.shape, window_size=24, overlap=12 )


quiver(x,y,u,v)
show()

#u, v, mask = openpiv.validation.sig2noise_val( u, v, sig2noise, threshold = 100.0 )
#
#u, v = openpiv.filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)
##
#x, y, u, v = openpiv.scaling.uniform(x, y, u, v, scaling_factor = 96.52 )
#
#openpiv.tools.save(x, y, u, v, mask, 'exp1_001.txt' )
#
#openpiv.tools.display_vector_field('exp1_001.txt', scale=100, width=0.0025)

#
#
#u, v= openpiv.pyprocess.piv( frame_a, frame_b, corr_method='fft', window_size=24, overlap=12, dt=0.02, sig2noise_method='peak2peak' )
#x, y = openpiv.pyprocess.get_coordinates( image_size=frame_a.shape, window_size=24, overlap=12 )
#
#
#u, v, mask = openpiv.validation.sig2noise_val( u, v, sig2noise, threshold = 2.5 )
#
#u, v = openpiv.filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2.5)
#
#x, y, u, v = openpiv.scaling.uniform(x, y, u, v, scaling_factor = 96.52 )
#
#openpiv.tools.save(x, y, u, v, mask, 'exp1_002.txt' )
#
#openpiv.tools.display_vector_field('exp1_002.txt', scale=100, width=0.0025)



#
#u, v= openpiv.pyprocess.piv( frame_a, frame_b, corr_method='direct', window_size=24, overlap=12, dt=0.02, sig2noise_method='peak2peak' )
#x, y = openpiv.pyprocess.get_coordinates( image_size=frame_a.shape, window_size=24, overlap=12 )
#
#
#u, v, mask = openpiv.validation.sig2noise_val( u, v, sig2noise, threshold = 2.5 )
#
#u, v = openpiv.filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2.5)
#
#x, y, u, v = openpiv.scaling.uniform(x, y, u, v, scaling_factor = 96.52 )
#
#openpiv.tools.save(x, y, u, v, mask, 'exp1_002.txt' )
#
#openpiv.tools.display_vector_field('exp1_002.txt', scale=100, width=0.0025)