"""Example of openpiv module usage"""

import openpiv.tools
import openpiv.process
import openpiv.scaling

frame_a  = openpiv.tools.imread( 'exp1_001_a.bmp' )
frame_b  = openpiv.tools.imread( 'exp1_001_b.bmp' )

# process it with Cython implementation
u, v, sig2noise = openpiv.process.extended_search_area_piv( frame_a, frame_b, window_size=24, overlap=12, dt=0.02, search_area_size=64 )


# or process image pair with the purepython implementation
# u, v = openpiv.pyprocess.piv( frame_a, frame_b, window_size=32, overlap=16, dt=0.02, corr_method = 'fft', sig2noise_method = 'peak2peak', sig2noise_lim = 1.0  )
    
# get window centers coordinates
x, y = openpiv.pyprocess.get_coordinates( image_size=frame_a.shape, window_size=32, overlap=16 )

u, v = openpiv.validation.sig2noise_val( u, v, sig2noise, threshold = 1.3 )

u, v = openpiv.filters.replace_outliers( u, v, method='localmean', n_iter=10, kernel_size=2)

x, y = openpiv.pyprocess.get_coordinates( image_size=frame_a.shape, window_size=24, overlap=12 )

x, y, u, v = openpiv.scaling.uniform(x, y, u, v, scaling_factor = 96.52 )

openpiv.tools.save(x, y, u, v, 'exp1_001.txt' )
    
openpiv.tools.display_vector_field( 'exp1_001.txt' )    
