#!/usr/bin/env ipython
import sys

if 'OpenPIV' not in sys.path:
    sys.path.append('/Users/alex/Documents/OpenPIV/alexlib/openpiv-python')

from openpiv import tools, validation, process, filters, scaling, pyprocess
import numpy as np

frame_a  = tools.imread( 'exp1_001_a.bmp' )
frame_b  = tools.imread( 'exp1_001_b.bmp' )

u, v, sig2noise = process.extended_search_area_piv( frame_a.astype(np.int32), 
frame_b.astype(np.int32), window_size=24, overlap=12, dt=0.02, search_area_size=64, 
sig2noise_method='peak2peak' )
x, y = process.get_coordinates( image_size=frame_a.shape, window_size=24, overlap=12 )
u, v, mask = validation.sig2noise_val( u, v, sig2noise, threshold = 2.5 )
u, v = filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)
x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor = 96.52 )
tools.save(x, y, u, v, mask, 'exp1_001.txt' )
tools.display_vector_field('exp1_001.txt', scale=100, width=0.0025)



u, v, s2n= pyprocess.piv(frame_a, frame_b, corr_method='fft', window_size=24, overlap=12, 
dt=0.02, sig2noise_method='peak2peak' )
x, y = pyprocess.get_coordinates( image_size=frame_a.shape, window_size=24, overlap=12 )
u, v, mask = validation.sig2noise_val( u, v, s2n, threshold = 2.5 )
u, v = filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2.5)
x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor = 96.52 )
tools.save(x, y, u, v, mask, 'exp1_002.txt' )
tools.display_vector_field('exp1_002.txt', scale=100, width=0.0025)




u,v,s2n = pyprocess.piv( frame_a, frame_b, corr_method='direct', window_size=24, 
overlap=12, dt=0.02, sig2noise_method='peak2peak' )
x, y = pyprocess.get_coordinates( image_size=frame_a.shape, window_size=24, overlap=12 )
u, v, mask = validation.sig2noise_val( u, v, sig2noise, threshold = 2.5 )
u, v = filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2.5)
x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor = 96.52 )
tools.save(x, y, u, v, mask, 'exp1_002.txt' )
tools.display_vector_field('exp1_002.txt', scale=100, width=0.0025)